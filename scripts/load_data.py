import logging
import os
import sys
import uuid
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, event, inspect as sa_inspect, text

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from logging_utils import setup_logging
from schema_contracts import validate_dataframe, SCHEMA_CONTRACTS
from constants import CONSTRUCTOR_ID

_log = logging.getLogger("f1_analytics")

try:
    from config import DB_CONFIG, DATA_PATHS
except ImportError:
    _log.warning("config.py not found; using SQLite defaults. Copy scripts/config.example.py to scripts/config.py.")
    DB_CONFIG = {
        "type": "sqlite",
        "filename": "f1_analytics.db",
    }
    DATA_PATHS = {
        "processed_data": "data/processed/",
    }


def _build_connection_string(config: dict) -> str:
    if config.get("type") == "sqlite":
        return f"sqlite:///{config.get('filename', 'f1_analytics.db')}"
    if config.get("type") == "duckdb":
        return f"duckdb:///{config.get('filename', 'f1_analytics.duckdb')}"
    return (
        f"mysql+pymysql://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['database']}"
    )


class F1DataLoader:
    def __init__(
        self,
        config=None,
        processed_data_path=None,
        mode: str = "full_refresh",
        strict_schema: bool = True,
        run_id: str | None = None,
        source_url: str | None = None,
    ):
        self.config = config or DB_CONFIG
        self.processed_path = processed_data_path or DATA_PATHS.get("processed_data", "data/processed/")
        self.raw_path = os.path.normpath(os.path.join(self.processed_path, "..", "raw"))
        self.engine = None
        self.mode = mode
        self.strict_schema = strict_schema
        self.run_id = run_id or str(uuid.uuid4())
        self.source_url = source_url
        self.logger = logging.getLogger("f1_analytics")
        self._rb_driver_ids: set = set()
        self._connect()
        self._ensure_metadata_tables()

    def _connect(self) -> None:
        try:
            db_type = self.config.get("type")
            if db_type in ("sqlite", "duckdb"):
                db_file = self.config.get("filename", "f1_analytics.duckdb")
                if self.mode == "full_refresh" and os.path.exists(db_file):
                    bak_file = db_file + ".bak"
                    os.replace(db_file, bak_file)
                    self.logger.warning("Full refresh: renamed %s → %s.", db_file, bak_file)
                if not os.path.isabs(db_file) and "/" in db_file:
                    os.makedirs(os.path.dirname(db_file), exist_ok=True)
                self.logger.info("Connecting to %s database at %s.", db_type, db_file)
            else:
                self.logger.info("Connecting to MySQL at %s.", self.config.get("host"))

            self.engine = create_engine(_build_connection_string(self.config))

            if db_type == "sqlite":
                # Enable FK enforcement on every new connection — SQLite disables it by default.
                @event.listens_for(self.engine, "connect")
                def set_fk_pragma(dbapi_conn, _):
                    dbapi_conn.execute("PRAGMA foreign_keys=ON")

            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            self._apply_schema()

            if self._is_file_db and self.mode == "incremental":
                self._check_schema_drift()

            self.logger.info("Database connection established.")

        except Exception as exc:
            self.logger.error("Error connecting to database: %s", exc)
            self.logger.error("Check your database configuration in scripts/config.py.")
            raise

    @property
    def _is_file_db(self) -> bool:
        return self.config.get("type") in ("sqlite", "duckdb")

    def _apply_schema_file(self, filename: str) -> None:
        schema_path = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "database", "schema", filename))
        try:
            with open(schema_path, "r") as handle:
                schema_sql = handle.read()
        except FileNotFoundError:
            self.logger.warning("Schema file not found: %s", schema_path)
            return
        with self.engine.connect() as conn:
            for statement in schema_sql.split(";"):
                stmt = statement.strip()
                if stmt:
                    conn.execute(text(stmt))
            conn.commit()

    def _apply_schema(self) -> None:
        db_type = self.config.get("type")
        filenames = {"sqlite": "create_tables_sqlite.sql", "duckdb": "create_tables_duckdb.sql"}
        if db_type in filenames:
            self._apply_schema_file(filenames[db_type])

    def _check_schema_drift(self) -> None:
        inspector = sa_inspect(self.engine)
        existing_tables = set(inspector.get_table_names())
        table_cols = {t: {c["name"] for c in inspector.get_columns(t)} for t in existing_tables}
        for table_name, _, cols, _, _ in self._TABLE_SPECS:
            if table_name not in existing_tables:
                self.logger.warning(
                    "Schema drift: table '%s' is missing. Run without --incremental to recreate.",
                    table_name,
                )
                continue
            if not cols:
                continue
            missing = [c for c in cols if c not in table_cols[table_name]]
            if missing:
                self.logger.warning(
                    "Schema drift: '%s' is missing columns %s. Run without --incremental to apply changes.",
                    table_name, missing,
                )

    def _ensure_metadata_tables(self) -> None:
        if self._is_file_db:
            return

        with self.engine.connect() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS pipeline_runs (
                        run_id VARCHAR(36) PRIMARY KEY,
                        started_at DATETIME,
                        ended_at DATETIME,
                        status VARCHAR(20),
                        source_url VARCHAR(255),
                        mode VARCHAR(20)
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS pipeline_run_tables (
                        run_id VARCHAR(36),
                        table_name VARCHAR(50),
                        rows_loaded INT,
                        PRIMARY KEY (run_id, table_name)
                    )
                    """
                )
            )
            conn.commit()

    def _record_run_start(self) -> None:
        started_at = datetime.now(timezone.utc).isoformat()
        with self.engine.connect() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO pipeline_runs (run_id, started_at, status, source_url, mode)
                    VALUES (:run_id, :started_at, :status, :source_url, :mode)
                    """
                ),
                {
                    "run_id": self.run_id,
                    "started_at": started_at,
                    "status": "running",
                    "source_url": self.source_url,
                    "mode": self.mode,
                },
            )
            conn.commit()

    def _record_run_end(self, status: str) -> None:
        ended_at = datetime.now(timezone.utc).isoformat()
        with self.engine.connect() as conn:
            conn.execute(
                text(
                    """
                    UPDATE pipeline_runs
                    SET ended_at = :ended_at, status = :status
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": self.run_id, "ended_at": ended_at, "status": status},
            )
            conn.commit()

    def _record_table_load(self, table_name: str, rows: int) -> None:
        if self._is_file_db:
            upsert_sql = (
                """
                INSERT INTO pipeline_run_tables (run_id, table_name, rows_loaded)
                VALUES (:run_id, :table_name, :rows_loaded)
                ON CONFLICT(run_id, table_name)
                DO UPDATE SET rows_loaded = excluded.rows_loaded
                """
            )
        else:
            upsert_sql = (
                """
                INSERT INTO pipeline_run_tables (run_id, table_name, rows_loaded)
                VALUES (:run_id, :table_name, :rows_loaded)
                ON DUPLICATE KEY UPDATE rows_loaded = VALUES(rows_loaded)
                """
            )

        with self.engine.connect() as conn:
            conn.execute(
                text(upsert_sql),
                {"run_id": self.run_id, "table_name": table_name, "rows_loaded": rows},
            )
            conn.commit()

    def _quote(self, identifier: str) -> str:
        return f'"{identifier}"' if self._is_file_db else f"`{identifier}`"

    def _validate_df(self, df: pd.DataFrame, table_name: str) -> None:
        issues = validate_dataframe(table_name, df)
        if issues:
            message = f"Schema validation issues for {table_name}: {issues}"
            if self.strict_schema:
                raise ValueError(message)
            self.logger.warning(message)

    def _coerce_df(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        contract = SCHEMA_CONTRACTS.get(table_name)
        if not contract or df.empty:
            return df

        for col in contract.get("string", []):
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        for col in contract.get("numeric", []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in contract.get("datetime", []):
            if col in df.columns:
                before = int(df[col].notna().sum())
                df[col] = pd.to_datetime(df[col], errors="coerce")
                lost = before - int(df[col].notna().sum())
                if lost:
                    self.logger.warning("Coerced %s invalid values to NaT in %s.%s.", lost, table_name, col)

        return df

    def _load_table_full_refresh(self, df: pd.DataFrame, table_name: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(f"DELETE FROM {self._quote(table_name)}"))
            df.to_sql(table_name, conn, if_exists="append", index=False)

    def _load_table_incremental(self, df: pd.DataFrame, table_name: str) -> None:
        staging_table = f"_stg_{table_name}"

        columns = [self._quote(col) for col in df.columns]
        column_list = ", ".join(columns)
        select_list = ", ".join(columns)

        if self._is_file_db:
            upsert_sql = (
                f"INSERT OR REPLACE INTO {table_name} ({column_list}) "
                f"SELECT {select_list} FROM {staging_table}"
            )
        else:
            update_clause = ", ".join([f"{self._quote(col)}=VALUES({self._quote(col)})" for col in df.columns])
            upsert_sql = (
                f"INSERT INTO {table_name} ({column_list}) "
                f"SELECT {select_list} FROM {staging_table} "
                f"ON DUPLICATE KEY UPDATE {update_clause}"
            )

        try:
            with self.engine.begin() as conn:
                df.to_sql(staging_table, conn, if_exists="replace", index=False)
                conn.execute(text(upsert_sql))
        finally:
            with self.engine.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {staging_table}"))

    def _load_table(self, df: pd.DataFrame, table_name: str) -> None:
        if df.empty:
            self.logger.info("Skipping %s: no rows to load.", table_name)
            return

        df = self._coerce_df(df, table_name)
        self._validate_df(df, table_name)

        try:
            if self.mode == "incremental":
                self._load_table_incremental(df, table_name)
            else:
                self._load_table_full_refresh(df, table_name)
            self.logger.info("Loaded %s rows into %s.", len(df), table_name)
            self._record_table_load(table_name, len(df))
        except Exception as exc:
            self.logger.error("Error loading %s: %s", table_name, exc)
            raise

    # Each spec: (table_name, csv_path, columns_to_keep, datetime_cols, fillna_defaults)
    # csv_path is relative to processed_path unless prefixed with "raw:"
    _TABLE_SPECS = [
        ("seasons",               "raw:seasons.csv",                 None,  [], {}),
        ("circuits",              "circuits_clean.csv",              None,  [], {}),
        ("constructors",          "raw:constructors.csv",            None,  [], {}),
        ("drivers",               "drivers_clean.csv",               None,  ["dob"], {}),
        ("races",                 "races_clean.csv",
            ["race_id", "year", "round", "circuit_id", "race_name", "race_date", "race_time", "url"],
            ["race_date"], {"race_time": "00:00:00"}),
        ("results",               "results_clean.csv",
            ["race_id", "driver_id", "constructor_id", "number", "grid", "position",
             "position_text", "position_order", "points", "laps", "time_result",
             "milliseconds", "fastest_lap", "fastest_lap_rank", "fastest_lap_time",
             "fastest_lap_speed", "status"],
            [], {}),
        ("qualifying",            "qualifying_clean.csv",
            ["race_id", "driver_id", "constructor_id", "number", "position", "q1", "q2", "q3"],
            [], {}),
        ("pit_stops",             "pit_stops_clean.csv",
            ["race_id", "driver_id", "stop", "lap", "time_of_day", "duration", "milliseconds"],
            [], {"time_of_day": "00:00:00"}),
        ("constructor_standings", "constructor_standings_clean.csv",
            ["race_id", "constructor_id", "points", "position", "position_text", "wins"],
            [], {}),
        ("driver_standings",      "driver_standings_clean.csv",
            ["race_id", "driver_id", "points", "position", "position_text", "wins"],
            [], {}),
    ]

    def _filter_team(self, df: pd.DataFrame, table: str) -> pd.DataFrame:
        if "constructor_id" in df.columns:
            df = df[df["constructor_id"] == CONSTRUCTOR_ID].copy()
            if table == "results":
                self._rb_driver_ids = set(df["driver_id"].dropna().astype(int))
            return df
        if table in ("pit_stops", "driver_standings") and self._rb_driver_ids:
            return df[df["driver_id"].isin(self._rb_driver_ids)].copy()
        return df

    def _load_from_spec(self, table: str, csv_name: str, cols, datetime_cols, fillna_defaults) -> None:
        if csv_name.startswith("raw:"):
            path = os.path.join(self.raw_path, csv_name[4:])
        else:
            path = os.path.join(self.processed_path, csv_name)

        if not os.path.exists(path) or os.path.getsize(path) < 10:
            self.logger.info("Skipping %s: file missing or empty.", table)
            return

        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            self.logger.warning("%s has no columns; skipping load.", csv_name)
            return

        df = self._filter_team(df, table)

        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        for col, default in fillna_defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)
        if cols:
            df = df[[c for c in cols if c in df.columns]]

        self._load_table(df, table)

    def load_all(self) -> None:
        self.logger.info("Starting data loading into database.")
        self._record_run_start()

        try:
            for spec in self._TABLE_SPECS:
                self.logger.info("Loading %s...", spec[0])
                self._load_from_spec(*spec)

            self._record_run_end("success")
            self.logger.info("All data loaded successfully into database.")

        except Exception:
            self._record_run_end("failed")
            self.logger.exception("Error during loading.")
            raise


def main() -> None:
    loader = F1DataLoader()
    loader.load_all()


if __name__ == "__main__":
    main()
