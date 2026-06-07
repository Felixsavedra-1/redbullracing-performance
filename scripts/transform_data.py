"""Clean raw F1 CSVs and resolve ``*_ref`` string keys to integer ``*_id`` surrogate keys for load."""

import os
import sys
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from logging_utils import setup_logging
from constants import DNF_POSITION_ORDER


class F1DataTransformer:
    def __init__(self, raw_data_path: str = "data/raw/", processed_data_path: str = "data/processed/") -> None:
        self.raw_path = raw_data_path
        self.processed_path = processed_data_path
        self.logger = setup_logging()
        os.makedirs(raw_data_path, exist_ok=True)
        os.makedirs(processed_data_path, exist_ok=True)

    def _read_csv_safe(self, filename: str) -> pd.DataFrame | None:
        """Return a DataFrame, or None with a warning if the file is missing or empty."""
        path = os.path.join(self.raw_path, filename)
        if not os.path.exists(path) or os.path.getsize(path) < 10:
            self.logger.warning("%s is missing or empty.", filename)
            return None
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            self.logger.warning("%s has no parseable data.", filename)
            return None

    def _load_ref_map(self, filename: str, ref_col: str, id_col: str) -> dict:
        df = pd.read_csv(os.path.join(self.raw_path, filename))
        if id_col not in df.columns:
            df[id_col] = range(1, len(df) + 1)
        return dict(zip(df[ref_col], df[id_col]))

    def _apply_ref_map(self, df: pd.DataFrame, ref_col: str, id_col: str, filename: str) -> pd.DataFrame:
        """Map ref strings to integer IDs; drops rows that cannot be mapped."""
        try:
            ref_map = self._load_ref_map(filename, ref_col, id_col)
        except (FileNotFoundError, KeyError):
            self.logger.warning("Ref map file %s not found; dropping all %s rows.", filename, ref_col)
            return df.iloc[0:0].copy()

        mapped = df[ref_col].map(ref_map)
        unmapped_count = int(mapped.isna().sum())
        if unmapped_count:
            dropout_pct = 100.0 * unmapped_count / len(df)
            msg = "Dropping %s rows (%.1f%%) with unmapped %s values (not in %s)."
            if dropout_pct > 5.0:
                self.logger.error(msg, unmapped_count, dropout_pct, ref_col, filename)
            else:
                self.logger.warning(msg, unmapped_count, dropout_pct, ref_col, filename)
            df = df[mapped.notna()].copy()
            mapped = mapped[mapped.notna()]

        df[id_col] = mapped.astype(int)
        return df

    def _coerce_datetime(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        before = int(df[col].notna().sum())
        df[col] = pd.to_datetime(df[col], errors="coerce")
        lost = before - int(df[col].notna().sum())
        if lost:
            self.logger.warning("Coerced %s invalid values to NaT in column '%s'.", lost, col)
        return df

    def transform_circuits(self) -> pd.DataFrame:
        df = self._read_csv_safe("circuits.csv")
        if df is None:
            empty = pd.DataFrame()
            empty.to_csv(f"{self.processed_path}circuits_clean.csv", index=False)
            return empty

        df["circuit_id"] = range(1, len(df) + 1)
        # Leave altitude as NaN when missing — 0 is valid (sea level) and must stay distinguishable.
        df = df[[
            "circuit_id",
            "circuit_ref",
            "circuit_name",
            "location",
            "country",
            "lat",
            "lng",
            "altitude",
            "url",
        ]]
        df.to_csv(f"{self.processed_path}circuits_clean.csv", index=False)
        self.logger.info("Transformed %s circuits.", len(df))
        return df

    def transform_drivers(self) -> pd.DataFrame:
        df = self._read_csv_safe("drivers.csv")
        if df is None:
            empty = pd.DataFrame()
            empty.to_csv(f"{self.processed_path}drivers_clean.csv", index=False)
            return empty

        if "driver_id" not in df.columns:
            df["driver_id"] = range(1, len(df) + 1)

        if "dob" in df.columns:
            df = self._coerce_datetime(df, "dob")

        if "driver_number" in df.columns:
            df["driver_number"] = df["driver_number"].fillna(0).astype(int)
        else:
            df["driver_number"] = 0

        required_cols = [
            "driver_id",
            "driver_ref",
            "driver_number",
            "code",
            "forename",
            "surname",
            "dob",
            "nationality",
            "url",
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = "" if col in {"code", "url"} else None

        df = df[required_cols]
        df.to_csv(f"{self.processed_path}drivers_clean.csv", index=False)
        self.logger.info("Transformed %s drivers.", len(df))
        return df

    def transform_races(self) -> pd.DataFrame:
        df = self._read_csv_safe("races.csv")
        if df is None:
            empty = pd.DataFrame()
            empty.to_csv(f"{self.processed_path}races_clean.csv", index=False)
            return empty

        if "race_date" in df.columns:
            df = self._coerce_datetime(df, "race_date")

        if "race_time" in df.columns:
            df["race_time"] = df["race_time"].fillna("00:00:00")
        else:
            df["race_time"] = "00:00:00"

        if "circuit_ref" in df.columns:
            df = self._apply_ref_map(df, "circuit_ref", "circuit_id", "circuits.csv")

        if "race_id" in df.columns:
            df["race_id"] = df["race_id"].astype(int)
        else:
            df["race_id"] = (
                df["year"].astype(str) + df["round"].astype(str).str.zfill(2)
            ).astype(int)

        required_cols = [
            "race_id",
            "year",
            "round",
            "circuit_id",
            "race_name",
            "race_date",
            "race_time",
            "url",
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        df = df[required_cols]

        df.to_csv(f"{self.processed_path}races_clean.csv", index=False)
        self.logger.info("Transformed %s races.", len(df))
        return df

    def transform_results(self) -> pd.DataFrame:
        results_columns = [
            "race_id", "driver_ref", "constructor_ref", "number", "grid",
            "position", "position_text", "position_order", "points", "laps",
            "time_result", "milliseconds", "fastest_lap", "fastest_lap_rank",
            "fastest_lap_time", "fastest_lap_speed", "status",
        ]
        df = self._read_csv_safe("results.csv")
        if df is None:
            empty = pd.DataFrame(columns=results_columns)
            empty.to_csv(f"{self.processed_path}results_clean.csv", index=False)
            return empty

        df = self._apply_ref_map(df, "driver_ref", "driver_id", "drivers.csv")
        df = self._apply_ref_map(df, "constructor_ref", "constructor_id", "constructors.csv")

        if "position" in df.columns:
            df["position"] = pd.to_numeric(df["position"], errors="coerce")

        if "position_text" in df.columns:
            df["position_text"] = df["position_text"].fillna("").astype(str)
        else:
            df["position_text"] = ""

        for col in ["points", "laps", "number", "fastest_lap", "fastest_lap_rank"]:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        # grid=0 is a valid F1 value (pit-lane start); preserve NULL for missing data.
        if "grid" in df.columns:
            df["grid"] = pd.to_numeric(df["grid"], errors="coerce")

        if "fastest_lap_speed" in df.columns:
            df["fastest_lap_speed"] = df["fastest_lap_speed"].fillna("").astype(str)
        else:
            df["fastest_lap_speed"] = ""

        if "milliseconds" in df.columns:
            df["milliseconds"] = pd.to_numeric(df["milliseconds"], errors="coerce").fillna(0).astype(int)
        else:
            df["milliseconds"] = 0

        if "status" not in df.columns:
            df["status"] = ""

        if "position_order" not in df.columns:
            # Derive from position as fallback. Non-numeric position (R=retired, D/DQ=disqualified)
            # all map to the same sentinel; use the `status` column to distinguish DNF vs DSQ downstream.
            df["position_order"] = df["position"].fillna(DNF_POSITION_ORDER)
        df["position_order"] = pd.to_numeric(
            df["position_order"], errors="coerce"
        ).fillna(DNF_POSITION_ORDER).astype(int)

        df.to_csv(f"{self.processed_path}results_clean.csv", index=False)
        self.logger.info("Transformed %s results.", len(df))
        return df

    def transform_qualifying(self) -> pd.DataFrame:
        qualifying_columns = [
            "race_id", "driver_ref", "constructor_ref", "number", "position", "q1", "q2", "q3",
        ]
        df = self._read_csv_safe("qualifying.csv")
        if df is None:
            empty = pd.DataFrame(columns=qualifying_columns)
            empty.to_csv(f"{self.processed_path}qualifying_clean.csv", index=False)
            return empty

        df = self._apply_ref_map(df, "driver_ref", "driver_id", "drivers.csv")
        df = self._apply_ref_map(df, "constructor_ref", "constructor_id", "constructors.csv")

        for col in ["q1", "q2", "q3"]:
            if col in df.columns:
                df[col] = df[col].fillna("")
            else:
                df[col] = ""

        for col in ["position", "number"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        df.to_csv(f"{self.processed_path}qualifying_clean.csv", index=False)
        self.logger.info("Transformed %s qualifying results.", len(df))
        return df

    def transform_pit_stops(self) -> pd.DataFrame:
        df = self._read_csv_safe("pit_stops.csv")
        if df is None:
            empty = pd.DataFrame()
            empty.to_csv(f"{self.processed_path}pit_stops_clean.csv", index=False)
            return empty

        df = self._apply_ref_map(df, "driver_ref", "driver_id", "drivers.csv")

        if "time_of_day" in df.columns:
            df["time_of_day"] = df["time_of_day"].fillna("00:00:00")
        else:
            df["time_of_day"] = "00:00:00"

        if "milliseconds" not in df.columns or df["milliseconds"].isna().all():
            if "duration" in df.columns:
                df["milliseconds"] = pd.to_numeric(df["duration"], errors="coerce") * 1000
            else:
                df["milliseconds"] = 0

        df["milliseconds"] = pd.to_numeric(df["milliseconds"], errors="coerce").fillna(0).astype(int)

        df.to_csv(f"{self.processed_path}pit_stops_clean.csv", index=False)
        self.logger.info("Transformed %s pit stops.", len(df))
        return df

    def _transform_standings_df(
        self, filename: str, ref_col: str, id_col: str, ref_filename: str, out_filename: str
    ) -> pd.DataFrame:
        df = self._read_csv_safe(filename)
        if df is None or df.empty:
            self.logger.info("No %s to transform.", filename)
            return pd.DataFrame()
        df = self._apply_ref_map(df, ref_col, id_col, ref_filename)
        df["points"] = df["points"].fillna(0)
        df["wins"] = df["wins"].fillna(0)
        df.to_csv(f"{self.processed_path}{out_filename}", index=False)
        self.logger.info("Transformed %s %s.", len(df), filename)
        return df

    def transform_standings(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_const = self._transform_standings_df(
            "constructor_standings.csv", "constructor_ref", "constructor_id",
            "constructors.csv", "constructor_standings_clean.csv",
        )
        df_driver = self._transform_standings_df(
            "driver_standings.csv", "driver_ref", "driver_id",
            "drivers.csv", "driver_standings_clean.csv",
        )
        return df_const, df_driver

    def transform_all(self) -> None:
        self.logger.info("Starting data transformation.")
        try:
            self.transform_circuits()
            self.transform_drivers()
            self.transform_races()
            self.transform_results()
            self.transform_qualifying()
            self.transform_pit_stops()
            self.transform_standings()
            self.logger.info("All transformations completed.")
            self.logger.info("Cleaned data written to: %s", self.processed_path)
        except Exception:
            self.logger.exception("Error during transformation.")
            raise


def main() -> None:
    transformer = F1DataTransformer()
    transformer.transform_all()


if __name__ == "__main__":
    main()
