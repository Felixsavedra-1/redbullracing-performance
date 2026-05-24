import argparse
import json
import logging
import os
import subprocess
import sys

import pandas as pd
from sqlalchemy import text

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from logging_utils import setup_logging, format_table
from constants import DEFAULT_START_YEAR, DEFAULT_END_YEAR, TEAM_NAME, TEAM_REFS
from extract_data import F1DataExtractor
from transform_data import F1DataTransformer
from load_data import F1DataLoader, DB_CONFIG
from data_quality import run_quality_checks
from extract_telemetry import extract_all as extract_telemetry


def _load_skipped(name: str) -> dict:
    for path in (os.path.join("data", "cache", name), os.path.join("data", "raw", name)):
        if not os.path.exists(path):
            continue
        try:
            with open(path) as handle:
                data = json.load(handle)
            return data.get("skipped", {}) if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError, ValueError):
            return {}
    return {}


_DRIVER_SUMMARY_SQL = """
SELECT
    COALESCE(d.forename, '') || ' ' || COALESCE(d.surname, '') AS driver,
    STRING_AGG(DISTINCT con.constructor_name, ',') AS team,
    COUNT(*) AS races,
    SUM(res.points) AS points,
    COUNT(*) FILTER (WHERE res.position = 1)  AS wins,
    COUNT(*) FILTER (WHERE res.position <= 3) AS podiums,
    ROUND(AVG(res.position_order) FILTER (WHERE res.position_order < 999), 1) AS avg_finish,
    COUNT(*) FILTER (WHERE res.position_order = 999) AS dnfs,
    MIN(r.year) AS from_yr,
    MAX(r.year) AS to_yr
FROM results res
JOIN races        r   ON res.race_id        = r.race_id
JOIN drivers      d   ON res.driver_id      = d.driver_id
JOIN constructors con ON res.constructor_id = con.constructor_id
WHERE con.constructor_ref IN ({team_refs})
GROUP BY d.driver_id, d.forename, d.surname
ORDER BY points DESC
"""

_DRIVER_SUMMARY_SQL_SQLITE = """
SELECT
    COALESCE(d.forename, '') || ' ' || COALESCE(d.surname, '') AS driver,
    GROUP_CONCAT(DISTINCT con.constructor_name) AS team,
    COUNT(*) AS races,
    SUM(res.points) AS points,
    COUNT(CASE WHEN res.position = 1  THEN 1 END) AS wins,
    COUNT(CASE WHEN res.position <= 3 THEN 1 END) AS podiums,
    ROUND(AVG(CASE WHEN res.position_order < 999 THEN res.position_order END), 1) AS avg_finish,
    COUNT(CASE WHEN res.position_order = 999 THEN 1 END) AS dnfs,
    MIN(r.year) AS from_yr,
    MAX(r.year) AS to_yr
FROM results res
JOIN races        r   ON res.race_id        = r.race_id
JOIN drivers      d   ON res.driver_id      = d.driver_id
JOIN constructors con ON res.constructor_id = con.constructor_id
WHERE con.constructor_ref IN ({team_refs})
GROUP BY d.driver_id, d.forename, d.surname
ORDER BY points DESC
"""

_TEAM_SHORT = {
    "Oracle Red Bull Racing": "Red Bull",
    "Red Bull Racing":        "Red Bull",
}


def _shorten_teams(raw: str) -> str:
    parts = [_TEAM_SHORT.get(t.strip(), t.strip()) for t in raw.split(",")]
    return " · ".join(dict.fromkeys(parts))


def _print_driver_summary(engine) -> None:
    _log = logging.getLogger("f1_analytics")
    try:
        placeholders = ", ".join(f":r{i}" for i in range(len(TEAM_REFS)))
        params = {f"r{i}": r for i, r in enumerate(TEAM_REFS)}
        template = _DRIVER_SUMMARY_SQL if (DB_CONFIG or {}).get("type") == "duckdb" else _DRIVER_SUMMARY_SQL_SQLITE
        sql = template.format(team_refs=placeholders)
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
        if df.empty:
            return

        df["team"]   = df["team"].apply(_shorten_teams)
        df["period"] = df.apply(
            lambda r: f"{int(r.from_yr)}–{int(r.to_yr) % 100:02d}", axis=1
        )
        df["points"] = df["points"].astype(int)
        year_range = f"{int(df['from_yr'].min())}–{int(df['to_yr'].max())}"
        df = df[["driver", "team", "period", "races", "points",
                 "wins", "podiums", "avg_finish", "dnfs"]]

        headers = ["Driver", "Team", "Period", "Races", "Pts", "Wins", "Pods", "Avg", "DNFs"]
        rows = df.values.tolist()
        right_cols = {3, 4, 5, 6, 7, 8}

        rule = "─" * 70
        print(f"\n{rule}")
        print(f"  {TEAM_NAME} Drivers  {year_range}")
        print(rule)
        print(format_table(headers, rows, right_cols))
        print(f"{rule}\n")
    except Exception as e:
        _log.warning("driver summary skipped: %s", e)


def _normalize_year_range(start_year: int, end_year: int) -> tuple[int, int, bool]:
    clamped = False
    if start_year < DEFAULT_START_YEAR:
        start_year = DEFAULT_START_YEAR
        clamped = True
    if end_year > DEFAULT_END_YEAR:
        end_year = DEFAULT_END_YEAR
        clamped = True
    return start_year, end_year, clamped


def _dry_run_preview() -> None:
    _log = logging.getLogger("f1_analytics")
    _log.info("DRY RUN — extract and transform complete. Would load:")
    headers = ["Table", "Rows", "Status"]
    rows = []
    for table_name, csv_name, *_ in F1DataLoader._TABLE_SPECS:
        if csv_name.startswith("raw:"):
            path = os.path.join("data", "raw", csv_name[4:])
        else:
            path = os.path.join("data", "processed", csv_name)
        if not os.path.exists(path):
            rows.append([table_name, "—", "missing"])
            continue
        try:
            with open(path) as fh:
                count = max(0, sum(1 for _ in fh) - 1)
            rows.append([table_name, str(count), "ready"])
        except OSError:
            rows.append([table_name, "—", "error"])
    print(format_table(headers, rows, {1}))
    _log.info("Re-run without --dry-run to load the above into the database.")


def run_full_pipeline(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    skip_extract: bool = False,
    skip_transform: bool = False,
    skip_load: bool = False,
    skip_pit_stops: bool = False,
    skip_quality: bool = False,
    skip_dbt: bool = False,
    include_telemetry: bool = False,
    dry_run: bool = False,
    mode: str = "full_refresh",
    strict_schema: bool = True,
    base_delay: float = 1.5,
    max_retries: int = 6,
    max_base_delay: float = 8.0,
) -> None:
    """Run extraction, transformation, and loading for the requested year range."""

    logger = setup_logging()

    start_year, end_year, clamped = _normalize_year_range(start_year, end_year)
    if start_year > end_year:
        raise ValueError(
            f"Invalid year range after clamping to {DEFAULT_START_YEAR}-{DEFAULT_END_YEAR}."
        )

    logger.info("F1 Red Bull Analytics - Complete Pipeline")
    if clamped:
        logger.warning(
            "Year range clamped to %s-%s for this project scope.",
            DEFAULT_START_YEAR,
            DEFAULT_END_YEAR,
        )

    if not skip_extract:
        logger.info("[1/3] EXTRACTING DATA FROM API")
        extractor = F1DataExtractor(
            output_path="data/raw/",
            base_delay=base_delay,
            max_retries=max_retries,
            max_base_delay=max_base_delay,
        )
        extractor.extract_all(
            start_year=start_year,
            end_year=end_year,
            skip_pit_stops=skip_pit_stops,
        )
    else:
        logger.info("[1/3] SKIPPING EXTRACTION (--skip-extract flag)")

    if not skip_transform:
        logger.info("[2/3] TRANSFORMING DATA")
        transformer = F1DataTransformer(
            raw_data_path="data/raw/",
            processed_data_path="data/processed/",
        )
        transformer.transform_all()
    else:
        logger.info("[2/3] SKIPPING TRANSFORMATION (--skip-transform flag)")

    if dry_run:
        _dry_run_preview()
        return

    if not skip_load:
        logger.info("[3/3] LOADING DATA INTO DATABASE")
        loader = F1DataLoader(
            mode=mode,
            strict_schema=strict_schema,
            source_url=F1DataExtractor.BASE_URL,
        )
        loader.load_all()

        _print_driver_summary(loader.engine)

        if not skip_quality:
            skipped_rounds = {
                "results": _load_skipped("results_progress.json"),
                "qualifying": _load_skipped("qualifying_progress.json"),
            }

            failures = run_quality_checks(
                loader.engine,
                start_year=start_year,
                end_year=end_year,
                skipped_rounds=skipped_rounds,
            )
            if failures:
                logger.error("Data quality checks failed: %s", failures)
                raise RuntimeError("Data quality checks failed")
            else:
                logger.info("Data quality checks passed")

        if not skip_dbt:
            dbt_dir = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "dbt"))
            if os.path.isdir(dbt_dir):
                logger.info("BUILDING dbt ANALYTICAL MODELS")
                result = subprocess.run(
                    ["dbt", "run", "--profiles-dir", dbt_dir, "--project-dir", dbt_dir],
                )
                if result.returncode != 0:
                    raise RuntimeError("dbt run failed")
                logger.info("dbt models built successfully.")
    else:
        logger.info("[3/3] SKIPPING DATABASE LOAD (--skip-load flag)")

    if include_telemetry and not skip_load:
        logger.info("[4/4] EXTRACTING FASTF1 LAP TELEMETRY")
        try:
            extract_telemetry(loader.engine, start_year=start_year, end_year=end_year)
        except Exception:
            logger.exception("Telemetry extraction failed — pipeline continues without lap data.")

    logger.info("Pipeline completed successfully.")
    logger.info("Next steps:")
    logger.info("  - Run queries: python scripts/run_queries.py --list")
    logger.info("  - Export results: python scripts/run_queries.py --query kpi_summary --export")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete F1 Red Bull Analytics pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/run_pipeline.py

  # Run only extraction
  python scripts/run_pipeline.py --skip-transform --skip-load

  # Run only transformation and loading (skip extraction)
  python scripts/run_pipeline.py --skip-extract

  # Custom year range
  python scripts/run_pipeline.py --start-year 2010 --end-year 2023

  # Incremental load
  python scripts/run_pipeline.py --incremental

  # Fast demo run
  python scripts/run_pipeline.py --fast
        """,
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help=f"Start year for data extraction (default: {DEFAULT_START_YEAR})",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        help=f"End year for data extraction (default: {DEFAULT_END_YEAR})",
    )
    parser.add_argument("--skip-extract", action="store_true", help="Skip data extraction step")
    parser.add_argument("--skip-transform", action="store_true", help="Skip data transformation step")
    parser.add_argument("--skip-load", action="store_true", help="Skip database loading step")
    parser.add_argument("--skip-pit-stops", action="store_true", help="Skip pit stop extraction")
    parser.add_argument("--skip-quality", action="store_true", help="Skip data quality checks")
    parser.add_argument("--skip-dbt", action="store_true", help="Skip dbt analytical model build")
    parser.add_argument("--incremental", action="store_true", help="Use incremental load instead of full refresh")
    parser.add_argument("--no-strict-schema", action="store_true", help="Do not fail on schema contract warnings")
    parser.add_argument("--telemetry", action="store_true",
                        help="Extract FastF1 lap telemetry after load (sector times, tyre data)")
    parser.add_argument("--base-delay", type=float, default=1.5, help="Delay between API requests in seconds")
    parser.add_argument("--max-retries", type=int, default=6, help="Max retries on API errors or rate limits")
    parser.add_argument("--max-base-delay", type=float, default=8.0, help="Upper bound for adaptive delay")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a faster demo extraction (2021–2025, reduced retries/backoff)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and transform data but skip database load — prints row counts per table",
    )

    args = parser.parse_args()

    mode = "incremental" if args.incremental else "full_refresh"
    strict_schema = not args.no_strict_schema

    if args.fast:
        args.start_year = max(args.start_year, 2021)
        args.base_delay = min(args.base_delay, 0.3)
        args.max_retries = min(args.max_retries, 3)
        args.max_base_delay = min(args.max_base_delay, 2.0)
        args.skip_pit_stops = True

    try:
        run_full_pipeline(
            start_year=args.start_year,
            end_year=args.end_year,
            skip_extract=args.skip_extract,
            skip_transform=args.skip_transform,
            skip_load=args.skip_load,
            skip_pit_stops=args.skip_pit_stops,
            skip_quality=args.skip_quality,
            skip_dbt=args.skip_dbt,
            include_telemetry=args.telemetry,
            dry_run=args.dry_run,
            mode=mode,
            strict_schema=strict_schema,
            base_delay=args.base_delay,
            max_retries=args.max_retries,
            max_base_delay=args.max_base_delay,
        )
    except KeyboardInterrupt:
        logging.getLogger("f1_analytics").warning("Pipeline interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger = setup_logging()
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
