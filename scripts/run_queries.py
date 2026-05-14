import os
import sys
import argparse

import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from logging_utils import setup_logging, format_table
from constants import CONSTRUCTOR_ID, TEAM_REFS
from load_data import _build_connection_string

try:
    from config import DB_CONFIG, DATA_PATHS
except ImportError:
    DB_CONFIG = {"type": "sqlite", "filename": "f1_analytics.db"}
    DATA_PATHS = {"processed_data": "data/processed/"}

logger = setup_logging()

_DEFAULT_QUERIES_FILE = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "database", "queries", "analytical_queries.yaml")
)


def create_db_connection(config: dict | None = None) -> Engine:
    return create_engine(_build_connection_string(config or DB_CONFIG))


def execute_query(
    engine: Engine, query_name: str, query_text: str, params: dict | None = None
) -> pd.DataFrame | None:
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query_text), conn, params=params)
        return df
    except Exception as e:
        logger.error("Error executing %s: %s", query_name, e)
        return None


def export_results(df: pd.DataFrame, filename: str, output_path: str | None = None) -> None:
    output_path = output_path or DATA_PATHS.get("processed_data", "data/processed/")
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)
    df.to_csv(filepath, index=False)
    logger.info("Exported results to %s.", filepath)


def load_queries_from_yaml(
    query_file: str = _DEFAULT_QUERIES_FILE,
) -> dict[str, str]:
    if not os.path.exists(query_file):
        logger.warning("Query file %s not found.", query_file)
        return {}
    with open(query_file, "r") as fh:
        return yaml.safe_load(fh) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run F1 analytical queries")
    parser.add_argument("--query", type=str, help='Query name to execute (or "all" for all queries)')
    parser.add_argument("--export", action="store_true", help="Export results to CSV")
    parser.add_argument("--list", action="store_true", help="List available queries")
    parser.add_argument(
        "--file",
        type=str,
        default=_DEFAULT_QUERIES_FILE,
        help="YAML file to load queries from",
    )

    args = parser.parse_args()
    engine = create_db_connection()
    ref_placeholders = ", ".join(f":r{i}" for i in range(len(TEAM_REFS)))
    ref_params = {f"r{i}": r for i, r in enumerate(TEAM_REFS)}
    params = {"cid": CONSTRUCTOR_ID, **ref_params}

    _is_duckdb = (DB_CONFIG or {}).get("type") == "duckdb"

    queries = load_queries_from_yaml(args.file)

    if args.list:
        logger.info("Available queries:")
        for query_name in sorted(queries.keys()):
            logger.info("  - %s", query_name)
        return

    if args.query:
        targets = list(queries.items()) if args.query == "all" else [(args.query, queries.get(args.query))]

        for query_name, query_text in targets:
            if not query_text:
                logger.warning("Query '%s' not found. Use --list to see available queries.", query_name)
                continue

            logger.info("Executing %s...", query_name)
            resolved = query_text.replace("{team_refs}", ref_placeholders)
            if _is_duckdb:
                resolved = resolved.replace(
                    "GROUP_CONCAT(DISTINCT ", "STRING_AGG(DISTINCT "
                ).replace(
                    "STRING_AGG(DISTINCT con.constructor_name)",
                    "STRING_AGG(DISTINCT con.constructor_name, ',')",
                )
            df = execute_query(engine, query_name, resolved, params=params)

            if df is not None and not df.empty:
                headers = list(df.columns)
                rows = df.values.tolist()
                right_cols = {i for i, col in enumerate(df.columns) if pd.api.types.is_numeric_dtype(df[col])}
                logger.info("\n%s", format_table(headers, rows, right_cols))
                if args.export:
                    export_results(df, f"{query_name}_results.csv")
            else:
                logger.warning("No results returned for %s.", query_name)
    else:
        logger.info("Use --query <name> or --query all. Use --list to see available queries.")


if __name__ == "__main__":
    main()
