from __future__ import annotations

import os
import sys

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from constants import DEFAULT_START_YEAR, DEFAULT_END_YEAR, DNF_POSITION_ORDER


def _as_round_set(skipped: dict | None) -> set:
    if not skipped:
        return set()
    rounds = set()
    for year, values in skipped.items():
        for r in (values or []):
            try:
                rounds.add((int(year), int(r)))
            except (TypeError, ValueError):
                pass
    return rounds


def run_quality_checks(
    engine: Engine,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    skipped_rounds: dict | None = None,
) -> list[dict]:
    """Run minimal quality checks and return failures."""
    checks = []

    def add_check(name: str, query: str, expected_zero: bool = True) -> None:
        checks.append({"name": name, "query": query, "expected_zero": expected_zero})

    for table in ("results", "drivers", "races"):
        add_check(f"{table}_non_empty", f"SELECT COUNT(*) AS value FROM {table}", expected_zero=False)

    add_check(
        "races_outside_year_range",
        "SELECT COUNT(*) AS value FROM races WHERE year < :start_year OR year > :end_year",
    )

    for table, pk in [("drivers", "driver_id"), ("constructors", "constructor_id"),
                      ("circuits", "circuit_id"), ("races", "race_id")]:
        add_check(f"{table}_unique", f"SELECT COUNT(*) - COUNT(DISTINCT {pk}) AS value FROM {table}")

    for child, fk, parent in [
        ("results",   "race_id",        "races"),
        ("results",   "driver_id",      "drivers"),
        ("results",   "constructor_id", "constructors"),
        ("qualifying", "race_id",        "races"),
        ("pit_stops", "race_id",        "races"),
    ]:
        label = fk.replace("_id", "")
        add_check(
            f"{child}_{label}_fk",
            f"SELECT COUNT(*) AS value FROM {child} c"
            f" LEFT JOIN {parent} p ON c.{fk} = p.{fk} WHERE p.{fk} IS NULL",
        )

    for col in ("points", "laps", "grid", "position_order"):
        add_check(f"results_{col}_non_negative", f"SELECT COUNT(*) AS value FROM results WHERE {col} < 0")
    add_check(
        "results_position_order_consistency",
        f"""
        SELECT COUNT(*) AS value FROM results
        WHERE position IS NOT NULL
          AND CAST(position AS INTEGER) BETWEEN 1 AND 20
          AND position_order = {DNF_POSITION_ORDER}
        """,
    )

    failures: list[dict] = []
    with engine.connect() as conn:
        for check in checks:
            result = conn.execute(
                text(check["query"]),
                {"start_year": start_year, "end_year": end_year},
            ).fetchone()
            value = result[0] if result else 0
            if check["expected_zero"]:
                if value != 0:
                    failures.append({
                        "check": check["name"],
                        "value": str(value),
                        "expected": "0",
                    })
            else:
                if value == 0:
                    failures.append({
                        "check": check["name"],
                        "value": str(value),
                        "expected": "> 0",
                    })

        try:
            year_rows = conn.execute(
                text(
                    "SELECT DISTINCT year FROM races WHERE year BETWEEN :start_year AND :end_year"
                ),
                {"start_year": start_year, "end_year": end_year},
            ).fetchall()
            present_years = {row[0] for row in year_rows}
            expected_years = set(range(start_year, end_year + 1))
            missing_years = sorted(expected_years - present_years)
            if missing_years:
                failures.append(
                    {
                        "check": "missing_race_years",
                        "value": ", ".join(str(y) for y in missing_years),
                        "expected": f"All years {start_year}-{end_year}",
                    }
                )
        except SQLAlchemyError as exc:
            failures.append(
                {
                    "check": "missing_race_years",
                    "value": f"error: {exc}",
                    "expected": "query_success",
                }
            )

        def _check_missing_data(
            conn, join_table: str, join_col: str, check_name: str, skipped_key: str
        ) -> None:
            sql = f"""
                SELECT ra.year, ra.round
                FROM races ra
                LEFT JOIN {join_table} t ON t.race_id = ra.race_id
                WHERE t.{join_col} IS NULL AND ra.year BETWEEN :start_year AND :end_year
            """
            try:
                rows = conn.execute(
                    text(sql), {"start_year": start_year, "end_year": end_year}
                ).fetchall()
                skipped = _as_round_set((skipped_rounds or {}).get(skipped_key))
                unaccounted = [(r[0], r[1]) for r in rows if (r[0], r[1]) not in skipped]
                if unaccounted:
                    failures.append({"check": check_name, "value": str(len(unaccounted)), "expected": "0"})
            except SQLAlchemyError as exc:
                failures.append({"check": check_name, "value": f"error: {exc}", "expected": "query_success"})

        _check_missing_data(conn, "results",   "race_id", "races_missing_results",   "results")
        _check_missing_data(conn, "qualifying", "race_id", "races_missing_qualifying", "qualifying")

        try:
            laps_row = conn.execute(text("SELECT COUNT(*) AS value FROM laps")).fetchone()
            laps_count = laps_row[0] if laps_row else 0
        except SQLAlchemyError:
            laps_count = 0

        if laps_count > 0:
            lap_checks = [
                {
                    "name": "laps_lap_time_bounds",
                    "query": "SELECT COUNT(*) AS value FROM laps WHERE lap_time_s IS NOT NULL AND (lap_time_s < 60 OR lap_time_s > 300)",
                    "expected_zero": True,
                },
                {
                    "name": "laps_invalid_compound",
                    "query": "SELECT COUNT(*) AS value FROM laps WHERE compound IS NOT NULL AND compound NOT IN ('SOFT','MEDIUM','HARD','INTERMEDIATE','WET','UNKNOWN')",
                    "expected_zero": True,
                },
                {
                    "name": "laps_driver_fk",
                    "query": "SELECT COUNT(*) AS value FROM laps WHERE driver_id NOT IN (SELECT driver_id FROM drivers)",
                    "expected_zero": True,
                },
                {
                    "name": "laps_race_fk",
                    "query": "SELECT COUNT(*) AS value FROM laps WHERE race_id NOT IN (SELECT race_id FROM races)",
                    "expected_zero": True,
                },
            ]
            for check in lap_checks:
                try:
                    result = conn.execute(text(check["query"])).fetchone()
                    value = result[0] if result else 0
                    if value != 0:
                        failures.append({"check": check["name"], "value": str(value), "expected": "0"})
                except SQLAlchemyError as exc:
                    failures.append({"check": check["name"], "value": f"error: {exc}", "expected": "query_success"})

    return failures
