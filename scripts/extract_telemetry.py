"""
FastF1 lap-by-lap telemetry extractor.
Writes sector times, tyre compound, and stint data to the laps table.
Runs after the main Ergast pipeline (needs race_id / driver_id from DB).

Usage:
    python scripts/extract_telemetry.py
    python scripts/extract_telemetry.py --start-year 2023 --end-year 2024
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict

import fastf1
import pandas as pd
import requests
from sqlalchemy import create_engine, text

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from analytics import ref_params
from constants import DEFAULT_START_YEAR, DEFAULT_END_YEAR, TEAM_REFS
from load_data import _build_connection_string, DB_CONFIG
from logging_utils import setup_logging

_log = logging.getLogger("f1_analytics")
_CACHE_DIR = os.path.join("data", "fastf1_cache")

_COLS = [
    "race_id", "driver_id", "lap_number",
    "lap_time_s", "sector1_s", "sector2_s", "sector3_s",
    "compound", "tyre_life", "stint",
    "is_personal_best", "pit_in", "pit_out", "track_status",
]


def _td_s(series: pd.Series) -> pd.Series:
    """Timedelta series → float seconds; NaT becomes NaN."""
    return pd.to_numeric(series.dt.total_seconds(), errors="coerce")


def _load_maps(
    engine, start_year: int, end_year: int
) -> tuple[pd.DataFrame, dict[str, int], dict[int, set[str]]]:
    """Return races_df, driver code→id map, race_id→set[driver_code] map."""
    placeholders, ref_p = ref_params(TEAM_REFS)
    with engine.connect() as conn:
        races = pd.read_sql(
            text("SELECT race_id, year, round FROM races WHERE year BETWEEN :y1 AND :y2 ORDER BY year, round"),
            conn, params={"y1": start_year, "y2": end_year},
        )
        drivers = pd.read_sql(
            text("SELECT driver_id, code FROM drivers WHERE code IS NOT NULL AND code != ''"),
            conn,
        )
        team_results = pd.read_sql(
            text(f"""
                SELECT res.race_id, d.code
                FROM results res
                JOIN drivers      d ON res.driver_id      = d.driver_id
                JOIN constructors c ON res.constructor_id = c.constructor_id
                WHERE c.constructor_ref IN ({placeholders})
                  AND d.code IS NOT NULL AND d.code != ''
            """),
            conn, params=ref_p,
        )

    driver_map: dict[str, int] = dict(zip(drivers["code"], drivers["driver_id"]))

    race_to_codes: dict[int, set[str]] = defaultdict(set)
    for row in team_results.itertuples(index=False):
        race_to_codes[int(row.race_id)].add(row.code)

    return races, driver_map, race_to_codes


def _session_laps(
    session,
    race_id: int,
    team_codes: set[str],
    driver_map: dict[str, int],
) -> pd.DataFrame:
    """Extract accurate laps for team drivers from a loaded FastF1 session."""
    raw = session.laps
    mask = raw["Driver"].isin(team_codes) & raw["IsAccurate"].fillna(False)
    raw = raw[mask].copy()
    if raw.empty:
        return pd.DataFrame(columns=_COLS)

    df = pd.DataFrame(index=raw.index)
    df["race_id"]          = race_id
    df["driver_id"]        = raw["Driver"].map(driver_map)
    df["lap_number"]       = raw["LapNumber"].astype(int)
    df["lap_time_s"]       = _td_s(raw["LapTime"])
    df["sector1_s"]        = _td_s(raw["Sector1Time"])
    df["sector2_s"]        = _td_s(raw["Sector2Time"])
    df["sector3_s"]        = _td_s(raw["Sector3Time"])
    df["compound"]         = raw["Compound"].str.upper().where(raw["Compound"].notna())
    df["tyre_life"]        = pd.to_numeric(raw["TyreLife"], errors="coerce").astype("Int64")
    df["stint"]            = pd.to_numeric(raw["Stint"],    errors="coerce").astype("Int64")
    df["is_personal_best"] = raw["IsPersonalBest"].fillna(False).astype(int)
    df["pit_in"]           = raw["PitInTime"].notna().astype(int)
    df["pit_out"]          = raw["PitOutTime"].notna().astype(int)
    df["track_status"]     = raw["TrackStatus"].astype(str)

    return df.dropna(subset=["driver_id", "lap_time_s"])[_COLS].reset_index(drop=True)


def extract_all(
    engine,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> int:
    """Extract FastF1 lap data for team drivers → laps table. Returns total rows written."""
    fastf1.Cache.enable_cache(_CACHE_DIR)
    os.makedirs(_CACHE_DIR, exist_ok=True)

    races, driver_map, race_to_codes = _load_maps(engine, start_year, end_year)

    with engine.begin() as conn:
        conn.execute(text(
            "DELETE FROM laps WHERE race_id IN "
            "(SELECT race_id FROM races WHERE year BETWEEN :y1 AND :y2)"
        ), {"y1": start_year, "y2": end_year})

    total = 0
    for race in races.itertuples(index=False):
        year, rnd, race_id = int(race.year), int(race.round), int(race.race_id)
        team_codes = race_to_codes.get(race_id, set())
        if not team_codes:
            continue

        try:
            session = fastf1.get_session(year, rnd, "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            df = _session_laps(session, race_id, team_codes, driver_map)
            if df.empty:
                _log.info("  %s R%02d — no accurate laps found.", year, rnd)
                continue
            df.to_sql("laps", engine, if_exists="append", index=False)
            total += len(df)
            _log.info("  %s R%02d — %s laps loaded.", year, rnd, len(df))
        except requests.exceptions.RequestException as exc:
            _log.warning("  %s R%02d — network error, skipped: %s.", year, rnd, exc)
        except (KeyError, ValueError, TypeError) as exc:
            _log.warning("  %s R%02d — data error, skipped: %s.", year, rnd, exc)
        except Exception as exc:
            _log.warning("  %s R%02d — unexpected error, skipped: %s.", year, rnd, exc)

    _log.info("Telemetry extraction complete: %s total laps.", total)
    return total


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Extract FastF1 lap telemetry into laps table")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year",   type=int, default=DEFAULT_END_YEAR)
    args = parser.parse_args()

    engine = create_engine(_build_connection_string(DB_CONFIG))
    extract_all(engine, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
