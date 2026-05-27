import argparse
import json
import os
import random
import sys
import time
from collections.abc import Callable, Iterator
from datetime import datetime

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from logging_utils import setup_logging
from constants import DEFAULT_START_YEAR, DEFAULT_END_YEAR, DNF_POSITION_ORDER


def _safe_float(val) -> float | None:
    try:
        return float(val) if val else None
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    try:
        return int(val) if val else None
    except (ValueError, TypeError):
        return None


class F1DataExtractor:
    BASE_URL = "https://api.jolpi.ca/ergast/f1"

    def __init__(
        self,
        output_path: str = "data/raw/",
        base_delay: float = 1.5,
        max_retries: int = 6,
        max_backoff: float = 20.0,
        max_base_delay: float = 8.0,
        timeout: int = 30,
        circuit_breaker_limit: int | None = 50,
    ):
        self.output_path = output_path
        base_dir = os.path.dirname(os.path.normpath(output_path)) or "."
        self.cache_path = os.path.join(base_dir, "cache")
        self.base_delay = base_delay
        self.max_retries = max_retries
        self.max_backoff = max_backoff
        self.max_base_delay = max_base_delay
        self.timeout = timeout
        self.circuit_breaker_limit = circuit_breaker_limit
        self.session = requests.Session()
        self._last_request_ts = 0.0
        self._consecutive_rate_limits = 0
        self._total_rate_limits = 0
        self.logger = setup_logging()
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(self.cache_path, exist_ok=True)

    def _backoff(self, attempt: int, retry_after: str | None) -> None:
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = self.base_delay
        else:
            delay = min(self.base_delay * (2 ** attempt), self.max_backoff)
        if retry_after:
            self.base_delay = min(max(self.base_delay, delay), self.max_base_delay)
        else:
            self.base_delay = min(self.base_delay * 1.25, self.max_base_delay)
        jitter = random.uniform(0, min(0.25, self.base_delay))
        wait_for = delay + jitter
        self.logger.info("Backoff %.1fs (base_delay=%.2fs)", wait_for, self.base_delay)
        time.sleep(wait_for)
    
    def _rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self._last_request_ts
        if elapsed < self.base_delay:
            time.sleep(self.base_delay - elapsed)
        self._last_request_ts = time.time()

    def _get_total(self, json_data: dict | None) -> int:
        if not json_data:
            return 0
        try:
            total = json_data.get("MRData", {}).get("total")
            return int(total) if total is not None else 0
        except (TypeError, ValueError):
            return 0

    def _parse_duration_ms(self, duration: str) -> int | None:
        if not duration:
            return None
        value = duration.strip()
        try:
            return int(float(value) * 1000)
        except ValueError:
            pass

        try:
            parts = value.split(":")
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return int((minutes * 60 + seconds) * 1000)
        except ValueError:
            pass
        return None

    def _output_file_empty(self, filename: str) -> bool:
        path = os.path.join(self.output_path, filename)
        return not os.path.exists(path) or os.path.getsize(path) < 10

    def _output_has_rows(self, filename: str) -> bool:
        path = os.path.join(self.output_path, filename)
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r") as handle:
                handle.readline()
                return bool(handle.readline())
        except OSError:
            return False

    def _count_rows(self, filename: str) -> int:
        path = os.path.join(self.output_path, filename)
        if not os.path.exists(path):
            return 0
        try:
            with open(path, "r") as handle:
                count = -1
                for count, _ in enumerate(handle):
                    pass
                return max(count, 0)
        except OSError:
            return 0

    def _write_csv_atomic(self, df: pd.DataFrame, filename: str) -> None:
        path = os.path.join(self.output_path, filename)
        tmp_path = f"{path}.tmp"
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
        if os.path.getsize(path) < 10:
            self.logger.warning("%s was written but appears empty.", filename)

    def _make_request(self, endpoint: str, limit: int = 1000, offset: int = 0) -> dict | None:
        url = f"{self.BASE_URL}/{endpoint}.json?limit={limit}&offset={offset}"

        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    self._consecutive_rate_limits += 1
                    self._total_rate_limits += 1
                    if self.circuit_breaker_limit and self._total_rate_limits >= self.circuit_breaker_limit:
                        raise RuntimeError(
                            f"Circuit breaker: {self._total_rate_limits} rate limit responses received. "
                            "Increase --base-delay or try again later."
                        )
                    self.logger.info(
                        "Rate limited on %s (total: %s); retrying...",
                        endpoint, self._total_rate_limits,
                    )
                    if self._consecutive_rate_limits >= 3:
                        self.base_delay = min(self.base_delay * 1.5, self.max_base_delay)
                    self._backoff(attempt, retry_after)
                    continue
                if response.status_code in {400, 404}:
                    self.logger.warning("Invalid request for %s: %s. Skipping.", endpoint, response.status_code)
                    return None
                response.raise_for_status()
                self._consecutive_rate_limits = 0
                return response.json()
            except requests.exceptions.RequestException as e:
                self.logger.warning("Error fetching %s: %s", endpoint, e)
                self._backoff(attempt, None)

        self.logger.error("Failed to fetch %s after %s retries.", endpoint, self.max_retries)
        return None
    
    def _extract_table(self, json_data: dict, table_name: str) -> list[dict]:
        if not json_data or 'MRData' not in json_data:
            return []
        
        mr_data = json_data['MRData']
        table_data = mr_data.get(table_name)
        if not table_data:
            for key in mr_data.keys():
                if key.endswith('Table'):
                    table_data = mr_data[key]
                    break
        if not table_data:
            return []
        
        for key, value in table_data.items():
            if isinstance(value, list):
                return value

        return []
    
    def _paginate(
        self, endpoint: str, table_key: str, limit: int = 100
    ) -> Iterator[list]:
        offset = 0
        while True:
            data = self._make_request(endpoint, limit=limit, offset=offset)
            if not data:
                break
            records = self._extract_table(data, table_key) or []
            if not records:
                break
            yield records
            if len(records) < limit:
                break
            offset += limit

    def extract_circuits(self) -> pd.DataFrame:
        self.logger.info("Extracting circuits...")
        all_circuits = []
        for circuits in self._paginate("circuits", "CircuitsTable"):
            for circuit in circuits:
                location = circuit.get('Location', {})
                all_circuits.append({
                    'circuit_ref': circuit.get('circuitId', ''),
                    'circuit_name': circuit.get('circuitName', ''),
                    'location': location.get('locality', ''),
                    'country': location.get('country', ''),
                    'lat': _safe_float(location.get('lat')),
                    'lng': _safe_float(location.get('long')),
                    'altitude': None,
                    'url': circuit.get('url', ''),
                })
        df = pd.DataFrame(all_circuits)
        self._write_csv_atomic(df, "circuits.csv")
        self.logger.info("Extracted %s circuits.", len(df))
        return df
    
    def extract_seasons(self) -> pd.DataFrame:
        self.logger.info("Extracting seasons...")
        all_seasons = []
        for seasons in self._paginate("seasons", "SeasonTable"):
            for season in seasons:
                all_seasons.append({
                    "year": int(season.get("season", 0)),
                    "url": season.get("url", ""),
                })
        df = pd.DataFrame(all_seasons)
        self._write_csv_atomic(df, "seasons.csv")
        self.logger.info("Extracted %s seasons.", len(df))
        return df
    
    def extract_constructors(self) -> pd.DataFrame:
        self.logger.info("Extracting constructors...")
        all_constructors = []
        for constructors in self._paginate("constructors", "ConstructorTable"):
            for constructor in constructors:
                all_constructors.append({
                    'constructor_ref': constructor.get('constructorId', ''),
                    'constructor_name': constructor.get('name', ''),
                    'nationality': constructor.get('nationality', ''),
                    'url': constructor.get('url', ''),
                })
        df = pd.DataFrame(all_constructors)
        df.insert(0, "constructor_id", range(1, len(df) + 1))
        self._write_csv_atomic(df, "constructors.csv")
        self.logger.info("Extracted %s constructors.", len(df))
        return df
    
    def extract_drivers(self) -> pd.DataFrame:
        self.logger.info("Extracting drivers...")
        all_drivers = []
        for drivers in self._paginate("drivers", "DriverTable"):
            for driver in drivers:
                dob = driver.get('dateOfBirth', '')
                all_drivers.append({
                    'driver_ref': driver.get('driverId', ''),
                    'driver_number': None,
                    'code': driver.get('code', ''),
                    'forename': driver.get('givenName', ''),
                    'surname': driver.get('familyName', ''),
                    'dob': dob if dob else None,
                    'nationality': driver.get('nationality', ''),
                    'url': driver.get('url', ''),
                })
        df = pd.DataFrame(all_drivers)
        df.insert(0, "driver_id", range(1, len(df) + 1))
        self._write_csv_atomic(df, "drivers.csv")
        self.logger.info("Extracted %s drivers.", len(df))
        return df
    
    def extract_races(
        self, start_year: int = DEFAULT_START_YEAR, end_year: int = DEFAULT_END_YEAR
    ) -> pd.DataFrame:
        self.logger.info("Extracting races (%s-%s)...", start_year, end_year)
        all_races = []
        for year in range(start_year, end_year + 1):
            for races in self._paginate(f"{year}/races", "RaceTable"):
                for race in races:
                    circuit = race.get('Circuit', {})
                    all_races.append({
                        'year': year,
                        'round': int(race.get('round', 0)),
                        'race_id': int(f"{year}{int(race.get('round', 0)):02d}"),
                        'circuit_ref': circuit.get('circuitId', ''),
                        'race_name': race.get('raceName', ''),
                        'race_date': race.get('date', ''),
                        'race_time': race.get('time', '00:00:00Z').replace('Z', ''),
                        'url': race.get('url', ''),
                    })
        df = pd.DataFrame(all_races)
        self._write_csv_atomic(df, "races.csv")
        self.logger.info("Extracted %s races.", len(df))
        return df
    
    def _extract_per_round(
        self,
        start_year: int,
        end_year: int,
        rounds_by_year: dict[int, list[int]],
        endpoint: str,
        progress_file: str,
        output_file: str,
        parse_race: Callable,  # (race_dict, race_id) -> list[dict]
        label: str,
        min_rows_per_race: int = 0,
    ) -> list[dict]:
        rows: list[dict] = []
        progress = self._load_progress(progress_file, start_year, end_year)
        races_count = sum(len(rounds_by_year.get(y) or []) for y in range(start_year, end_year + 1))

        if self._output_file_empty(output_file) or not self._output_has_rows(output_file):
            self.logger.warning("%s is missing or empty; rebuilding extraction state.", output_file)
            progress = {"years": {}, "skipped": {}}
        elif min_rows_per_race and races_count:
            row_count = self._count_rows(output_file)
            if row_count < races_count * min_rows_per_race:
                self.logger.warning(
                    "%s looks incomplete (%s rows for %s races); rebuilding.",
                    output_file, row_count, races_count,
                )
                progress = {"years": {}, "skipped": {}}

        for year in range(start_year, end_year + 1):
            rounds = rounds_by_year.get(year) or list(range(1, 25))
            progress_years = progress.get("years", {})
            progress_skipped = progress.get("skipped", {})
            done_rounds = set(progress_years.get(str(year), [])) | set(progress_skipped.get(str(year), []))
            total = len(rounds)

            for round_num in rounds:
                if round_num in done_rounds:
                    continue
                self.logger.info("%s %s R%s/%s", label, year, round_num, total)
                data = self._make_request(endpoint.format(year=year, round=round_num), limit=1000, offset=0)
                races = self._extract_table(data, "RaceTable") if data else []

                if not races:
                    progress_skipped.setdefault(str(year), [])
                    progress_skipped[str(year)] = sorted(set(progress_skipped[str(year)] + [round_num]))
                else:
                    for race in races:
                        race_id = int(f"{year}{int(race.get('round', round_num)):02d}")
                        rows.extend(parse_race(race, race_id))
                    progress_years.setdefault(str(year), [])
                    progress_years[str(year)] = sorted(set(progress_years[str(year)] + [round_num]))

                self._save_progress(progress_file, progress, start_year, end_year)

        return rows

    def _parse_results_race(self, race: dict, race_id: int) -> list[dict]:
        results = race.get("Results", [])
        if not isinstance(results, list):
            results = [results]
        rows = []
        for result in results:
            driver = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            fastest_lap = result.get("FastestLap", {})
            position = result.get("position", "")
            rows.append({
                "race_id": race_id,
                "driver_ref": driver.get("driverId", ""),
                "constructor_ref": constructor.get("constructorId", ""),
                "number": _safe_int(result.get("number")),
                "grid": _safe_int(result.get("grid")),
                "position": int(position) if position.isdigit() else None,
                "position_text": result.get("positionText", ""),
                "position_order": int(position) if position.isdigit() else DNF_POSITION_ORDER,
                "points": float(result.get("points", 0)),
                "laps": int(result.get("laps", 0)) if result.get("laps") else None,
                "time_result": result.get("Time", {}).get("time", "") if result.get("Time") else None,
                "milliseconds": int(result["Time"]["millis"]) if result.get("Time", {}).get("millis") else None,
                "fastest_lap": int(fastest_lap.get("lap", 0)) if fastest_lap.get("lap") else None,
                "fastest_lap_rank": int(fastest_lap.get("rank", 0)) if fastest_lap.get("rank") else None,
                "fastest_lap_time": fastest_lap.get("Time", {}).get("time", "") if fastest_lap.get("Time") else None,
                "fastest_lap_speed": fastest_lap.get("AverageSpeed", {}).get("speed", "") if fastest_lap.get("AverageSpeed") else None,
                "status": result.get("status", "Finished"),
            })
        return rows

    def _parse_qualifying_race(self, race: dict, race_id: int) -> list[dict]:
        qualifying_results = race.get("QualifyingResults", [])
        if not isinstance(qualifying_results, list):
            qualifying_results = [qualifying_results]
        rows = []
        for q in qualifying_results:
            driver = q.get("Driver", {})
            constructor = q.get("Constructor", {})
            rows.append({
                "race_id": race_id,
                "driver_ref": driver.get("driverId", ""),
                "constructor_ref": constructor.get("constructorId", ""),
                "number": int(q.get("number", 0)) if q.get("number") else None,
                "position": int(q.get("position", 0)) if q.get("position") else None,
                "q1": q.get("Q1", ""),
                "q2": q.get("Q2", ""),
                "q3": q.get("Q3", ""),
            })
        return rows

    def _parse_pit_stops_race(self, race: dict, race_id: int) -> list[dict]:
        pit_stops = race.get("PitStops", [])
        if not isinstance(pit_stops, list):
            pit_stops = [pit_stops]
        rows = []
        for pit_stop in pit_stops:
            duration = pit_stop.get("duration", "")
            rows.append({
                "race_id": race_id,
                "driver_ref": pit_stop.get("driverId", ""),
                "stop": int(pit_stop.get("stop", 0)),
                "lap": int(pit_stop.get("lap", 0)),
                "time_of_day": pit_stop.get("time", ""),
                "duration": duration,
                "milliseconds": self._parse_duration_ms(duration),
            })
        return rows

    def extract_results(self, start_year: int = DEFAULT_START_YEAR, end_year: int = DEFAULT_END_YEAR) -> pd.DataFrame:
        self.logger.info("Extracting results (%s-%s)...", start_year, end_year)
        rounds_by_year = self._get_rounds_by_year(start_year, end_year)
        rows = self._extract_per_round(
            start_year, end_year, rounds_by_year,
            endpoint="{year}/{round}/results",
            progress_file="results_progress.json",
            output_file="results.csv",
            parse_race=self._parse_results_race,
            label="Results",
            min_rows_per_race=10,
        )
        df = pd.DataFrame(rows)
        self._write_csv_atomic(df, "results.csv")
        self.logger.info("Extracted %s results.", len(df))
        return df

    def extract_qualifying(self, start_year: int = DEFAULT_START_YEAR, end_year: int = DEFAULT_END_YEAR) -> pd.DataFrame:
        self.logger.info("Extracting qualifying (%s-%s)...", start_year, end_year)
        rounds_by_year = self._get_rounds_by_year(start_year, end_year)
        rows = self._extract_per_round(
            start_year, end_year, rounds_by_year,
            endpoint="{year}/{round}/qualifying",
            progress_file="qualifying_progress.json",
            output_file="qualifying.csv",
            parse_race=self._parse_qualifying_race,
            label="Qualifying",
            min_rows_per_race=10,
        )
        qualifying_columns = ["race_id", "driver_ref", "constructor_ref", "number", "position", "q1", "q2", "q3"]
        df = pd.DataFrame(rows, columns=qualifying_columns)
        self._write_csv_atomic(df, "qualifying.csv")
        self.logger.info("Extracted %s qualifying results.", len(df))
        return df
    
    def _normalize_progress(
        self, data: dict, start_year: int, end_year: int
    ) -> dict[str, dict[str, list[int]]]:
        years = data.get("years", data) if isinstance(data, dict) else {}
        skipped = data.get("skipped", {}) if isinstance(data, dict) else {}

        def clean(rounds) -> list[int]:
            if not isinstance(rounds, list):
                return []
            return sorted({int(r) for r in rounds if str(r).isdigit()})

        return {
            "years":   {str(y): clean(years.get(str(y), []))   for y in range(start_year, end_year + 1)},
            "skipped": {str(y): clean(skipped.get(str(y), [])) for y in range(start_year, end_year + 1)},
        }

    def _progress_path(self, filename: str) -> str:
        return os.path.join(self.cache_path, filename)

    def _legacy_progress_path(self, filename: str) -> str:
        return os.path.join(self.output_path, filename)

    def _load_progress(self, filename: str, start_year: int, end_year: int) -> dict[str, dict[str, list[int]]]:
        path = self._progress_path(filename)
        legacy_path = self._legacy_progress_path(filename)
        try_paths = [path, legacy_path]
        for candidate in try_paths:
            if not os.path.exists(candidate):
                continue
            try:
                with open(candidate, "r") as handle:
                    data = json.load(handle)
                normalized = self._normalize_progress(data, start_year, end_year)
                if candidate == legacy_path and not os.path.exists(path):
                    self._save_progress(filename, normalized, start_year, end_year)
                return normalized
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                self.logger.warning(
                    "Progress file %s is corrupt or unreadable (%s); re-extracting from scratch.",
                    candidate, exc,
                )
                return {"years": {}, "skipped": {}}
        return {"years": {}, "skipped": {}}

    def _save_progress(
        self,
        filename: str,
        data: dict[str, dict[str, list[int]]],
        start_year: int,
        end_year: int,
    ) -> None:
        path = self._progress_path(filename)
        normalized = self._normalize_progress(data, start_year, end_year)
        payload = {
            "version": 1,
            "updated_at": datetime.now().strftime("%Y-%m-%d"),
            "years": normalized.get("years", {}),
            "skipped": normalized.get("skipped", {}),
        }
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, path)

    def extract_pit_stops(self, start_year: int = DEFAULT_START_YEAR, end_year: int = DEFAULT_END_YEAR) -> pd.DataFrame:
        # Pit stop data is only available from 2012 onward in the Ergast API.
        self.logger.info("Extracting pit stops (%s-%s)...", start_year, end_year)
        rounds_by_year = self._get_rounds_by_year(start_year, end_year)
        rows = self._extract_per_round(
            start_year, end_year, rounds_by_year,
            endpoint="{year}/{round}/pitstops",
            progress_file="pit_stops_progress.json",
            output_file="pit_stops.csv",
            parse_race=self._parse_pit_stops_race,
            label="Pit stops",
        )
        pit_stop_columns = ["race_id", "driver_ref", "stop", "lap", "time_of_day", "duration", "milliseconds"]
        df = pd.DataFrame(rows, columns=pit_stop_columns)
        self._write_csv_atomic(df, "pit_stops.csv")
        self.logger.info("Extracted %s pit stops.", len(df))
        return df
    
    def _get_rounds_by_year(self, start_year: int, end_year: int) -> dict[int, list[int]]:
        races_path = os.path.join(self.output_path, "races.csv")
        if not os.path.exists(races_path):
            return {}
        try:
            races_df = pd.read_csv(races_path)
            rounds_by_year: dict[int, list[int]] = {}
            for year in range(start_year, end_year + 1):
                year_rounds = races_df[races_df["year"] == year]["round"].dropna().astype(int).tolist()
                rounds_by_year[year] = sorted(set(year_rounds))
            return rounds_by_year
        except (OSError, pd.errors.EmptyDataError, KeyError):
            return {}

    def _collect_standings(
        self,
        start_year: int,
        end_year: int,
        endpoint: str,
        entries_key: str,
        entity_key: str,
        ref_field: str,
        id_attr: str,
    ) -> list[dict]:
        rows: list[dict] = []
        for year in range(start_year, end_year + 1):
            for batch in self._paginate(f"{year}/{endpoint}", "StandingsTable", limit=1000):
                for sl in batch:
                    round_num = int(sl.get("round", 0) or 0)
                    if not round_num:
                        continue
                    race_id = int(f"{year}{round_num:02d}")
                    entries = sl.get(entries_key, [])
                    if not isinstance(entries, list):
                        entries = [entries]
                    for entry in entries:
                        entity = entry.get(entity_key, {})
                        rows.append({
                            "race_id": race_id,
                            ref_field: entity.get(id_attr, ""),
                            "points": float(entry.get("points", 0)),
                            "position": int(entry.get("position", 0)),
                            "position_text": entry.get("positionText", ""),
                            "wins": int(entry.get("wins", 0)),
                        })
        return rows

    def extract_standings(
        self, start_year: int = DEFAULT_START_YEAR, end_year: int = DEFAULT_END_YEAR
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.logger.info("Extracting standings (%s-%s)...", start_year, end_year)
        df_const = pd.DataFrame(self._collect_standings(
            start_year, end_year,
            "constructorStandings", "ConstructorStandings", "Constructor", "constructor_ref", "constructorId",
        ))
        df_driver = pd.DataFrame(self._collect_standings(
            start_year, end_year,
            "driverStandings", "DriverStandings", "Driver", "driver_ref", "driverId",
        ))
        self._write_csv_atomic(df_const, "constructor_standings.csv")
        self._write_csv_atomic(df_driver, "driver_standings.csv")
        self.logger.info("Extracted %s constructor standings.", len(df_const))
        self.logger.info("Extracted %s driver standings.", len(df_driver))
        return df_const, df_driver
    
    def extract_all(
        self,
        start_year: int = DEFAULT_START_YEAR,
        end_year: int = DEFAULT_END_YEAR,
        skip_pit_stops: bool = False,
    ):
        self._total_rate_limits = 0
        self._consecutive_rate_limits = 0
        if start_year < DEFAULT_START_YEAR:
            self.logger.warning(
                "start_year %s is before project scope; clamping to %s.", start_year, DEFAULT_START_YEAR
            )
        if end_year > DEFAULT_END_YEAR:
            self.logger.warning(
                "end_year %s is in the future; clamping to current year %s.", end_year, DEFAULT_END_YEAR
            )
        start_year = max(DEFAULT_START_YEAR, start_year)
        end_year = min(DEFAULT_END_YEAR, end_year)
        if start_year > end_year:
            raise ValueError(f"Invalid year range after clamping to {DEFAULT_START_YEAR}-{DEFAULT_END_YEAR}.")
        self.logger.info("Starting F1 data extraction from the Ergast API")
        
        try:
            self.extract_circuits()
            self.extract_seasons()
            self.extract_constructors()
            self.extract_drivers()
            self.extract_races(start_year, end_year)
            self.extract_results(start_year, end_year)
            self.extract_qualifying(start_year, end_year)
            if skip_pit_stops:
                self.logger.info("Skipping pit stop extraction (--skip-pit-stops flag).")
            else:
                self.extract_pit_stops(start_year, end_year)
            self.extract_standings(start_year, end_year)
            
            self.logger.info("All data extraction steps completed successfully.")
            self.logger.info("Raw data saved to: %s", self.output_path)
            
        except Exception:
            self.logger.exception("Error during extraction.")
            raise

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract F1 data from Ergast API")
    parser.add_argument(
        '--start-year',
        type=int,
        default=DEFAULT_START_YEAR,
        help=f"Start year (default: {DEFAULT_START_YEAR})",
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=DEFAULT_END_YEAR,
        help=f"End year (default: {DEFAULT_END_YEAR})",
    )
    parser.add_argument('--output', type=str, default='data/raw/', help='Output directory (default: data/raw/)')
    parser.add_argument('--base-delay', type=float, default=1.5, help='Delay between API requests in seconds')
    parser.add_argument('--max-retries', type=int, default=6, help='Max retries on API errors or rate limits')
    
    args = parser.parse_args()
    
    extractor = F1DataExtractor(
        output_path=args.output,
        base_delay=args.base_delay,
        max_retries=args.max_retries,
    )
    extractor.extract_all(start_year=args.start_year, end_year=args.end_year)

if __name__ == "__main__":
    main()
