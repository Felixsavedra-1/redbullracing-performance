from __future__ import annotations

from collections.abc import Callable
from typing import Any
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_float_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

_TYPE_CHECK: dict[str, Callable] = {
    "numeric":  is_numeric_dtype,
    "integer":  is_integer_dtype,
    "float":    is_float_dtype,
    "string":   is_object_dtype,
    "datetime": is_datetime64_any_dtype,
}

SCHEMA_CONTRACTS: dict[str, dict[str, list[str]]] = {
    "circuits": {
        "required": [
            "circuit_id",
            "circuit_ref",
            "circuit_name",
            "location",
            "country",
            "lat",
            "lng",
            "altitude",
            "url",
        ],
        "numeric": ["circuit_id", "lat", "lng", "altitude"],
        "string": ["circuit_ref", "circuit_name", "location", "country", "url"],
    },
    "seasons": {
        "required": ["year", "url"],
        "numeric": ["year"],
        "string": ["url"],
    },
    "constructors": {
        "required": [
            "constructor_id",
            "constructor_ref",
            "constructor_name",
            "nationality",
            "url",
        ],
        "numeric": ["constructor_id"],
        "string": ["constructor_ref", "constructor_name", "nationality", "url"],
    },
    "drivers": {
        "required": [
            "driver_id",
            "driver_ref",
            "driver_number",
            "code",
            "forename",
            "surname",
            "dob",
            "nationality",
            "url",
        ],
        "numeric": ["driver_id", "driver_number"],
        "string": ["driver_ref", "code", "forename", "surname", "nationality", "url"],
        "datetime": ["dob"],
    },
    "races": {
        "required": [
            "race_id",
            "year",
            "round",
            "circuit_id",
            "race_name",
            "race_date",
            "race_time",
            "url",
        ],
        "numeric": ["race_id", "year", "round", "circuit_id"],
        "string": ["race_name", "race_time", "url"],
        "datetime": ["race_date"],
    },
    "results": {
        "required": [
            "race_id", "driver_id", "constructor_id", "number", "grid",
            "position", "position_text", "position_order", "points", "laps",
            "time_result", "milliseconds", "fastest_lap", "fastest_lap_rank",
            "fastest_lap_time", "fastest_lap_speed", "status",
        ],
        "numeric": [
            "race_id", "driver_id", "constructor_id", "number", "grid",
            "position", "position_order", "points", "laps",
            "milliseconds", "fastest_lap", "fastest_lap_rank",
        ],
        "string": ["position_text", "time_result", "fastest_lap_time", "fastest_lap_speed", "status"],
        "constraints": {
            "points": ("min", 0),
            "laps": ("min", 0),
            "position_order": ("min", 0),
        },
    },
    "qualifying": {
        "required": [
            "race_id",
            "driver_id",
            "constructor_id",
            "number",
            "position",
            "q1",
            "q2",
            "q3",
        ],
        "numeric": ["race_id", "driver_id", "constructor_id", "number", "position"],
        "string": ["q1", "q2", "q3"],
    },
    "pit_stops": {
        "required": [
            "race_id",
            "driver_id",
            "stop",
            "lap",
            "time_of_day",
            "duration",
            "milliseconds",
        ],
        "numeric": ["race_id", "driver_id", "stop", "lap", "milliseconds"],
        "string": ["time_of_day", "duration"],
    },
    "constructor_standings": {
        "required": [
            "race_id",
            "constructor_id",
            "points",
            "position",
            "position_text",
            "wins",
        ],
        "numeric": ["race_id", "constructor_id", "points", "position", "wins"],
        "string": ["position_text"],
    },
    "driver_standings": {
        "required": [
            "race_id",
            "driver_id",
            "points",
            "position",
            "position_text",
            "wins",
        ],
        "numeric": ["race_id", "driver_id", "points", "position", "wins"],
        "string": ["position_text"],
    },
    "laps": {
        "required": [
            "race_id", "driver_id", "lap_number",
            "lap_time_s", "compound", "tyre_life", "stint",
        ],
        "numeric": [
            "race_id", "driver_id", "lap_number",
            "lap_time_s", "sector1_s", "sector2_s", "sector3_s",
            "tyre_life", "stint", "is_personal_best", "pit_in", "pit_out",
        ],
        "string": ["compound", "track_status"],
        "constraints": {
            "compound": ("enum", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"]),
        },
    },
}


def _check_types(df: pd.DataFrame, columns: list[str], type_name: str) -> list[str]:
    checker = _TYPE_CHECK.get(type_name)
    if not checker:
        raise ValueError(f"Unknown type_name: {type_name}")
    return [
        f"{col} is not {type_name}"
        for col in columns
        if col in df.columns and not checker(df[col])
    ]


def _check_constraints(df: pd.DataFrame, constraints: dict[str, tuple[str, Any]]) -> list[str]:
    issues = []
    for col, (kind, bound) in constraints.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if kind == "min":
            bad = int((series < bound).sum())
            if bad:
                issues.append(f"{col} has {bad} values below minimum {bound}")
        elif kind == "enum":
            bad_vals = set(series.unique()) - set(bound) - {""}
            if bad_vals:
                issues.append(f"{col} has unexpected values: {sorted(bad_vals)}")
    return issues


def validate_dataframe(table_name: str, df: pd.DataFrame) -> list[str]:
    """Validate dataframe against a simple schema contract."""
    contract = SCHEMA_CONTRACTS.get(table_name)
    if not contract:
        return [f"No schema contract defined for {table_name}"]

    issues = []
    required = contract.get("required", [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {', '.join(missing)}")

    issues.extend(_check_types(df, contract.get("numeric", []), "numeric"))
    issues.extend(_check_types(df, contract.get("string", []), "string"))
    issues.extend(_check_types(df, contract.get("datetime", []), "datetime"))
    issues.extend(_check_constraints(df, contract.get("constraints", {})))

    return issues
