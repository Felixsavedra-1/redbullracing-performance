from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from constants import TEAM_REFS, DNF_POSITION_ORDER

_log = logging.getLogger(__name__)


def ref_params(refs: list[str]) -> tuple[str, dict]:
    """Returns (IN-clause placeholders, params dict) for parameterized queries."""
    return ", ".join(f":r{i}" for i in range(len(refs))), {f"r{i}": r for i, r in enumerate(refs)}


def teammate_delta(
    engine: Engine,
    team_refs: list[str] = TEAM_REFS,
    min_shared_races: int = 5,
) -> pd.DataFrame:
    """
    Position delta for each driver vs their teammate in shared finished races.
    Returns: driver_a | driver_b | mean_delta | ci_lower | ci_upper | n | p_value
    Negative mean_delta = driver_a finishes ahead of driver_b on average.
    DNFs excluded from both sides to isolate pace, not reliability.
    """
    placeholders, params = ref_params(team_refs)
    sql = f"""
    SELECT
        COALESCE(da.forename, '') || ' ' || COALESCE(da.surname, '') AS driver_a,
        COALESCE(db.forename, '') || ' ' || COALESCE(db.surname, '') AS driver_b,
        CAST(ra.position_order AS INTEGER) - CAST(rb.position_order AS INTEGER) AS delta
    FROM results ra
    JOIN results rb
        ON  ra.race_id        = rb.race_id
        AND ra.constructor_id = rb.constructor_id
        AND ra.driver_id      < rb.driver_id
    JOIN constructors c ON ra.constructor_id = c.constructor_id
    JOIN drivers da ON ra.driver_id = da.driver_id
    JOIN drivers db ON rb.driver_id = db.driver_id
    WHERE c.constructor_ref IN ({placeholders})
      AND ra.position_order < {DNF_POSITION_ORDER}
      AND rb.position_order < {DNF_POSITION_ORDER}
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
    except SQLAlchemyError as exc:
        _log.warning("teammate_delta: query failed — %s", exc)
        return pd.DataFrame(columns=["driver_a", "driver_b", "mean_delta", "ci_lower", "ci_upper", "n", "p_value"])

    rows = []
    for (a, b), g in df.groupby(["driver_a", "driver_b"]):
        d = g["delta"].dropna().values.astype(float)
        n = len(d)
        if n < min_shared_races:
            continue
        mean = d.mean()
        _, p = stats.ttest_1samp(d, 0)
        ci = stats.t.interval(0.95, n - 1, loc=mean, scale=stats.sem(d))
        rows.append(dict(
            driver_a=a, driver_b=b,
            mean_delta=round(mean, 3),
            ci_lower=round(ci[0], 3),
            ci_upper=round(ci[1], 3),
            n=n, p_value=round(p, 6),
        ))

    if not rows:
        return pd.DataFrame(columns=["driver_a", "driver_b", "mean_delta", "ci_lower", "ci_upper", "n", "p_value"])
    return pd.DataFrame(rows).sort_values("mean_delta").reset_index(drop=True)


def qualifying_race_ols(
    engine: Engine,
    team_refs: list[str] = TEAM_REFS,
) -> tuple[dict, pd.DataFrame]:
    """
    OLS regression of grid position on race finish position.
    Returns (stats_dict, scatter_df). DNFs excluded.
    stats_dict: slope | intercept | r2 | p_value | n
    """
    placeholders, params = ref_params(team_refs)
    sql = f"""
    SELECT
        COALESCE(da.forename, '') || ' ' || COALESCE(da.surname, '') AS driver,
        CAST(r.grid          AS INTEGER) AS grid,
        CAST(r.position_order AS INTEGER) AS finish
    FROM results r
    JOIN constructors c ON r.constructor_id = c.constructor_id
    JOIN drivers da ON r.driver_id = da.driver_id
    WHERE c.constructor_ref IN ({placeholders})
      AND r.grid           > 0
      AND r.position_order < {DNF_POSITION_ORDER}
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
    except SQLAlchemyError as exc:
        _log.warning("qualifying_race_ols: query failed — %s", exc)
        return {"slope": None, "intercept": None, "r2": None, "p_value": None, "n": 0}, pd.DataFrame(columns=["driver", "grid", "finish"])

    if len(df) < 2:
        empty_df = pd.DataFrame(columns=["driver", "grid", "finish"])
        return {"slope": None, "intercept": None, "r2": None, "p_value": None, "n": 0}, empty_df

    slope, intercept, r, p, _ = stats.linregress(df["grid"], df["finish"])
    return dict(slope=slope, intercept=intercept, r2=r ** 2, p_value=p, n=len(df)), df


def pit_stop_efficiency(
    engine: Engine,
    team_refs: list[str] = TEAM_REFS,
    min_stops: int = 5,
) -> pd.DataFrame:
    """
    Z-score each stop against the season field distribution, return per-driver aggregates.
    Returns: driver | mean_z | std_z | n_stops  (sorted ascending by mean_z).
    Stops outside [15 s, 60 s] dropped — safety-car pits and data errors skew the baseline.
    """
    placeholders, params = ref_params(team_refs)
    sql = f"""
    WITH season_stats AS (
        SELECT
            ra.year,
            AVG(p.milliseconds) AS mu,
            SQRT(AVG(p.milliseconds * p.milliseconds)
                 - AVG(p.milliseconds) * AVG(p.milliseconds)) AS sigma
        FROM pit_stops p
        JOIN races ra ON p.race_id = ra.race_id
        WHERE p.milliseconds BETWEEN 15000 AND 60000
        GROUP BY ra.year
    )
    SELECT
        COALESCE(da.forename, '') || ' ' || COALESCE(da.surname, '') AS driver,
        (p.milliseconds - ss.mu) / NULLIF(ss.sigma, 0) AS z
    FROM pit_stops p
    JOIN races ra         ON p.race_id          = ra.race_id
    JOIN season_stats ss  ON ss.year             = ra.year
    JOIN results res      ON res.race_id         = p.race_id
                         AND res.driver_id       = p.driver_id
    JOIN constructors c   ON res.constructor_id  = c.constructor_id
    JOIN drivers da       ON p.driver_id         = da.driver_id
    WHERE c.constructor_ref IN ({placeholders})
      AND p.milliseconds BETWEEN 15000 AND 60000
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params).dropna(subset=["z"])
    except SQLAlchemyError as exc:
        _log.warning("pit_stop_efficiency: query failed — %s", exc)
        return pd.DataFrame(columns=["driver", "mean_z", "std_z", "n_stops"])

    agg = (
        df.groupby("driver")["z"]
        .agg(mean_z="mean", std_z="std", n_stops="count")
        .reset_index()
    )
    return agg[agg["n_stops"] >= min_stops].sort_values("mean_z").reset_index(drop=True)


def championship_trajectory(engine: Engine, team_refs: list[str] = TEAM_REFS) -> pd.DataFrame:
    """
    Cumulative championship points per driver per round, all seasons.
    Returns: year | round | driver | points | position
    """
    placeholders, params = ref_params(team_refs)
    sql = f"""
    SELECT
        ra.year, ra.round,
        COALESCE(da.forename, '') || ' ' || COALESCE(da.surname, '') AS driver,
        ds.points, ds.position
    FROM driver_standings ds
    JOIN races ra       ON ds.race_id          = ra.race_id
    JOIN drivers da     ON ds.driver_id        = da.driver_id
    JOIN results res    ON res.race_id         = ra.race_id
                       AND res.driver_id       = ds.driver_id
    JOIN constructors c ON res.constructor_id  = c.constructor_id
    WHERE c.constructor_ref IN ({placeholders})
    ORDER BY da.driver_id, ra.year, ra.round
    """
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)
    except SQLAlchemyError as exc:
        _log.warning("championship_trajectory: query failed — %s", exc)
        return pd.DataFrame(columns=["year", "round", "driver", "points", "position"])


def dnf_rate_model(
    engine: Engine,
    team_refs: list[str] = TEAM_REFS,
    min_races: int = 10,
) -> pd.DataFrame:
    """
    Poisson MLE for DNF rate per driver with exact 95% confidence intervals.
    Returns: driver | races | dnfs | rate | ci_lower | ci_upper  (sorted desc by rate).
    CI uses chi-squared exact method: lower = χ²(0.025, 2k)/(2n), upper = χ²(0.975, 2k+2)/(2n).
    """
    placeholders, params = ref_params(team_refs)
    params["min_races"] = min_races
    sql = f"""
    SELECT
        COALESCE(da.forename, '') || ' ' || COALESCE(da.surname, '') AS driver,
        COUNT(*) AS races,
        SUM(CASE WHEN r.position_order = {DNF_POSITION_ORDER} THEN 1 ELSE 0 END) AS dnfs
    FROM results r
    JOIN drivers da     ON r.driver_id        = da.driver_id
    JOIN constructors c ON r.constructor_id   = c.constructor_id
    WHERE c.constructor_ref IN ({placeholders})
    GROUP BY r.driver_id, da.forename, da.surname
    HAVING races >= :min_races
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
    except SQLAlchemyError as exc:
        _log.warning("dnf_rate_model: query failed — %s", exc)
        return pd.DataFrame(columns=["driver", "races", "dnfs", "rate", "ci_lower", "ci_upper"])

    dnfs = df["dnfs"].to_numpy(dtype=float)
    races = df["races"].to_numpy(dtype=float)
    df["rate"] = dnfs / races
    df["ci_lower"] = np.where(
        dnfs > 0,
        stats.chi2.ppf(0.025, 2 * dnfs) / (2 * races),
        0.0,
    )
    df["ci_upper"] = stats.chi2.ppf(0.975, 2 * (dnfs + 1)) / (2 * races)
    return df.sort_values("rate", ascending=False).reset_index(drop=True)


def tyre_degradation(
    engine: Engine,
    team_refs: list[str] = TEAM_REFS,
    min_laps: int = 5,
) -> pd.DataFrame:
    """
    OLS degradation rate (seconds lost per additional lap on tyre) per driver per compound.
    Green-flag laps only (track_status='1'), tyre_life > 1.
    Returns: driver | compound | deg_rate_s | r2 | n
    Positive deg_rate_s = lap time grows with tyre age (expected).
    """
    placeholders, params = ref_params(team_refs)
    sql = f"""
    SELECT COALESCE(d.forename,'') || ' ' || COALESCE(d.surname,'') AS driver,
           l.compound, l.tyre_life, l.lap_time_s
    FROM laps l
    JOIN results      res ON l.race_id         = res.race_id
                         AND l.driver_id       = res.driver_id
    JOIN constructors c   ON res.constructor_id = c.constructor_id
    JOIN drivers      d   ON l.driver_id        = d.driver_id
    WHERE c.constructor_ref IN ({placeholders})
      AND l.compound     IN ('SOFT','MEDIUM','HARD')
      AND l.lap_time_s   IS NOT NULL
      AND l.tyre_life    > 1
      AND l.track_status = '1'
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
    except SQLAlchemyError as exc:
        _log.warning("tyre_degradation: query failed — %s", exc)
        return pd.DataFrame(columns=["driver", "compound", "deg_rate_s", "r2", "n"])
    if df.empty:
        return pd.DataFrame(columns=["driver", "compound", "deg_rate_s", "r2", "n"])

    rows = []
    for (driver, compound), g in df.groupby(["driver", "compound"]):
        g = g.dropna(subset=["lap_time_s", "tyre_life"])
        if len(g) < min_laps:
            continue
        slope, _, r_val, _, _ = stats.linregress(g["tyre_life"], g["lap_time_s"])
        rows.append(dict(
            driver=driver, compound=compound,
            deg_rate_s=round(slope, 4), r2=round(r_val ** 2, 3), n=len(g),
        ))

    if not rows:
        return pd.DataFrame(columns=["driver", "compound", "deg_rate_s", "r2", "n"])
    return pd.DataFrame(rows).sort_values(["compound", "deg_rate_s"]).reset_index(drop=True)


def pit_strategy(
    engine: Engine,
    team_refs: list[str] = TEAM_REFS,
    year: int | None = None,
) -> pd.DataFrame:
    """
    Stint structure for the most recent race with lap data.
    Returns: race_name | year | driver | stint | compound | start_lap | end_lap | stint_laps | finish_pos
    Sorted by finish position then stint number.
    """
    placeholders, params = ref_params(team_refs)
    year_clause = "AND ra.year = :year" if year else ""
    if year:
        params["year"] = year
    sql = f"""
    WITH team_cons AS (
        SELECT constructor_id FROM constructors
        WHERE constructor_ref IN ({placeholders})
    ),
    latest_race AS (
        SELECT l.race_id
        FROM laps l
        JOIN results res ON l.race_id   = res.race_id
                       AND l.driver_id = res.driver_id
        WHERE res.constructor_id IN (SELECT constructor_id FROM team_cons)
          {year_clause}
        ORDER BY l.race_id DESC
        LIMIT 1
    )
    SELECT
        ra.race_name,
        ra.year,
        COALESCE(d.forename,'') || ' ' || COALESCE(d.surname,'') AS driver,
        l.stint,
        UPPER(l.compound) AS compound,
        MIN(l.lap_number) AS start_lap,
        MAX(l.lap_number) AS end_lap,
        COUNT(*)           AS stint_laps,
        COALESCE(res.position_order, 99) AS finish_pos
    FROM laps l
    JOIN latest_race  lr  ON l.race_id         = lr.race_id
    JOIN races        ra  ON l.race_id         = ra.race_id
    JOIN results      res ON l.race_id         = res.race_id
                         AND l.driver_id       = res.driver_id
    JOIN drivers      d   ON l.driver_id        = d.driver_id
    WHERE res.constructor_id IN (SELECT constructor_id FROM team_cons)
      AND l.compound IS NOT NULL
      AND l.compound NOT IN ('UNKNOWN', '')
    GROUP BY ra.race_name, ra.year, l.driver_id, d.forename, d.surname,
             l.stint, l.compound, res.position_order
    ORDER BY finish_pos, l.stint
    """
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)
    except SQLAlchemyError as exc:
        _log.warning("pit_strategy: query failed — %s", exc)
        return pd.DataFrame(columns=[
            "race_name", "year", "driver", "stint", "compound",
            "start_lap", "end_lap", "stint_laps", "finish_pos",
        ])


def sector_deltas(
    engine: Engine,
    team_refs: list[str] = TEAM_REFS,
    min_laps: int = 10,
) -> pd.DataFrame:
    """
    Mean sector times per driver on green-flag laps.
    Returns: driver | s1_mean | s2_mean | s3_mean | n
    Sorted by combined sector time (fastest first).
    """
    placeholders, params = ref_params(team_refs)
    params["min_laps"] = min_laps
    sql = f"""
    SELECT COALESCE(d.forename,'') || ' ' || COALESCE(d.surname,'') AS driver,
           AVG(l.sector1_s) AS s1_mean,
           AVG(l.sector2_s) AS s2_mean,
           AVG(l.sector3_s) AS s3_mean,
           COUNT(*)          AS n
    FROM laps l
    JOIN results      res ON l.race_id         = res.race_id AND l.driver_id = res.driver_id
    JOIN constructors c   ON res.constructor_id = c.constructor_id
    JOIN drivers      d   ON l.driver_id        = d.driver_id
    WHERE c.constructor_ref IN ({placeholders})
      AND l.track_status  = '1'
      AND l.lap_time_s   IS NOT NULL
      AND l.sector1_s    IS NOT NULL
      AND l.sector2_s    IS NOT NULL
      AND l.sector3_s    IS NOT NULL
    GROUP BY l.driver_id, d.forename, d.surname
    HAVING COUNT(*) >= :min_laps
    ORDER BY s1_mean + s2_mean + s3_mean
    """
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)
    except SQLAlchemyError as exc:
        _log.warning("sector_deltas: query failed — %s", exc)
        return pd.DataFrame(columns=["driver", "s1_mean", "s2_mean", "s3_mean", "n"])
