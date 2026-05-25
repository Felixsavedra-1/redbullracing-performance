"""Tests for analytics.py statistical models."""
from __future__ import annotations

import unittest

from sqlalchemy import create_engine, text

from scripts.analytics import (
    championship_trajectory,
    dnf_rate_model,
    pit_stop_efficiency,
    pit_strategy,
    qualifying_race_ols,
    sector_deltas,
    teammate_delta,
    tyre_degradation,
)

_DDL = """
CREATE TABLE seasons (year INTEGER PRIMARY KEY, url TEXT);
CREATE TABLE circuits (circuit_id INTEGER PRIMARY KEY, circuit_ref TEXT,
    circuit_name TEXT, location TEXT, country TEXT,
    lat REAL, lng REAL, altitude INTEGER, url TEXT);
CREATE TABLE constructors (constructor_id INTEGER PRIMARY KEY,
    constructor_ref TEXT UNIQUE, constructor_name TEXT,
    nationality TEXT, url TEXT);
CREATE TABLE drivers (driver_id INTEGER PRIMARY KEY, driver_ref TEXT UNIQUE,
    driver_number INTEGER, code TEXT, forename TEXT, surname TEXT,
    dob TEXT, nationality TEXT, url TEXT);
CREATE TABLE races (race_id INTEGER PRIMARY KEY, year INTEGER, round INTEGER,
    circuit_id INTEGER, race_name TEXT, race_date TEXT,
    race_time TEXT, url TEXT);
CREATE TABLE results (race_id INTEGER NOT NULL, driver_id INTEGER NOT NULL,
    constructor_id INTEGER NOT NULL, number INTEGER, grid INTEGER,
    position INTEGER, position_text TEXT, position_order INTEGER,
    points REAL, laps INTEGER, time_result TEXT, milliseconds INTEGER,
    fastest_lap INTEGER, fastest_lap_rank INTEGER,
    fastest_lap_time TEXT, fastest_lap_speed TEXT, status TEXT,
    PRIMARY KEY (race_id, driver_id, constructor_id));
CREATE TABLE pit_stops (race_id INTEGER NOT NULL, driver_id INTEGER NOT NULL,
    stop INTEGER, lap INTEGER, time_of_day TEXT, duration TEXT,
    milliseconds INTEGER, PRIMARY KEY (race_id, driver_id, stop));
CREATE TABLE driver_standings (race_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL, points REAL, position INTEGER,
    position_text TEXT, wins INTEGER, PRIMARY KEY (race_id, driver_id));
CREATE TABLE laps (race_id INTEGER NOT NULL, driver_id INTEGER NOT NULL,
    lap_number INTEGER NOT NULL, lap_time_s REAL,
    sector1_s REAL, sector2_s REAL, sector3_s REAL,
    compound TEXT, tyre_life INTEGER, stint INTEGER,
    is_personal_best INTEGER, pit_in INTEGER, pit_out INTEGER,
    track_status TEXT, PRIMARY KEY (race_id, driver_id, lap_number));
"""

# 2 drivers, 1 constructor (ref='red_bull'), 6 races.
# Driver 1 always finishes P1 (grid=2); driver 2 always finishes P2 (grid=1).
# This gives: delta = pos_order_a - pos_order_b = 1 - 2 = -1 (driver_a ahead on average).
_FIXTURE = """
INSERT INTO seasons VALUES (2024, '');
INSERT INTO circuits VALUES (1,'silverstone','Silverstone','Silverstone','UK',52.07,-1.02,153,'');
INSERT INTO constructors VALUES (9,'red_bull','Red Bull','Austrian','');
INSERT INTO drivers VALUES (1,'ver',1,'VER','Max','Verstappen','1997-09-30','Dutch','');
INSERT INTO drivers VALUES (2,'per',11,'PER','Sergio','Perez','1990-01-26','Mexican','');
INSERT INTO races VALUES (202401,2024,1,1,'R1','2024-03-02','15:00','');
INSERT INTO races VALUES (202402,2024,2,1,'R2','2024-03-09','15:00','');
INSERT INTO races VALUES (202403,2024,3,1,'R3','2024-03-16','15:00','');
INSERT INTO races VALUES (202404,2024,4,1,'R4','2024-03-23','15:00','');
INSERT INTO races VALUES (202405,2024,5,1,'R5','2024-03-30','15:00','');
INSERT INTO races VALUES (202406,2024,6,1,'R6','2024-04-06','15:00','');
INSERT INTO results VALUES (202401,1,9,1,2,1,'1',1,25,57,'',5400000,0,0,'','','Finished');
INSERT INTO results VALUES (202401,2,9,11,1,2,'2',2,18,57,'',5410000,0,0,'','','Finished');
INSERT INTO results VALUES (202402,1,9,1,2,1,'1',1,25,57,'',5400000,0,0,'','','Finished');
INSERT INTO results VALUES (202402,2,9,11,1,2,'2',2,18,57,'',5410000,0,0,'','','Finished');
INSERT INTO results VALUES (202403,1,9,1,2,1,'1',1,25,57,'',5400000,0,0,'','','Finished');
INSERT INTO results VALUES (202403,2,9,11,1,2,'2',2,18,57,'',5410000,0,0,'','','Finished');
INSERT INTO results VALUES (202404,1,9,1,2,1,'1',1,25,57,'',5400000,0,0,'','','Finished');
INSERT INTO results VALUES (202404,2,9,11,1,2,'2',2,18,57,'',5410000,0,0,'','','Finished');
INSERT INTO results VALUES (202405,1,9,1,2,1,'1',1,25,57,'',5400000,0,0,'','','Finished');
INSERT INTO results VALUES (202405,2,9,11,1,2,'2',2,18,57,'',5410000,0,0,'','','Finished');
INSERT INTO results VALUES (202406,1,9,1,2,1,'1',1,25,57,'',5400000,0,0,'','','Finished');
INSERT INTO results VALUES (202406,2,9,11,1,2,'2',2,18,57,'',5410000,0,0,'','','Finished');
INSERT INTO pit_stops VALUES (202401,1,1,20,'14:00:00','25.123',25123);
INSERT INTO pit_stops VALUES (202402,1,1,22,'14:00:00','23.456',23456);
INSERT INTO pit_stops VALUES (202403,1,1,18,'14:00:00','26.789',26789);
INSERT INTO pit_stops VALUES (202404,1,1,25,'14:00:00','24.000',24000);
INSERT INTO pit_stops VALUES (202405,1,1,19,'14:00:00','25.500',25500);
INSERT INTO pit_stops VALUES (202406,1,1,21,'14:00:00','22.000',22000);
INSERT INTO driver_standings VALUES (202401,1,25,1,'1',1);
INSERT INTO driver_standings VALUES (202401,2,18,2,'2',0);
INSERT INTO driver_standings VALUES (202402,1,50,1,'1',2);
INSERT INTO driver_standings VALUES (202402,2,36,2,'2',0);
INSERT INTO laps VALUES (202401,1,1,90.1,28.0,31.0,31.1,'SOFT',1,1,0,0,1,'1');
INSERT INTO laps VALUES (202401,1,2,90.3,28.1,31.1,31.1,'SOFT',2,1,0,0,0,'1');
INSERT INTO laps VALUES (202401,1,3,90.5,28.2,31.2,31.1,'SOFT',3,1,0,1,0,'1');
INSERT INTO laps VALUES (202401,1,4,90.7,28.3,31.2,31.2,'MEDIUM',1,2,0,0,1,'1');
INSERT INTO laps VALUES (202401,1,5,90.9,28.4,31.3,31.2,'MEDIUM',2,2,0,0,0,'1');
INSERT INTO laps VALUES (202401,2,1,91.0,28.5,31.4,31.1,'SOFT',1,1,0,0,1,'1');
INSERT INTO laps VALUES (202401,2,2,91.2,28.6,31.5,31.1,'SOFT',2,1,0,0,0,'1');
INSERT INTO laps VALUES (202401,2,3,91.4,28.7,31.5,31.2,'SOFT',3,1,0,1,0,'1');
INSERT INTO laps VALUES (202401,2,4,91.6,28.8,31.6,31.2,'MEDIUM',1,2,0,0,1,'1');
INSERT INTO laps VALUES (202401,2,5,91.8,28.9,31.6,31.3,'MEDIUM',2,2,0,0,0,'1');
"""


def _make_engine():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        for stmt in _DDL.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))
        for stmt in _FIXTURE.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))
    return engine


class TestTeammateDelta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = _make_engine()

    def test_returns_expected_columns(self):
        df = teammate_delta(self.engine, team_refs=["red_bull"], min_shared_races=5)
        self.assertFalse(df.empty)
        for col in ("driver_a", "driver_b", "mean_delta", "ci_lower", "ci_upper", "n", "p_value"):
            self.assertIn(col, df.columns)

    def test_driver_a_finishes_ahead(self):
        df = teammate_delta(self.engine, team_refs=["red_bull"], min_shared_races=5)
        # Driver 1 always P1, driver 2 always P2 → delta = 1-2 = -1 (negative = ahead)
        self.assertLess(df.iloc[0]["mean_delta"], 0)

    def test_below_min_shared_races_returns_empty(self):
        df = teammate_delta(self.engine, team_refs=["red_bull"], min_shared_races=99)
        self.assertTrue(df.empty)


class TestQualifyingRaceOls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = _make_engine()

    def test_returns_stats_dict_and_dataframe(self):
        stats_dict, df = qualifying_race_ols(self.engine, team_refs=["red_bull"])
        for key in ("slope", "intercept", "r2", "p_value", "n"):
            self.assertIn(key, stats_dict)
        self.assertIn("grid", df.columns)
        self.assertIn("finish", df.columns)

    def test_r2_in_unit_interval(self):
        stats_dict, _ = qualifying_race_ols(self.engine, team_refs=["red_bull"])
        self.assertGreaterEqual(stats_dict["r2"], 0.0)
        self.assertLessEqual(stats_dict["r2"], 1.0)

    def test_empty_constructor_returns_none_stats(self):
        stats_dict, df = qualifying_race_ols(self.engine, team_refs=["nonexistent"])
        self.assertIsNone(stats_dict["slope"])
        self.assertEqual(stats_dict["n"], 0)
        self.assertTrue(df.empty)


class TestPitStopEfficiency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = _make_engine()

    def test_returns_expected_columns(self):
        df = pit_stop_efficiency(self.engine, team_refs=["red_bull"], min_stops=1)
        self.assertFalse(df.empty)
        for col in ("driver", "mean_z", "std_z", "n_stops"):
            self.assertIn(col, df.columns)

    def test_below_min_stops_returns_empty(self):
        df = pit_stop_efficiency(self.engine, team_refs=["red_bull"], min_stops=999)
        self.assertTrue(df.empty)


class TestChampionshipTrajectory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = _make_engine()

    def test_returns_expected_columns(self):
        df = championship_trajectory(self.engine, team_refs=["red_bull"])
        self.assertFalse(df.empty)
        for col in ("year", "round", "driver", "points", "position"):
            self.assertIn(col, df.columns)

    def test_points_non_negative(self):
        df = championship_trajectory(self.engine, team_refs=["red_bull"])
        self.assertTrue((df["points"] >= 0).all())


class TestDnfRateModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = _make_engine()

    def test_returns_expected_columns(self):
        df = dnf_rate_model(self.engine, team_refs=["red_bull"], min_races=1)
        self.assertFalse(df.empty)
        for col in ("driver", "races", "dnfs", "rate", "ci_lower", "ci_upper"):
            self.assertIn(col, df.columns)

    def test_rate_in_unit_interval(self):
        df = dnf_rate_model(self.engine, team_refs=["red_bull"], min_races=1)
        self.assertTrue((df["rate"] >= 0).all())
        self.assertTrue((df["rate"] <= 1).all())

    def test_zero_dnfs_gives_zero_rate(self):
        df = dnf_rate_model(self.engine, team_refs=["red_bull"], min_races=1)
        # All results are 'Finished' → DNF rate should be 0
        self.assertTrue((df["dnfs"] == 0).all())
        self.assertTrue((df["rate"] == 0.0).all())


class TestTyreDegradationAndSectorDeltas(unittest.TestCase):
    """Laps table is empty in the fixture — both functions must return empty gracefully."""

    @classmethod
    def setUpClass(cls):
        cls.engine = _make_engine()

    def test_tyre_degradation_empty_returns_correct_schema(self):
        df = tyre_degradation(self.engine, team_refs=["red_bull"])
        self.assertTrue(df.empty)
        for col in ("driver", "compound", "deg_rate_s", "r2", "n"):
            self.assertIn(col, df.columns)

    def test_sector_deltas_empty_returns_empty(self):
        df = sector_deltas(self.engine, team_refs=["red_bull"])
        self.assertTrue(df.empty)


class TestPitStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = _make_engine()

    def test_returns_expected_columns(self):
        df = pit_strategy(self.engine, team_refs=["red_bull"])
        for col in ("race_name", "year", "driver", "stint", "compound",
                    "start_lap", "end_lap", "stint_laps", "finish_pos"):
            self.assertIn(col, df.columns)

    def test_returns_two_stints_per_driver(self):
        df = pit_strategy(self.engine, team_refs=["red_bull"])
        self.assertFalse(df.empty)
        stints_per_driver = df.groupby("driver")["stint"].nunique()
        self.assertTrue((stints_per_driver == 2).all())

    def test_nonexistent_team_returns_empty(self):
        df = pit_strategy(self.engine, team_refs=["nonexistent"])
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
