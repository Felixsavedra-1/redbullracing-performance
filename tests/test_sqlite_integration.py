"""
Integration test: real SQLite schema + quality gates + analytics.

Fills the gap where test_analytics.py uses hand-rolled DDL and
test_quality_checks.py only covers DuckDB. This exercises the actual
create_tables_sqlite.sql schema against the quality checks and analytics
query layer end-to-end.
"""
import os
import unittest

from sqlalchemy import create_engine, event, text

from scripts.analytics import championship_trajectory, qualifying_race_ols, teammate_delta
from scripts.data_quality import run_quality_checks

_SCHEMA = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "database", "schema", "create_tables_sqlite.sql")
)

_FIXTURE = """
INSERT INTO seasons VALUES (2024,'');
INSERT INTO circuits VALUES (1,'silverstone','Silverstone','Silverstone','UK',52.07,-1.02,153,'');
INSERT INTO constructors VALUES (9,'red_bull','Red Bull Racing','Austrian','');
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
INSERT INTO qualifying VALUES (202401,1,9,1,1,'1:21','1:20','1:19');
INSERT INTO qualifying VALUES (202401,2,9,11,2,'1:22','1:21','1:20');
INSERT INTO qualifying VALUES (202402,1,9,1,1,'1:21','1:20','1:19');
INSERT INTO qualifying VALUES (202402,2,9,11,2,'1:22','1:21','1:20');
INSERT INTO qualifying VALUES (202403,1,9,1,1,'1:21','1:20','1:19');
INSERT INTO qualifying VALUES (202403,2,9,11,2,'1:22','1:21','1:20');
INSERT INTO qualifying VALUES (202404,1,9,1,1,'1:21','1:20','1:19');
INSERT INTO qualifying VALUES (202404,2,9,11,2,'1:22','1:21','1:20');
INSERT INTO qualifying VALUES (202405,1,9,1,1,'1:21','1:20','1:19');
INSERT INTO qualifying VALUES (202405,2,9,11,2,'1:22','1:21','1:20');
INSERT INTO qualifying VALUES (202406,1,9,1,1,'1:21','1:20','1:19');
INSERT INTO qualifying VALUES (202406,2,9,11,2,'1:22','1:21','1:20');
INSERT INTO driver_standings VALUES (202401,1,25,1,'1',1);
INSERT INTO driver_standings VALUES (202401,2,18,2,'2',0);
INSERT INTO driver_standings VALUES (202402,1,50,1,'1',2);
INSERT INTO driver_standings VALUES (202402,2,36,2,'2',0);
INSERT INTO driver_standings VALUES (202403,1,75,1,'1',3);
INSERT INTO driver_standings VALUES (202403,2,54,2,'2',0);
INSERT INTO driver_standings VALUES (202404,1,100,1,'1',4);
INSERT INTO driver_standings VALUES (202404,2,72,2,'2',0);
INSERT INTO driver_standings VALUES (202405,1,125,1,'1',5);
INSERT INTO driver_standings VALUES (202405,2,90,2,'2',0);
INSERT INTO driver_standings VALUES (202406,1,150,1,'1',6);
INSERT INTO driver_standings VALUES (202406,2,108,2,'2',0);
"""


def _make_engine():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def set_fk_pragma(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    with open(_SCHEMA) as fh:
        schema_sql = fh.read()
    with engine.begin() as conn:
        for stmt in schema_sql.split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))
        for stmt in _FIXTURE.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))
    return engine


class TestSQLiteIntegration(unittest.TestCase):
    """Real SQLite schema + FK enforcement + quality checks + analytics."""

    @classmethod
    def setUpClass(cls):
        cls.engine = _make_engine()

    def test_quality_checks_pass(self):
        failures = run_quality_checks(self.engine, start_year=2024, end_year=2024)
        self.assertEqual(failures, [], failures)

    def test_championship_trajectory_returns_data(self):
        df = championship_trajectory(self.engine, team_refs=["red_bull"])
        self.assertFalse(df.empty)
        self.assertEqual(set(df.columns) & {"year", "round", "driver", "points"}, {"year", "round", "driver", "points"})
        self.assertTrue((df["points"] >= 0).all())

    def test_teammate_delta_ver_ahead_of_per(self):
        df = teammate_delta(self.engine, team_refs=["red_bull"], min_shared_races=5)
        self.assertFalse(df.empty)
        self.assertLess(df.iloc[0]["mean_delta"], 0)

    def test_qualifying_regression_perfect_fit(self):
        stats, scatter = qualifying_race_ols(self.engine, team_refs=["red_bull"])
        self.assertIsNotNone(stats["slope"])
        self.assertAlmostEqual(stats["r2"], 1.0, places=5)
        self.assertEqual(stats["n"], 12)

    def test_nonexistent_team_returns_empty(self):
        df = championship_trajectory(self.engine, team_refs=["nonexistent"])
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
