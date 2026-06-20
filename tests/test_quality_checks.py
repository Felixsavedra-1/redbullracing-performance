import os
import tempfile
import unittest

from sqlalchemy import create_engine, text

from scripts.data_quality import run_quality_checks


def apply_duckdb_schema(engine) -> None:
    schema_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "database", "schema", "create_tables_duckdb.sql",
    ))
    with open(schema_path, "r") as handle:
        schema_sql = handle.read()
    with engine.connect() as conn:
        for statement in schema_sql.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()


class TestQualityChecks(unittest.TestCase):
    def test_quality_checks_pass_for_minimal_valid_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "quality.duckdb")
            engine = create_engine(f"duckdb:///{db_path}")
            apply_duckdb_schema(engine)

            with engine.connect() as conn:
                conn.execute(
                    text(
                        "INSERT INTO circuits (circuit_id, circuit_ref, circuit_name, location, country, lat, lng, altitude, url) "
                        "VALUES (1, 'silverstone', 'Silverstone Circuit', 'Silverstone', 'UK', 52.07, -1.02, 0, 'http://example.com')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO seasons (year, url) VALUES (2024, 'http://example.com/season/2024')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO constructors (constructor_id, constructor_ref, constructor_name, nationality, url) "
                        "VALUES (1, 'red_bull', 'Red Bull', 'Austrian', 'http://example.com')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO drivers (driver_id, driver_ref, driver_number, code, forename, surname, dob, nationality, url) "
                        "VALUES (1, 'max_verstappen', 33, 'VER', 'Max', 'Verstappen', '1997-09-30', 'Dutch', 'http://example.com')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO races (race_id, year, round, circuit_id, race_name, race_date, race_time, url) "
                        "VALUES (202401, 2024, 1, 1, 'British Grand Prix', '2024-07-07', '14:00:00', 'http://example.com')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO results (race_id, driver_id, constructor_id, number, grid, position, position_text, position_order, points, laps, "
                        "time_result, milliseconds, fastest_lap, fastest_lap_rank, fastest_lap_time, fastest_lap_speed, status) "
                        "VALUES (202401, 1, 1, 33, 1, 1, '1', 1, 25, 52, '1:30:00', 5400000, 12, 1, '1:20.000', '220.5', 'Finished')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO qualifying (race_id, driver_id, constructor_id, number, position, q1, q2, q3) "
                        "VALUES (202401, 1, 1, 33, 1, '1:21.0', '1:20.5', '1:20.0')"
                    )
                )
                conn.commit()

            failures = run_quality_checks(engine, start_year=2024, end_year=2024)
            self.assertEqual(failures, [])


class TestQualityCheckFailures(unittest.TestCase):
    """Each test inserts exactly one violation and verifies the specific check fires."""

    def _engine_with_base_data(self):
        engine = create_engine("duckdb:///:memory:")
        apply_duckdb_schema(engine)
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO circuits VALUES (1,'silverstone','Silverstone','Silverstone','UK',52.07,-1.02,0,'')"
            ))
            conn.execute(text("INSERT INTO seasons VALUES (2024,'')"))
            conn.execute(text(
                "INSERT INTO constructors VALUES (1,'red_bull','Red Bull','Austrian','')"
            ))
            conn.execute(text(
                "INSERT INTO drivers VALUES (1,'ver',33,'VER','Max','Verstappen','1997-09-30','Dutch','')"
            ))
            conn.execute(text(
                "INSERT INTO races VALUES (202401,2024,1,1,'British GP','2024-07-07','14:00','')"
            ))
        return engine

    def _check_names(self, failures):
        return {f["check"] for f in failures}

    def test_results_empty_fires_when_no_results(self):
        engine = self._engine_with_base_data()
        failures = run_quality_checks(engine, start_year=2024, end_year=2024)
        self.assertIn("results_non_empty", self._check_names(failures))

    def test_race_outside_year_range_fires(self):
        engine = self._engine_with_base_data()
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO seasons VALUES (2099,'')"))
            conn.execute(text(
                "INSERT INTO races VALUES (209901,2099,1,1,'Future GP','2099-01-01','14:00','')"
            ))
            conn.execute(text(
                "INSERT INTO results VALUES "
                "(202401,1,1,33,1,1,'1',1,25,52,'',0,0,0,'','','Finished')"
            ))
            conn.execute(text(
                "INSERT INTO qualifying VALUES (202401,1,1,33,1,'1:21','1:20','1:20')"
            ))
        failures = run_quality_checks(engine, start_year=2024, end_year=2024)
        self.assertIn("races_outside_year_range", self._check_names(failures))

    def test_position_order_consistency_fires(self):
        engine = self._engine_with_base_data()
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO results VALUES "
                "(202401,1,1,33,1,1,'1',999,25,52,'',0,0,0,'','','Finished')"
            ))
            conn.execute(text(
                "INSERT INTO qualifying VALUES (202401,1,1,33,1,'1:21','1:20','1:20')"
            ))
        failures = run_quality_checks(engine, start_year=2024, end_year=2024)
        self.assertIn("results_position_order_consistency", self._check_names(failures))

    def test_missing_race_year_fires(self):
        engine = self._engine_with_base_data()
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO results VALUES "
                "(202401,1,1,33,1,1,'1',1,25,52,'',0,0,0,'','','Finished')"
            ))
            conn.execute(text(
                "INSERT INTO qualifying VALUES (202401,1,1,33,1,'1:21','1:20','1:20')"
            ))
        failures = run_quality_checks(engine, start_year=2023, end_year=2024)
        self.assertIn("missing_race_years", self._check_names(failures))


if __name__ == "__main__":
    unittest.main()
