"""Unit tests for ETL edge cases: missing files, unmapped refs, schema violations, DNF sentinel."""
import os
import tempfile
import unittest

import pandas as pd
from sqlalchemy import inspect as sa_inspect

from scripts.transform_data import F1DataTransformer
from scripts.load_data import F1DataLoader
from scripts.constants import DNF_POSITION_ORDER
from tests.utils import write_csv


class TestReadCsvSafe(unittest.TestCase):
    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = F1DataTransformer(raw_data_path=tmp + "/", processed_data_path=tmp + "/")
            self.assertIsNone(t._read_csv_safe("does_not_exist.csv"))

    def test_empty_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.csv")
            with open(path, "w"):
                pass
            t = F1DataTransformer(raw_data_path=tmp + "/", processed_data_path=tmp + "/")
            self.assertIsNone(t._read_csv_safe("empty.csv"))

    def test_valid_file_returns_dataframe(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_csv(os.path.join(tmp, "data.csv"), ["a", "b"], [[1, 2]])
            t = F1DataTransformer(raw_data_path=tmp + "/", processed_data_path=tmp + "/")
            df = t._read_csv_safe("data.csv")
            self.assertIsNotNone(df)
            self.assertEqual(len(df), 1)


class TestTransformMissingFiles(unittest.TestCase):
    def _transformer(self, tmp):
        return F1DataTransformer(raw_data_path=tmp + "/", processed_data_path=tmp + "/")

    def test_transform_races_missing_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self._transformer(tmp).transform_races()
            self.assertTrue(df.empty)

    def test_transform_circuits_missing_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self._transformer(tmp).transform_circuits()
            self.assertTrue(df.empty)

    def test_transform_drivers_missing_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self._transformer(tmp).transform_drivers()
            self.assertTrue(df.empty)


class TestApplyRefMap(unittest.TestCase):
    def test_unmapped_refs_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_csv(os.path.join(tmp, "constructors.csv"),
                      ["constructor_id", "constructor_ref"],
                      [[9, "red_bull"]])
            t = F1DataTransformer(raw_data_path=tmp + "/", processed_data_path=tmp + "/")
            df = pd.DataFrame({"constructor_ref": ["red_bull", "unknown_team"]})
            result = t._apply_ref_map(df, "constructor_ref", "constructor_id", "constructors.csv")
            self.assertEqual(result.loc[result["constructor_ref"] == "red_bull", "constructor_id"].iloc[0], 9)
            self.assertEqual(len(result[result["constructor_ref"] == "unknown_team"]), 0)
            self.assertEqual(len(result), 1)

    def test_missing_ref_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = F1DataTransformer(raw_data_path=tmp + "/", processed_data_path=tmp + "/")
            df = pd.DataFrame({"driver_ref": ["max_verstappen"]})
            result = t._apply_ref_map(df, "driver_ref", "driver_id", "drivers.csv")
            self.assertTrue(result.empty)

    def test_known_refs_are_mapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_csv(os.path.join(tmp, "drivers.csv"),
                      ["driver_id", "driver_ref"],
                      [[1, "max_verstappen"], [2, "perez"]])
            t = F1DataTransformer(raw_data_path=tmp + "/", processed_data_path=tmp + "/")
            df = pd.DataFrame({"driver_ref": ["max_verstappen", "perez"]})
            result = t._apply_ref_map(df, "driver_ref", "driver_id", "drivers.csv")
            self.assertListEqual(result["driver_id"].tolist(), [1, 2])


class TestDNFSentinel(unittest.TestCase):
    def test_dnf_position_order_constant(self):
        self.assertEqual(DNF_POSITION_ORDER, 999)

    def test_non_finisher_gets_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_csv(
                os.path.join(tmp, "results.csv"),
                ["race_id", "driver_ref", "constructor_ref", "number", "grid",
                 "position", "position_text", "position_order", "points", "laps",
                 "time_result", "milliseconds", "fastest_lap", "fastest_lap_rank",
                 "fastest_lap_time", "fastest_lap_speed", "status"],
                [[202401, "max_verstappen", "red_bull", 33, 1,
                  "", "R", DNF_POSITION_ORDER, 0, 10, "", 0, 0, 0, "", "", "Engine"]],
            )
            write_csv(os.path.join(tmp, "drivers.csv"),
                      ["driver_id", "driver_ref"], [[1, "max_verstappen"]])
            write_csv(os.path.join(tmp, "constructors.csv"),
                      ["constructor_id", "constructor_ref"], [[9, "red_bull"]])

            t = F1DataTransformer(raw_data_path=tmp + "/", processed_data_path=tmp + "/")
            df = t.transform_results()
            self.assertEqual(df["position_order"].iloc[0], DNF_POSITION_ORDER)


class TestStrictSchema(unittest.TestCase):
    def test_strict_schema_controls_raise_behavior(self):
        bad_df = pd.DataFrame({"circuit_name": ["Silverstone"]})
        for strict, should_raise in [(True, True), (False, False)]:
            with self.subTest(strict=strict), tempfile.TemporaryDirectory() as tmp:
                loader = F1DataLoader(
                    config={"type": "sqlite", "filename": os.path.join(tmp, "test.db")},
                    processed_data_path=tmp + "/",
                    strict_schema=strict,
                )
                if should_raise:
                    with self.assertRaises(ValueError):
                        loader._validate_df(bad_df, "circuits")
                else:
                    loader._validate_df(bad_df, "circuits")


class TestNormalizeProgress(unittest.TestCase):
    def _extractor(self, tmp):
        from scripts.extract_data import F1DataExtractor
        return F1DataExtractor(output_path=tmp + "/")

    def test_canonical_format_roundtrips(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = self._extractor(tmp)
            data = {"years": {"2020": [1, 2, 3]}, "skipped": {"2020": [4]}}
            result = e._normalize_progress(data, 2020, 2020)
            self.assertEqual(result["years"]["2020"], [1, 2, 3])
            self.assertEqual(result["skipped"]["2020"], [4])

    def test_legacy_flat_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = self._extractor(tmp)
            data = {"2020": [1, 2]}
            result = e._normalize_progress(data, 2020, 2020)
            self.assertEqual(result["years"]["2020"], [1, 2])

    def test_deduplicates_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = self._extractor(tmp)
            data = {"years": {"2020": [3, 1, 2, 1]}, "skipped": {}}
            result = e._normalize_progress(data, 2020, 2020)
            self.assertEqual(result["years"]["2020"], [1, 2, 3])

    def test_out_of_range_years_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = self._extractor(tmp)
            data = {"years": {"2019": [1], "2020": [1], "2021": [1]}, "skipped": {}}
            result = e._normalize_progress(data, 2020, 2020)
            self.assertNotIn("2019", result["years"])
            self.assertNotIn("2021", result["years"])


class TestIncrementalStagingCleanup(unittest.TestCase):
    def test_staging_table_dropped_on_upsert_failure(self):
        """Staging table must not persist when the upsert fails."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            loader = F1DataLoader(
                config={"type": "sqlite", "filename": db_path},
                processed_data_path=tmp + "/",
                mode="incremental",
            )
            df = pd.DataFrame({"col_a": [1, 2]})
            try:
                loader._load_table_incremental(df, "nonexistent_table")
            except Exception:
                pass
            inspector = sa_inspect(loader.engine)
            self.assertNotIn("_stg_nonexistent_table", inspector.get_table_names())

    def test_full_refresh_creates_backup(self):
        """Full-refresh must rename old DB to .bak rather than deleting it."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            with open(db_path, "w") as f:
                f.write("old data")
            F1DataLoader(
                config={"type": "sqlite", "filename": db_path},
                processed_data_path=tmp + "/",
                mode="full_refresh",
            )
            self.assertTrue(os.path.exists(db_path + ".bak"),
                            "backup file should exist after full refresh of an existing DB")


class TestDatetimeCoercionLogging(unittest.TestCase):
    def test_invalid_dob_logged_not_silently_dropped(self):
        """Invalid date values should produce NaT with a warning, not silently vanish."""
        with tempfile.TemporaryDirectory() as tmp:
            write_csv(os.path.join(tmp, "drivers.csv"),
                      ["driver_id", "driver_ref", "forename", "surname",
                       "dob", "nationality", "url", "code"],
                      [[1, "test_driver", "Test", "Driver", "not-a-date", "British", "", ""]])
            t = F1DataTransformer(raw_data_path=tmp + "/", processed_data_path=tmp + "/")
            df = t.transform_drivers()
            self.assertEqual(len(df), 1)
            self.assertTrue(pd.isna(df["dob"].iloc[0]))


class TestInvalidInputs(unittest.TestCase):
    """Negative tests: bad inputs must fail loudly, never silently corrupt data."""

    def test_header_only_csv_returns_empty_not_crash(self):
        """A CSV with all required headers but no data rows produces an empty DataFrame."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "circuits.csv")
            with open(path, "w") as f:
                f.write("circuit_ref,circuit_name,location,country,lat,lng,altitude,url\n")
            t = F1DataTransformer(raw_data_path=tmp + "/", processed_data_path=tmp + "/")
            df = t.transform_circuits()
            self.assertTrue(df.empty)

    def test_unmapped_ref_logs_warning_does_not_inject_zero(self):
        """An unmapped ref must drop the row entirely — not substitute driver_id=0."""
        with tempfile.TemporaryDirectory() as tmp:
            write_csv(os.path.join(tmp, "drivers.csv"),
                      ["driver_id", "driver_ref"], [[1, "max_verstappen"]])
            t = F1DataTransformer(raw_data_path=tmp + "/", processed_data_path=tmp + "/")
            df = pd.DataFrame({"driver_ref": ["ghost_driver"]})
            result = t._apply_ref_map(df, "driver_ref", "driver_id", "drivers.csv")
            self.assertTrue(result.empty)

    def test_strict_schema_rejects_missing_required_column(self):
        """Loading a DataFrame missing a required column raises ValueError in strict mode."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            loader = F1DataLoader(
                config={"type": "sqlite", "filename": db_path},
                processed_data_path=tmp + "/",
                strict_schema=True,
            )
            bad_df = pd.DataFrame({"not_circuit_id": [1]})
            with self.assertRaises(ValueError):
                loader._validate_df(bad_df, "circuits")

    def test_duplicate_primary_keys_in_ref_map_last_wins(self):
        """Duplicate refs in the lookup file: last value wins (documented behaviour)."""
        with tempfile.TemporaryDirectory() as tmp:
            write_csv(os.path.join(tmp, "drivers.csv"),
                      ["driver_id", "driver_ref"],
                      [[1, "max_verstappen"], [99, "max_verstappen"]])
            t = F1DataTransformer(raw_data_path=tmp + "/", processed_data_path=tmp + "/")
            df = pd.DataFrame({"driver_ref": ["max_verstappen"]})
            result = t._apply_ref_map(df, "driver_ref", "driver_id", "drivers.csv")
            self.assertEqual(len(result), 1)
            self.assertEqual(result["driver_id"].iloc[0], 99)


if __name__ == "__main__":
    unittest.main()
