"""
Tests for BatchMonitor in src.common.batch_monitor.
"""

import unittest
import tempfile
import os
import json
import pandas as pd
from src.common.batch_monitor import BatchMonitor


class TestBatchMonitor(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def test_batch_lifecycle_accounting(self):
        monitor = BatchMonitor(batch_id="batch-unit-test", batch_file="test_upcs.txt")
        monitor.set_total_upcs(3)

        # 1. Start and complete UPC 1
        monitor.start_upc("UPC_1", "B", "1", 2)
        monitor.complete_upc("UPC_1", duration_ms=1200.0, tables_written=14, records_written=50000)

        # 2. Start and fail UPC 2
        monitor.start_upc("UPC_2", "B", "1", 2)
        monitor.fail_upc("UPC_2", duration_ms=500.0, error_message="File not found")

        # 3. Skip UPC 3
        monitor.skip_upc("UPC_3", "B", "1", 2, reason="Invalid survey parameters")

        summary = monitor.finish_batch()
        self.assertEqual(summary["total_upcs_received"], 3)
        self.assertEqual(summary["upcs_completed"], 1)
        self.assertEqual(summary["upcs_failed"], 1)
        self.assertEqual(summary["upcs_skipped"], 1)
        self.assertEqual(summary["upcs_not_processed"], 0)
        self.assertEqual(summary["total_tables_written"], 14)

        # Save and verify
        csv_path = os.path.join(self.tmp_dir, "batch_summary.csv")
        json_path = os.path.join(self.tmp_dir, "run_summary.json")
        monitor.save_batch_summary(csv_path, json_path=json_path)

        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))

        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 3)

        with open(json_path, "r") as f:
            j = json.load(f)
        self.assertEqual(j["batch_id"], "batch-unit-test")
        self.assertEqual(j["upcs_completed"], 1)


if __name__ == "__main__":
    unittest.main()
