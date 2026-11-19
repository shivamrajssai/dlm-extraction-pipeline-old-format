"""
Tests for IssueReporter in src.common.issue_reporter.
"""

import unittest
import tempfile
import os
import pandas as pd
from src.common.run_context import init_run_context
from src.common.issue_reporter import IssueReporter


class TestIssueReporter(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        init_run_context(
            run_id="run-test",
            batch_id="batch-test",
            upc="N_04006_03001_GJ",
            zone="B",
            phase="1",
            cycle=2,
        )

    def test_record_technical_exception(self):
        reporter = IssueReporter(tracebacks_dir=self.tmp_dir)
        try:
            raise ValueError("Test technical error")
        except Exception as e:
            rec = reporter.record_technical_exception(e, stage_id="INGEST_D2")

        self.assertEqual(len(reporter.issues), 1)
        self.assertEqual(rec.issue_type, "TECHNICAL_EXCEPTION")
        self.assertEqual(rec.exception_type, "ValueError")
        self.assertEqual(rec.stage_id, "INGEST_D2")
        self.assertTrue(rec.issue_id.startswith("ISS-"))
        self.assertIsNotNone(rec.traceback_ref)
        self.assertTrue(os.path.exists(rec.traceback_ref))

    def test_record_validation_discrepancy(self):
        reporter = IssueReporter(tracebacks_dir=self.tmp_dir)
        rec = reporter.record_validation_discrepancy(
            check_name="display_defect_id unique in table",
            scope="master_roughness_parent",
            failed_count=3,
            checked_count=238,
            detail="Duplicates found",
            section="V8 IDs",
        )
        self.assertEqual(len(reporter.issues), 1)
        self.assertEqual(rec.issue_type, "BUSINESS_VALIDATION_DISCREPANCY")
        self.assertEqual(rec.status, "OPEN")
        self.assertIn("3/238", rec.issue_description)

    def test_export_to_csv(self):
        reporter = IssueReporter(tracebacks_dir=self.tmp_dir)
        reporter.record_validation_discrepancy("check_1", "tbl", 1, 10)
        out_csv = os.path.join(self.tmp_dir, "issues.csv")
        reporter.save_issue_report(out_csv)
        self.assertTrue(os.path.exists(out_csv))
        df = pd.read_csv(out_csv)
        self.assertEqual(len(df), 1)
        self.assertEqual(df["issue_type"].iloc[0], "BUSINESS_VALIDATION_DISCREPANCY")


if __name__ == "__main__":
    unittest.main()
