"""
Tests for RunContext and stage_scope in src.common.run_context.
"""

import unittest
import time
from src.common.run_context import (
    RunContext,
    init_run_context,
    get_current_context,
    stage_scope,
)


class TestRunContext(unittest.TestCase):

    def test_init_and_get_context(self):
        ctx = init_run_context(
            batch_id="batch-001",
            upc="N_01001_02001_GJ",
            zone="B",
            phase="1",
            cycle=2,
        )
        self.assertEqual(ctx.batch_id, "batch-001")
        self.assertEqual(ctx.upc, "N_01001_02001_GJ")
        self.assertEqual(ctx.zone, "B")
        self.assertEqual(ctx.phase, "1")
        self.assertEqual(ctx.cycle, 2)

        cur = get_current_context()
        self.assertEqual(cur.upc, "N_01001_02001_GJ")
        d = cur.to_dict()
        self.assertEqual(d["upc"], "N_01001_02001_GJ")
        self.assertEqual(d["batch_id"], "batch-001")

    def test_stage_scope_timing(self):
        init_run_context(upc="N_TEST")
        with stage_scope("TEST_STAGE", "Test Stage Description") as stg:
            cur = get_current_context()
            self.assertEqual(cur.stage_id, "TEST_STAGE")
            self.assertEqual(cur.stage_name, "Test Stage Description")
            time.sleep(0.01)

        self.assertGreater(stg.duration_ms, 5.0)
        cur_after = get_current_context()
        self.assertIsNone(cur_after.stage_id)


if __name__ == "__main__":
    unittest.main()
