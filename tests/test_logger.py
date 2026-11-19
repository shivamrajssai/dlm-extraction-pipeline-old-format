"""
Unit test for src/common/logger.py
"""

import io
import logging
import os
import sys
import tempfile
import unittest

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.logger import setup_logger, logger


class TestLogger(unittest.TestCase):
    def test_default_logger_exists(self):
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "dlm_pipeline")
        self.assertEqual(logger.level, logging.INFO)

    def test_console_logging(self):
        test_logger = setup_logger("test_console", log_level=logging.DEBUG)
        self.assertEqual(len(test_logger.handlers), 1)
        self.assertIsInstance(test_logger.handlers[0], logging.StreamHandler)

    def test_file_logging(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            test_logger = setup_logger("test_file", log_level=logging.INFO, log_file=tmp_path)
            self.assertEqual(len(test_logger.handlers), 2)  # Console + File
            test_msg = "DLM Pipeline test log entry"
            test_logger.info(test_msg)

            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn(test_msg, content)
            self.assertIn("[INFO]", content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_json_lines_logging(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            import json
            test_logger = setup_logger("test_json", log_level=logging.INFO, json_log_file=tmp_path)
            test_logger.info("Structured event test")

            with open(tmp_path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]

            self.assertGreaterEqual(len(lines), 1)
            event = lines[0]
            self.assertEqual(event["level"], "INFO")
            self.assertEqual(event["activity"], "Structured event test")
            self.assertIn("timestamp", event)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
