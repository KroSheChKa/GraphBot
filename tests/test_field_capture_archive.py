import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core.field_capture_archive import save_clean_field_capture


class FieldCaptureArchiveTests(unittest.TestCase):
    def test_saves_lossless_clean_png(self):
        source = np.zeros((12, 20, 3), dtype=np.uint8)
        source[3:9, 5:15] = (11, 77, 233)

        with tempfile.TemporaryDirectory() as temp_dir:
            info = save_clean_field_capture(
                source,
                Path(temp_dir),
            )
            saved = cv2.imread(str(Path(temp_dir) / info["filename"]), cv2.IMREAD_COLOR)

        self.assertIsNotNone(saved)
        np.testing.assert_array_equal(saved, source)
        self.assertTrue(info["filename"].startswith("field_"))
        self.assertEqual(info["width"], 20)
        self.assertEqual(info["height"], 12)

    def test_avoids_timestamp_collision(self):
        source = np.zeros((2, 2, 3), dtype=np.uint8)
        captured_at = datetime(2026, 1, 2, 3, 4, 5, 6)

        with tempfile.TemporaryDirectory() as temp_dir:
            first = save_clean_field_capture(source, temp_dir, captured_at)
            second = save_clean_field_capture(source, temp_dir, captured_at)

        self.assertNotEqual(first["filename"], second["filename"])
        self.assertTrue(second["filename"].endswith("_01.png"))


if __name__ == "__main__":
    unittest.main()
