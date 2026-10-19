import unittest

import cv2
import numpy as np

from core.detection import find_all_players


class EnemyDetectionTests(unittest.TestCase):
    @staticmethod
    def synthetic_scene():
        image = np.full((180, 240, 3), 255, dtype=np.uint8)

        # Two enemy sprites on the right side.
        for center in ((150, 90), (200, 110)):
            cv2.circle(image, center, 8, (100, 100, 100), -1)

        # One allied sprite on the left side.
        cv2.circle(image, (60, 100), 8, (100, 100, 100), -1)
        return image

    def test_enemies_are_detected_and_sorted(self):
        result = find_all_players(self.synthetic_scene(), 240)
        self.assertEqual(len(result["enemies"]), 2)
        xs = [point[0] for point in result["enemies"]]
        self.assertEqual(xs, sorted(xs))
        self.assertTrue(all(x > 120 for x in xs))

    def test_no_enemies_returns_empty(self):
        image = np.full((180, 240, 3), 255, dtype=np.uint8)
        cv2.circle(image, (60, 100), 8, (100, 100, 100), -1)
        result = find_all_players(image, 240)
        self.assertEqual(result["enemies"], [])


if __name__ == "__main__":
    unittest.main()
