import unittest

import cv2
import numpy as np

from core.detection import glow_centroid, refine_active_center


class ActiveRefinementTests(unittest.TestCase):
    def test_refinement_keeps_ring_center(self):
        height, width = 160, 220
        bgr = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (93, 71)
        cv2.circle(mask, center, 13, 255, 2)
        cv2.circle(bgr, center, 13, (0, 0, 255), 2)

        result = refine_active_center(
            bgr,
            mask,
            (93, 71, 8),
            (93.0, 71.0),
            {"refine_scale": 10, "review_uncertainty_game": 0.05},
        )

        self.assertLessEqual(np.hypot(result["center"][0] - center[0], result["center"][1] - center[1]), 0.5)
        self.assertEqual(result["method"], "red-ring+player-circle")
        self.assertGreater(result["confidence"], 0.65)

    def test_non_integer_center_survives_noise_and_blur(self):
        height, width = 180, 240
        base_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(base_mask, (96, 74), 13, 255, 2)
        shift = np.float32([[1, 0, 0.37], [0, 1, -0.42]])
        clean_mask = cv2.warpAffine(base_mask, shift, (width, height), flags=cv2.INTER_LINEAR)

        blurred_mask = cv2.GaussianBlur(clean_mask, (5, 5), 0)
        blurred_mask[blurred_mask > 40] = 255
        seed = (96.3, 73.6, 13)
        clean = refine_active_center(
            np.zeros((height, width, 3), dtype=np.uint8),
            clean_mask,
            seed,
            seed[:2],
            {"refine_scale": 10},
        )
        blurred = refine_active_center(
            np.zeros((height, width, 3), dtype=np.uint8),
            blurred_mask,
            seed,
            seed[:2],
            {"refine_scale": 10},
        )

        self.assertLess(np.hypot(clean["center"][0] - 96.37, clean["center"][1] - 73.58), 0.75)
        self.assertLess(np.hypot(blurred["center"][0] - 96.37, blurred["center"][1] - 73.58), 0.75)
        self.assertLess(np.hypot(clean["center"][0] - blurred["center"][0], clean["center"][1] - blurred["center"][1]), 0.2)

    def test_red_component_wins_over_distant_hough_candidate(self):
        height, width = 160, 220
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (48, 112), 13, 255, 2)

        result = refine_active_center(
            np.zeros((height, width, 3), dtype=np.uint8),
            mask,
            (72, 112, 8),
            (48.0, 112.0),
            {"refine_scale": 10},
        )

        self.assertLessEqual(np.hypot(result["center"][0] - 48, result["center"][1] - 112), 1.0)

    def test_left_only_rejects_red_components_on_right(self):
        mask = np.zeros((120, 200), dtype=np.uint8)
        cv2.rectangle(mask, (20, 50), (38, 68), 255, -1)
        cv2.rectangle(mask, (150, 50), (199, 68), 255, -1)

        center, area = glow_centroid(mask, 20, max_x=100)

        self.assertIsNotNone(center)
        self.assertGreater(area, 0)
        self.assertLess(center[0], 100)


if __name__ == "__main__":
    unittest.main()
