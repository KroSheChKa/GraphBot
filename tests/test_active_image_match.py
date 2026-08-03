import unittest

import cv2
import numpy as np

from core.detection import (
    detect_yellow_player_candidates,
    find_active_player,
    load_active_params,
    load_players_params,
)


class ActiveImageMatchTests(unittest.TestCase):
    @staticmethod
    def synthetic_scene():
        image = np.full((180, 240, 3), 255, dtype=np.uint8)

        # Active sprite: the inner appearance is deliberately irrelevant to
        # the detector; only the stable yellow body and outer red marker matter.
        cv2.circle(image, (40, 100), 13, (0, 0, 255), 2)
        cv2.ellipse(image, (40, 104), (7, 6), 0, 0, 360, (0, 220, 255), -1)
        cv2.rectangle(image, (34, 91), (46, 96), (255, 180, 0), -1)

        # Distractor: another player with a red name outline, but no circular
        # active marker. This is the failure mode of the old red-centroid path.
        cv2.ellipse(image, (100, 104), (7, 6), 0, 0, 360, (0, 220, 255), -1)
        cv2.rectangle(image, (94, 91), (106, 96), (255, 180, 0), -1)
        cv2.rectangle(image, (72, 74), (128, 88), (0, 0, 255), 1)
        return image

    def test_yellow_candidates_ignore_red_name_outline(self):
        candidates, _ = detect_yellow_player_candidates(
            self.synthetic_scene(), load_active_params()
        )
        self.assertEqual(len(candidates), 2)
        self.assertLess(abs(candidates[0][0] - 40), 1)
        self.assertLess(abs(candidates[1][0] - 100), 1)

    def test_circular_marker_beats_red_name_outline(self):
        result = find_active_player(
            self.synthetic_scene(),
            240,
            active_params=load_active_params(),
            players_params=load_players_params(),
        )
        self.assertEqual(result["method"], "yellow-body+red-ring-match")
        self.assertLess(np.hypot(result["active"][0] - 40, result["active"][1] - 100), 1.0)
        self.assertGreater(result["active_detection"]["ring_angular_coverage"], 0.9)


if __name__ == "__main__":
    unittest.main()
