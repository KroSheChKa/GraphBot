import unittest

import numpy as np

from core.game_capture import (
    crop_client_field,
    auto_crop_white_grid,
    resize_to_grid,
    apply_size_caps,
    load_grid_capture_params,
)


class ClientFieldCropTests(unittest.TestCase):
    def test_crops_client_relative_margins(self):
        client = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)

        field = crop_client_field(
            client,
            {"margin_left": 2, "margin_top": 1, "margin_right": 3, "margin_bottom": 2},
        )

        np.testing.assert_array_equal(field, client[1:4, 2:5])

    def test_rejects_empty_crop(self):
        client = np.zeros((4, 4, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            crop_client_field(client, {"margin_left": 4})


class AutoCropWhiteGridTests(unittest.TestCase):
    def test_crops_to_largest_white_region(self):
        # 100x200 image: a white 50x50 block plus a small white speck.
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[20:70, 40:90] = 255
        img[90, 150] = 255

        cropped, box = auto_crop_white_grid(img, {"min_component": 1})

        self.assertIsNotNone(box)
        x, y, w, h = box
        self.assertEqual((x, y, w, h), (40, 20, 50, 50))
        self.assertEqual(cropped.shape, (50, 50, 3))
        self.assertTrue(np.all(cropped == 255))

    def test_returns_copy_when_no_white_region(self):
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        cropped, box = auto_crop_white_grid(img)
        self.assertIsNone(box)
        np.testing.assert_array_equal(cropped, img)


class ResizeToGridTests(unittest.TestCase):
    def test_preserves_isotropic_scale_and_centers(self):
        # 200x100 source (2:1) into 1000x600 (5:3): scale limited by height.
        src = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
        out = resize_to_grid(src, 1000, 600)
        self.assertEqual(out.shape, (600, 1000, 3))
        # scale = max(1000/200, 600/100) = 6 -> 1200x600, center-cropped to 1000x600
        # No padding: canvas fully filled, overflow cropped symmetrically.
        self.assertFalse(np.all(out[:50, :, :] == 255))
        self.assertFalse(np.all(out[550:, :, :] == 255))

    def test_exact_match_returns_target_size(self):
        src = np.full((600, 1000, 3), 128, dtype=np.uint8)
        out = resize_to_grid(src, 1000, 600)
        self.assertEqual(out.shape, (600, 1000, 3))


class GridCaptureParamsTests(unittest.TestCase):
    def test_defaults_are_target_1000x600(self):
        params = load_grid_capture_params(path="__nonexistent__.json")
        self.assertEqual(params["target_width"], 1000)
        self.assertEqual(params["target_height"], 600)


class ApplySizeCapsTests(unittest.TestCase):
    def test_noop_when_within_caps(self):
        img = np.full((300, 500, 3), 128, dtype=np.uint8)
        out = apply_size_caps(img)
        np.testing.assert_array_equal(out, img)
        self.assertEqual(out.shape, (300, 500, 3))

    def test_downscales_when_exceeding_max_preserves_aspect(self):
        img = np.zeros((2000, 3000, 3), dtype=np.uint8)
        img[100:1900, 200:2800] = 128
        out = apply_size_caps(img)
        h, w = out.shape[:2]
        self.assertLessEqual(w, 1600)
        self.assertLessEqual(h, 960)
        # Aspect preserved (3000x2000 -> 1.5; output w/h should equal 1.5).
        self.assertAlmostEqual(w / h, 3000 / 2000, places=2)
        # Graph content preserved, not truncated to white.
        self.assertTrue(np.any(out == 128))

    def test_upscales_when_below_min(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        out = apply_size_caps(img)
        h, w = out.shape[:2]
        self.assertGreaterEqual(w, 400)
        self.assertGreaterEqual(h, 240)
        self.assertTrue(np.any(out == 128))


if __name__ == "__main__":
    unittest.main()
