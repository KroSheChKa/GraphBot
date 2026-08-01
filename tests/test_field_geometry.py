import unittest

from core.field_geometry import game_to_pixel, pixel_radius_to_game, pixel_to_game


class FieldGeometryTests(unittest.TestCase):
    def test_pixel_to_game_uses_independent_axes(self):
        self.assertEqual(pixel_to_game(0, 0, 770, 450), (-25.0, 15.0))
        self.assertEqual(pixel_to_game(770, 450, 770, 450), (25.0, -15.0))
        self.assertEqual(pixel_to_game(385, 225, 770, 450), (0.0, 0.0))

    def test_game_to_pixel_is_inverse_at_field_bounds(self):
        self.assertEqual(game_to_pixel(-25, 15, 770, 450), (0.0, 0.0))
        self.assertEqual(game_to_pixel(25, -15, 770, 450), (770.0, 450.0))

    def test_radius_uses_both_scales(self):
        radius = pixel_radius_to_game(10, 770, 450)
        expected = (10 * 50 / 770 + 10 * 30 / 450) / 2
        self.assertAlmostEqual(radius, expected)


if __name__ == "__main__":
    unittest.main()
