import unittest

import numpy as np

from core.game_capture import crop_client_field


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


if __name__ == "__main__":
    unittest.main()
