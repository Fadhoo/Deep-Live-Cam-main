"""A body transfer should affect the person while preserving the scene."""

import unittest
from types import SimpleNamespace

import numpy as np

from modules.processors.frame import body_swapper


class _Mask:
    def __init__(self, pixels):
        self.pixels = pixels

    def numpy_view(self):
        return self.pixels


class BodySwapperTests(unittest.TestCase):
    def test_composite_transfers_person_and_preserves_background(self):
        source = np.full((200, 160, 3), (0, 0, 220), dtype=np.uint8)
        target = np.full((200, 160, 3), (220, 0, 0), dtype=np.uint8)
        mask = np.zeros((200, 160), dtype=np.float32)
        mask[20:180, 35:125] = 1
        positions = {
            0: (.5, .16), 11: (.35, .3), 12: (.65, .3),
            13: (.28, .45), 14: (.72, .45), 15: (.24, .6), 16: (.76, .6),
            23: (.4, .58), 24: (.6, .58), 25: (.39, .72), 26: (.61, .72),
            27: (.38, .88), 28: (.62, .88), 29: (.38, .9), 30: (.62, .9),
            31: (.38, .91), 32: (.62, .91),
        }
        landmarks = [
            SimpleNamespace(x=positions.get(i, (.5, .5))[0],
                            y=positions.get(i, (.5, .5))[1], visibility=1.0)
            for i in range(33)
        ]
        result = SimpleNamespace(
            pose_landmarks=[landmarks], segmentation_masks=[_Mask(mask)]
        )

        output = body_swapper._composite(source, result, target, result)

        self.assertGreater(int(output[100, 80, 2]), 150)
        self.assertEqual(output[32, 80].tolist(), [220, 0, 0])
        self.assertEqual(output[0, 0].tolist(), [220, 0, 0])

        shifted_mask = np.zeros_like(mask)
        shifted_mask[20:180, 51:141] = 1
        shifted_landmarks = [
            SimpleNamespace(x=point.x + .1, y=point.y, visibility=1.0)
            for point in landmarks
        ]
        shifted_result = SimpleNamespace(
            pose_landmarks=[shifted_landmarks],
            segmentation_masks=[_Mask(shifted_mask)],
        )
        shifted = body_swapper._composite(source, result, target, shifted_result)
        self.assertGreater(int(shifted[100, 96, 2]), 150)
        self.assertEqual(shifted[100, 40].tolist(), [220, 0, 0])


if __name__ == "__main__":
    unittest.main()
