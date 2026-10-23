"""Skin transfer must leave the target clothes and scene untouched."""

import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from modules.processors.frame import body_swapper


class _Mask:
    def __init__(self, pixels):
        self.pixels = pixels

    def numpy_view(self):
        return self.pixels


def _person(skin, clothing, background=(12, 18, 25)):
    image = np.full((200, 160, 3), background, dtype=np.uint8)
    mask = np.zeros((200, 160), dtype=np.float32)
    cv2.rectangle(image, (35, 20), (125, 179), clothing, -1)
    cv2.rectangle(mask, (35, 20), (125, 179), 1, -1)
    cv2.circle(image, (80, 32), 14, skin, -1)
    cv2.line(image, (56, 60), (45, 90), skin, 10)
    cv2.line(image, (45, 90), (38, 120), skin, 9)
    cv2.line(image, (104, 60), (115, 90), skin, 10)
    cv2.line(image, (115, 90), (122, 120), skin, 9)
    cv2.line(image, (64, 140), (60, 175), skin, 13)
    cv2.line(image, (96, 140), (100, 175), skin, 13)
    positions = {
        0: (.5, .16), 2: (.46, .13), 5: (.54, .13),
        7: (.42, .16), 8: (.58, .16),
        11: (.35, .30), 12: (.65, .30),
        13: (.28, .45), 14: (.72, .45),
        15: (.24, .60), 16: (.76, .60),
        19: (.22, .64), 20: (.78, .64),
        23: (.40, .58), 24: (.60, .58),
        25: (.39, .72), 26: (.61, .72),
        27: (.38, .88), 28: (.62, .88),
        31: (.38, .91), 32: (.62, .91),
    }
    landmarks = [
        SimpleNamespace(x=positions.get(i, (.5, .5))[0],
                        y=positions.get(i, (.5, .5))[1], visibility=1.0)
        for i in range(33)
    ]
    result = SimpleNamespace(pose_landmarks=[landmarks],
                             segmentation_masks=[_Mask(mask)])
    return image, result


class BodySwapperTests(unittest.TestCase):
    def test_recolors_exposed_skin_but_keeps_target_clothes_and_background(self):
        source, source_result = _person((35, 65, 105), (0, 0, 220))
        target, target_result = _person((80, 130, 175), (220, 0, 0))
        output = body_swapper._composite(source, source_result,
                                         target, target_result)

        self.assertEqual(output[100, 80].tolist(), target[100, 80].tolist())
        self.assertEqual(output[0, 0].tolist(), target[0, 0].tolist())
        self.assertLess(int(output[90, 45, 2]), int(target[90, 45, 2]) - 15)
        self.assertLess(int(output[160, 61, 2]), int(target[160, 61, 2]) - 15)
        # The source's red shirt must not appear anywhere in the target.
        self.assertEqual(output[100, 80, 2], 0)

    def test_uncertain_face_leaves_frame_unchanged(self):
        source, source_result = _person((35, 65, 105), (0, 0, 220))
        target, target_result = _person((80, 130, 175), (220, 0, 0))
        target_result.pose_landmarks[0][0].visibility = 0.0
        output = body_swapper._composite(source, source_result,
                                         target, target_result)
        np.testing.assert_array_equal(output, target)

    def test_face_only_source_can_supply_skin_reference(self):
        source = np.full((100, 100, 3), (35, 65, 105), np.uint8)
        face = SimpleNamespace(bbox=np.array([20, 10, 80, 90]))
        reference = body_swapper._skin_reference(source, None, face)
        self.assertIsNotNone(reference)
        expected = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)[0, 0]
        np.testing.assert_allclose(reference, expected, atol=1)

    def test_skin_coloured_shirt_outside_limb_regions_is_untouched(self):
        source, source_result = _person((35, 65, 105), (0, 0, 220))
        target, target_result = _person((80, 130, 175), (80, 130, 175))
        output = body_swapper._composite(source, source_result,
                                         target, target_result)
        np.testing.assert_array_equal(output[100, 80], target[100, 80])


if __name__ == "__main__":
    unittest.main()
