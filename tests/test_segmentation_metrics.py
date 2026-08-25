"""Unit tests for the shared segmentation metrics."""

import unittest

import numpy as np

from segmentation_metrics import (
    centerline_dice,
    classwise_iou,
    dice_score,
    instance_detection_metrics,
    intersection_over_union,
)


class SegmentationMetricsTests(unittest.TestCase):
    def test_empty_masks_are_a_perfect_match(self):
        empty = np.zeros((4, 4), dtype=np.uint8)
        self.assertEqual(intersection_over_union(empty, empty), 1.0)
        self.assertEqual(dice_score(empty, empty), 1.0)
        self.assertEqual(centerline_dice(empty, empty), 1.0)

    def test_binary_overlap(self):
        prediction = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        target = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        self.assertAlmostEqual(intersection_over_union(prediction, target), 1 / 3)
        self.assertAlmostEqual(dice_score(prediction, target), 0.5)

    def test_classwise_iou_preserves_class_order(self):
        prediction = np.array([[1, 2, 0]], dtype=np.uint8)
        target = np.array([[1, 0, 3]], dtype=np.uint8)
        self.assertEqual(classwise_iou(prediction, target), (1.0, 0.0, 0.0))

    def test_instance_metrics_count_overlapping_components(self):
        target = np.array([[1, 0, 1, 0], [1, 0, 1, 0]], dtype=np.uint8)
        prediction = np.array([[1, 0, 0, 1], [0, 0, 0, 1]], dtype=np.uint8)
        recalls, true_positives, false_positives, fdr = instance_detection_metrics(
            prediction, target
        )
        self.assertEqual(recalls, (0.5, 1.0, 1.0))
        self.assertEqual(true_positives, 1)
        self.assertEqual(false_positives, 1)
        self.assertEqual(fdr, 0.5)


if __name__ == "__main__":
    unittest.main()
