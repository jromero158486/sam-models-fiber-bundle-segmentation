"""Reusable metrics for binary and class-labelled segmentation masks."""

from __future__ import annotations

import numpy as np
from skimage.measure import label
from skimage.morphology import skeletonize


def intersection_over_union(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return binary intersection over union, treating two empty masks as equal."""
    prediction = np.asarray(prediction) > 0
    target = np.asarray(target) > 0
    intersection = np.count_nonzero(prediction & target)
    union = np.count_nonzero(prediction | target)
    return 1.0 if union == 0 else float(intersection / union)


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return the Sørensen-Dice score for two binary masks."""
    prediction = np.asarray(prediction) > 0
    target = np.asarray(target) > 0
    denominator = np.count_nonzero(prediction) + np.count_nonzero(target)
    if denominator == 0:
        return 1.0
    intersection = np.count_nonzero(prediction & target)
    return float(2 * intersection / denominator)


def centerline_dice(
    prediction: np.ndarray, target: np.ndarray, smooth: float = 1e-6
) -> float:
    """Return clDice, which measures agreement between mask centerlines."""
    prediction = np.asarray(prediction) > 0
    target = np.asarray(target) > 0
    if not prediction.any() and not target.any():
        return 1.0

    prediction_skeleton = skeletonize(prediction)
    target_skeleton = skeletonize(target)
    precision = (
        np.count_nonzero(prediction_skeleton & target) + smooth
    ) / (np.count_nonzero(prediction_skeleton) + smooth)
    sensitivity = (
        np.count_nonzero(target_skeleton & prediction) + smooth
    ) / (np.count_nonzero(target_skeleton) + smooth)
    denominator = precision + sensitivity
    return 0.0 if denominator == 0 else float(2 * precision * sensitivity / denominator)


def classwise_iou(
    prediction: np.ndarray, target: np.ndarray, class_ids: tuple[int, ...] = (1, 2, 3)
) -> tuple[float, ...]:
    """Return IoU for each requested class ID in the supplied order."""
    return tuple(
        intersection_over_union(prediction == class_id, target == class_id)
        for class_id in class_ids
    )


def instance_detection_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    class_ids: tuple[int, ...] = (1, 2, 3),
) -> tuple[tuple[float, ...], int, int, float]:
    """Measure per-class target recall and false discoveries by component overlap."""
    recalls = []
    for class_id in class_ids:
        target_labels, target_count = label(target == class_id, return_num=True)
        detected = np.unique(target_labels[np.asarray(prediction) > 0])
        true_positives = np.count_nonzero(detected)
        recalls.append(1.0 if target_count == 0 else true_positives / target_count)

    prediction_labels, prediction_count = label(
        np.asarray(prediction) > 0, return_num=True
    )
    overlapping = np.unique(prediction_labels[np.asarray(target) > 0])
    true_positives = int(np.count_nonzero(overlapping))
    false_positives = prediction_count - true_positives
    false_discovery_rate = (
        0.0 if prediction_count == 0 else false_positives / prediction_count
    )
    return tuple(recalls), true_positives, false_positives, false_discovery_rate

