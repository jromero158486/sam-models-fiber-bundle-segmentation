"""Image and mask shape-conversion helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def resize_mask_nearest(mask: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Resize a 2D mask with nearest-neighbor index sampling."""
    source_height, source_width = mask.shape[:2]
    target_height, target_width = target_shape[:2]
    if (source_height, source_width) == (target_height, target_width):
        return mask

    row_indices = np.linspace(0, source_height - 1, target_height).round().astype(int)
    column_indices = np.linspace(0, source_width - 1, target_width).round().astype(int)
    return mask[row_indices][:, column_indices]

