"""
Evaluate SAM pipeline predictions against fiber bundle ground truth.

Predictions are expected as:
    slice_{z:03d}_pred.png
from ``sam_multi_instance.py``.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
from PIL import Image
from image_utils import resize_mask_nearest
from segmentation_metrics import (
    centerline_dice as compute_cldice,
    classwise_iou as calculate_class_wise_iou,
    dice_score,
    instance_detection_metrics,
    intersection_over_union as calculate_iou,
)


def get_clusterwise_metrics_typewise(pred, col_target):
    sens, tp, fp, fdr = instance_detection_metrics(pred, col_target)
    sens_light, sens_mod, sens_dense = sens
    return sens_dense, sens_mod, sens_light, tp, fp, fdr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted-dir", required=True, help="Directory containing slice_*_pred.png")
    parser.add_argument("--base-dir", required=True, help="Root containing sub-<subject>/micr")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--level", default="4")
    parser.add_argument("--start-slice", type=int, default=0)
    parser.add_argument("--end-slice", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = args.base_dir
    subject = args.subject
    level = args.level
    predicted_folder = args.predicted_dir

    dirname = Path(base_dir) / f"sub-{subject}" / "micr"
    histology_paths = sorted(dirname.glob(f"sub-{subject}_*_stain-LY_DF.ome.zarr"))
    if not histology_paths:
        raise FileNotFoundError(f"No LY_DF OME-Zarr found in {dirname}")
    histology = zarr.open_group(histology_paths[0], mode="r")
    outline = zarr.open_group(os.path.join(dirname, "masks", "Outline.ome.zarr"), mode="r")
    dense_masks = zarr.open_group(os.path.join(dirname, "masks", "Fiber_dense_bundle.ome.zarr"), mode="r")
    moderate_masks = zarr.open_group(os.path.join(dirname, "masks", "Fiber_moderate_bundle.ome.zarr"), mode="r")
    light_masks = zarr.open_group(os.path.join(dirname, "masks", "Fiber_light_bundle.ome.zarr"), mode="r")

    hist_z = histology[level]              # (C, Z, Y, X)
    outline_z = outline[level]             # (C, Z, Y, X) or compatible
    dense_z = dense_masks[level]           # (C, Z, Y, X)
    moderate_z = moderate_masks[level]     # (C, Z, Y, X)
    light_z = light_masks[level]           # (C, Z, Y, X)

    # Use lazy shapes; do not materialize full arrays in memory.
    num_slices = min(
        hist_z.shape[1],
        outline_z.shape[1],
        dense_z.shape[1],
        moderate_z.shape[1],
        light_z.shape[1],
    )
    start_slice = max(0, args.start_slice)
    end_slice = num_slices if args.end_slice is None else min(num_slices, args.end_slice)

    results = []

    for z in range(start_slice, end_slice):
        pred_path = os.path.join(predicted_folder, f"slice_{z:03d}_pred.png")
        if not os.path.isfile(pred_path):
            continue

        image = np.transpose(hist_z[:, z, :, :], (1, 2, 0))
        roi_mask = outline_z[0, z]
        if roi_mask.shape != image.shape[:2]:
            roi_mask = resize_mask_nearest(roi_mask, image.shape[:2])
        roi_mask = (roi_mask > 0)

        target_db = dense_z[0, z]
        target_mb = moderate_z[0, z]
        target_lb = light_z[0, z]
        if target_db.shape != image.shape[:2]:
            target_db = resize_mask_nearest(target_db, image.shape[:2])
        if target_mb.shape != image.shape[:2]:
            target_mb = resize_mask_nearest(target_mb, image.shape[:2])
        if target_lb.shape != image.shape[:2]:
            target_lb = resize_mask_nearest(target_lb, image.shape[:2])

        predicted_np = np.array(Image.open(pred_path))
        predicted_np = (predicted_np > 127).astype(np.uint8)

        target_map = np.zeros_like(target_db, dtype=np.uint8)
        target_map[target_lb > 0] = 1
        target_map[target_mb > 0] = 2
        target_map[target_db > 0] = 3

        target = target_map * roi_mask
        pred = predicted_np * roi_mask

        sensds, sensms, sensls, tps, fps, fdrs = get_clusterwise_metrics_typewise(pred, target)
        score = dice_score(pred, target)
        iou_score = calculate_iou(pred, target)
        iou_light, iou_moderate, iou_dense = calculate_class_wise_iou(pred, target)
        cldice = compute_cldice(pred, target)

        results.append({
            "slice": z,
            "file": f"{z:03d}",
            "sensitivity_dense": sensds,
            "sensitivity_moderate": sensms,
            "sensitivity_light": sensls,
            "false_positives_vaan_fun": fps,
            "true_positives_vaan_fun": tps,
            "fdr": fdrs,
            "dice_score": score,
            "iou_score": iou_score,
            "iou_dense": iou_dense,
            "iou_moderate": iou_moderate,
            "iou_light": iou_light,
            "cldice": cldice,
        })

    df = pd.DataFrame(results)
    out_csv = os.path.join(predicted_folder, "bundle_evaluation_results_inside_mask.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(df)} slices)")


if __name__ == "__main__":
    main()
