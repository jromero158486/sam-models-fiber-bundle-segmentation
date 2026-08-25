"""
FINAL robust lazy evaluation script for fiber bundle segmentation.

✔ Lazy OME-Zarr reading
✔ ROI from OUTLINE (channel-agnostic, scale-agnostic)
✔ Slice-wise processing
✔ Safe cropping
✔ Skips empty slices cleanly
✔ Object + pixel metrics
✔ CSV export
"""

import argparse
import os
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import zarr
from skimage.measure import label

from segmentation_metrics import (
    centerline_dice as compute_cldice,
    dice_score,
    intersection_over_union as calculate_iou,
)
from image_cropping import crop_mask_like_data


# =============================================================================
# METRICS
# =============================================================================

def calculate_class_wise_pixel_recall(pred, target):
    pred_bin = (pred > 0)
    recalls = []
    for class_id in [1, 2, 3]:
        gt = (target == class_id)
        total = np.sum(gt)
        recalls.append(np.nan if total == 0 else float(np.sum(pred_bin & gt) / total))
    return tuple(recalls)


def get_clusterwise_metrics_typewise(pred, target_map):
    sens = []
    for ct in [1, 2, 3]:
        labeltarget, num_target = label(target_map == ct, return_num=True)
        if num_target > 0:
            touched = np.unique(labeltarget[pred > 0])
            touched = touched[touched != 0]
            sens.append(float(len(touched) / num_target))
        else:
            sens.append(np.nan)

    labelpred, num_pred = label(pred > 0, return_num=True)
    touched_pred = np.unique(labelpred[target_map > 0])
    touched_pred = touched_pred[touched_pred != 0]
    tp = int(len(touched_pred))
    fp = int(num_pred - tp)
    fdr = 0.0 if num_pred == 0 else float(fp / num_pred)

    return sens[2], sens[1], sens[0], tp, fp, fdr


# =============================================================================
# ZARR HELPERS
# =============================================================================

def open_ome_zarr_level(path, preferred_level="4"):
    print(f"[DEBUG] Opening: {path}")
    root = zarr.open(path, mode="r")

    if isinstance(root, zarr.Array):
        return root

    keys = list(root.array_keys())
    print("[DEBUG] Found array keys:", keys)

    if preferred_level in keys:
        print(f"[DEBUG] Using level: {preferred_level}")
        return root[preferred_level]
    if "0" in keys:
        print("[DEBUG] Using level: 0")
        return root["0"]
    return root[keys[0]]


def get_num_slices(arr):
    if arr.ndim == 4 and arr.shape[0] in (1,2,3,4):
        return arr.shape[1]
    if arr.ndim == 4:
        return arr.shape[0]
    if arr.ndim == 3:
        return arr.shape[0]
    raise ValueError("Unsupported zarr shape:", arr.shape)


def get_slice_hw_c(arr, z_idx):
    if arr.ndim == 4 and arr.shape[0] in (1,2,3,4):
        sl = arr.oindex[:, z_idx, :, :]
        return np.transpose(np.asarray(sl), (1,2,0))
    if arr.ndim == 4:
        return np.asarray(arr.oindex[z_idx, :, :, :])
    if arr.ndim == 3:
        return np.asarray(arr.oindex[z_idx, :, :])[..., None]
    raise ValueError("Unsupported zarr shape:", arr.shape)

def get_slice_crop_hw_c(arr, z_idx, y0, y1, x0, x1):
    if arr.ndim == 4 and arr.shape[0] in (1,2,3,4):
        sl = arr.oindex[:, z_idx, y0:y1+1, x0:x1+1]
        return np.transpose(np.asarray(sl), (1,2,0))
    if arr.ndim == 4:
        return np.asarray(arr.oindex[z_idx, y0:y1+1, x0:x1+1, :])
    if arr.ndim == 3:
        return np.asarray(arr.oindex[z_idx, y0:y1+1, x0:x1+1])[..., None]
    raise ValueError("Unsupported zarr shape:", arr.shape)


# =============================================================================
# SUBJECT LOADER
# =============================================================================

def load_subject(subject, base_dir, preferred_level="4"):
    subject_dir = Path(base_dir) / f"sub-{subject}" / "micr"
    histology_paths = sorted(subject_dir.glob(f"sub-{subject}_*_stain-LY_DF.ome.zarr"))
    if not histology_paths:
        raise FileNotFoundError(f"No LY_DF OME-Zarr found in {subject_dir}")

    masks_dir = subject_dir / "masks"
    outline_path = masks_dir / "Outline_mask.ome.zarr"
    if not outline_path.exists():
        outline_path = masks_dir / "Outline.ome.zarr"

    hist = open_ome_zarr_level(histology_paths[0], preferred_level)
    outline = open_ome_zarr_level(outline_path, preferred_level)
    dense = open_ome_zarr_level(masks_dir / "Fiber_dense_bundle.ome.zarr", preferred_level)
    moderate = open_ome_zarr_level(masks_dir / "Fiber_moderate_bundle.ome.zarr", preferred_level)
    light = open_ome_zarr_level(masks_dir / "Fiber_light_bundle.ome.zarr", preferred_level)

    z = min(get_num_slices(hist), get_num_slices(outline),
            get_num_slices(dense), get_num_slices(moderate),
            get_num_slices(light))

    print("[DEBUG] Z slices =", z)
    slides = [f"{i:03d}" for i in range(1, min(z,35)+1)]

    return hist, outline, dense, moderate, light, slides


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_subject(subject, predicted_root, base_dir, preferred_level="4"):
    print(f"\n================= Evaluating {subject} (FINAL) =================")

    hist, outline, dense, moderate, light, slides = load_subject(
        subject, base_dir, preferred_level
    )

    results_dir = os.path.join(predicted_root, subject)
    results = []
    skipped = []

    for ix, slide in enumerate(slides):
        outline_slice = get_slice_hw_c(outline, ix)

        # -------- ROBUST ROI FROM OUTLINE --------
        mask2d = np.max(outline_slice, axis=2).astype(np.float32)
        coords = np.where(mask2d > 1e-6)

        if coords[0].size == 0:
            print(f"[SKIP ROI EMPTY] slice {slide}")
            skipped.append(slide)
            continue

        y0, y1 = coords[0].min(), coords[0].max()
        x0, x1 = coords[1].min(), coords[1].max()
        boundaries = (y0, y1, x0, x1)

        brain_mask = outline_slice[y0:y1+1, x0:x1+1, 0]
        if brain_mask.size == 0 or brain_mask.sum()==0:
            print(f"[SKIP BRAIN MASK EMPTY] slice {slide}")
            skipped.append(slide)
            continue

        target_db = get_slice_crop_hw_c(dense, ix, y0, y1, x0, x1)[:, :, 0]
        target_mb = get_slice_crop_hw_c(moderate, ix, y0, y1, x0, x1)[:, :, 0]
        target_lb = get_slice_crop_hw_c(light, ix, y0, y1, x0, x1)[:, :, 0]

        target_map = np.zeros_like(target_db, dtype=np.uint8)
        target_map[target_lb>0]=1
        target_map[target_mb>0]=2
        target_map[target_db>0]=3

        pred_path = os.path.join(results_dir, f"bundle_{slide}_0000_pred.png")
        if not os.path.exists(pred_path):
            print(f"[SKIP NO PRED] slice {slide}")
            skipped.append(slide)
            continue

        pred_full = (np.array(Image.open(pred_path))>127).astype(np.uint8)
        pred = crop_mask_like_data(pred_full[...,None], boundaries)[:,:,0]

        if pred.shape != brain_mask.shape:
            print(f"[SKIP SHAPE MISMATCH] slice {slide}, pred={pred.shape}, mask={brain_mask.shape}")
            skipped.append(slide)
            continue

        pred *= brain_mask
        target_map *= brain_mask

        sensd, sensm, sensl, tp, fp, fdr = get_clusterwise_metrics_typewise(pred, target_map)
        dice = dice_score(pred, target_map)
        iou = calculate_iou(pred, target_map)
        cld = compute_cldice(pred, target_map)
        rec_l, rec_m, rec_d = calculate_class_wise_pixel_recall(pred, target_map)

        results.append({
            "slice": slide,
            "obj_sens_dense": sensd,
            "obj_sens_moderate": sensm,
            "obj_sens_light": sensl,
            "obj_TP": tp,
            "obj_FP": fp,
            "obj_FDR": fdr,
            "pixel_dice": dice,
            "pixel_iou": iou,
            "pixel_cldice": cld,
            "pixel_recall_dense": rec_d,
            "pixel_recall_moderate": rec_m,
            "pixel_recall_light": rec_l
        })

        print(f"[OK] slice {slide} evaluated")

    print("\nSkipped slices:", skipped)
    print("Total evaluated:", len(results))

    if len(results)==0:
        print("No valid slices; no CSV was created.")
        return

    df = pd.DataFrame(results)
    out_csv = os.path.join(results_dir,"bundle_evaluation_metrics.csv")
    df.to_csv(out_csv,index=False)
    print("\nCSV saved at:", out_csv)


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted-root", required=True)
    parser.add_argument("--base-dir", required=True, help="Root containing sub-<subject>/micr")
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--level", default="4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for subject in args.subjects:
        try:
            evaluate_subject(
                subject,
                args.predicted_root,
                args.base_dir,
                preferred_level=args.level,
            )
        except Exception as exc:
            print(f"Evaluation failed for {subject}: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
