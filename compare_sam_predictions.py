"""
Create side-by-side images with three columns:
SAM prediction | MedSAM prediction | Ground truth.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import zarr
from PIL import Image
import matplotlib.pyplot as plt
from image_utils import resize_mask_nearest


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 for visualization."""
    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    img = img.astype(np.float32)
    vmin, vmax = np.percentile(img, [1, 99])
    img = np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return (img * 255).astype(np.uint8)


def gt_to_rgb(target_db: np.ndarray, target_mb: np.ndarray, target_lb: np.ndarray) -> np.ndarray:
    """Map GT classes to RGB for visualization."""
    h, w = target_db.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    # Light = green, Moderate = yellow, Dense = red
    out[target_lb > 0] = (0, 255, 0)
    out[target_mb > 0] = (255, 255, 0)
    out[target_db > 0] = (255, 0, 0)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--level", default="4")
    parser.add_argument("--sam-dir", required=True)
    parser.add_argument("--medsam-dir", required=True)
    parser.add_argument("--sam2-dir")
    parser.add_argument("--medsam2-dir")
    parser.add_argument("--output-dir", default="prediction_comparison")
    parser.add_argument("--start-slice", type=int, default=0)
    parser.add_argument("--end-slice", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = args.base_dir
    subject = args.subject
    level = args.level
    pred_dir_sam = args.sam_dir
    pred_dir_medsam = args.medsam_dir
    pred_dir_sam2 = args.sam2_dir
    pred_dir_medsam2 = args.medsam2_dir
    out_dir = args.output_dir

    dirname = Path(base_dir) / f"sub-{subject}" / "micr"
    histology_paths = sorted(dirname.glob(f"sub-{subject}_*_stain-LY_DF.ome.zarr"))
    if not histology_paths:
        raise FileNotFoundError(f"No LY_DF OME-Zarr found in {dirname}")
    omz = zarr.open_group(histology_paths[0], mode="r")
    dense_masks = zarr.open_group(os.path.join(dirname, "masks", "Fiber_dense_bundle.ome.zarr"), mode="r")
    moderate_masks = zarr.open_group(os.path.join(dirname, "masks", "Fiber_moderate_bundle.ome.zarr"), mode="r")
    light_masks = zarr.open_group(os.path.join(dirname, "masks", "Fiber_light_bundle.ome.zarr"), mode="r")

    hist_z = omz[level]              # (C, Z, Y, X)
    dense_z = dense_masks[level]     # (C, Z, Y, X)
    moderate_z = moderate_masks[level]
    light_z = light_masks[level]

    num_slices = min(hist_z.shape[1], dense_z.shape[1], moderate_z.shape[1], light_z.shape[1])
    start = max(0, args.start_slice)
    end = num_slices if args.end_slice is None else min(num_slices, args.end_slice)
    if start >= end:
        raise ValueError(f"Invalid slice range: start={start}, end={end}")

    os.makedirs(out_dir, exist_ok=True)

    for z in range(start, end):
        img = np.transpose(hist_z[:, z, :, :], (1, 2, 0))
        target_db = dense_z[0, z]
        target_mb = moderate_z[0, z]
        target_lb = light_z[0, z]
        if target_db.shape != img.shape[:2]:
            target_db = resize_mask_nearest(target_db, img.shape[:2])
        if target_mb.shape != img.shape[:2]:
            target_mb = resize_mask_nearest(target_mb, img.shape[:2])
        if target_lb.shape != img.shape[:2]:
            target_lb = resize_mask_nearest(target_lb, img.shape[:2])

        gt_vis = gt_to_rgb(target_db, target_mb, target_lb)
        img_vis = normalize_to_uint8(img)

        columns = [("Original", img_vis)]

        pred_path_sam = os.path.join(pred_dir_sam, f"slice_{z:03d}_pred.png")
        pred_path_medsam = os.path.join(pred_dir_medsam, f"slice_{z:03d}_pred.png")
        if not (os.path.isfile(pred_path_sam) and os.path.isfile(pred_path_medsam)):
            continue

        pred_sam = np.array(Image.open(pred_path_sam))
        pred_medsam = np.array(Image.open(pred_path_medsam))
        if pred_sam.ndim == 2:
            pred_sam = np.repeat(pred_sam[:, :, None], 3, axis=2)
        if pred_medsam.ndim == 2:
            pred_medsam = np.repeat(pred_medsam[:, :, None], 3, axis=2)
        columns.append(("SAM", pred_sam.astype(np.uint8)))
        columns.append(("MedSAM", pred_medsam.astype(np.uint8)))

        if pred_dir_sam2:
            pred_path_sam2 = os.path.join(pred_dir_sam2, f"slice_{z:03d}_pred.png")
            if not os.path.isfile(pred_path_sam2):
                continue
            pred_sam2 = np.array(Image.open(pred_path_sam2))
            if pred_sam2.ndim == 2:
                pred_sam2 = np.repeat(pred_sam2[:, :, None], 3, axis=2)
            columns.append(("SAM2", pred_sam2.astype(np.uint8)))

        if pred_dir_medsam2:
            pred_path_medsam2 = os.path.join(pred_dir_medsam2, f"slice_{z:03d}_pred.png")
            if not os.path.isfile(pred_path_medsam2):
                continue
            pred_medsam2 = np.array(Image.open(pred_path_medsam2))
            if pred_medsam2.ndim == 2:
                pred_medsam2 = np.repeat(pred_medsam2[:, :, None], 3, axis=2)
            columns.append(("MedSAM2", pred_medsam2.astype(np.uint8)))

        columns.append(("Ground Truth", gt_vis))

        ncols = len(columns)
        fig_w = max(3.5 * ncols, 10)
        fig_h = 4.5
        fig, axes = plt.subplots(1, ncols, figsize=(fig_w, fig_h), facecolor="white")
        if ncols == 1:
            axes = [axes]

        for ax, (label, col_img) in zip(axes, columns):
            ax.imshow(col_img)
            ax.set_title(label, fontsize=12, color="black")
            ax.set_facecolor("white")
            ax.tick_params(axis="both", labelsize=8)

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"slice_{z:03d}_side_by_side.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
