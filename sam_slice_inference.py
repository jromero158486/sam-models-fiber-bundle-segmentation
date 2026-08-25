import argparse
import os
import time
from pathlib import Path
import numpy as np
import zarr
import torch
from PIL import Image
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects, binary_closing, disk

from segment_anything import sam_model_registry, SamPredictor
from image_cropping import crop_mask_like_data, tight_crop_data


# ======================================================
# ------------------ SAM HELPERS ------------------------
# ======================================================

def to_uint8_rgb(img):
    """
    Convert a normalized image to uint8 RGB.
    Args:
        img: NumPy array in [0, 1], 2D or 3D.
    Returns:
        RGB uint8 image with values in [0, 255].
    """
    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return (img * 255).clip(0, 255).astype(np.uint8)


def minmax_normalize(img, mask=None):
    """
    Min-max normalize an image per channel, optionally within a ROI mask.
    Args:
        img: Image array (H, W, C).
        mask: Binary mask (H, W) or None to use full image.
    Returns:
        Float32 image normalized to [0, 1].
    """
    img = img.astype(np.float32)
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        if mask is not None and mask.sum() > 0:
            v = ch[mask > 0]
            vmin, vmax = v.min(), v.max()
        else:
            vmin, vmax = ch.min(), ch.max()
        out[:, :, c] = (ch - vmin) / (vmax - vmin + 1e-8)
    return out


def compute_density(gray, mask, sigma=5):
    """
    Compute a smoothed density map inside a mask.
    Args:
        gray: Grayscale image (H, W).
        mask: Binary mask (H, W).
        sigma: Gaussian filter sigma.
    Returns:
        Density map normalized to [0, 1].
    """
    d = gaussian_filter(gray * mask, sigma)
    return d / (d.max() + 1e-8)


def sample_prompts(density, mask, percentile=92, k_pos=12, k_neg=6):
    """
    Sample positive/negative points from a density map.
    Args:
        density: Density map (H, W).
        mask: Binary mask (H, W) limiting points.
        percentile: Percentile threshold for positives.
        k_pos: Max number of positive points.
        k_neg: Max number of negative points.
    Returns:
        (pts, labels) with points (x, y) and labels 1/0, or (None, None).
    """
    vals = density[mask > 0]
    if vals.size == 0:
        return None, None

    thr = np.percentile(vals, percentile)
    pos = np.argwhere((density >= thr) & (mask > 0))
    if len(pos) == 0:
        return None, None

    if len(pos) > k_pos:
        pos = pos[np.random.choice(len(pos), k_pos, replace=False)]

    neg = []
    for y, x in pos:
        ny = np.clip(y + np.random.randint(-15, 15), 0, mask.shape[0] - 1)
        nx = np.clip(x + np.random.randint(-15, 15), 0, mask.shape[1] - 1)
        if mask[ny, nx] > 0:
            neg.append([ny, nx])

    neg = np.array(neg)[:k_neg] if len(neg) else np.zeros((0, 2))

    pts = np.vstack([pos, neg])
    labels = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])

    pts = pts[:, [1, 0]]  # (x, y)
    return pts.astype(int), labels.astype(int)


def save_debug(image, density, pos, neg, mask, outpath):
    """
    Save a debug figure with image, density, and prompts.
    Args:
        image: Original image (H, W, C).
        density: Density map (H, W).
        pos: Positive points (N, 2) in (x, y).
        neg: Negative points (M, 2) in (x, y).
        mask: Predicted binary mask (H, W).
        outpath: Output file path.
    Returns:
        None.
    """
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.title("Input image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Density map")
    plt.imshow(density, cmap="hot")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Prompts")
    plt.imshow(image)
    handles = []
    if len(pos):
        h_pos = plt.scatter(pos[:, 0], pos[:, 1], c="lime", s=15, label="pos")
        handles.append(h_pos)
    if len(neg):
        h_neg = plt.scatter(neg[:, 0], neg[:, 1], c="yellow", s=15, label="neg")
        handles.append(h_neg)
    if handles:
        plt.legend(handles=handles, loc="upper right", frameon=True)
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("SAM mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def downsample_mask_to_shape(mask, target_shape):
    """
    Downsample a mask to match the target spatial shape using nearest-neighbor.
    Args:
        mask: Mask array (H, W) or (H, W, C).
        target_shape: Target spatial shape (H, W) or (H, W, C).
    Returns:
        Mask downsampled to the target spatial shape.
    """
    src_h, src_w = mask.shape[:2]
    tgt_h, tgt_w = target_shape[:2]
    if (src_h, src_w) == (tgt_h, tgt_w):
        return mask
    y_idx = np.linspace(0, src_h - 1, tgt_h).round().astype(int)
    x_idx = np.linspace(0, src_w - 1, tgt_w).round().astype(int)
    if mask.ndim == 3:
        return mask[y_idx][:, x_idx, :]
    return mask[y_idx][:, x_idx]


# ======================================================
# ---------------- SEGMENT ONE SLICE -------------------
# ======================================================

def iter_patch_coords(height, width, patch_size, stride):
    """
    Yield overlapping patch coordinates for tiling.
    Args:
        height: Total height.
        width: Total width.
        patch_size: Square patch size.
        stride: Step between patches.
    Returns:
        Iterator of (y0, y1, x0, x1) tuples.
    """
    y = 0
    while y < height:
        x = 0
        y1 = min(y + patch_size, height)
        y0 = y1 - patch_size if y1 - patch_size > 0 else 0
        while x < width:
            x1 = min(x + patch_size, width)
            x0 = x1 - patch_size if x1 - patch_size > 0 else 0
            yield y0, y1, x0, x1
            if x + stride >= width:
                break
            x += stride
        if y + stride >= height:
            break
        y += stride


def segment_slice(image, roi, predictor, patch_size=2048, stride=1024):
    """
    Segment a slice using SAM over tiled patches.
    Args:
        image: Image array (H, W, C).
        roi: Binary ROI mask (H, W).
        predictor: SamPredictor instance.
        patch_size: Patch size.
        stride: Patch stride.
    Returns:
        (final, stats) where final is the mask and stats holds patch counts.
    """
    h, w = roi.shape
    final = np.zeros((h, w), dtype=np.uint8)
    patch_total = 0
    patch_used = 0

    img01_full = minmax_normalize(image, roi)
    gray_full = img01_full.mean(axis=2)
    density_full = compute_density(gray_full, roi)
    pts_full, labels_full = sample_prompts(density_full, roi)
    if pts_full is None:
        stats = {"patch_total": patch_total, "patch_used": patch_used}
        return final, stats

    for y0, y1, x0, x1 in iter_patch_coords(h, w, patch_size, stride):
        patch_total += 1
        patch = image[y0:y1, x0:x1]
        patch_roi = roi[y0:y1, x0:x1]
        if patch_roi.sum() == 0:
            continue
        # Select global prompts that fall inside this patch
        in_patch = (
            (pts_full[:, 0] >= x0) & (pts_full[:, 0] < x1) &
            (pts_full[:, 1] >= y0) & (pts_full[:, 1] < y1)
        )
        if not np.any(in_patch):
            continue
        pts = pts_full[in_patch].copy()
        labels = labels_full[in_patch].copy()
        pts[:, 0] -= x0
        pts[:, 1] -= y0
        patch_used += 1

        img01 = img01_full[y0:y1, x0:x1]
        predictor.set_image(to_uint8_rgb(img01))
        masks, scores, _ = predictor.predict(
            point_coords=pts,
            point_labels=labels,
            multimask_output=True
        )

        best = np.argmax(scores)
        binary = masks[best].astype(np.uint8)
        binary = remove_small_objects(binary.astype(bool), min_size=30)
        binary = binary_closing(binary, disk(1))
        binary = binary.astype(np.uint8)
        binary = binary * patch_roi

        final[y0:y1, x0:x1] = np.maximum(final[y0:y1, x0:x1], binary)

    stats = {"patch_total": patch_total, "patch_used": patch_used}
    return final, stats


def build_full_slice_debug(image, roi, pred_mask):
    """
    Build debug inputs for a full slice.
    Args:
        image: Image array (H, W, C).
        roi: Binary ROI mask (H, W).
        pred_mask: Predicted mask (H, W).
    Returns:
        density, pos, neg, pred_mask for save_debug.
    """
    img01 = minmax_normalize(image, roi)
    gray = img01.mean(axis=2)
    density = compute_density(gray, roi)
    pts, labels = sample_prompts(density, roi)
    if pts is None:
        # Still return density and empty prompts so we can save a debug image.
        return density, np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int), pred_mask
    pos = pts[labels == 1]
    neg = pts[labels == 0]
    return density, pos, neg, pred_mask


def build_auto_roi(image, min_size=5000):
    """
    Generate an automatic ROI when the outline mask is empty.
    Args:
        image: Image array (H, W, C).
        min_size: Minimum object size to keep.
    Returns:
        Binary ROI mask (H, W).
    """
    # Fallback tissue mask when outline ROI is empty.
    gray = minmax_normalize(image).mean(axis=2)
    thr = np.percentile(gray, 60)
    auto = (gray > thr)
    auto = remove_small_objects(auto, min_size=min_size)
    auto = binary_closing(auto, disk(2))
    return auto.astype(np.uint8)


# ======================================================
# ---------------------- MAIN --------------------------
# ======================================================

def main():
    """
    Run slice inference, save predictions, and optionally evaluate.
    Args:
        None.
    Returns:
        None.
    """

    base_dir = os.environ["BASE_DIR"]
    subject = os.environ.get("SUBJECT", "MF191")
    level = os.environ.get("PYRAMID_LEVEL", "4")
    outline_level = os.environ.get("OUTLINE_LEVEL", level)

    CHECKPOINTS = {
        "medsam": None,
        "sam": None,
        "medsam2": None,
        "sam2": None,
    }
    # Default to medsam only; override with SAM_MODEL env var.
    model_only = os.environ.get("SAM_MODEL", "medsam")
    if model_only not in CHECKPOINTS:
        valid_models = ", ".join(sorted(CHECKPOINTS))
        raise ValueError(f"Unknown SAM_MODEL={model_only!r}. Expected one of: {valid_models}")
    checkpoint = os.environ["SAM_CHECKPOINT"]
    CHECKPOINTS = {model_only: checkpoint}

    OUTROOT = os.environ.get("OUTPUT_DIR", "predictions")
    # Use "all" to save debug for every slice, or a list of indices like [0, 10, 20].
    DEBUG_SLICES = "all"
    PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "2048"))
    STRIDE = int(os.environ.get("STRIDE", "1024"))

    # -------- LOAD DATA (MR243) --------
    dirname = f"{base_dir}/sub-{subject}/micr/"

    histology_paths = sorted(Path(dirname).glob(f"sub-{subject}_*_stain-LY_DF.ome.zarr"))
    if not histology_paths:
        raise FileNotFoundError(f"No LY_DF OME-Zarr found in {dirname}")
    omz = zarr.open_group(histology_paths[0], mode="r")
    outline = zarr.open_group(dirname + "masks/Outline.ome.zarr", mode="r")
    outline_alt = None
    
    level = int(level)  # por seguridad

    # Some OME-Zarr stores have arrays at level keys ("0","1",...).
    # Others have a group "0/0". Return the array for the requested level.
    def _get_level_array(zg, level_key):
        """
        Get the array for a level inside an OME-Zarr group.
        Args:
            zg: Open Zarr group.
            level_key: Level key as string.
        Returns:
            Zarr array for the requested level.
        """
        if level_key in zg:
            return zg[level_key]
        if "0" in zg:
            arr0 = zg["0"]
            if isinstance(arr0, zarr.core.Array):
                return arr0
            if level_key in arr0:
                return arr0[level_key]
            if "0" in arr0:
                return arr0["0"]
        # Fallback: first array key
        keys = list(zg.array_keys())
        if keys:
            return zg[keys[0]]
        raise KeyError(f"No arrays found in zarr group for level {level_key}")

    level_key = str(level)
    outline_level_key = str(outline_level)
    hist_z = _get_level_array(omz, level_key)
    brain_z = _get_level_array(outline, outline_level_key)
    brain_alt_z = _get_level_array(outline_alt, outline_level_key) if outline_alt is not None else None

    def _get_slice(arr, idx):
        """
        Extract a 2D/3D slice from a Zarr array.
        Args:
            arr: Zarr array with ndim 3 or 4.
            idx: Slice index along Z.
        Returns:
            Slice as a NumPy array.
        """
        if arr.ndim == 4:
            return np.transpose(arr[:, idx, :, :], (1, 2, 0))
        if arr.ndim == 3:
            return arr[idx, :, :]
        raise ValueError(f"Unexpected array ndim={arr.ndim} for slice extraction")

    slice_count = hist_z.shape[1] if hist_z.ndim == 4 else hist_z.shape[0]
    start_slice = max(0, int(os.environ.get("START_SLICE") or 0))
    end_slice = min(slice_count, int(os.environ.get("END_SLICE") or slice_count))
    if start_slice >= end_slice:
        raise ValueError(
            f"Invalid slice range [{start_slice}, {end_slice}) for {slice_count} slices"
        )
    slice_indices = list(range(start_slice, end_slice))
    slides = [f"{index:03d}" for index in slice_indices]

    for model_name, checkpoint in CHECKPOINTS.items():
        out_dir = OUTROOT
        debug_dir = os.path.join(out_dir, "debug")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)

        print(f"=== MODEL {model_name} ===")

        if model_name in {"sam2", "medsam2"}:
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError as exc:
                raise ImportError(
                    "SAM2 requires its upstream repository on PYTHONPATH"
                ) from exc

            default_config = (
                "configs/sam2.1/sam2.1_hiera_l"
                if model_name == "sam2"
                else "configs/sam2.1/sam2.1_hiera_t512"
            )
            model = build_sam2(os.environ.get("SAM2_CONFIG", default_config), checkpoint)
            model.to("cuda")
            model.eval()
            predictor = SAM2ImagePredictor(model)
        else:
            model = sam_model_registry["vit_b"](checkpoint=checkpoint)
            model.to("cuda")
            model.eval()
            predictor = SamPredictor(model)

        for idx, slide in zip(slice_indices, slides):
            t0 = time.time()
            print(f"Slice {idx}")

            # OME-Zarr here is (C, Z, Y, X); slice lazily to avoid huge allocations.
            image = _get_slice(hist_z, idx)
            roi = _get_slice(brain_z, idx)
            if roi.ndim == 2:
                roi = roi[:, :, None]
            roi = downsample_mask_to_shape(roi, image.shape)

            image, bounds = tight_crop_data(image)
            roi = crop_mask_like_data(roi, bounds)[:, :, 0]
            if roi.sum() == 0 and brain_alt_z is not None:
                roi_alt = _get_slice(brain_alt_z, idx)
                if roi_alt.ndim == 2:
                    roi_alt = roi_alt[:, :, None]
                roi_alt = downsample_mask_to_shape(roi_alt, image.shape)
                roi_alt = crop_mask_like_data(roi_alt, bounds)[:, :, 0]
                if roi_alt.sum() > 0:
                    roi = roi_alt
                    print("  INFO: used Outline.ome.zarr as ROI (Outline_mask empty)")
            if roi.sum() == 0:
                roi = build_auto_roi(image)
                print("  INFO: used auto ROI (outline empty)")
            if image.size == 0 or roi.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
                print(f"  SKIP: empty crop for slice {idx}")
                continue

            pred, stats = segment_slice(
                image, roi, predictor, patch_size=PATCH_SIZE, stride=STRIDE
            )

            if pred.size == 0 or pred.shape[0] == 0 or pred.shape[1] == 0:
                print(f"  SKIP: empty prediction for slice {idx}")
                continue

            Image.fromarray((pred * 255).astype(np.uint8)).save(
                os.path.join(out_dir, f"slice_{slide}_pred.png")
            )

            save_dbg = (
                DEBUG_SLICES == "all"
                or DEBUG_SLICES is None
                or idx in DEBUG_SLICES
            )
            if save_dbg:
                dbg = build_full_slice_debug(image, roi, pred)
            if dbg is not None and save_dbg:
                density, pos, neg, dbg_mask = dbg
                save_debug(
                    image,
                    density,
                    pos,
                    neg,
                    dbg_mask,
                    os.path.join(debug_dir, f"slice_{slide}_debug.png"),
                )

            dt = time.time() - t0
            print(
                f"  patches={stats['patch_total']} used={stats['patch_used']} "
                f"roi_pixels={int(roi.sum())} density_max={float(compute_density(minmax_normalize(image, roi).mean(axis=2), roi).max()):.4f} "
                f"pred_pixels={int(pred.sum())} time={dt:.1f}s"
            )

    print("SAM slice inference finished successfully.")

def parse_args() -> argparse.Namespace:
    """Parse command-line options, using environment variables as defaults."""
    parser = argparse.ArgumentParser(
        description="Run slice-based fiber-bundle inference with a SAM-family model."
    )
    parser.add_argument("--model", choices=("sam", "medsam", "sam2", "medsam2"),
                        default=os.environ.get("SAM_MODEL", "medsam"))
    parser.add_argument("--checkpoint", default=os.environ.get("SAM_CHECKPOINT"))
    parser.add_argument("--base-dir", default=os.environ.get("BASE_DIR"))
    parser.add_argument("--subject", default=os.environ.get("SUBJECT", "MF191"))
    parser.add_argument("--level", default=os.environ.get("PYRAMID_LEVEL", "4"))
    parser.add_argument("--outline-level", default=os.environ.get("OUTLINE_LEVEL"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "predictions"))
    parser.add_argument("--start-slice", type=int, default=None)
    parser.add_argument("--end-slice", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=int(os.environ.get("PATCH_SIZE", "2048")))
    parser.add_argument("--stride", type=int, default=int(os.environ.get("STRIDE", "1024")))
    parser.add_argument("--sam2-config", default=os.environ.get("SAM2_CONFIG"))
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint is required (or set SAM_CHECKPOINT)")
    if not args.base_dir:
        parser.error("--base-dir is required (or set BASE_DIR)")
    if args.model in {"sam2", "medsam2"} and not args.sam2_config:
        parser.error("--sam2-config is required for SAM2 and MedSAM2")
    return args


def cli() -> None:
    """Translate CLI options to the environment-backed pipeline configuration."""
    args = parse_args()
    values = {
        "SAM_MODEL": args.model,
        "SAM_CHECKPOINT": args.checkpoint,
        "BASE_DIR": args.base_dir,
        "SUBJECT": args.subject,
        "PYRAMID_LEVEL": args.level,
        "OUTLINE_LEVEL": args.outline_level or args.level,
        "OUTPUT_DIR": args.output_dir,
        "START_SLICE": args.start_slice,
        "END_SLICE": args.end_slice,
        "PATCH_SIZE": args.patch_size,
        "STRIDE": args.stride,
        "SAM2_CONFIG": args.sam2_config,
    }
    for name, value in values.items():
        if value is not None:
            os.environ[name] = str(value)
    main()


if __name__ == "__main__":
    cli()
