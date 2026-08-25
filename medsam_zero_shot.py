"""
MedSAM zero-shot pipeline with fiber-aware prompting (Frangi -> skeleton points).

Outputs binary PNGs named bundle_{slide}_0000_pred.png.
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image
import torch
import zarr
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, dilation, disk, remove_small_objects
from segment_anything import sam_model_registry, SamPredictor

from image_cropping import crop_mask_like_data, tight_crop_data


def to_uint8_rgb(image):
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    else:
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        img = image.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        image = (img * 255).astype(np.uint8)
    return image


def resize_mask_to_match(mask, target_shape):
    return cv2.resize(
        mask.astype(np.uint8),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def zscore_normalize(image, mask):
    image = image.astype(np.float32)
    if image.ndim == 2:
        image = image[:, :, None]
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        channel = image[:, :, c]
        if mask is not None:
            valid = channel[mask > 0]
            if valid.size > 0:
                mean = valid.mean()
                std = valid.std()
                out[:, :, c] = (channel - mean) / (std + 1e-8)
            else:
                out[:, :, c] = channel
        else:
            mean = channel.mean()
            std = channel.std()
            out[:, :, c] = (channel - mean) / (std + 1e-8)
    return out


def minmax_normalize(image, mask):
    image = image.astype(np.float32)
    if image.ndim == 2:
        image = image[:, :, None]
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        channel = image[:, :, c]
        if mask is not None:
            valid = channel[mask > 0]
            if valid.size > 0:
                vmin = valid.min()
                vmax = valid.max()
            else:
                vmin = channel.min()
                vmax = channel.max()
        else:
            vmin = channel.min()
            vmax = channel.max()
        if vmax <= vmin:
            out[:, :, c] = 0.0
        else:
            out[:, :, c] = (channel - vmin) / (vmax - vmin)
    return out


def preprocess_section(image, brain_mask, downsample_factor, margin, clip_std, norm_mode, max_pixels):
    h, w = image.shape[:2]
    if downsample_factor > 1:
        new_w = max(1, w // downsample_factor)
        new_h = max(1, h // downsample_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        brain_mask = cv2.resize(
            brain_mask.astype(np.uint8),
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_coords = np.argwhere(brain_mask > 0)
    if mask_coords.size > 0:
        y_min, x_min = mask_coords.min(axis=0)
        y_max, x_max = mask_coords.max(axis=0)
        y_min = max(0, y_min - margin)
        x_min = max(0, x_min - margin)
        y_max = min(image.shape[0], y_max + margin)
        x_max = min(image.shape[1], x_max + margin)
    else:
        y_min, x_min = 0, 0
        y_max, x_max = image.shape[0], image.shape[1]

    image_cropped = image[y_min:y_max, x_min:x_max]
    mask_cropped = brain_mask[y_min:y_max, x_min:x_max]

    if max_pixels and image_cropped.size > max_pixels:
        # extra safety downsample to cap memory
        h, w = image_cropped.shape[:2]
        scale = (image_cropped.size / float(max_pixels)) ** 0.5
        new_w = max(1, int(w / scale))
        new_h = max(1, int(h / scale))
        image_cropped = cv2.resize(image_cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_cropped = cv2.resize(mask_cropped.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    if norm_mode == "minmax":
        norm = minmax_normalize(image_cropped, mask_cropped)
    else:
        norm = zscore_normalize(image_cropped, mask_cropped)
        norm = np.clip(norm, -clip_std, clip_std)
        norm = (norm + clip_std) / (2 * clip_std)
        norm = np.clip(norm, 0, 1)

    valid_region = {
        "y_min": y_min,
        "y_max": y_max,
        "x_min": x_min,
        "x_max": x_max,
        "original_shape": image.shape[:2],
    }
    return norm, mask_cropped, valid_region


def iter_patch_positions(height, width, patch_size, stride):
    if height <= 0 or width <= 0:
        return

    if height <= patch_size:
        ys = [0]
    else:
        ys = list(range(0, height - patch_size + 1, stride))
        last_y = height - patch_size
        if ys[-1] != last_y:
            ys.append(last_y)

    if width <= patch_size:
        xs = [0]
    else:
        xs = list(range(0, width - patch_size + 1, stride))
        last_x = width - patch_size
        if xs[-1] != last_x:
            xs.append(last_x)

    for y in ys:
        for x in xs:
            yield int(y), int(x)


def get_level_array(zarr_group, level_key, ref_shape=None):
    if not hasattr(zarr_group, "array_keys"):
        return zarr_group
    if level_key in zarr_group:
        return zarr_group[level_key]
    array_keys = list(zarr_group.array_keys())
    if not array_keys:
        raise KeyError(f"No arrays found in zarr group for level {level_key}")
    if ref_shape is not None:
        for key in array_keys:
            arr = zarr_group[key]
            if len(arr.shape) >= 4 and arr.shape[2] == ref_shape[2] and arr.shape[3] == ref_shape[3]:
                return arr
    return zarr_group[array_keys[0]]


def get_outline_group(dirname, prefer_mask=False):
    if prefer_mask:
        outline_mask_path = os.path.join(dirname, "masks", "Outline_mask.ome.zarr")
        if os.path.isdir(outline_mask_path):
            return zarr.open_group(outline_mask_path, mode="r")
    outline_path = os.path.join(dirname, "masks", "Outline.ome.zarr")
    if os.path.isdir(outline_path):
        return zarr.open_group(outline_path, mode="r")
    outline_mask_path = os.path.join(dirname, "masks", "Outline_mask.ome.zarr")
    if os.path.isdir(outline_mask_path):
        return zarr.open_group(outline_mask_path, mode="r")
    raise FileNotFoundError(f"Outline mask not found in {os.path.join(dirname, 'masks')}")


def load_subject(subject, base_dir, level):
    if subject == "MR256":
        dirname = f"{base_dir}/sub-MR256/micr/"
        omz = zarr.open_group(
            dirname + "/sub-MR256_sample-slice0000slice0038_stain-LY_DF.ome.zarr", mode="r"
        )
        outline = get_outline_group(dirname, prefer_mask=True)
        hist_arr = get_level_array(omz, level)
        brain_arr = get_level_array(outline, level, ref_shape=hist_arr.shape)
        slides = [f"{i:03d}" for i in range(1, 33)]
        slice_indices = list(range(0, 32))
    elif subject == "MR243":
        dirname = f"{base_dir}/sub-MR243/micr/"
        omz = zarr.open_group(
            dirname + "sub-MR243_sample-slice0000slice0072_stain-LY_DF.ome.zarr", mode="r"
        )
        outline = get_outline_group(dirname, prefer_mask=False)
        hist_arr = get_level_array(omz, level)
        brain_arr = get_level_array(outline, level, ref_shape=hist_arr.shape)
        slides = [f"{i:03d}" for i in range(1, 36)]
        slice_indices = list(range(0, 35))
    elif subject == "MN115":
        dirname = f"{base_dir}/sub-MN115/micr/"
        omz = zarr.open_group(
            dirname + "sub-MN115_sample-slice0000slice0026_stain-LY_DF.ome.zarr", mode="r"
        )
        outline = get_outline_group(dirname, prefer_mask=False)
        hist_arr = get_level_array(omz, level)
        brain_arr = get_level_array(outline, level, ref_shape=hist_arr.shape)
        slides = [f"{i:03d}" for i in range(33, 49)]
        slice_indices = list(range(0, 27))
    elif subject == "MR252":
        dirname = f"{base_dir}/sub-MR252/micr/"
        omz = zarr.open_group(
            dirname + "/sub-MR252_sample-slice0000slice0042_stain-FS_DF.ome.zarr", mode="r"
        )
        outline = get_outline_group(dirname, prefer_mask=False)
        hist_arr = get_level_array(omz, level)
        brain_arr = get_level_array(outline, level, ref_shape=hist_arr.shape)
        slides = [f"{i:03d}" for i in range(1, 44)]
        slice_indices = list(range(0, 43))
    elif subject == "MF191":
        dirname = f"{base_dir}/sub-MF191/micr/"
        omz = zarr.open_group(
            dirname + "/sub-MF191_sample-slice0000slice0049_stain-LY_DF.ome.zarr", mode="r"
        )
        outline = get_outline_group(dirname, prefer_mask=False)
        hist_arr = get_level_array(omz, level)
        brain_arr = get_level_array(outline, level, ref_shape=hist_arr.shape)
        slides = [f"{i:03d}" for i in range(49, 63)]
        slice_indices = list(range(0, 30))
    else:
        raise ValueError(f"Unknown subject: {subject}")

    return hist_arr, brain_arr, slides, slice_indices


def get_slice_from_arr(arr, idx):
    if arr.ndim == 5:
        slice_data = arr[0, 0, idx, :, :]
    elif arr.ndim == 4:
        slice_data = arr[0, idx, :, :]
    elif arr.ndim == 3:
        slice_data = arr[idx, :, :]
    else:
        raise ValueError(f"Unsupported zarr array ndim={arr.ndim}")
    if slice_data.ndim == 2:
        slice_data = slice_data[:, :, None]
    return slice_data


def get_gray(image_rgb):
    if image_rgb.ndim == 2:
        return image_rgb.astype(np.float32)
    return image_rgb.mean(axis=2).astype(np.float32)


def downsample_image(image, factor, interp):
    if factor <= 1:
        return image
    h, w = image.shape[:2]
    new_w = max(1, int(w // factor))
    new_h = max(1, int(h // factor))
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def compute_frangi_map(gray, brain_mask, sigmas, downsample_factor, log_prefix=""):
    if brain_mask is not None and gray.shape != brain_mask.shape:
        print(
            f"{log_prefix}frangi shape mismatch: gray={gray.shape} brain_mask={brain_mask.shape} -> resizing mask"
        )
        brain_mask = resize_mask_to_match(brain_mask, gray.shape[:2])
    if brain_mask is not None:
        assert gray.shape == brain_mask.shape, (
            f"{log_prefix}frangi shape mismatch after resize: gray={gray.shape} brain_mask={brain_mask.shape}"
        )

    norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    if downsample_factor > 1:
        gray_ds = downsample_image(norm, downsample_factor, cv2.INTER_LINEAR)
        mask_ds = downsample_image(brain_mask, downsample_factor, cv2.INTER_NEAREST) if brain_mask is not None else None
    else:
        gray_ds = norm
        mask_ds = brain_mask

    response = frangi(gray_ds, sigmas=sigmas, black_ridges=False).astype(np.float32)
    if mask_ds is not None:
        response *= (mask_ds > 0)
    if response.max() > 0:
        response = response / response.max()
    if downsample_factor > 1:
        response = cv2.resize(response, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return response


def sample_prompts(frangi_map, brain_mask, percentile, k_pos, k_neg, neg_radius, rng):
    mask = brain_mask > 0 if brain_mask is not None else np.ones_like(frangi_map, dtype=bool)
    vals = frangi_map[mask]
    if vals.size == 0:
        return None, None
    thr = np.percentile(vals, percentile)
    binary = (frangi_map >= thr) & mask
    if not np.any(binary):
        return None, None
    skel = skeletonize(binary)
    pos_coords = np.argwhere(skel)
    if pos_coords.size == 0:
        return None, None
    if len(pos_coords) > k_pos:
        idx = rng.choice(len(pos_coords), size=k_pos, replace=False)
        pos_coords = pos_coords[idx]
    neg_mask = dilation(binary, disk(neg_radius)) ^ binary
    neg_coords = np.argwhere(neg_mask)
    if len(neg_coords) > k_neg and k_neg > 0:
        idx = rng.choice(len(neg_coords), size=k_neg, replace=False)
        neg_coords = neg_coords[idx]

    pos_xy = np.stack([pos_coords[:, 1], pos_coords[:, 0]], axis=1)
    if len(neg_coords) > 0:
        neg_xy = np.stack([neg_coords[:, 1], neg_coords[:, 0]], axis=1)
    else:
        neg_xy = np.zeros((0, 2), dtype=np.int64)
    point_coords = np.concatenate([pos_xy, neg_xy], axis=0)
    point_labels = np.concatenate(
        [np.ones(len(pos_xy), dtype=np.int64), np.zeros(len(neg_xy), dtype=np.int64)],
        axis=0,
    )
    return point_coords, point_labels


def filter_fiber_like(mask, min_area, min_length, min_ecc):
    labeled = label(mask > 0)
    if labeled.max() == 0:
        return mask
    keep = np.zeros_like(mask, dtype=np.uint8)
    for region in regionprops(labeled):
        if region.area < min_area:
            continue
        if region.major_axis_length < min_length:
            continue
        if region.eccentricity < min_ecc:
            continue
        coords = region.coords
        keep[coords[:, 0], coords[:, 1]] = 1
    return keep


def continuity_filter(mask, prev_mask, min_overlap, max_isolation_area):
    if prev_mask is None or prev_mask.sum() == 0:
        return mask, 0
    if prev_mask.shape != mask.shape:
        prev_mask = resize_mask_to_match(prev_mask, mask.shape[:2])
    labeled = label(mask > 0)
    if labeled.max() == 0:
        return mask, 0
    keep = np.zeros_like(mask, dtype=np.uint8)
    removed = 0
    for region in regionprops(labeled):
        coords = region.coords
        overlap = prev_mask[coords[:, 0], coords[:, 1]].sum()
        overlap_frac = overlap / (region.area + 1e-8)
        if overlap_frac < min_overlap and region.area < max_isolation_area:
            removed += 1
            continue
        keep[coords[:, 0], coords[:, 1]] = 1
    return keep, removed


def run_subject(subject, args, predictor):
    hist_arr, brain_arr, slides, slice_indices = load_subject(subject, args.base_dir, args.level)
    out_dir = os.path.join(args.predicted_root, subject)
    os.makedirs(out_dir, exist_ok=True)
    debug_dir = None
    if args.save_debug:
        debug_dir = os.path.join(out_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
    debug_set = set(args.debug_slices)
    prev_mask = None

    for ix, slide in enumerate(slides):
        slice_idx = slice_indices[ix]
        hist_slice = get_slice_from_arr(hist_arr, slice_idx)
        brain_slice = get_slice_from_arr(brain_arr, slice_idx)

        if brain_slice.shape[:2] != hist_slice.shape[:2]:
            brain_plane = brain_slice[:, :, 0]
            resized = resize_mask_to_match(brain_plane, hist_slice.shape[:2])
            brain_slice = resized[:, :, None]

        slice_crop, boundaries = tight_crop_data(hist_slice)
        brain_crop = crop_mask_like_data(brain_slice, boundaries)[:, :, 0]
        brain_sum = float(brain_crop.sum())

        if args.log_slices:
            print(f"[{subject}] slice={slide} idx={ix} hist_shape={hist_slice.shape} brain_sum={int(brain_sum)}")

        if brain_sum == 0:
            if args.log_slices:
                print("  SKIP: empty brain mask after crop")
            continue

        image_proc, mask_proc, valid_region = preprocess_section(
            slice_crop,
            brain_crop,
            downsample_factor=args.downsample_factor,
            margin=args.crop_margin,
            clip_std=args.clip_std,
            norm_mode=args.norm,
            max_pixels=args.max_pixels,
        )

        gray = get_gray(image_proc)
        if args.log_slices:
            img_mb = image_proc.nbytes / (1024 * 1024)
            print(
                f"  shapes: image_proc={image_proc.shape} gray={gray.shape} "
                f"brain_crop={brain_crop.shape} mask_proc={mask_proc.shape} ~{img_mb:.1f}MB"
            )
        if gray.shape != mask_proc.shape:
            mask_proc = resize_mask_to_match(mask_proc, gray.shape[:2])
        assert gray.shape == mask_proc.shape, (
            f"gray/mask mismatch: gray={gray.shape} mask_proc={mask_proc.shape}"
        )
        frangi_map = compute_frangi_map(
            gray,
            mask_proc,
            args.frangi_sigmas,
            downsample_factor=args.frangi_downsample,
            log_prefix="  ",
        )

        h, w = image_proc.shape[:2]
        stride = max(1, int(args.patch_size * (1 - args.overlap)))
        prob_dtype = np.float16 if args.prob_fp16 else np.float32
        prob_map = np.zeros((h, w), dtype=prob_dtype)

        patches_total = 0
        patches_used = 0
        masks_total = 0
        prompts_total = 0
        rng = np.random.default_rng(args.seed + ix)
        prompt_pos = []
        prompt_neg = []

        for y, x in iter_patch_positions(h, w, args.patch_size, stride):
            patch = image_proc[y:y + args.patch_size, x:x + args.patch_size]
            mask_patch = mask_proc[y:y + args.patch_size, x:x + args.patch_size]
            frangi_patch = frangi_map[y:y + args.patch_size, x:x + args.patch_size]

            patch_h, patch_w = patch.shape[:2]
            if patch_h == 0 or patch_w == 0:
                continue

            patches_total += 1
            brain_ratio = float(mask_patch.sum()) / float(patch_h * patch_w)
            if brain_ratio < args.min_brain_ratio:
                continue
            patches_used += 1

            point_coords, point_labels = sample_prompts(
                frangi_patch,
                mask_patch,
                percentile=args.seed_percentile,
                k_pos=args.k_pos,
                k_neg=args.k_neg,
                neg_radius=args.neg_radius,
                rng=rng,
            )
            if point_coords is None:
                continue

            prompts_total += len(point_labels)
            if ix in debug_set:
                pos_idx = point_labels == 1
                neg_idx = point_labels == 0
                if np.any(pos_idx):
                    pts = point_coords[pos_idx] + np.array([x, y])
                    prompt_pos.append(pts)
                if np.any(neg_idx):
                    pts = point_coords[neg_idx] + np.array([x, y])
                    prompt_neg.append(pts)

            pad_h = max(0, args.patch_size - patch_h)
            pad_w = max(0, args.patch_size - patch_w)
            if pad_h or pad_w:
                patch_pad = np.zeros((patch_h + pad_h, patch_w + pad_w, patch.shape[2]), dtype=patch.dtype)
                patch_pad[:patch_h, :patch_w] = patch
            else:
                patch_pad = patch

            patch_uint8 = (patch_pad * 255).clip(0, 255).astype(np.uint8)
            patch_uint8 = to_uint8_rgb(patch_uint8)
            predictor.set_image(patch_uint8)
            try:
                with torch.no_grad():
                    masks, scores, _ = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=False,
                    )
            except (IndexError, RuntimeError, ValueError):
                continue
            masks_total += 1

            patch_mask = masks[0].astype(np.float32)
            score = float(scores[0]) if scores is not None and len(scores) else 1.0
            if patch_mask.shape != (patch_pad.shape[0], patch_pad.shape[1]):
                patch_mask = cv2.resize(
                    patch_mask,
                    (patch_pad.shape[1], patch_pad.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            patch_mask = patch_mask[:patch_h, :patch_w]
            prob_map[y:y + patch_h, x:x + patch_w] = np.maximum(
                prob_map[y:y + patch_h, x:x + patch_w],
                patch_mask * score,
            )

        nonzero = prob_map[prob_map > 0].astype(np.float32)
        if nonzero.size > 0:
            if args.percentile_sample and nonzero.size > args.percentile_sample:
                idx = np.random.choice(nonzero.size, size=args.percentile_sample, replace=False)
                sample = nonzero[idx]
            else:
                sample = nonzero
            thr = np.percentile(sample, args.prob_percentile)
            thr = max(thr, args.min_prob_threshold)
        else:
            thr = 1.0

        prob_map_smooth = gaussian_filter(prob_map.astype(np.float32), sigma=args.smooth_sigma)
        binary_mask = (prob_map_smooth > thr).astype(np.uint8)
        comps_before = int(label(binary_mask > 0).max())

        # Fallback: if empty but we had responses, relax threshold once
        if comps_before == 0 and nonzero.size > 0:
            fb_thr = np.percentile(sample, args.fallback_percentile)
            fb_thr = max(fb_thr, args.fallback_min_prob)
            binary_mask = (prob_map_smooth > fb_thr).astype(np.uint8)
            comps_before = int(label(binary_mask > 0).max())

        binary_mask = remove_small_objects(
            binary_mask.astype(bool),
            min_size=args.min_mask_region_area,
            connectivity=2,
        ).astype(np.uint8)
        binary_mask = filter_fiber_like(
            binary_mask,
            min_area=args.min_mask_region_area,
            min_length=args.min_fiber_length,
            min_ecc=args.min_fiber_ecc,
        )
        binary_mask, removed = continuity_filter(
            binary_mask,
            prev_mask,
            min_overlap=args.continuity_overlap,
            max_isolation_area=args.continuity_max_area,
        )
        comps_after = int(label(binary_mask > 0).max())
        prev_mask = binary_mask.copy()

        full_pred_down = np.zeros(valid_region["original_shape"], dtype=np.uint8)
        target_h = valid_region["y_max"] - valid_region["y_min"]
        target_w = valid_region["x_max"] - valid_region["x_min"]
        if binary_mask.shape[:2] != (target_h, target_w):
            binary_mask = cv2.resize(binary_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        full_pred_down[
            valid_region["y_min"]:valid_region["y_max"],
            valid_region["x_min"]:valid_region["x_max"],
        ] = binary_mask

        full_pred = cv2.resize(
            full_pred_down,
            (slice_crop.shape[1], slice_crop.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        full_pred = (full_pred > 0).astype(np.uint8)
        full_pred = full_pred * (brain_crop > 0).astype(np.uint8)

        out_path = os.path.join(out_dir, f"bundle_{slide}_0000_pred.png")
        Image.fromarray((full_pred * 255).astype(np.uint8)).save(out_path)

        if args.log_slices:
            max_prob = float(prob_map.max()) if prob_map.size else 0.0
            mean_prob = float(nonzero.mean()) if nonzero.size else 0.0
            nonzero_pixels = int(nonzero.size)
            prompts_avg = float(prompts_total) / float(patches_used) if patches_used else 0.0
            print(
                f"  patches_total={patches_total} patches_used={patches_used} "
                f"prompts_total={prompts_total} prompts_per_patch={prompts_avg:.2f} "
                f"masks_generated={masks_total} max_prob={max_prob:.4f} mean_prob={mean_prob:.4f} "
                f"nonzero_pixels={nonzero_pixels} thr={thr:.4f} comps_before={comps_before} comps_after={comps_after} "
                f"removed_by_continuity={removed}"
            )

        if args.save_debug and ix in debug_set and debug_dir:
            base_gray = (get_gray(image_proc) * 255).clip(0, 255).astype(np.uint8)
            frangi_vis = (frangi_map * 255).clip(0, 255).astype(np.uint8)
            prob_vis = (prob_map_smooth * 255).clip(0, 255).astype(np.uint8)
            bin_vis = (binary_mask * 255).astype(np.uint8)

            max_side = max(base_gray.shape[:2])
            scale = 1.0
            if args.debug_max_side > 0 and max_side > args.debug_max_side:
                scale = args.debug_max_side / float(max_side)
                new_w = max(1, int(base_gray.shape[1] * scale))
                new_h = max(1, int(base_gray.shape[0] * scale))
                base_gray = cv2.resize(base_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
                frangi_vis = cv2.resize(frangi_vis, (new_w, new_h), interpolation=cv2.INTER_AREA)
                prob_vis = cv2.resize(prob_vis, (new_w, new_h), interpolation=cv2.INTER_AREA)
                bin_vis = cv2.resize(bin_vis, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            Image.fromarray(base_gray).save(os.path.join(debug_dir, f"preprocessed_{slide}.png"))
            Image.fromarray(frangi_vis).save(os.path.join(debug_dir, f"frangi_{slide}.png"))
            Image.fromarray(prob_vis).save(os.path.join(debug_dir, f"probmap_{slide}.png"))
            Image.fromarray(bin_vis).save(os.path.join(debug_dir, f"binary_{slide}.png"))

            overlay = np.dstack([base_gray, base_gray, base_gray])
            overlay[bin_vis > 0] = [255, 0, 0]
            if prompt_pos:
                pts = np.vstack(prompt_pos)
                for (px, py) in pts:
                    if scale != 1.0:
                        px = int(px * scale)
                        py = int(py * scale)
                    cv2.circle(overlay, (int(px), int(py)), 1, (0, 255, 0), -1)
            if prompt_neg:
                pts = np.vstack(prompt_neg)
                for (px, py) in pts:
                    if scale != 1.0:
                        px = int(px * scale)
                        py = int(py * scale)
                    cv2.circle(overlay, (int(px), int(py)), 1, (255, 255, 0), -1)
            Image.fromarray(overlay).save(os.path.join(debug_dir, f"overlay_{slide}.png"))

            # Histogram of prob_map (nonzero) for threshold tuning
            if nonzero.size > 0:
                if args.hist_sample and nonzero.size > args.hist_sample:
                    idx = np.random.choice(nonzero.size, size=args.hist_sample, replace=False)
                    hist_data = nonzero[idx]
                else:
                    hist_data = nonzero
                hist, bin_edges = np.histogram(hist_data, bins=args.hist_bins, range=(0.0, 1.0))
                hist_path = os.path.join(debug_dir, f"probmap_hist_{slide}.csv")
                with open(hist_path, "w", encoding="utf-8") as f:
                    f.write("bin_left,bin_right,count\n")
                    for i in range(len(hist)):
                        f.write(f"{bin_edges[i]:.6f},{bin_edges[i+1]:.6f},{int(hist[i])}\n")


def main():
    parser = argparse.ArgumentParser(description="MedSAM zero-shot pipeline (Frangi -> prompts).")
    parser.add_argument("--checkpoint", required=True, help="Path to MedSAM .pth")
    parser.add_argument("--model-type", default="vit_b", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--predicted-root", default="fiber-bundle-segmentation-benchmarking/predictions/MedSAM_zero_shot")
    parser.add_argument("--level", default="4")
    parser.add_argument("--subjects", nargs="+", default=["MR243"])
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--min-brain-ratio", type=float, default=0.1)
    parser.add_argument("--downsample-factor", type=int, default=1)
    parser.add_argument("--crop-margin", type=int, default=10)
    parser.add_argument("--clip-std", type=float, default=3.0)
    parser.add_argument("--norm", default="minmax", choices=["zscore", "minmax"])
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--min-mask-region-area", type=int, default=25)
    parser.add_argument("--min-fiber-length", type=float, default=50.0)
    parser.add_argument("--min-fiber-ecc", type=float, default=0.8)
    parser.add_argument("--prob-percentile", type=float, default=95.0)
    parser.add_argument("--min-prob-threshold", type=float, default=0.2)
    parser.add_argument("--fallback-percentile", type=float, default=80.0)
    parser.add_argument("--fallback-min-prob", type=float, default=0.05)
    parser.add_argument("--frangi-sigmas", type=str, default="1,2,4,8")
    parser.add_argument("--frangi-downsample", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=60000000)
    parser.add_argument("--percentile-sample", type=int, default=2000000)
    parser.add_argument("--seed-percentile", type=float, default=97.0)
    parser.add_argument("--k-pos", type=int, default=20)
    parser.add_argument("--k-neg", type=int, default=10)
    parser.add_argument("--neg-radius", type=int, default=5)
    parser.add_argument("--continuity-overlap", type=float, default=0.02)
    parser.add_argument("--continuity-max-area", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-slices", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--debug-slices", type=str, default="0,10,20")
    parser.add_argument("--debug-max-side", type=int, default=4096)
    parser.add_argument("--prob-fp16", action="store_true")
    parser.add_argument("--hist-bins", type=int, default=50)
    parser.add_argument("--hist-sample", type=int, default=2000000)
    args = parser.parse_args()

    args.frangi_sigmas = tuple(float(s.strip()) for s in args.frangi_sigmas.split(",") if s.strip())
    args.debug_slices = [int(s) for s in args.debug_slices.split(",") if s.strip()]

    device = args.device if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    for subject in args.subjects:
        run_subject(subject, args, predictor)


if __name__ == "__main__":
    main()
