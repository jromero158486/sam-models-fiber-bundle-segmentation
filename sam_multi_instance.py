"""Multi-instance SAM-family inference for fiber-bundle segmentation.

The pipeline detects candidate instances, generates adaptive prompts, segments
each instance, post-processes masks independently, and writes one combined PNG
per slice. Use ``evaluate_sam_predictions.py`` to evaluate its output.
"""

import argparse
import os
import numpy as np
import zarr
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional

from scipy.ndimage import gaussian_filter, binary_dilation, label
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, disk
from skimage.filters import sobel

from segment_anything import sam_model_registry, SamPredictor
from image_utils import resize_mask_nearest


# ======================================================================
# CONFIGURATION
# ======================================================================

class Config:
    """Configuration parameters for the segmentation pipeline"""
    
    # Data paths
    BASE_DIR = ""
    SUBJECT = "MF191"
    PYRAMID_LEVEL = "4"
    
    # Model checkpoint
    SAM_CHECKPOINT = ""
    SAM_MODEL_TYPE = "vit_b"
    SAM2_CONFIG = ""
    SAM_MODEL_NAME = "default"
    
    # Output directories
    OUTPUT_DIR = "sam_predictions"
    START_SLICE = None          # Inclusive start index (None = full volume)
    END_SLICE = None            # Exclusive end index (None = full volume)
    
    # Pre-segmentation parameters
    PRESEG_PERCENTILE = 98      # Stronger bright-region threshold for sparse targets
    PRESEG_MIN_SIZE = 200       # Remove small noisy detections early
    PRESEG_MARGIN = 10          # Tighter margin to avoid oversized boxes
    
    # Multi-instance detection parameters
    MIN_INSTANCE_SIZE = 250     # Keep only meaningful connected components
    MAX_INSTANCES_PER_SLICE = 4   # Limit over-fragmentation per slice
    MERGE_DISTANCE = 30         # Merge nearby fragments likely from same bundle
    
    # Prompt generation parameters (ADAPTIVE)
    MIN_POSITIVE_POINTS = 3     # For very small bundles
    MAX_POSITIVE_POINTS = 6     # For large bundles
    POINTS_PER_100_PIXELS = 1   # Scale prompts with bundle area
    N_NEGATIVE_POINTS = 2       # Number of negative prompts per instance
    IMPORTANCE_PERCENTILE = 90  # Focus prompts on most confident regions
    EXCLUSION_DILATION = 14     # Larger exclusion around positives
    
    # Feature weights for importance map
    DENSITY_WEIGHT = 0.4        # Weight for intensity density
    EDGE_WEIGHT = 0.6           # Weight for edge detection
    
    # SAM ensemble parameters
    MULTIMASK_OUTPUT = False    # Faster when using best-mask strategy
    CONFIDENCE_THRESHOLD = 0.75 # Minimum confidence to accept a mask
    COMBINE_STRATEGY = "best"   # Options: "best", "union", "weighted"

    # Prompt strategy
    USE_POINTS = True           # If False, use bounding box only (no point prompts)
    
    # Post-processing parameters (PER INSTANCE)
    MIN_OBJECT_SIZE = 15        # Minimum size for connected components
    CLOSING_DISK_SIZE = 2       # Morphological closing disk radius
    
    # Bounding box parameters
    BOX_MARGIN = 15             # Margin for bounding box
    MIN_BOX_SIZE = 40           # Minimum bounding box dimension
    ALLOW_SHAPE_ADJUST = True   # If True, nearest-neighbor resize ROI mask when shapes mismatch
    FINAL_KEEP_TOP_COMPONENTS = 3  # Keep largest components in final mask (0 disables)


def apply_env_overrides(config: Config) -> str:
    """
    Override config from environment variables.
    Use SAM_MODEL to switch presets from Slurm without editing this file.
    """
    model_presets = {name: "vit_b" for name in ("sam", "medsam", "sam2", "medsam2")}

    selected_model = os.environ.get("SAM_MODEL", "").strip().lower()
    if selected_model:
        if selected_model not in model_presets:
            raise ValueError(
                f"Unknown SAM_MODEL='{selected_model}'. "
                f"Valid options: {', '.join(sorted(model_presets.keys()))}"
            )
        config.SAM_MODEL_TYPE = model_presets[selected_model]
        config.SAM_MODEL_NAME = selected_model

    # Explicit overrides (highest priority)
    if os.environ.get("SAM_CHECKPOINT"):
        config.SAM_CHECKPOINT = os.environ["SAM_CHECKPOINT"]
    if os.environ.get("SAM_MODEL_TYPE"):
        config.SAM_MODEL_TYPE = os.environ["SAM_MODEL_TYPE"]
    if os.environ.get("SAM2_CONFIG"):
        config.SAM2_CONFIG = os.environ["SAM2_CONFIG"]
    if os.environ.get("BASE_DIR"):
        config.BASE_DIR = os.environ["BASE_DIR"]
    if os.environ.get("OUTPUT_DIR"):
        config.OUTPUT_DIR = os.environ["OUTPUT_DIR"]
    if os.environ.get("USE_POINTS"):
        config.USE_POINTS = os.environ["USE_POINTS"].strip().lower() in ("1", "true", "yes")
    if os.environ.get("SUBJECT"):
        config.SUBJECT = os.environ["SUBJECT"]
    if os.environ.get("PYRAMID_LEVEL"):
        config.PYRAMID_LEVEL = os.environ["PYRAMID_LEVEL"]
    if os.environ.get("START_SLICE"):
        config.START_SLICE = int(os.environ["START_SLICE"])
    if os.environ.get("END_SLICE"):
        config.END_SLICE = int(os.environ["END_SLICE"])

    return selected_model if selected_model else "default"


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def normalize_image(img: np.ndarray) -> np.ndarray:
    """Robust normalization using percentiles to handle outliers."""
    img = img.astype(np.float32)
    
    if len(img.shape) == 2:
        vmin, vmax = np.percentile(img, [1, 99])
        return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)
    
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        vmin, vmax = np.percentile(ch, [1, 99])
        out[:, :, c] = np.clip((ch - vmin) / (vmax - vmin + 1e-8), 0, 1)
    
    return out


# ======================================================================
# INSTANCE DETECTION
# ======================================================================

class BundleInstance:
    """Container for a single fiber bundle instance."""
    def __init__(self, instance_id: int, mask: np.ndarray, bbox: np.ndarray):
        self.id = instance_id
        self.mask = mask
        self.bbox = bbox  # (x_min, y_min, x_max, y_max)
        self.area = mask.sum()
        self.centroid = self._compute_centroid()
        
    def _compute_centroid(self) -> Tuple[int, int]:
        """Compute instance centroid (x, y)"""
        ys, xs = np.where(self.mask)
        if len(xs) == 0:
            return (0, 0)
        return (int(xs.mean()), int(ys.mean()))
    
    def __repr__(self):
        return f"BundleInstance(id={self.id}, area={self.area}, centroid={self.centroid})"


def find_bundle_instances(
    image: np.ndarray, 
    roi_mask: np.ndarray, 
    config: Config
) -> List[BundleInstance]:
    """
    Multi-instance pre-segmentation: Detect all fiber bundle instances.
    
    Uses intensity thresholding + connected components analysis.
    
    Returns:
        List of BundleInstance objects, sorted by area (largest first)
    """
    
    # Convert to grayscale and normalize
    gray = image.mean(axis=2) if len(image.shape) == 3 else image
    gray = gray.astype(np.float32)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    
    # Apply ROI mask
    gray_roi = gray * roi_mask
    
    # Threshold to find bright regions
    vals = gray_roi[roi_mask > 0]
    if vals.size == 0:
        return []
    
    threshold = np.percentile(vals, config.PRESEG_PERCENTILE)
    binary = (gray_roi > threshold) & (roi_mask > 0)
    
    # Clean up noise
    binary = remove_small_objects(binary, min_size=config.PRESEG_MIN_SIZE)
    binary = binary_opening(binary, disk(2))
    
    if binary.sum() == 0:
        return []
    
    # Connected components analysis
    labeled_array, n_components = label(binary)
    
    instances = []
    
    for component_id in range(1, n_components + 1):
        component_mask = (labeled_array == component_id)
        area = component_mask.sum()
        
        if area < config.MIN_INSTANCE_SIZE:
            continue
        
        # Compute bounding box with margin
        ys, xs = np.where(component_mask)
        margin = config.PRESEG_MARGIN
        
        x_min = max(0, xs.min() - margin)
        x_max = min(image.shape[1], xs.max() + margin)
        y_min = max(0, ys.min() - margin)
        y_max = min(image.shape[0], ys.max() + margin)
        
        bbox = np.array([x_min, y_min, x_max, y_max])
        instance = BundleInstance(component_id, component_mask, bbox)
        instances.append(instance)
    
    # Optional merging of nearby instances
    instances = merge_nearby_instances(instances, config.MERGE_DISTANCE)
    
    # Limit number of instances
    if len(instances) > config.MAX_INSTANCES_PER_SLICE:
        instances = sorted(instances, key=lambda x: x.area, reverse=True)
        instances = instances[:config.MAX_INSTANCES_PER_SLICE]
    
    # Sort by area and reassign IDs
    instances = sorted(instances, key=lambda x: x.area, reverse=True)
    for i, inst in enumerate(instances, start=1):
        inst.id = i
    
    return instances


def merge_nearby_instances(
    instances: List[BundleInstance], 
    distance_threshold: int
) -> List[BundleInstance]:
    """
    Merge instances that are spatially close.
    Uses centroid distance as proximity metric.
    
    Note: This is a simple greedy approach. For production use,
    consider graph-based connectivity analysis.
    """
    if len(instances) <= 1:
        return instances
    
    merged = []
    used = set()
    
    for i, inst_i in enumerate(instances):
        if i in used:
            continue
        
        merged_mask = inst_i.mask.copy()
        
        for j, inst_j in enumerate(instances[i+1:], start=i+1):
            if j in used:
                continue
            
            # Compute centroid distance
            cx_i, cy_i = inst_i.centroid
            cx_j, cy_j = inst_j.centroid
            distance = np.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
            
            if distance < distance_threshold:
                merged_mask |= inst_j.mask
                used.add(j)
        
        # Create merged instance
        ys, xs = np.where(merged_mask)
        if len(xs) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            bbox = np.array([x_min, y_min, x_max, y_max])
            
            merged_instance = BundleInstance(inst_i.id, merged_mask, bbox)
            merged.append(merged_instance)
        
        used.add(i)
    
    return merged


# ======================================================================
# ADAPTIVE PROMPT GENERATION
# ======================================================================

def compute_importance_map(
    image: np.ndarray, 
    bundle_mask: np.ndarray, 
    config: Config
) -> np.ndarray:
    """
    Compute importance map combining intensity and edge information.
    For thin fiber bundles, edges are more important than intensity.
    """
    gray = image.mean(axis=2) if len(image.shape) == 3 else image
    
    # Intensity density (Gaussian smoothed)
    density = gaussian_filter(gray * bundle_mask, sigma=2)
    density = density / (density.max() + 1e-8)
    
    # Edge detection (Sobel operator)
    edges = sobel(gray)
    edges = edges / (edges.max() + 1e-8)
    
    # Combine with weights
    importance = (config.DENSITY_WEIGHT * density + 
                  config.EDGE_WEIGHT * edges)
    
    # Restrict to bundle region
    importance = importance * bundle_mask
    
    return importance


def farthest_point_sampling(coords: np.ndarray, k: int) -> np.ndarray:
    """
    Sample k points with maximum spatial diversity using Farthest Point Sampling.
    Ensures points are spread across the bundle rather than clustered.
    """
    if len(coords) <= k:
        return coords
    
    selected_indices = [np.random.randint(len(coords))]
    selected = [coords[selected_indices[0]]]
    
    for _ in range(k - 1):
        dists_to_selected = []
        for i in range(len(coords)):
            if i in selected_indices:
                dists_to_selected.append(0)
            else:
                min_dist = min([np.linalg.norm(coords[i] - s) for s in selected])
                dists_to_selected.append(min_dist)
        
        farthest_idx = np.argmax(dists_to_selected)
        selected_indices.append(farthest_idx)
        selected.append(coords[farthest_idx])
    
    return np.array(selected)


def compute_adaptive_n_prompts(instance: BundleInstance, config: Config) -> int:
    """
    Adaptively determine number of prompts based on bundle size.
    
    FIXED: Now correctly uses MIN/MAX bounds, not N_POSITIVE_POINTS.
    """
    # Scale with square root of area (geometric scaling)
    base_prompts = int(np.sqrt(instance.area) * config.POINTS_PER_100_PIXELS / 10)
    
    # FIXED: Clamp to MIN/MAX, not default N_POSITIVE_POINTS
    n_prompts = np.clip(
        base_prompts, 
        config.MIN_POSITIVE_POINTS, 
        config.MAX_POSITIVE_POINTS
    )
    
    return int(n_prompts)


def sample_prompts_for_instance(
    image: np.ndarray, 
    instance: BundleInstance,
    roi_mask: np.ndarray, 
    config: Config
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate positive and negative prompts for a single bundle instance.
    
    Returns:
        points: (N, 2) array of (x, y) coordinates
        labels: (N,) array of labels (1=positive, 0=negative)
    """
    
    # Compute importance map
    importance = compute_importance_map(image, instance.mask, config)
    
    vals = importance[instance.mask > 0]
    if vals.size == 0:
        return None, None
    
    # Threshold for high-importance regions
    thr = np.percentile(vals, config.IMPORTANCE_PERCENTILE)
    
    # === POSITIVE POINTS ===
    pos_candidates = np.argwhere((importance >= thr) & (instance.mask > 0))
    
    if len(pos_candidates) == 0:
        pos_candidates = np.argwhere(instance.mask > 0)
    
    if len(pos_candidates) == 0:
        return None, None
    
    # Adaptive number of prompts
    n_pos = compute_adaptive_n_prompts(instance, config)
    
    # Apply Farthest Point Sampling
    if len(pos_candidates) > n_pos:
        pos = farthest_point_sampling(pos_candidates, n_pos)
    else:
        pos = pos_candidates
    
    # === NEGATIVE POINTS ===
    # Exclusion zone around THIS instance
    exclusion_zone = binary_dilation(
        instance.mask, 
        iterations=config.EXCLUSION_DILATION
    )
    
    neg_candidates = np.argwhere((roi_mask > 0) & (~exclusion_zone))
    
    if len(neg_candidates) == 0:
        # Fallback: points far from instance
        cx, cy = instance.centroid
        h, w = instance.mask.shape
        
        border_coords = []
        for y in [0, h//4, h//2, 3*h//4, h-1]:
            for x in [0, w//4, w//2, 3*w//4, w-1]:
                if 0 <= y < h and 0 <= x < w and roi_mask[y, x] > 0:
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist > 50:
                        border_coords.append([y, x])
        
        neg_candidates = np.array(border_coords) if border_coords else np.array([[0, 0]])
    
    # Sample negative points
    n_neg = config.N_NEGATIVE_POINTS
    if len(neg_candidates) > n_neg:
        neg = farthest_point_sampling(neg_candidates, n_neg)
    else:
        neg = neg_candidates[:n_neg] if len(neg_candidates) > 0 else np.array([[0, 0]])
    
    # === COMBINE PROMPTS ===
    # Convert from (y, x) to (x, y) for SAM format
    points = np.vstack([pos, neg])[:, [1, 0]]
    labels = np.hstack([
        np.ones(len(pos), dtype=int), 
        np.zeros(len(neg), dtype=int)
    ])
    
    return points.astype(int), labels.astype(int)


def compute_bounding_box(mask: np.ndarray, margin: int = 10) -> Optional[np.ndarray]:
    """Compute tight bounding box around mask with margin."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    
    h, w = mask.shape
    x_min = max(0, xs.min() - margin)
    y_min = max(0, ys.min() - margin)
    x_max = min(w, xs.max() + margin)
    y_max = min(h, ys.max() + margin)
    
    return np.array([x_min, y_min, x_max, y_max])


# ======================================================================
# SAM INFERENCE
# ======================================================================

def predict_instance_with_sam(
    predictor: SamPredictor,
    image: np.ndarray,
    instance: BundleInstance,
    points: np.ndarray,
    labels: np.ndarray,
    config: Config
) -> np.ndarray:
    """
    Run SAM inference for a single bundle instance with ensemble strategy.
    
    Returns:
        pred_mask: Final predicted mask
    """
    
    # Compute bounding box
    bbox = compute_bounding_box(instance.mask, margin=config.BOX_MARGIN)
    
    # Ensure minimum box size
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        w = x_max - x_min
        h = y_max - y_min
        
        if w < config.MIN_BOX_SIZE or h < config.MIN_BOX_SIZE:
            cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
            half_size = config.MIN_BOX_SIZE // 2
            bbox = np.array([
                max(0, cx - half_size),
                max(0, cy - half_size),
                min(image.shape[1], cx + half_size),
                min(image.shape[0], cy + half_size)
            ])
    
    # Run SAM prediction
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=bbox,
        multimask_output=config.MULTIMASK_OUTPUT
    )
    
    # === ENSEMBLE STRATEGY ===
    
    if config.COMBINE_STRATEGY == "best":
        best_idx = np.argmax(scores)
        pred_mask = masks[best_idx].astype(bool)
        
    elif config.COMBINE_STRATEGY == "union":
        pred_mask = np.zeros_like(masks[0], dtype=bool)
        valid_scores = []
        
        for mask, score in zip(masks, scores):
            if score >= config.CONFIDENCE_THRESHOLD:
                pred_mask |= mask
                valid_scores.append(score)
        
        if not valid_scores:
            best_idx = np.argmax(scores)
            pred_mask = masks[best_idx].astype(bool)
            
    elif config.COMBINE_STRATEGY == "weighted":
        pred_prob = np.zeros(masks[0].shape, dtype=np.float32)
        total_weight = 0
        
        for mask, score in zip(masks, scores):
            if score >= config.CONFIDENCE_THRESHOLD:
                pred_prob += mask.astype(np.float32) * score
                total_weight += score
        
        if total_weight > 0:
            pred_prob /= total_weight
            pred_mask = pred_prob > 0.5
        else:
            best_idx = np.argmax(scores)
            pred_mask = masks[best_idx].astype(bool)
    
    else:
        raise ValueError(f"Unknown combine strategy: {config.COMBINE_STRATEGY}")
    
    return pred_mask


# ======================================================================
# POST-PROCESSING
# ======================================================================

def post_process_mask(
    mask: np.ndarray, 
    roi_mask: np.ndarray, 
    config: Config
) -> np.ndarray:
    """
    Post-process predicted mask to remove noise and smooth boundaries.
    IMPORTANT: Apply this PER INSTANCE before combining.
    """
    # Restrict to ROI
    mask = mask & (roi_mask > 0)
    
    # Remove small objects (noise)
    mask = remove_small_objects(mask, min_size=config.MIN_OBJECT_SIZE)
    
    # Morphological closing to smooth boundaries
    mask = binary_closing(mask, disk(config.CLOSING_DISK_SIZE))
    
    # Final ROI restriction
    mask = mask & (roi_mask > 0)
    
    return mask


def keep_largest_components(mask: np.ndarray, top_k: int) -> np.ndarray:
    """Keep only the largest connected components in a binary mask."""
    if top_k <= 0:
        return mask
    labeled, n_comp = label(mask)
    if n_comp <= top_k:
        return mask

    areas = []
    for comp_id in range(1, n_comp + 1):
        areas.append((comp_id, int((labeled == comp_id).sum())))
    areas.sort(key=lambda x: x[1], reverse=True)
    keep_ids = {comp_id for comp_id, _ in areas[:top_k]}

    out = np.zeros_like(mask, dtype=bool)
    for comp_id in keep_ids:
        out |= (labeled == comp_id)
    return out


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    """
    Main multi-instance SAM inference pipeline.
    
    Outputs:
    - Combined prediction masks (PNG)
    """
    
    config = Config()
    apply_env_overrides(config)
    
    # Construct paths
    dirname = Path(config.BASE_DIR) / f"sub-{config.SUBJECT}" / "micr"
    
    # Load zarr groups
    histology_paths = sorted(dirname.glob(f"sub-{config.SUBJECT}_*_stain-LY_DF.ome.zarr"))
    if not histology_paths:
        raise FileNotFoundError(f"No LY_DF OME-Zarr found in {dirname}")
    histology = zarr.open_group(histology_paths[0], mode="r")
    outline = zarr.open_group(dirname / "masks" / "Outline.ome.zarr", mode="r")
    
    hist_z = histology[config.PYRAMID_LEVEL]
    
    # Load outline mask directly at the same pyramid level as histology
    if config.PYRAMID_LEVEL not in outline:
        raise KeyError(
            f"Outline level {config.PYRAMID_LEVEL} not found. "
            "This pipeline expects all data at the same level."
        )
    outline_data = outline[config.PYRAMID_LEVEL][0, :]
    
    # Initialize model + predictor
    if config.SAM_MODEL_NAME == "sam2":
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise ImportError(
                "SAM2 selected but sam2 package is not available. "
                "Install/clone SAM2 and ensure it's on PYTHONPATH."
            ) from exc

        if not config.SAM2_CONFIG:
            raise ValueError(
                "SAM2 selected but SAM2_CONFIG is empty. "
                "Set SAM2_CONFIG to the SAM2 config yaml."
            )

        sam2_model = build_sam2(config.SAM2_CONFIG, config.SAM_CHECKPOINT)
        sam2_model.to("cuda")
        sam2_model.eval()
        predictor = SAM2ImagePredictor(sam2_model)
    elif config.SAM_MODEL_NAME == "medsam2":
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise ImportError(
                "MedSAM2 selected but sam2 package from the MedSAM2 repo is not available. "
                "Ensure the MedSAM2 repository is on PYTHONPATH."
            ) from exc

        if not config.SAM2_CONFIG:
            raise ValueError(
                "MedSAM2 selected but SAM2_CONFIG is empty. "
                "Set SAM2_CONFIG to the MedSAM2 config yaml."
            )

        medsam2_model = build_sam2(config.SAM2_CONFIG, config.SAM_CHECKPOINT)
        medsam2_model.to("cuda")
        medsam2_model.eval()
        predictor = SAM2ImagePredictor(medsam2_model)
    else:
        sam = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=config.SAM_CHECKPOINT)
        sam.to("cuda")
        sam.eval()
        predictor = SamPredictor(sam)
    
    # Setup output directories
    output_dir = Path(config.OUTPUT_DIR)
    
    output_dir.mkdir(exist_ok=True)
    
    # Process slices (guard against length mismatch across modalities)
    num_slices = min(hist_z.shape[1], outline_data.shape[0])
    
    start_slice = 0 if config.START_SLICE is None else max(0, config.START_SLICE)
    end_slice = num_slices if config.END_SLICE is None else min(num_slices, config.END_SLICE)
    if start_slice >= end_slice:
        raise ValueError(
            f"Invalid slice range: START_SLICE={config.START_SLICE}, "
            f"END_SLICE={config.END_SLICE}, num_slices={num_slices}"
        )

    for z in range(start_slice, end_slice):
        # Load slice data
        image = np.transpose(hist_z[:, z, :, :], (1, 2, 0))
        roi_mask = outline_data[z]
        
        if roi_mask.shape != image.shape[:2]:
            if not config.ALLOW_SHAPE_ADJUST:
                raise ValueError(
                    f"Shape mismatch at slice {z}: roi={roi_mask.shape}, image={image.shape[:2]}. "
                    "Set ALLOW_SHAPE_ADJUST=True to auto-resize ROI."
                )
            roi_mask = resize_mask_nearest(roi_mask, image.shape[:2])
        
        if roi_mask.sum() == 0:
            continue
        
        # Stage 1: Multi-instance detection
        img_normalized = normalize_image(image)
        instances = find_bundle_instances(image, roi_mask, config)
        
        if len(instances) == 0:
            # Save empty prediction
            empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            Image.fromarray(empty_mask).save(output_dir / f"slice_{z:03d}_pred.png")
            continue
        
        # Stage 2-4: Process each instance
        predictor.set_image((img_normalized * 255).astype(np.uint8))
        
        instance_masks = []
        
        for instance in instances:
            if config.USE_POINTS:
                # Generate prompts
                points, labels = sample_prompts_for_instance(
                    img_normalized, instance, roi_mask, config
                )

                if points is None:
                    instance_masks.append(np.zeros_like(instance.mask))
                    continue
            else:
                points, labels = None, None

            # SAM inference
            pred_mask = predict_instance_with_sam(
                predictor, img_normalized, instance, points, labels, config
            )
            
            # Post-processing PER INSTANCE (FIXED)
            pred_mask = post_process_mask(pred_mask, roi_mask, config)
            
            instance_masks.append(pred_mask)
        
        # Stage 5: Combine instances
        combined_mask = np.zeros_like(instance_masks[0], dtype=bool)
        for mask in instance_masks:
            combined_mask |= mask
        combined_mask = keep_largest_components(
            combined_mask, config.FINAL_KEEP_TOP_COMPONENTS
        )
        
        # Save masks
        Image.fromarray((combined_mask * 255).astype(np.uint8)).save(
            output_dir / f"slice_{z:03d}_pred.png"
        )


def parse_args() -> argparse.Namespace:
    """Parse options for multi-instance SAM-family inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("sam", "medsam", "sam2", "medsam2"),
                        default=os.environ.get("SAM_MODEL", "sam"))
    parser.add_argument("--checkpoint", default=os.environ.get("SAM_CHECKPOINT"))
    parser.add_argument("--base-dir", default=os.environ.get("BASE_DIR", Config.BASE_DIR))
    parser.add_argument("--subject", default=os.environ.get("SUBJECT", Config.SUBJECT))
    parser.add_argument("--level", default=os.environ.get("PYRAMID_LEVEL", Config.PYRAMID_LEVEL))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", Config.OUTPUT_DIR))
    parser.add_argument("--start-slice", type=int, default=None)
    parser.add_argument("--end-slice", type=int, default=None)
    parser.add_argument("--sam2-config", default=os.environ.get("SAM2_CONFIG"))
    parser.add_argument("--box-only", action="store_true", help="Use bounding boxes without point prompts")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint is required (or set SAM_CHECKPOINT)")
    if not args.base_dir:
        parser.error("--base-dir is required (or set BASE_DIR)")
    if args.model in {"sam2", "medsam2"} and not args.sam2_config:
        parser.error("--sam2-config is required for SAM2 and MedSAM2")
    return args


def cli() -> None:
    args = parse_args()
    values = {
        "SAM_MODEL": args.model,
        "SAM_CHECKPOINT": args.checkpoint,
        "BASE_DIR": args.base_dir,
        "SUBJECT": args.subject,
        "PYRAMID_LEVEL": args.level,
        "OUTPUT_DIR": args.output_dir,
        "START_SLICE": args.start_slice,
        "END_SLICE": args.end_slice,
        "SAM2_CONFIG": args.sam2_config,
        "USE_POINTS": not args.box_only,
    }
    for name, value in values.items():
        if value is not None:
            os.environ[name] = str(value)
    main()


if __name__ == "__main__":
    cli()
