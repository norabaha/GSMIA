import numpy as np
import cv2
from src.utils import cluster, superpixels, smooth, timer
from src.utils.config import AWBConfig


def run_awb_method_3_tiles(img, cfg: AWBConfig):
    """
    Advanced AWB method using tile-based clustering and smoothing.

    Args:
        img: HxWx3 float32 image in [0,1]
        cfg: AWBConfig

    Returns:
        dict with keys:
        'image', 'mask', 'mask_inside', 'gain_map', 'smoothed_gain_map',
        'region_illuminants', 'cluster_labels', 'cluster_centers'
    """
    T = timer.Timer()
    # ------------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------------
    T.tic("Downsample")
    orig_h, orig_w = img.shape[:2]
    # Optional pre-downsample for faster processing
    if cfg.downsample:
        scale = cfg.downsample_scale
        work_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    else:
        scale = 1.0
        work_img = img
    T.toc()  # Downsample
    # ------------------------------------------------------------
    # MASK CREATION
    # ------------------------------------------------------------
    T.tic("Mask creation")
    # Mask saturated/black pixels
    mask = ~(np.any(work_img >= cfg.mask_saturation_threshold, axis=2) |
             np.all(work_img <= cfg.mask_black_threshold, axis=2))
    # max_vals = work_img.max(axis=2)
    # min_vals = work_img.min(axis=2)
    # mask = (max_vals < cfg.mask_saturation_threshold) & (min_vals > cfg.mask_black_threshold)

    # Ensure there are valid pixels
    if not np.any(mask):
        raise ValueError("No valid pixels found: all pixels are either too bright or too dark.")
    T.toc()  # Mask creation

    # ------------------------------------------------------------
    # TILE SEGMENTATION
    # ------------------------------------------------------------
    T.tic("Region segmentation")

    h, w, _ = work_img.shape
    ty = tx = cfg.num_tiles
    tile_h = h // ty
    tile_w = w // tx

    T.toc()  # Region segmentation

    # ------------------------------------------------------------
    # FAST REGION MEANS
    # ------------------------------------------------------------
    T.tic("Region illuminants")

    # Compute per-tile illuminants (vectorized)
    # Crop image to exact tile grid size to enable reshaping
    cropped_h, cropped_w = tile_h * ty, tile_w * tx
    work_img_cropped = work_img[:cropped_h, :cropped_w]
    mask_cropped = mask[:cropped_h, :cropped_w]
    
    # Reshape into tiles: (ty, tile_h, tx, tile_w, 3) -> (ty, tx, tile_h, tile_w, 3)
    tiles = work_img_cropped.reshape(ty, tile_h, tx, tile_w, 3).transpose(0, 2, 1, 3, 4)
    mask_tiles = mask_cropped.reshape(ty, tile_h, tx, tile_w).transpose(0, 2, 1, 3)
    
    # Flatten to (num_tiles, pixels_per_tile, 3) and (num_tiles, pixels_per_tile)
    tiles_flat = tiles.reshape(ty * tx, -1, 3)
    mask_tiles_flat = mask_tiles.reshape(ty * tx, -1)
    
    # Compute masked sums and counts per tile
    mask_expanded = mask_tiles_flat[:, :, None].astype(np.float32)
    masked_pixels = tiles_flat * mask_expanded
    tile_sums = masked_pixels.sum(axis=1)  # (num_tiles, 3)
    tile_counts = mask_tiles_flat.sum(axis=1, keepdims=True).astype(np.float32)  # (num_tiles, 1)
    
    # Compute mean RGB per tile (avoid division by zero)
    valid_mask = tile_counts.squeeze() > 0
    mean_rgb_tiles = np.zeros((ty * tx, 3), dtype=np.float32)
    mean_rgb_tiles[valid_mask] = tile_sums[valid_mask] / tile_counts[valid_mask]
    
    # Compute r/g and b/g ratios
    region_illuminants_full = np.full((ty * tx, 2), np.nan, dtype=np.float32)
    green_valid = (valid_mask) & (mean_rgb_tiles[:, 1] >= 1e-6)
    region_illuminants_full[green_valid, 0] = mean_rgb_tiles[green_valid, 0] / mean_rgb_tiles[green_valid, 1]
    region_illuminants_full[green_valid, 1] = mean_rgb_tiles[green_valid, 2] / mean_rgb_tiles[green_valid, 1]
    # Tiles with valid pixels but green < 1e-6 get [0, 0]
    low_green = valid_mask & ~green_valid
    region_illuminants_full[low_green] = [0.0, 0.0]
    
    valid_tiles = ~np.isnan(region_illuminants_full).any(axis=1)

    T.toc()  # Region illuminants

    # ------------------------------------------------------------
    # CCT CURVE BOUNDARY CHECK
    # ------------------------------------------------------------

    T.tic("CCT boundary check")

    # Check which tiles are inside a boundary around the curve
    if cfg.cct_curve_rb is None:
        raise ValueError("cct_curve_rb must be initialized in AWBConfig")
    
    distances = np.linalg.norm(
        cfg.cct_curve_rb[:, None, :] - region_illuminants_full[None, :, :],
        axis=2
    )
    # Remove NaNs before computing min → no warnings
    distances_clean = np.nan_to_num(distances, nan=np.inf)
    min_dist = distances_clean.min(axis=0)

    mask_inside_full = min_dist < cfg.cct_tolerance

    inside_points = region_illuminants_full[mask_inside_full]
    
    # Handle empty inside_points
    if len(inside_points) == 0:
        if cfg.debug:
            print("No tiles inside CCT curve. Using single-illuminant fallback.")
        valid = ~np.isnan(region_illuminants_full).any(axis=1)
        mean_illum = np.nanmean(region_illuminants_full[valid], axis=0)
        gain_vec = np.array([1.0/(mean_illum[0]+cfg.eps), 1.0, 1.0/(mean_illum[1]+cfg.eps)])
        gain_map_small = np.ones((h, w, 3), dtype=np.float32) * gain_vec
        tile_gain_map = np.ones((ty, tx, 3), dtype=np.float32) * gain_vec

        if cfg.downsample and scale != 1.0:
            smoothed_gain_map = cv2.resize(gain_map_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            mask_out = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            smoothed_gain_map = gain_map_small
            mask_out = mask

        img_awb = np.clip(img * smoothed_gain_map, 0, 1)
        
        # Create full-length cluster_labels to match region_illuminants_full
        cluster_labels_full = np.zeros(len(region_illuminants_full), dtype=int)
        
        return {
            'image': img_awb,
            'mask': mask_out,
            'mask_inside': mask_inside_full,
            'gain_map': tile_gain_map,
            'smoothed_gain_map': smoothed_gain_map,
            'region_illuminants': region_illuminants_full,
            'cluster_labels': cluster_labels_full,
            'cluster_centers': np.array([mean_illum])
        }

    T.toc()  # CCT boundary check

    # ------------------------------------------------------------
    # Cluster centers based on tiles inside the CCT boundary
    # ------------------------------------------------------------
    T.tic("Clustering")
    cl_labels, cl_centers = cluster.cluster(inside_points, cfg.cluster_method, **cfg.cluster_kwargs)
   

    # Two-cluster ratio check

    N = len(inside_points)
    unique, counts = np.unique(cl_labels, return_counts=True)
    if cfg.debug:
        ratios = counts / N
        print(f"[Method3-Tiles] Cluster distribution: {counts} (ratios: {ratios})")

    # Should be exactly 2 clusters
    if len(counts) == 2:
        ratios = counts / N
        min_ratio = cfg.cluster_min_ratio  # e.g. 0.20

        # Either cluster too small --> fallback
        if np.any(ratios < min_ratio):
            if cfg.debug:
                print(
                    f"[Method3-Tiles] Cluster ratio check FAILED: ratios={ratios}, "
                    f"min required={min_ratio}. Using single-illuminant fallback."
                )
            mean_illum = cl_centers.mean(axis=0)
            gain_vec = np.array([
                1.0 / (mean_illum[0] + cfg.eps),
                1.0,
                1.0 / (mean_illum[1] + cfg.eps)
            ])
            gain_map_small = np.ones((h, w, 3), dtype=np.float32) * gain_vec
            tile_gain_map = np.ones((ty, tx, 3), dtype=np.float32) * gain_vec

            if cfg.downsample and scale != 1.0:
                smoothed_gain_map = cv2.resize(gain_map_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                mask_out = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                smoothed_gain_map = gain_map_small
                mask_out = mask

            img_awb = np.clip(img * smoothed_gain_map, 0, 1)

            # Create full-length cluster_labels to match region_illuminants_full
            cluster_labels_full = np.zeros(len(region_illuminants_full), dtype=int)

            return {
                'image': img_awb,
                'mask': mask_out,
                'mask_inside': mask_inside_full,
                'gain_map': tile_gain_map,
                'smoothed_gain_map': smoothed_gain_map,
                'region_illuminants': region_illuminants_full,
                'cluster_labels': cluster_labels_full,
                'cluster_centers': np.array([mean_illum])
            }


    # Single-illuminant fallback if clusters too close
    cluster_dist = np.linalg.norm(cl_centers[0] - cl_centers[1]) if len(cl_centers) > 1 else 0
    if len(cl_centers) > 1 and cluster_dist < cfg.cluster_min_center_distance:
        mean_illum = cl_centers.mean(axis=0)
        gain_vec = np.array([1.0/(mean_illum[0]+cfg.eps), 1.0, 1.0/(mean_illum[1]+cfg.eps)])
        gain_map_small = np.ones((h, w, 3), dtype=np.float32) * gain_vec
        tile_gain_map = np.ones((ty, tx, 3), dtype=np.float32) * gain_vec

        if cfg.downsample and scale != 1.0:
            smoothed_gain_map = cv2.resize(gain_map_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            mask_out = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            smoothed_gain_map = gain_map_small
            mask_out = mask

        img_awb = np.clip(img * smoothed_gain_map, 0, 1)
        
        # Create full-length cluster_labels to match region_illuminants_full
        cluster_labels_full = np.zeros(len(region_illuminants_full), dtype=int)
        
        return {
            'image': img_awb,
            'mask': mask_out,
            'mask_inside': mask_inside_full,
            'gain_map': tile_gain_map,
            'smoothed_gain_map': smoothed_gain_map,
            'region_illuminants': region_illuminants_full,
            'cluster_labels': cluster_labels_full,
            'cluster_centers': np.array([mean_illum])
        }
    T.toc()  # Clustering

    # ------------------------------------------------------------
    # Assign per-tile gains using weighted average from cluster centers (vectorized)
    # ------------------------------------------------------------
    T.tic("Gain map computation")

    gains = np.ones((ty * tx, 3), dtype=np.float32)
    valid_pts = ~np.isnan(region_illuminants_full).any(axis=1)
    pts = region_illuminants_full[valid_pts]  # (N_valid, 2)
    
    # Compute distances from each valid point to each cluster center
    # pts: (N_valid, 2), cl_centers: (K, 2) -> dists: (N_valid, K)
    dists = np.linalg.norm(pts[:, None, :] - cl_centers[None, :, :], axis=2)
    weights = 1.0 / (dists + cfg.eps)  # (N_valid, K)
    weights_sum = weights.sum(axis=1, keepdims=True)  # (N_valid, 1)
    
    # Weighted average of cluster centers: (N_valid, K) @ (K, 2) -> (N_valid, 2)
    assigned = (weights @ cl_centers) / weights_sum  # (N_valid, 2)
    
    # Compute gains for valid tiles
    gains[valid_pts, 0] = 1.0 / (assigned[:, 0] + cfg.eps)
    gains[valid_pts, 2] = 1.0 / (assigned[:, 1] + cfg.eps)
    gains = gains.reshape(ty, tx, 3)
  
    # Build gain map at working resolution, smooth, upscale, and apply

    gain_map_small = cv2.resize(gains, (w, h), interpolation=cv2.INTER_LINEAR) # cv2.INTER_AREA or cv2.INTER_LINEAR

    T.toc()  # Gain map computation

    # ------------------------------------------------------------
    # SMOOTH GAINS
    # ------------------------------------------------------------

    T.tic("Gain map smoothing")
    # Two-stage smoothing pipeline
    if cfg.smooth1_method != 'none':
        smoothed_gain_map_small = smooth.smooth_gain_map(gain_map_small, img=work_img, method=cfg.smooth1_method, **cfg.smooth1_kwargs)
    else:
        smoothed_gain_map_small = gain_map_small
    T.toc()  # Gain map smoothing

    T.tic("Gain map upscaling")
    if cfg.downsample and scale != 1.0:
        smoothed_gain_map = cv2.resize(smoothed_gain_map_small, (orig_w, orig_h), interpolation=cv2.INTER_AREA) # cv2.INTER_AREA or cv2.INTER_LINEAR
        if not cfg.fast:
            mask_out = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            mask_out = None
    else:
        smoothed_gain_map = smoothed_gain_map_small
        if not cfg.fast:
            mask_out = mask

    T.toc()  # Gain map upscaling

    T.tic("Applying gains")

    img_awb = img * smoothed_gain_map
    np.clip(img_awb, 0, 1, out=img_awb)  # In-place clipping
    T.toc()  # Applying gains

    if not cfg.fast:
        # Create full-length cluster_labels to match region_illuminants_full
        cluster_labels_full = np.zeros(len(region_illuminants_full), dtype=int)
        cluster_labels_full[mask_inside_full] = cl_labels
    else:
        cluster_labels_full = None

    return {
        'image': img_awb,
        'mask': mask_out,
        'mask_inside': mask_inside_full,
        'gain_map': gain_map_small,  # Reuse already computed resize
        'smoothed_gain_map': smoothed_gain_map,
        'region_illuminants': region_illuminants_full,
        'cluster_labels': cluster_labels_full,
        'cluster_centers': cl_centers,
        'timings': T.get_times()
    }
