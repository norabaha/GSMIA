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
    T.tic("tiles: downsampling")
    orig_h, orig_w = img.shape[:2]
    # Optional pre-downsample for faster processing
    if cfg.downsample:
        scale = cfg.downsample_scale
        work_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if cfg.debug:
            print(f"[Method3-Tiles] Downsampled to {work_img.shape[0]}x{work_img.shape[1]} (scale={scale})")
    else:
        scale = 1.0
        work_img = img
        if cfg.debug:
            print(f"[Method3-Tiles] No downsampling")
    T.toc()
    
    h, w, _ = work_img.shape
    ty = tx = cfg.num_tiles
    tile_h = h // ty
    tile_w = w // tx
    if cfg.debug:
        print(f"[Method3-Tiles] Tile grid: {ty}x{tx}, tile size: {tile_h}x{tile_w}")
    T.tic("tiles: masking")
    # Mask saturated/black pixels
    mask = ~(np.any(work_img >= cfg.mask_saturation_threshold, axis=2) |
             np.all(work_img <= cfg.mask_black_threshold, axis=2))
    if cfg.debug:
        print(f"[Method3-Tiles] Valid pixels: {np.sum(mask)}/{mask.size} ({100*np.sum(mask)/mask.size:.1f}%)")

    # Ensure there are valid pixels
    if not np.any(mask):
        raise ValueError("No valid pixels found: all pixels are either too bright or too dark.")
    T.toc()

    # Compute per-tile illuminants
    T.tic("tiles: region statistics")
    region_illuminants_full = np.full((ty*tx, 2), np.nan, dtype=np.float32)
    if cfg.debug:
        print(f"[Method3-Tiles] Computing illuminants for {ty*tx} tiles...")
    for i in range(ty):
        for j in range(tx):
            idx = i*tx + j
            tile = work_img[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            tile_mask = mask[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]

            if np.any(tile_mask):
                vals = tile[tile_mask]
                if vals.shape[1] != 3:
                    raise ValueError(f"Invalid tile RGB shape: {vals.shape}")
                mean_rgb = vals.mean(axis=0)
                if mean_rgb[1] < 1e-6:
                    region_illuminants_full[idx] = [0.0, 0.0]
                else:
                    region_illuminants_full[idx] = [mean_rgb[0]/mean_rgb[1],
                                                    mean_rgb[2]/mean_rgb[1]]
    T.toc()

    valid_tiles = ~np.isnan(region_illuminants_full).any(axis=1)
    if cfg.debug:
        print(f"[Method3-Tiles] Valid tiles: {np.sum(valid_tiles)}/{ty*tx}")

    # Check which tiles are inside a boundary around the curve
    if cfg.cct_curve_rb is None:
        raise ValueError("cct_curve_rb must be initialized in AWBConfig")
    
    T.tic("tiles: CCT distance")
    distances = np.linalg.norm(
        cfg.cct_curve_rb[:, None, :] - region_illuminants_full[None, :, :],
        axis=2
    )
    # Remove NaNs before computing min --> no warnings
    distances_clean = np.nan_to_num(distances, nan=np.inf)
    min_dist = distances_clean.min(axis=0)
    T.toc()
    mask_inside_full = min_dist < cfg.cct_tolerance

    inside_points = region_illuminants_full[mask_inside_full]
    if cfg.debug:
        print(f"[Method3-Tiles] Tiles inside CCT curve (tol={cfg.cct_tolerance}): {len(inside_points)}/{ty*tx}")
        if len(inside_points) > 0:
            print(f"[Method3-Tiles] Inside points range: r=[{inside_points[:,0].min():.3f}, {inside_points[:,0].max():.3f}], b=[{inside_points[:,1].min():.3f}, {inside_points[:,1].max():.3f}]")

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
    if cfg.debug:
        print(f"Clustering {len(inside_points)}/{tx*ty} valid tiles...")

    # ------------------------------------------------------------
    # Cluster centers based on tiles inside the CCT boundary
    T.tic("tiles: clustering")
    cl_labels, cl_centers = cluster.cluster(inside_points, cfg.cluster_method, **cfg.cluster_kwargs)
    T.toc()
    if cfg.debug:
        print(f"[Method3-Tiles] Clustering done: {len(cl_centers)} clusters found")
        for i, center in enumerate(cl_centers):
            print(f"[Method3-Tiles]   Cluster {i}: center=(r={center[0]:.4f}, b={center[1]:.4f})")

    # ------------------------------------------------------------
    # Two-cluster ratio check
    # ------------------------------------------------------------
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
    if cfg.debug and len(cl_centers) > 1:
        print(f"[Method3-Tiles] Cluster distance: {cluster_dist:.4f} (min={cfg.cluster_min_center_distance})")
    if len(cl_centers) > 1 and cluster_dist < cfg.cluster_min_center_distance:
        if cfg.debug:
            print(f"[Method3-Tiles] Clusters too close; using single illuminant fallback.")
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

    # ------------------------------------------------------------
    # HYBRID ASSIGNMENT: Direct for inside, weighted for outside
    # ------------------------------------------------------------
    if cfg.debug:
        print(f"[Method3-Tiles] Using multi-illuminant mode with {len(cl_centers)} clusters")
        print(f"[Method3-Tiles] Computing per-tile gains with hybrid assignment...")
        inside_count = np.sum(mask_inside_full)
        outside_count = np.sum(valid_tiles & ~mask_inside_full)
        print(f"[Method3-Tiles]   Inside CCT tolerance (direct assignment): {inside_count}")
        print(f"[Method3-Tiles]   Outside CCT tolerance (weighted average): {outside_count}")
    
    
    # Track which cluster each inside point belongs to for direct assignment
    inside_indices = np.where(mask_inside_full)[0]
    cluster_assignment = np.zeros(len(region_illuminants_full), dtype=int)
    cluster_assignment[inside_indices] = cl_labels

    # Compute average cluster center for invalid regions
    mean_illuminant = cl_centers.mean(axis=0)
    mean_gain_r = 1.0 / (mean_illuminant[0] + cfg.eps)
    mean_gain_b = 1.0 / (mean_illuminant[1] + cfg.eps)

    # NEW CODE: per-pixel gain assignment loop
    gain_map_small = np.ones((h, w, 3), dtype=np.float32)

    for idx in range(ty * tx):
        pt = region_illuminants_full[idx]
        i, j = divmod(idx, tx)

        y0, y1 = i * tile_h, (i + 1) * tile_h
        x0, x1 = j * tile_w, (j + 1) * tile_w

        if np.isnan(pt).any():
            gain_map_small[y0:y1, x0:x1, 0] = mean_gain_r
            gain_map_small[y0:y1, x0:x1, 2] = mean_gain_b
            continue

        if mask_inside_full[idx]:
            dists = np.linalg.norm(cl_centers - pt, axis=1)
            assigned = cl_centers[np.argmin(dists)]
        else:
            dists = np.linalg.norm(cl_centers - pt, axis=1)
            weights = 1.0 / (dists + cfg.eps)
            assigned = (weights[:, None] * cl_centers).sum(axis=0) / weights.sum()

        gain_map_small[y0:y1, x0:x1, 0] = 1.0 / (assigned[0] + cfg.eps)
        gain_map_small[y0:y1, x0:x1, 2] = 1.0 / (assigned[1] + cfg.eps)

    # ------------------------------------------------------------

    # Two-stage smoothing pipeline
    T.tic("tiles: smooth gain")
    smoothed_gain_map_small = gain_map_small.copy()
    if cfg.debug:
        gain_range = (gain_map_small.min(), gain_map_small.max())
        print(f"[Method3-Tiles] Gain map range before smoothing: [{gain_range[0]:.3f}, {gain_range[1]:.3f}]")
    
    if cfg.smooth1_method != 'none':
        if cfg.debug:
            print(f"[Method3-Tiles] Smoothing stage1 using {cfg.smooth1_method} filter...")
        smoothed_gain_map_small = smooth.smooth_gain_map(smoothed_gain_map_small, img=work_img, method=cfg.smooth1_method, **cfg.smooth1_kwargs)
    if cfg.smooth2_method != 'none':
        if cfg.debug:
            print(f"[Method3-Tiles] Smoothing stage2 using {cfg.smooth2_method} filter...")
        smoothed_gain_map_small = smooth.smooth_gain_map(smoothed_gain_map_small, img=work_img, method=cfg.smooth2_method, **cfg.smooth2_kwargs)
    T.toc()
    if cfg.downsample and scale != 1.0:
        T.tic("tiles: smooth gain upsample")
        smoothed_gain_map = cv2.resize(smoothed_gain_map_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) # cv2.INTER_AREA, cv2.INTER_LINEAR, INTER_NEAREST
        T.toc()
        T.tic("tiles: mask upsample")
        mask_out = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR).astype(bool)
        T.toc()
    else:
        smoothed_gain_map = smoothed_gain_map_small
        mask_out = mask
    
    img_awb = np.clip(img * smoothed_gain_map, 0, 1)

    # Create full-length cluster_labels to match region_illuminants_full
    cluster_labels_full = np.zeros(len(region_illuminants_full), dtype=int)
    cluster_labels_full[mask_inside_full] = cl_labels
    cluster_labels_full = cluster_assignment.copy()

    if cfg.debug:
        print(f"[Method3-Tiles] Cluster label assignment:")
        print(f"  Total regions: {len(region_illuminants_full)}")
        print(f"  Regions inside CCT: {np.sum(mask_inside_full)}")
        print(f"  cl_labels shape: {cl_labels.shape}, unique values: {np.unique(cl_labels)}")
        print(f"  cluster_labels_full shape: {cluster_labels_full.shape}")
        unique_full, counts_full = np.unique(cluster_labels_full, return_counts=True)
        print(f"  cluster_labels_full distribution: {dict(zip(unique_full, counts_full))}")
        # Check a few examples
        inside_indices = np.where(mask_inside_full)[0][:5]
        print(f"  First 5 inside indices: {inside_indices}")
        print(f"  Their cl_labels: {cl_labels[:5]}")
        print(f"  Their cluster_labels_full: {cluster_labels_full[inside_indices]}")
    
    # T.report()

    return {
        'image': img_awb,
        'mask': None,#mask_out,
        'mask_inside': mask_inside_full,
        'gain_map': None,#cv2.resize(gains, (w, h), interpolation=cv2.INTER_AREA),
        'smoothed_gain_map': smoothed_gain_map,
        'region_illuminants': region_illuminants_full,
        'cluster_labels': cluster_labels_full,
        'cluster_centers': cl_centers,
        'timings': T.get_times()
    }


def run_awb_method_3_superpixels(img, cfg: AWBConfig):
    """
    Advanced AWB method using superpixel-based clustering and smoothing.

    Args:
        img: HxWx3 float32 image in [0,1]
        cfg: AWBConfig

    Returns:
        dict with keys:
        'image', 'mask', 'mask_inside', 'gain_map', 'smoothed_gain_map',
        'region_illuminants', 'cluster_labels', 'cluster_centers'
    """
    T = timer.Timer()
    T.tic("sp: Preprocessing")
    orig_h, orig_w = img.shape[:2]
    
    # Optional pre-downsample for faster processing
    if cfg.downsample:
        scale = cfg.downsample_scale
        work_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        work_img = img

    h, w, _ = work_img.shape

    # ------------------------------------------------------------
    # Mask saturated/too-dark pixels
    # ------------------------------------------------------------
    mask = ~(np.any(work_img >= cfg.mask_saturation_threshold, axis=2) |
             np.all(work_img <= cfg.mask_black_threshold, axis=2))

    if not np.any(mask):
        raise ValueError("No valid pixels found: all pixels are either too bright or too dark.")
    T.toc() # Preprocessing
    # ------------------------------------------------------------
    # Region segmentation
    # ------------------------------------------------------------
    # Compute superpixels (fast mode: fewer iterations, no connectivity enforcement)
    T.tic("sp: Region segmentation")
    labels, num_regions = superpixels.compute_superpixels_cv2(
        work_img,
        n_segments=cfg.num_superpixels,
        compactness=cfg.sp_compactness,
        num_iterations=4,
        enforce_connectivity=False
    )
    T.toc()
    # ------------------------------------------------------------
    # REGION ILLUMINANT ESTIMATION
    # ------------------------------------------------------------
    T.tic("sp: Region illuminants")
    # Compute per-superpixel illuminants (vectorized for speed)
    flat_labels = labels.reshape(-1)
    flat_img = work_img.reshape(-1, 3)
    flat_mask = mask.reshape(-1)

    region_illuminants_full = np.full((num_regions, 2), np.nan, dtype=np.float32)
    
    # Vectorized computation using bincount
    valid_pixels = flat_mask
    valid_labels = flat_labels[valid_pixels]
    valid_rgb = flat_img[valid_pixels]
    
    # Sum RGB values per superpixel
    
    for c in range(3):
        sums = np.bincount(valid_labels, weights=valid_rgb[:, c], minlength=num_regions)
        counts = np.bincount(valid_labels, minlength=num_regions)
        
        if c == 0:
            mean_r = np.divide(sums, counts, where=counts>0, out=np.zeros_like(sums))
        elif c == 1:
            mean_g = np.divide(sums, counts, where=counts>0, out=np.zeros_like(sums))
        else:
            mean_b = np.divide(sums, counts, where=counts>0, out=np.zeros_like(sums))

    # Compute R/G and B/G ratios
    valid_regions = (counts > 0) & (mean_g > 1e-6)
    region_illuminants_full[valid_regions, 0] = mean_r[valid_regions] / mean_g[valid_regions]
    region_illuminants_full[valid_regions, 1] = mean_b[valid_regions] / mean_g[valid_regions]
    
    T.toc() # Region illuminants

    # ------------------------------------------------------------
    # Cluster region illuminants
    # ------------------------------------------------------------
    T.tic("sp: Clustering")

    # Check which superpixels are inside CCT boundary
    if cfg.cct_curve_rb is None:
        raise ValueError("cct_curve_rb must be initialized in AWBConfig")
    
    distances = np.linalg.norm(
        cfg.cct_curve_rb[:, None, :] - region_illuminants_full[None, :, :],
        axis=2
    )
    distances_clean = np.nan_to_num(distances, nan=np.inf)
    min_dist = distances_clean.min(axis=0)
    mask_inside_full = min_dist < cfg.cct_tolerance
    inside_points = region_illuminants_full[mask_inside_full]

    # Handle empty inside_points fallback
    if len(inside_points) == 0:
        if cfg.debug:
            print("No superpixels inside CCT curve. Using single-illuminant fallback.")
        valid = ~np.isnan(region_illuminants_full).any(axis=1)
        mean_illum = np.nanmean(region_illuminants_full[valid], axis=0)
        gain_vec = np.array([1.0 / (mean_illum[0] + cfg.eps), 1.0, 1.0 / (mean_illum[1] + cfg.eps)])
        gain_map_small = np.ones((h, w, 3), dtype=np.float32) * gain_vec

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
            'gain_map': gain_map_small,
            'smoothed_gain_map': smoothed_gain_map,
            'region_illuminants': region_illuminants_full,
            'cluster_labels': cluster_labels_full,
            'cluster_centers': np.array([mean_illum])
        }

    # Cluster inside points
    cl_labels, cl_centers = cluster.cluster(inside_points, cfg.cluster_method, **cfg.cluster_kwargs)
    if cfg.debug:
        print(f"[Method3-Superpixels] Clustering done: {len(cl_centers)} clusters found")
        for i, center in enumerate(cl_centers):
            print(f"[Method3-Superpixels]   Cluster {i}: center=(r={center[0]:.4f}, b={center[1]:.4f})")

    # Two-cluster ratio check
    N = len(inside_points)
    unique, counts = np.unique(cl_labels, return_counts=True)
    if cfg.debug:
        ratios = counts / N
        print(f"[Method3-Superpixels] Cluster distribution: {counts} (ratios: {ratios})")

    if len(counts) == 2:
        ratios = counts / N
        if np.any(ratios < cfg.cluster_min_ratio):
            if cfg.debug:
                print(f"[Method3-Superpixels] Cluster ratio check FAILED: ratios={ratios}, min required={cfg.cluster_min_ratio}. Using single-illuminant fallback.")
            mean_illum = cl_centers.mean(axis=0)
            gain_vec = np.array([1.0 / (mean_illum[0] + cfg.eps), 1.0, 1.0 / (mean_illum[1] + cfg.eps)])
            gain_map_small = np.ones((h, w, 3), dtype=np.float32) * gain_vec

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
                'gain_map': gain_map_small,
                'smoothed_gain_map': smoothed_gain_map,
                'region_illuminants': region_illuminants_full,
                'cluster_labels': cluster_labels_full,
                'cluster_centers': np.array([mean_illum])
            }

    # Clusters too close fallback
    cluster_dist = np.linalg.norm(cl_centers[0] - cl_centers[1]) if len(cl_centers) > 1 else 0
    if cfg.debug and len(cl_centers) > 1:
        print(f"[Method3-Superpixels] Cluster distance: {cluster_dist:.4f} (min={cfg.cluster_min_center_distance})")
    if len(cl_centers) > 1 and cluster_dist < cfg.cluster_min_center_distance:
        if cfg.debug:
            print(f"[Method3-Superpixels] Clusters too close; using single illuminant fallback.")
        mean_illum = cl_centers.mean(axis=0)
        gain_vec = np.array([1.0 / (mean_illum[0] + cfg.eps), 1.0, 1.0 / (mean_illum[1] + cfg.eps)])
        gain_map_small = np.ones((h, w, 3), dtype=np.float32) * gain_vec

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
            'gain_map': gain_map_small,
            'smoothed_gain_map': smoothed_gain_map,
            'region_illuminants': region_illuminants_full,
            'cluster_labels': cluster_labels_full,
            'cluster_centers': np.array([mean_illum])
        }
    T.toc()  # Clustering

    # ------------------------------------------------------------
    # ASSIGN PER-SUPERPIXEL GAINS
    # ------------------------------------------------------------
    T.tic("sp: Gain assignment")
    # HYBRID ASSIGNMENT: Direct for inside, weighted for outside
    
    gains_sp = np.ones((num_regions, 3), dtype=np.float32)
    
    # Track which cluster each inside point belongs to for direct assignment
    inside_indices = np.where(mask_inside_full)[0]
    cluster_assignment = np.zeros(len(region_illuminants_full), dtype=int)
    cluster_assignment[inside_indices] = cl_labels
    
    # Compute average cluster center for invalid regions
    mean_illuminant = cl_centers.mean(axis=0)
    mean_gain_r = 1.0 / (mean_illuminant[0] + cfg.eps)
    mean_gain_b = 1.0 / (mean_illuminant[1] + cfg.eps)
    
    # Find valid regions (non-NaN illuminants)
    valid_mask = ~np.isnan(region_illuminants_full).any(axis=1)
    valid_idx = np.where(valid_mask)[0]
    invalid_idx = np.where(~valid_mask)[0]
    
    # Assign average cluster gains to invalid regions
    gains_sp[invalid_idx, 0] = mean_gain_r
    gains_sp[invalid_idx, 2] = mean_gain_b
    
    for idx in valid_idx:
        pt = region_illuminants_full[idx]
        
        if mask_inside_full[idx]:
            # DIRECT ASSIGNMENT: Use nearest cluster center
            dists = np.linalg.norm(cl_centers - pt, axis=1)
            nearest_cluster = np.argmin(dists)
            assigned = cl_centers[nearest_cluster]
        else:
            # WEIGHTED AVERAGE: Blend all cluster centers based on distance
            dists = np.linalg.norm(cl_centers - pt, axis=1)
            weights = 1.0 / (dists + cfg.eps)
            assigned = np.sum(weights[:, None] * cl_centers, axis=0) / weights.sum()
        
        gains_sp[idx, 0] = 1.0 / (assigned[0] + cfg.eps)
        gains_sp[idx, 2] = 1.0 / (assigned[1] + cfg.eps)
    T.toc()
    # ------------------------------------------------------------
    # BUILD AND SMOOTH GAIN MAP
    # ------------------------------------------------------------
    T.tic("sp: Build gain map")
    # Expand to full resolution
    gain_map_small = gains_sp[flat_labels].reshape(h, w, 3)
    T.toc()

    # ------------------------------------------------------------
    # SMOOTHING
    # ------------------------------------------------------------
    # Two-stage smoothing pipeline
    T.tic("sp: gain smoothing")
    
    smoothed_gain_map_small = gain_map_small
    if cfg.smooth1_method != 'none':
            smoothed_gain_map_small = smooth.smooth_gain_map(smoothed_gain_map_small, img=work_img, method=cfg.smooth1_method, **cfg.smooth1_kwargs)
    T.toc()
    T.tic("sp: Upsample gain map")

    if cfg.downsample and scale != 1.0:
        smoothed_gain_map = cv2.resize(smoothed_gain_map_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        # mask_out = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        smoothed_gain_map = smoothed_gain_map_small
        # mask_out = mask
    T.toc()
    # ------------------------------------------------------------
    # APPLY GAIN MAP
    # ------------------------------------------------------------
    T.tic("sp: Apply gain map")
    img_awb = img * smoothed_gain_map
    np.clip(img_awb, 0, 1, out=img_awb)  # In-place clipping to save memory
    
    T.toc()
    # T.report()
    
    # Create full-length cluster_labels to match region_illuminants_full
    cluster_labels_full = cluster_assignment.copy()
    if cfg.debug:
        print(f"[Method3-Superpixels] Cluster label assignment:")
        print(f"  Total regions: {len(region_illuminants_full)}")
        print(f"  Regions inside CCT: {np.sum(mask_inside_full)}")
        print(f"  cl_labels shape: {cl_labels.shape}, unique values: {np.unique(cl_labels)}")
        print(f"  cluster_labels_full shape: {cluster_labels_full.shape}")
        unique_full, counts_full = np.unique(cluster_labels_full, return_counts=True)
        print(f"  cluster_labels_full distribution: {dict(zip(unique_full, counts_full))}")
        # Check a few examples
        inside_indices = np.where(mask_inside_full)[0][:5]
        print(f"  First 5 inside indices: {inside_indices}")
        print(f"  Their cl_labels: {cl_labels[:5]}")
        print(f"  Their cluster_labels_full: {cluster_labels_full[inside_indices]}")
    
    
    return {
        'image': img_awb,
        'mask': None, #mask_out,
        'mask_inside': mask_inside_full,
        'gain_map': gain_map_small,
        'smoothed_gain_map': smoothed_gain_map,
        'region_illuminants': region_illuminants_full,
        'cluster_labels': cluster_labels_full,
        'cluster_centers': cl_centers,
        'timings': T.get_times()
    }


def run_awb_method_3(img, cfg: AWBConfig):
    """
    General method 3 for AWB, supporting both tiles and superpixels.
    img: HxWx3 float32 in [0,1]
    cfg: AWBConfig
    Returns: dict with 'image', 'mask', and other outputs
    """
    if cfg.region_method == 'tiles':
        result = run_awb_method_3_tiles(img, cfg)
    elif cfg.region_method == 'superpixels':
        result = run_awb_method_3_superpixels(img, cfg)
    else:
        raise ValueError(f"Unknown region method '{cfg.region_method}'")

    return result
