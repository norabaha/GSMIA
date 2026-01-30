import numpy as np
import cv2
from src.utils import cluster, superpixels, smooth, timer
from src.utils.config import AWBConfig

def run_awb_method_3(img, cfg: AWBConfig):
    """
    Advanced AWB method using superpixels or tiles with clustering and smoothing.
    Args:
        img: HxWx3 float32 image in [0,1]
        config: AWBConfig
    Returns:
        dict: {'image': AWB-corrected image, 'gains': gain map, 'mask': valid pixel mask}
    """
    T = timer.Timer()
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
    # Mask saturated/too-dark pixels
    # ------------------------------------------------------------
    T.tic("Mask creation")
    mask = ~(np.any(work_img >= cfg.mask_saturation_threshold, axis=2) | 
         np.all(work_img <= cfg.mask_black_threshold, axis=2))

    # if not np.any(mask):
    #     raise ValueError("No valid pixels found: all pixels are either too bright or too dark.")
    T.toc()  # Mask creation

    # ------------------------------------------------------------
    # Superpixels OR Tiles (superpixels recommended)
    # ------------------------------------------------------------
    T.tic("Region segmentation")
    if cfg.region_method == 'superpixels':
        region_labels = superpixels.compute_superpixels_cv2(
            work_img,
            n_segments=cfg.num_superpixels,
            compactness=cfg.sp_compactness
        )[0]
    else:
        h, w, _ = work_img.shape
        ty = tx = cfg.num_tiles
        tile_h = h // ty
        tile_w = w // tx
        ys = np.arange(h) // tile_h
        xs = np.arange(w) // tile_w
        region_labels = ys[:, None] * tx + xs[None, :]

    num_regions = region_labels.max() + 1
    T.toc()  # Region segmentation

    # ------------------------------------------------------------
    # FAST REGION MEANS
    # ------------------------------------------------------------
    T.tic("Region illuminants")
    flat_img = work_img.reshape(-1, 3)
    flat_labels = region_labels.reshape(-1)
    # flat_mask = mask.reshape(-1)
    flat_mask = mask.ravel()

    sums = np.zeros((num_regions, 3), dtype=np.float32)
    np.add.at(sums, flat_labels, flat_img * flat_mask[:, None])

    counts = np.bincount(flat_labels, weights=flat_mask, minlength=num_regions).astype(np.float32)
    counts[counts == 0] = 1  # avoid divide-by-zero

    region_means = sums / counts[:, None]

    # Avoid divide-by-zero when computing R/G and B/G by clamping G
    G_safe = np.where(np.abs(region_means[:, 1]) < cfg.eps, cfg.eps, region_means[:, 1])
    region_illuminants = np.column_stack((region_means[:, 0] / G_safe, region_means[:, 2] / G_safe))
    T.toc()  # Region illuminants

    # ------------------------------------------------------------
    # Filter by CCT tolerance and cluster region illuminants
    # ------------------------------------------------------------
    T.tic("CCT boundary check")
    # Exclude regions with no valid pixels (counts==0 before clamping) and non-finite values
    counts_raw = np.bincount(flat_labels, weights=flat_mask, minlength=num_regions).astype(np.float32)
    valid = (counts_raw > 0) & np.isfinite(region_illuminants).all(axis=1)

    if np.sum(valid) < 1:
        raise ValueError("No valid region illuminants")

    # Filter by CCT tolerance
    if cfg.cct_curve_rb is None:
        raise ValueError("cct_curve_rb must be initialized in AWBConfig")
    
    distances = np.linalg.norm(
        cfg.cct_curve_rb[:, None, :] - region_illuminants[None, :, :],
        axis=2
    )
    distances_clean = np.nan_to_num(distances, nan=np.inf)
    min_dist = distances_clean.min(axis=0)
    mask_inside = min_dist < cfg.cct_tolerance

    # Cluster only points inside CCT tolerance
    inside_points = region_illuminants[mask_inside & valid]
    
    if len(inside_points) < 1:
        # Fallback: use all valid points
        inside_points = region_illuminants[valid]
        mask_inside = valid
    T.toc()  # CCT boundary check

    # ------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------
    T.tic("Clustering")

    if inside_points.shape[0] < 2 or np.allclose(inside_points, inside_points[0]):
        cl_centers = np.array([inside_points[0]])
        cl_labels_inside = np.zeros(inside_points.shape[0], dtype=int)
    else:
        cl_labels_inside, cl_centers = cluster.cluster(
            inside_points,
            method=cfg.cluster_method,
            **cfg.cluster_kwargs
        )

    # Map cluster labels back to all regions
    cl_labels = np.zeros(len(region_illuminants), dtype=int)
    inside_indices = np.where(mask_inside & valid)[0]
    cl_labels[inside_indices] = cl_labels_inside
    T.toc()  # Clustering

    # ------------------------------------------------------------
    # Per-pixel gain map
    # ------------------------------------------------------------
    T.tic("Gain map computation")
    h, w, _ = work_img.shape
    
    # Pre-compute gains and use advanced indexing more efficiently:
    all_gains = np.ones((len(cl_centers), 3), dtype=np.float32)
    all_gains[:, 0] = 1 / np.clip(cl_centers[:, 0], cfg.eps, None) # R gains
    all_gains[:, 2] = 1 / np.clip(cl_centers[:, 1], cfg.eps, None) # B gains

    # Single fancy indexing operation:
    gain_map = all_gains[cl_labels[region_labels]]
    # gain_map shape: h, w, 3 (work image size)

    # Set average cluster gains in regions with no valid pixels
    invalid_regions = ~valid

    if np.any(invalid_regions):
        mean_illuminant = cl_centers.mean(axis=0)
        mean_gain_R = 1 / np.clip(mean_illuminant[0], cfg.eps, None)
        mean_gain_B = 1 / np.clip(mean_illuminant[1], cfg.eps, None)
        
        # Vectorized instead of loop:
        invalid_mask = np.isin(region_labels, np.where(invalid_regions)[0])
        gain_map[invalid_mask] = (mean_gain_R, 1.0, mean_gain_B) # Set G gain to 1.0 implicitly?
    T.toc()  # Gain map computation

    # ------------------------------------------------------------
    # SMOOTH GAINS (two-stage pipeline)
    # ------------------------------------------------------------
    T.tic("Gain map smoothing")

    if cfg.smooth1_method != 'none':
        smoothed_gain_map = smooth.smooth_gain_map(gain_map, img=work_img, method=cfg.smooth1_method, **cfg.smooth1_kwargs)
    else:
        smoothed_gain_map = gain_map
    T.toc()  # Gain map smoothing

    # ------------------------------------------------------------
    # UPSCALE GAINS TO FULL RES (if downsampled)
    # ------------------------------------------------------------
    T.tic("Gain map upscaling")
    if cfg.downsample and scale != 1.0:
        smoothed_gain_map = cv2.resize(smoothed_gain_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        if cfg.fast:
            gain_map_out = None
            mask_out = None
        else:
            gain_map_out = cv2.resize(gain_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            mask_out = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        if cfg.fast:
            gain_map_out = None
            mask_out = None
        else:
            mask_out = mask
    T.toc()  # Gain map upscaling

    # ------------------------------------------------------------
    # APPLY GAINS
    # ------------------------------------------------------------
    T.tic("Applying gains")

    img_awb = np.empty_like(img)
    np.multiply(img, smoothed_gain_map, out=img_awb)
    np.minimum(img_awb, 1.0, out=img_awb)

    T.toc()  # Applying gains

    #T.report()

    return {
        'image': img_awb,
        'mask': mask_out,
        'mask_inside': mask_inside,
        'gain_map': gain_map_out,
        'smoothed_gain_map': smoothed_gain_map,
        'region_illuminants': region_illuminants,
        'cluster_labels': cl_labels,
        'cluster_centers': cl_centers,
        'timings': T.get_times()
    }