import numpy as np
import cv2
from src.utils import superpixels, smooth, timer
from src.utils.config import AWBConfig

def run_awb_method_1(img, cfg: AWBConfig):
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

    # compute gains directly: G/R and G/B
    R = np.maximum(region_means[:, 0], cfg.eps)
    B = np.maximum(region_means[:, 2], cfg.eps)
    gR = region_means[:, 1] / R
    gB = region_means[:, 1] / B
    region_gains3 = np.stack([gR, np.ones_like(gR), gB], axis=1).astype(np.float32)

    if cfg.fast:
        region_illuminants = None
    else:
        # Avoid divide-by-zero when computing R/G and B/G by clamping G
        G_safe = np.where(np.abs(region_means[:, 1]) < cfg.eps, cfg.eps, region_means[:, 1])
        region_illuminants = np.column_stack((region_means[:, 0] / G_safe, region_means[:, 2] / G_safe))

    T.toc()  # Region illuminants

    # ------------------------------------------------------------
    # Per-pixel gain map
    # ------------------------------------------------------------
    T.tic("Gain map computation")
    h, w, _ = work_img.shape
    
    # per-pixel gain map
    gain_map = region_gains3[flat_labels].reshape(h, w, 3)

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
        'gain_map': gain_map_out,
        'smoothed_gain_map': smoothed_gain_map,
        'region_illuminants': region_illuminants,
        'cluster_labels': None,
        'cluster_centers': None,
        'timings': T.get_times()
    }
