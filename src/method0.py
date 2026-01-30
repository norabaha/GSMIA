import numpy as np
import cv2
from src.utils.config import AWBConfig
from src.utils import timer

def run_awb_method_0(img, cfg: AWBConfig):
    """Gray-world AWB with green gain fixed to 1.
    Args:
        img: HxWx3 float32 image in [0,1]
        config: AWBConfig
    """
    T = timer.Timer()
    T.tic("Preprocessing")

    orig_h, orig_w = img.shape[:2]

    # Optional pre-downsample for faster processing
    if cfg.downsample:
        scale = cfg.downsample_scale
        work_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if cfg.debug:
            print(f"[Method0] Downsampled to {work_img.shape[0]}x{work_img.shape[1]} (scale={scale})")
    else:
        scale = 1.0
        work_img = img
        if cfg.debug:
            print(f"[Method0] No downsampling")

    T.toc()  # Preprocessing
    T.tic("Compute means")
    # Valid-pixel mask
    mask = ~(np.any(work_img >= cfg.mask_saturation_threshold, axis=2) | 
         np.all(work_img <= cfg.mask_black_threshold, axis=2))

    # Ensure there are valid pixels
    if not np.any(mask):
        raise ValueError("No valid pixels found: all pixels are either too bright or too dark.")

    # Compute means for valid pixels
    if np.any(mask):
        means = work_img[mask].mean(axis=0)
    else:
        means = work_img.mean(axis=0)

    # Avoid zeros
    means = np.clip(means, 1e-6, None)
    T.toc()  # Compute means
    T.tic("Compute gain")
    # Fix green channel to 1
    # Scale R and B to match green mean
    gain = np.ones(3, dtype=np.float32)
    gain[0] = means[1] / means[0]  # R gain
    gain[2] = means[1] / means[2]  # B gain
    gain[1] = 1.0                  # G gain stays 1
    
    T.toc()  # Compute gain

    T.tic("Apply gain")
    # Apply
    # img_awb = img * gain
    # img_awb = np.clip(img_awb, 0, 1)
    img_awb = np.empty_like(img)
    np.multiply(img, gain, out=img_awb)
    np.minimum(img_awb, 1.0, out=img_awb)

    T.toc()  # Apply gain

    if cfg.fast:
        gain_map = None
    else:
        gain_map = np.ones(work_img.shape, dtype=np.float32) * gain
        
        # Upscale mask and gain_map back to original if downsampled
        if cfg.downsample and scale != 1.0:
            mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            gain_map = cv2.resize(gain_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return {
        'image': img_awb,
        'gain_map': gain_map,
        'smoothed_gain_map': gain_map,
        'region_illuminants':np.array([[1/gain[0], 1/gain[2]]]),
        'mask': mask,
        'cluster_centers': np.array([[1/gain[0], 1/gain[2]]]),
        'timings': T.get_times(),
    }
