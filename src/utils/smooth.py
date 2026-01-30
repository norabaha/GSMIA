import numpy as np
import cv2
from scipy.ndimage import median_filter


def smooth_gain_map(gain_map, img=None, method='none', **kwargs):
    """
    Smooth a gain map using different methods. Works with float32 images in [0,1].
    
    Parameters:
        gain_map: np.ndarray, shape (H,W,C), float32
        img: np.ndarray, original image (required for bilateral/guided)
        method: 'none', 'gaussian', 'median', 'bilateral', 'guided'
        kwargs: method-specific parameters
    Returns:
        smoothed_gain_map: np.ndarray, same shape as gain_map
    """
    if method == 'none':
        return gain_map
    
    smoothed = np.zeros_like(gain_map)

    if method == 'gaussian':
        ksize = kwargs.get('gaussian_ksize', (7,7))
        sigmaX = kwargs.get('gaussian_sigmaX', 0)
        # OpenCV GaussianBlur expects ksize as odd integers
        ksize = tuple((k if k%2==1 else k+1) for k in ksize)
        smoothed = cv2.GaussianBlur(gain_map, ksize=ksize, sigmaX=sigmaX)
        return smoothed

    elif method == 'median':
        ksize = kwargs.get('median_ksize', 5)
        for ch in range(gain_map.shape[2]):
            smoothed[..., ch] = median_filter(gain_map[..., ch], size=ksize)
        return smoothed

    elif method == 'bilateral':
        if img is None:
            raise ValueError("Original image required for bilateral filter")
        d = kwargs.get('bilateral_d', 5)
        sigmaColor = kwargs.get('bilateral_sigmaColor', 0.2)
        sigmaSpace = kwargs.get('bilateral_sigmaSpace', 5)
        for ch in range(gain_map.shape[2]):
            temp = np.clip(gain_map[..., ch]*255, 0, 255).astype(np.uint8)
            temp_blur = cv2.bilateralFilter(temp, d=d, sigmaColor=sigmaColor*255, sigmaSpace=sigmaSpace)
            smoothed[..., ch] = temp_blur.astype(np.float32) / 255.0
        return smoothed

    elif method == 'guided':
        if img is None:
            raise ValueError("Original image required for guided filter")
        radius = kwargs.get('guided_radius', 8)
        eps = kwargs.get('guided_eps', 1e-2)
        guide = img.astype(np.float32) if img.dtype != np.float32 else img
        if guide.ndim == 3 and guide.shape[2] == 3:
            guide_gray = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
        else:
            guide_gray = guide
        for ch in range(gain_map.shape[2]):
            smoothed[..., ch] = cv2.ximgproc.guidedFilter(guide_gray, gain_map[..., ch], radius, eps)
        return smoothed

    else:
        raise ValueError(f"Unknown smoothing method '{method}'")
