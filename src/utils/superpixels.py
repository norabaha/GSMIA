import numpy as np
import cv2

def compute_superpixels_cv2(img, n_segments=300, compactness=20, num_iterations=4, enforce_connectivity=False):
    """
    img: float or uint8, RGB order, shape (H,W,3)
    returns: superpixel labels, shape (H,W)
    """
    h, w = img.shape[:2]

    # OpenCV requires 8-bit BGR
    img_bgr = (img * 255).astype(np.uint8)[..., ::-1]

    slic = cv2.ximgproc.createSuperpixelSLIC(
        img_bgr,
        algorithm=cv2.ximgproc.SLIC,
        region_size=int(np.sqrt(h*w / n_segments)),
        ruler=compactness,
    )

    slic.iterate(num_iterations=num_iterations)
    if enforce_connectivity:
        slic.enforceLabelConnectivity(min_element_size=10)

    labels = slic.getLabels()
    return labels, slic.getNumberOfSuperpixels()

def compute_superpixels_cv2_slico(img, n_segments=300, compactness=20):
    """
    img: float or uint8, RGB order, shape (H,W,3)
    returns: superpixel labels, shape (H,W) and number of superpixels
    """
    h, w = img.shape[:2]

    # OpenCV requires 8-bit BGR
    img_bgr = (img * 255).astype(np.uint8)[..., ::-1]

    slico = cv2.ximgproc.createSuperpixelSLIC(
        img_bgr,
        algorithm=cv2.ximgproc.SLICO,
        region_size=int(np.sqrt(h*w / n_segments)),
        ruler=compactness,
    )

    slico.iterate(num_iterations=10)
    slico.enforceLabelConnectivity(min_element_size=10)

    return slico.getLabels(), slico.getNumberOfSuperpixels()

def compute_superpixels_cv2_seeds(img, num_superpixels=400, num_levels=4, prior=2, num_iterations=2):
    """
    Ultra-fast superpixel segmentation using OpenCV SEEDS.
    img: float [0-1] RGB image, HxWx3
    Returns: labels (HxW)
    """

    h, w = img.shape[:2]

    # SEEDS needs 8-bit BGR
    img_bgr = (img * 255).astype(np.uint8)[..., ::-1]

    seeds = cv2.ximgproc.createSuperpixelSEEDS(
        w,
        h,
        3,
        num_superpixels,
        num_levels,
        prior
    )

    # run a few iterations (SEEDS converges very quickly)
    for _ in range(num_iterations):
        seeds.iterate(img_bgr)

    labels = seeds.getLabels()
    return labels, seeds.getNumberOfSuperpixels()

