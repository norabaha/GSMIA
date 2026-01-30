import cv2
import numpy as np

# ISP STEPS
isp_steps = [
    'black_level_correction',
    'demosaic',
    'lens_shading_correction',
    'color_correction',
    'gamma_correction'
]

def black_level_correction(bayer_img, black_level=4096):
    """
    Subtract black level from a Bayer image.
    Args:
        bayer_img: HxW single-channel uint16 image in [0,65535]
        black_level: int, black level to subtract
    Returns:
        HxW single-channel uint16 image with black level subtracted
    """
    bayer_img = bayer_img.astype(np.int32)  # to avoid underflow
    bayer_img -= black_level
    bayer_img = np.clip(bayer_img, 0, 65535).astype(np.uint16)
    return bayer_img

def demosaic(bayer_img):
    """
    Demosaic a Bayer RGGB image to RGB using OpenCV.
    Args:
        bayer_img: HxW single-channel uint16 image in [0,65535]
    Returns:
        HxWx3 float32 RGB image in [0,1]
    """
    # OpenCV expects 8-bit or 16-bit images for demosaicing
    rgb_16bit = cv2.cvtColor(bayer_img, cv2.COLOR_BAYER_RG2RGB)
    rgb_img = (rgb_16bit.astype(np.float32)) / 65535.0
    rgb_img = np.clip(rgb_img, 0, 1)
    return rgb_img

def lens_shading_correction_map(img, lsc_map):
    """
    Apply lens shading correction to an RGB image.
    Args:
        img: HxWx3 float32 RGB image in [0,1]
        lsc_map: HxWx3 float32 gain map in [0,inf)
    Returns:
        HxWx3 float32 RGB image after LSC in [0,1]
    """
    corrected = img * lsc_map
    corrected = np.clip(corrected, 0, 1)
    return corrected

def lens_shading_correction_radial(raw, edge_gain=2.5, power=3.5):
    """
    Uniform radial lens-shading correction applied to RAW Bayer data.
    raw:   HxW uint16 (or uint32) Bayer mosaic
    edge_gain: gain at the frame corners
    power: exponent controlling falloff shape
    """
    h, w = raw.shape
    cx, cy = w/2, h/2

    y, x = np.indices((h, w))
    r = ((x - cx)**2 + (y - cy)**2)**0.5
    r = r / r.max()

    gain = 1.0 + (edge_gain - 1.0) * (r ** power)

    corrected = raw.astype(np.float32) * gain
    return corrected.clip(0, 65535).astype(raw.dtype)


def color_correction(img, cc_matrix):
    """
    Apply color correction matrix to an RGB image.
    Args:
        img: HxWx3 float32 RGB image in [0,1]
        cc_matrix: 3x3 float32 color correction matrix
    Returns:
        HxWx3 float32 RGB image after color correction in [0,1]
    """
    h, w, _ = img.shape
    flat_img = img.reshape(-1, 3)
    corrected = flat_img @ cc_matrix.T
    corrected = corrected.reshape(h, w, 3)
    corrected = np.clip(corrected, 0, 1)
    return corrected

def gamma_correction(img, gamma=2.2):
    """
    Apply gamma correction to an RGB image.
    Args:
        img: HxWx3 float32 RGB image in [0,1]
        gamma: float, gamma value
    Returns:
        HxWx3 float32 RGB image after gamma correction in [0,1]
    """
    corrected = np.power(img, 1.0 / gamma)
    corrected = np.clip(corrected, 0, 1)
    return corrected

def normalize_8bit(img):
    """
    Normalize a float32 RGB image in [0,1] to uint8 [0,255].
    Args:
        img: HxWx3 float32 RGB image in [0,1]
    Returns:
        HxWx3 uint8 RGB image in [0,255]
    """
    img_8bit = (img * 255.0).round().astype(np.uint8)
    img_8bit = np.clip(img_8bit, 0, 255)
    return img_8bit

def downscale(img, scale=0.5):
    """
    Downscale an image by a given scale factor using area interpolation.
    Args:
        img: HxWxC float32 image
        scale: float, downscale factor
    Returns:
        downscaled image
    """
    h, w = img.shape[:2]
    new_h = int(h * scale)
    new_w = int(w * scale)
    downscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return downscaled

def upscale(img, scale=2.0):
    """
    Upscale an image by a given scale factor using linear interpolation.
    Args:
        img: HxWxC float32 image
        scale: float, upscale factor
    Returns:
        upscaled image
    """
    h, w = img.shape[:2]
    new_h = int(h * scale)
    new_w = int(w * scale)
    upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return upscaled