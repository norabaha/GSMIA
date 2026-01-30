import numpy as np
import matplotlib.pyplot as plt
from src.utils import isp

NUM_METHODS = 4
NUM_ILLUMINANTS = 5

NUM_ALL_ILLUMINANTS = 6
ALL_ILLUMINANTS = ['A', 'CWF', 'D65', 'HZ', 'TL84', 'S']

ILLUMINANTS = ['A', 'CWF', 'D65', 'HZ', 'TL84']

SINGLE_FILES = [f'capture/lightbox/img_{ill}.npy' for ill in ILLUMINANTS]
MULTI_FILES = [f'capture/lightbox/img_multi_{ill}.npy' for ill in ILLUMINANTS]

MODES = ['single', 'multi_big', 'multi_small']
REGIONS = ['sp', 't']

alg_names = {0: 'GW',
             1: 'LGW',
             2: 'CLGW',
             3: 'CCT-AWB'}

reg_names = {'superpixels': 'S',
             'tiles': 'T'}

def get_image(ill_idx: int, mode: str='multi'):
    if mode == 'single':
        img_file = SINGLE_FILES[ill_idx]
    elif mode == 'multi':
        img_file = MULTI_FILES[ill_idx]
    else:
        raise ValueError(f'Unknown mode: {mode}')
    img = np.load(img_file)
    img = isp.black_level_correction(img)
    img = isp.demosaic(img)
    return img

illuminants = ['A', 'CWF', 'D65', 'HZ', 'TL84']
singlefiles = [f'capture/lightbox/img_{mode}.npy' for mode in illuminants]
multifiles = [f'capture/lightbox/img_multi_{mode}.npy' for mode in illuminants]

small_light_ill = np.array([0.9, 0.25])

bb_single_wp = [
    [1070, 442, 1110, 480],
    [1070, 442, 1110, 480],
    [1070, 442, 1110, 480],
    [1070, 442, 1110, 480],
    [1070, 442, 1110, 480]
]
bb_multi_wp_big = [
    [1070, 460, 1100, 496],
    [1020, 460, 1050, 494],
    [1150, 470, 1185, 505],
    [1070, 460, 1110, 500],
    [980, 465, 1020, 500]
]
bb_multi_wp_small = [
    [749, 780, 759, 790],
    [688, 775, 700, 787],
    [837, 780, 847, 791],
    [748, 774, 757, 788],
    [655, 776, 665, 788]
]

ccm = np.array([[1.5, -0.5, 0.0],
                [-0.1, 1.2, -0.1],
                [0.0, -0.5, 1.5]])

def get_updated_cct(include_small_light=True):
    '''Returns array of shape (illuminants, (R/G, B/G))'''
    base_array = [
        [0.7, 0.32],
        [0.4921, 0.3672],  # CWF
        [0.4160, 0.6208], # D65
        [0.9829, 0.2585], # HZ
        [0.5422, 0.4049], # TL84
    ]
    if include_small_light:
        base_array.append(small_light_ill.tolist())
    return np.array(base_array)

def get_gt_gain_map(ill):
    bb_small = np.array([
        [300, 350, 900, 1150],
        [250, 350, 850, 1150],
        [450, 350, 1000, 1150],
        [300, 350, 900, 1150],
        [200, 350, 800, 1150]
    ])
    cct = get_updated_cct()
    g_big = 1 / cct[ill]
    g_small = 1 / small_light_ill
    gt_gain_map = np.ones((1296, 2304, 3), dtype=np.float32)
    gt_gain_map[:] = [g_big[0], 1.0, g_big[1]]
    x0, y0, x1, y1 = bb_small[ill]
    gt_gain_map[y0:y1, x0:x1] = [g_small[0], 1.0, g_small[1]]
    gt_gain_map[int(y0+(y1-y0)*0.6):y1, x0:int(x0+(x1-x0)*0.5)] = [g_big[0], 1.0, g_big[1]]
    return gt_gain_map



def rgb_to_chroma(rgb):
    r = rgb[:,0] / rgb[:,1]
    b = rgb[:,2] / rgb[:,1]
    return r, b

def get_wp_single(img, ill):
    x0, y0, x1, y1 = bb_single_wp[ill]
    patch = img[y0:y1, x0:x1]
    wp = np.mean(patch, axis=(0,1))
    return wp

def get_wp_multi_big(img, ill):
    x0, y0, x1, y1 = bb_multi_wp_big[ill]
    patch = img[y0:y1, x0:x1]
    return np.mean(patch, axis=(0,1))

def get_wp_multi_small(img, ill):
    x0, y0, x1, y1 = bb_multi_wp_small[ill]
    patch = img[y0:y1, x0:x1]
    return np.mean(patch, axis=(0,1))


def angular_error(v, gt=np.array([1.0, 1.0])):
    """
    Compute angular error between estimated white points and ground truth.

    Parameters:
    - v: np.ndarray of shape (N, 2) with values (R/G, B/G)
    - gt: ground truth white point (default [1, 1])

    Returns:
    - errors: np.ndarray of angular errors in degrees
    """
    dot = np.sum(v * gt, axis=1)
    cross = v[:, 0] * gt[1] - v[:, 1] * gt[0]
    return np.degrees(np.arctan2(np.abs(cross), dot))


def get_wp_error(img, ill, mode='single'):
    """
    Get white patch and compute angular error.

    Parameters:
    - img: input image (H, W, 3)
    - ill: index of illuminant
    - mode: 'single', 'multi_big', 'multi_small'

    Returns:
    - error: angular error in degrees
    """
    if mode == 'single':
        wp = get_wp_single(img, ill)[None, :]  # shape (1,3)
    elif mode == 'multi_big':
        wp = get_wp_multi_big(img, ill)
        wp = wp[None, :]
    elif mode == 'multi_small':
        wp = get_wp_multi_small(img, ill)
        wp = wp[None, :]
    else:
        raise ValueError("mode must be 'single', 'multi_big' or 'multi_small'")

    eps = 1e-8
    r = wp[:, 0] / (wp[:, 1] + eps)
    b = wp[:, 2] / (wp[:, 1] + eps)

    v = np.stack([r, b], axis=1)   # shape (1, 2)
    return wp, angular_error(v)

def plot_result(result, gt_ill_idx=None):
    if gt_ill_idx is not None:
        ill = get_updated_cct()[gt_ill_idx]
    else:
        ill = 'unknown'

    img = result['image']
    valid_mask = result['mask']
    gain_map = result['gain_map']
    smoothed_gain_map = result['smoothed_gain_map']
    region_illuminants = result['region_illuminants'] # shape (n regions, 2). (R/G, B/G) 

    plt.figure(figsize=(18,10))
    plt.suptitle(f'Results of image with illuminant {illuminants[gt_ill_idx]}', fontsize=16)
    
    # img, mask, reg ill
    # gainmap , sm gain map
    disp = isp.color_correction(img, ccm)
    disp = isp.gamma_correction(img)
    
    plt.subplot(2, 3, 1)
    plt.title('Image after processing (CC+GC for disp.)')
    plt.imshow(disp)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Region illuminant estimates, before cluster/smooth')
    plt.scatter(region_illuminants[:,0], region_illuminants[:,1], c='black', label="Reg ill ests")

    # Plot cluster centers if present (methods 2 or 3)
    cluster_centers = result.get('cluster_centers')
    if cluster_centers is not None:
        plt.scatter(cluster_centers[:,0], cluster_centers[:,1], c='orange', marker='x', label='Cluster centers')

    cct = get_updated_cct()
    plt.scatter(cct[:,0], cct[:,1], label=f'CCT')
    if gt_ill_idx is not None:
        plt.scatter(ill[0], ill[1], c='red', label=f'GT ill: {illuminants[gt_ill_idx]}')
        plt.scatter(small_light_ill[0], small_light_ill[1], c='blue', label=f'GT small ill')


    plt.grid('on', which='both')
    plt.legend()
    plt.xlim(0,1.2)
    plt.ylim(0,1.1)

    plt.subplot(2, 3, 3)
    plt.title('Mask of tiles used for AWB. Invalid in black.')
    plt.imshow(valid_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Gain map (normalized)')
    plt.imshow(gain_map / gain_map.max())
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Smoothed gain map (normalized)')
    plt.imshow(smoothed_gain_map / smoothed_gain_map.max())
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title('GT gain map')
    gt_gain_map = get_gt_gain_map(gt_ill_idx)
    plt.imshow(gt_gain_map / gt_gain_map.max())
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def neutral_line_error(rgb):
    """ for a white patch: err = neutral_line_error(wp_rgb)
    rgb: (..., 3) linear RGB
    returns: normalized distance to neutral line
    """
    eps=1e-8
    mean = np.mean(rgb, axis=-1, keepdims=True)
    diff = rgb - mean
    dist = np.linalg.norm(diff, axis=-1)
    return dist / (mean.squeeze(-1) + eps)


def angular_error_rg_bg(est_rg_bg, gt_rg_bg):
    """
    est_rg_bg, gt_rg_bg: arrays of shape (..., 2) with [r/g, b/g]
    returns angular error in degrees
    """
    eps=0#1e-12

    est = np.concatenate([est_rg_bg[:, :1], np.ones((est_rg_bg.shape[0], 1)), est_rg_bg[:, 1:2]], axis=1)
    gt  = np.concatenate([gt_rg_bg[:,  :1], np.ones((gt_rg_bg.shape[0],  1)), gt_rg_bg[:,  1:2]], axis=1)

    est_n = est / (np.linalg.norm(est, axis=1, keepdims=True) + eps)
    gt_n  = gt  / (np.linalg.norm(gt,  axis=1, keepdims=True) + eps)

    cosang = np.clip(np.sum(est_n * gt_n, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def angular_distance_3d(point1, point2):
    """
    Calculates the angular distance (angle) in 3D between two points.
    angle = arccos(dot(u,v)/ (norm(u)*norm(v))

    Args:
        point1 (list or np.array): The first point [x1, y1, z1].
        point2 (list or np.array): The second point [x2, y2, z2].

    Returns:
        float: The angle between the two points in radians.
    """
    v1 = np.array(point1)
    v2 = np.array(point2)
    dot_product = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)




def create_synthetic_image_and_gt_gain_map(big_ill_idx: int, small_ill_idx: int):
    """Create a synthetic image with a neutral gray patch for testing in each illuminant.
    The rest should be a noneutral color.
    Add a circle area with one illuminant and the rest with another illuminant.
    """
    h, w = 1200, 2000
    
    # CREATE GROUND TRUTH IMAGE
    # Create base image with neutral gray
    img_gt = np.ones((h, w, 3), dtype=np.float32) * 0.5

    # Create blue sky
    img_gt[:h//2, :, :] = [0.5, 0.7, 1.0]  # Light blue sky
    img_gt = np.clip(img_gt, 0.0, 1.0)

    # Create green ground
    img_gt[h//2:h, :, :] = [0.3, 0.6, 0.2]  # Darker green ground

    # Create mountain background for tunnel entrance. Brownish gray
    img_gt[h//5:h*3//4, :w*3//4, :] = [0.4, 0.4, 0.2]  # Brownish gray mountains

    # Create tunnel entrance effect (a circle in the middle) with gradual darkening
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 4
    y_coords, x_coords = np.ogrid[:h, :w]
    circle_mask = (y_coords - center_y)**2 + (x_coords - center_x)**2 <= radius**2
    img_gt[circle_mask] = 0.2  # Darker area inside the circle

    # Create road. A triangular shape at the bottom center
    for y in range(h//2, h):
        road_width = (y - h//2) * (w // 2) // (h // 2)
        x_start = (w // 2) - road_width // 2
        x_end = (w // 2) + road_width // 2
        img_gt[y, x_start:x_end, :] = [0.4, 0.4, 0.4]  # Gray road

    # Create red car on the road
    car_w =300
    car_h =car_w // 2
    car_x_start = (w // 2) + car_w // 3
    car_y_start = h - car_h - 50
    img_gt[car_y_start:car_y_start+car_h, car_x_start:car_x_start+car_w, :] = [1.0, 0.0, 0.0]  # Red car

    # Create yellow car on the road
    car2_w =200
    car2_h =car2_w // 2
    car2_x_start = (w // 2) - car2_w
    car2_y_start = h//2 + 250
    img_gt[car2_y_start:car2_y_start+car2_h, car2_x_start:car2_x_start+car2_w, :] = [1.0, 1.0, 0.0]  # Yellow car

    # Create white car on road inside tunnel
    car3_w =100
    car3_h =60
    car3_x_start = (w // 2)-30
    car3_y_start = h//2 + 50
    img_gt[car3_y_start:car3_y_start+car3_h, car3_x_start:car3_x_start+car3_w, :] = [1.0, 1.0, 1.0]

    for y in range(h):
        for x in range(w):
            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            if dist < radius:
                brightness_increase = 0.8 * (1 - dist / radius)
                img_gt[y, x, :] += brightness_increase
    img_gt = np.clip(img_gt, 0.0, 1.0)

    img = img_gt.copy()

    gt_gain = np.ones((h, w, 3), dtype=np.float32)
    big_illum = get_updated_cct()[big_ill_idx]
    gt_gain[:, :, 0] = 1.0 / big_illum[0]
    gt_gain[:, :, 2] = 1.0 / big_illum[1]
    
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 4
    y_coords, x_coords = np.ogrid[:h, :w]
    circle_mask = (y_coords - center_y)**2 + (x_coords - center_x)**2 <= radius**2
    
    small_illum = get_updated_cct()[small_ill_idx]
    gt_gain[circle_mask, 0] = 1.0 / small_illum[0]
    gt_gain[circle_mask, 2] = 1.0 / small_illum[1]
    
    img_cast = img / (gt_gain + 1e-6)
    img_cast = np.clip(img_cast, 0.0, 1.0)
    
    return img_gt, img_cast, gt_gain, big_illum, small_illum


import src.methods as methods
from src.utils import isp
import src.utils.config as config
import numpy as np
import matplotlib.pyplot as plt

def plot_results_minimal( cfg: config.AWBConfig, mode: str='multi', show_plots: bool = True, save_file: str = None):
    wps_big = np.zeros((NUM_ILLUMINANTS, 3))
    wps_small = np.zeros((NUM_ILLUMINANTS, 3))

    fig, axs = plt.subplots(NUM_ILLUMINANTS, 4, figsize=(20, 3 * NUM_ILLUMINANTS))

    for ill_idx, ill in enumerate(ILLUMINANTS):
        img = get_image(ill_idx, mode)

        result = methods.run_awb(img, cfg, cfg.awb_method)
        img_awb = result['image']
        img_corr = isp.gamma_correction(img_awb)
            
        # Plot image and gain map
        axs[ill_idx, 0].imshow(img_corr)
        axs[ill_idx, 0].set_title(f'Result of Algorithm {cfg.awb_method} {cfg.region_method[0]} for Illuminant {ILLUMINANTS[ill_idx]}')
        axs[ill_idx, 0].axis('off')

        gain_map = result['smoothed_gain_map']
        gain_map_vis = gain_map / np.max(gain_map)
        axs[ill_idx, 1].imshow(gain_map_vis)
        axs[ill_idx, 1].set_title('Smoothed Gain Map')
        axs[ill_idx, 1].axis('off')
        

        
        # Plot cluster centers and region illuminants
        axs[ill_idx, 2].set_title('Estimated illuminants')
        cct = get_updated_cct()
        axs[ill_idx, 2].scatter(cct[:, 0], cct[:, 1], c='gray', marker='o', label='CCT Illuminants', s=30, alpha=0.5)
        
        # Plot region illuminants (input to clustering) colored by cluster assignment
        region_illuminants = result.get('region_illuminants')
        cluster_labels = result.get('cluster_labels')
        mask_inside = result.get('mask_inside')  # Get which regions were inside CCT
        
        if region_illuminants is not None and len(region_illuminants) > 0:
            if cluster_labels is not None and len(cluster_labels) == len(region_illuminants):
                # Filter to only show regions that were inside CCT curve (actually clustered)
                if mask_inside is not None:
                    # Plot regions inside CCT (clustered)
                    valid_regions_inside = mask_inside & ~np.isnan(region_illuminants).any(axis=1)
                    
                    if np.any(valid_regions_inside):
                        region_illums_inside = region_illuminants[valid_regions_inside]
                        cluster_labels_inside = cluster_labels[valid_regions_inside]
                        
                        # Color by cluster assignment
                        unique_labels = np.unique(cluster_labels_inside)
                        colors = ['red', 'green']
                        for idx, label in enumerate(unique_labels):
                            mask = cluster_labels_inside == label
                            axs[ill_idx, 2].scatter(region_illums_inside[mask, 0], region_illums_inside[mask, 1], 
                                                  c=[colors[idx]], label=f'Cluster {label} regions ({np.sum(mask)})', 
                                                  s=50, alpha=0.6, marker='.')
                    
                    # Plot regions outside CCT tolerance (not clustered)
                    valid_regions_outside = ~mask_inside & ~np.isnan(region_illuminants).any(axis=1)
                    
                    if np.any(valid_regions_outside):
                        region_illums_outside = region_illuminants[valid_regions_outside]
                        axs[ill_idx, 2].scatter(region_illums_outside[:, 0], region_illums_outside[:, 1], 
                                              c='gray', label=f'Outside CCT ({np.sum(valid_regions_outside)})', 
                                              s=30, alpha=0.4, marker='x')
                else:
                    # No mask_inside available, plot all
                    unique_labels = np.unique(cluster_labels)
                    colors = ['red', 'green']
                    for idx, label in enumerate(unique_labels):
                        mask = cluster_labels == label
                        axs[ill_idx, 2].scatter(region_illuminants[mask, 0], region_illuminants[mask, 1], 
                                              c=[colors[idx]], label=f'Cluster {label} regions', 
                                              s=50, alpha=0.6, marker='.')
            else:
                # Fallback if no labels available
                axs[ill_idx, 2].scatter(region_illuminants[:, 0], region_illuminants[:, 1], c='lightblue', 
                                      label='Region illuminants', s=50, alpha=0.6, marker='.')
        
        # Plot cluster centers
        cluster_centers = result.get('cluster_centers')
        if cluster_centers is not None and len(cluster_centers) > 0:
            axs[ill_idx, 2].scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='blue', label='Cluster centers', s=10, marker='o', linewidths=1.5)
        
        gt_ill_1_rb = cct[ill_idx]
        gt_ill_2_rb = small_light_ill
        axs[ill_idx, 2].scatter(gt_ill_1_rb[0], gt_ill_1_rb[1], c='black', marker='x', s=50, label=f'GT {ILLUMINANTS[ill_idx]}', linewidths=1.5)
        if mode == 'multi':
            axs[ill_idx, 2].scatter(gt_ill_2_rb[0], gt_ill_2_rb[1], c='black', marker='x', s=50, label='GT secondary', linewidths=1.5)
        axs[ill_idx, 2].set_xlim(0, 1.8)
        axs[ill_idx, 2].set_ylim(0, 1.0)
        axs[ill_idx, 2].legend(fontsize=7, loc='upper right')
        axs[ill_idx, 2].grid(alpha=0.3)


        # Compute white points on the AWB corrected image (not gamma corrected!)
        if mode == 'multi':
            wps_small[ill_idx] = get_wp_multi_small(img_awb, ill_idx)
            wps_big[ill_idx] = get_wp_multi_big(img_awb, ill_idx)
        elif mode == 'single':
            wps_big[ill_idx] = get_wp_single(img_awb, ill_idx)
        
        # Plot white point estimation in (R/G, B/G) space
        axs[ill_idx, 3].set_title('White Point Estimation')
        if mode == 'multi':
            axs[ill_idx, 3].scatter(wps_small[ill_idx,0]/wps_small[ill_idx,1], wps_small[ill_idx,2]/wps_small[ill_idx,1], c='blue', label='WP secondary', s=100)
            axs[ill_idx, 3].scatter(wps_big[ill_idx,0]/wps_big[ill_idx,1], wps_big[ill_idx,2]/wps_big[ill_idx,1], c='black', label='WP primary', s=100)
        elif mode == 'single':
            axs[ill_idx, 3].scatter(wps_big[ill_idx,0]/wps_big[ill_idx,1], wps_big[ill_idx,2]/wps_big[ill_idx,1], c='black', label='WP primary', s=100)
        axs[ill_idx, 3].set_xlim(0.4, 1.8)
        axs[ill_idx, 3].set_ylim(0.4, 1.8)

        # Pure white point for reference
        pure_wp = np.array([1.0, 1.0, 1.0])
        axs[ill_idx, 3].scatter(pure_wp[0]/pure_wp[1], pure_wp[2]/pure_wp[1], c='red', label='Pure neutral', s=100, marker='x')
        axs[ill_idx, 3].legend()
        axs[ill_idx, 3].grid()

    plt.tight_layout()
    if show_plots:
        plt.show()
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return wps_big, wps_small
