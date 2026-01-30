import numpy as np
import cv2

def _fix_centers(centers):
    centers = np.asarray(centers, dtype=np.float32)

    # Fix cases like (2,1)
    if centers.ndim == 2 and centers.shape[1] == 1:
        centers = np.hstack([centers, centers])

    # Convert shape (1,2) to (2,2) by duplication
    if centers.shape == (1, 2):
        centers = np.vstack([centers, centers])

    # Convert shape (2,) to (2,2)
    if centers.ndim == 1 and centers.shape[0] == 2:
        centers = np.vstack([centers, centers])

    # Convert completely malformed shapes
    if centers.shape != (2, 2):
        raise ValueError(f"Cluster centers must be (2,2), got {centers.shape}")

    return centers



def cluster_kmeans(points, **kwargs):
    _, labels, centers = cv2.kmeans(
        data=points.astype(np.float32),
        K=2,
        bestLabels=None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4),
        attempts=10,
        flags=cv2.KMEANS_PP_CENTERS
    )
    return labels.flatten(), centers


def cluster_median(points, **kwargs):
    median_val = np.median(points[:, 0])
    labels = (points[:, 0] > median_val).astype(int)

    c0 = points[labels == 0]
    c1 = points[labels == 1]

    if len(c0) == 0:
        c0_mean = c1.mean(axis=0)
        c1_mean = c1.mean(axis=0)
    elif len(c1) == 0:
        c0_mean = c0.mean(axis=0)
        c1_mean = c0.mean(axis=0)
    else:
        c0_mean = c0.mean(axis=0)
        c1_mean = c1.mean(axis=0)

    centers = np.vstack([c0_mean, c1_mean])
    return labels, _fix_centers(centers)


def cluster_threshold(points, **kwargs):
    axis = np.ptp(points, axis=0)
    if np.all(axis == 0):
        # all points identical — both centers = same point
        return np.zeros(len(points), dtype=int), np.vstack([points[0], points[0]])

    projection = points @ axis
    threshold = 0.5 * (projection.min() + projection.max())

    labels = (projection > threshold).astype(int)

    c0 = points[labels == 0]
    c1 = points[labels == 1]

    if len(c0) == 0:
        c0_mean = c1.mean(axis=0)
        c1_mean = c1.mean(axis=0)
    elif len(c1) == 0:
        c0_mean = c0.mean(axis=0)
        c1_mean = c0.mean(axis=0)
    else:
        c0_mean = c0.mean(axis=0)
        c1_mean = c1.mean(axis=0)

    centers = np.vstack([c0_mean, c1_mean])
    return labels, _fix_centers(centers)


def cluster(points, method, **kwargs):
    """
    points: (num_regions, 2) array. E.g. of (R/G, B/G) gains
    method: 'kmeans', 'median', 'threshold'
    kwargs: dict of properties for cluster methods. only for Kmeans currently.
    Returns: cluster labels for each region, cluster centers
    """
    # Too few regions. single illuminant fallback
    if points.shape[0] < 2:
        labels = np.zeros(points.shape[0], dtype=int)
        centers = np.vstack([points[0], points[0]])
        return labels, centers
    # All points identical. single illuminant
    if np.allclose(points, points[0]):
        labels = np.zeros(points.shape[0], dtype=int)
        centers = np.vstack([points[0], points[0]])
        return labels, centers
    
    if method == 'kmeans':
        cluster_func = cluster_kmeans
    elif method == 'median':
        cluster_func = cluster_median
    elif method == 'threshold':
        cluster_func = cluster_threshold
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    return cluster_func(points, **kwargs)