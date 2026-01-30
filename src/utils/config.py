from dataclasses import dataclass, field, fields
import numpy as np


def _format_value(v, indent=2):
    pad = " " * indent

    if isinstance(v, dict):
        if not v:
            return "{}"
        lines = ["{"]
        for k, val in v.items():
            lines.append(f"{pad}{k}: {_format_value(val, indent + 2)}")
        lines.append(" " * (indent - 2) + "}")
        return "\n".join(lines)

    if isinstance(v, np.ndarray):
        return f"ndarray(shape={v.shape}, dtype={v.dtype})"

    return repr(v)

@dataclass
class AWBConfig:
    debug: bool = False
    fast: bool = False
    awb_method: int = 1  # 0-3
    num_tiles: int = 16
    num_superpixels: int = 256  # Default number of superpixels
    mask_saturation_threshold: float = 0.9
    mask_black_threshold: float = 0.05
    eps: float = 1e-6  # small value to avoid division by zero
    cct_curve_rb: np.ndarray = None  # Nx2 array of R/G, B/G points along CCT curve
    cct_tolerance: float = 0.05     # distance in RB space to CCT curve
    downsample: bool = False
    downsample_scale: float = 0.5
    region_method: str = 'tiles' # 'tiles' or 'superpixels'
    sp_compactness: int = 10
    cluster_method: str = 'kmeans'
    cluster_kwargs: dict = field(default_factory=dict)
    cluster_min_center_distance: float = 0.1
    cluster_min_ratio: float = 0.20
    smooth1_method: str = 'none'
    smooth1_kwargs: dict = field(default_factory=dict)
    smooth2_method: str = 'none'
    smooth2_kwargs: dict = field(default_factory=dict)

    def __str__(self):
        lines = []
        for f in fields(self):
            val = getattr(self, f.name)
            lines.append(f"  {f.name}: {_format_value(val, indent=4)}")
        return "AWBConfig(\n" + "\n".join(lines) + "\n)"

    __repr__ = __str__