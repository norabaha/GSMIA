from src.utils.config import AWBConfig

def run_awb(img, cfg: AWBConfig, method: int):
    """Run AWB using specified method.
    img: HxWx3 float32 in [0,1]
    cfg: AWBConfig
    method: int (0-3)
    Returns: dict with 'image', 'gains', and 'mask'
    """
    if method == 0:
        from src.method0 import run_awb_method_0
        return run_awb_method_0(img, cfg)
    elif method == 1:
        from src.method1 import run_awb_method_1
        return run_awb_method_1(img, cfg)
    elif method == 2:
        from src.method2 import run_awb_method_2
        return run_awb_method_2(img, cfg)
    elif method == 3:
        from src.method3_new import run_awb_method_3
        return run_awb_method_3(img, cfg)
        if cfg.region_method == 'tiles':
            from src.method3_new import run_awb_method_3_new
            return run_awb_method_3_new(img, cfg)
        else:
            from src.method3 import run_awb_method_3
            return run_awb_method_3(img, cfg)
    else:
        raise ValueError(f"Invalid AWB method: {method}")