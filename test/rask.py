from src.methods import run_awb
from src.method3_new import run_awb_method_3
from src.method3_tiles import run_awb_method_3_tiles
from cfgs import ALL_CONFIGS
import src.test as test
from src.utils import config
import os


cfg = config.AWBConfig(
    debug=False,
    fast=True,
    awb_method=1,
    num_tiles=8,
    num_superpixels=24,
    mask_saturation_threshold=0.99,
    mask_black_threshold=0.01,
    eps=1e-6,
    cct_curve_rb=test.get_updated_cct(),
    cct_tolerance=0.2,
    downsample=True,
    downsample_scale=0.1,
    region_method='tiles',
    sp_compactness=10,
    cluster_method='kmeans',
    cluster_kwargs={},
    cluster_min_center_distance=0.1,
    cluster_min_ratio=0.05,
    smooth1_method='guided',
    smooth1_kwargs={'guided_radius': 6, 'guided_eps': 1e-4},
    smooth2_method='none',
    smooth2_kwargs={},
)

def run_one_cfg(cfg):
    img = test.get_image(ill_idx=0, mode='multi')

    print("Running AWB method", cfg.awb_method, cfg.region_method)

    # running once to warm up
    print("Warm-up runs...")
    n = 30
    for _ in range(n):
        run_awb(img, cfg, cfg.awb_method)
    
    # Now measuring time
    print("Measuring runs...")

    # Run the AWB method n times and average
    n = 50
    timings = []
    for _ in range(n):
        result = run_awb(img, cfg, cfg.awb_method)
        timings.append(result['timings'])

    avg_timings = {}
    for key in timings[0].keys():
        avg_timings[key] = sum(t[key] for t in timings) / n

    print("\nAverage timings over", n, "runs:")
    total_time = sum(avg_timings.values())
    for k, v in sorted(avg_timings.items(), key=lambda x: -x[1]):
        print(f"{k:30s}: {v*1e-6:7.2f} ms  ({100*v/total_time:5.1f}%)")
    print(f"{'TOTAL':30s}: {total_time*1e-6:7.2f} ms  ({1e9/total_time:.0f} FPS)")

    median_timings = {}
    for key in timings[0].keys():
        median_timings[key] = sorted(t[key] for t in timings)[n // 2]
    print("\nMedian timings over", n, "runs:")
    total_time = sum(median_timings.values())
    print(f"{'TOTAL':30s}: {total_time*1e-6:7.2f} ms  ({1e9/total_time:.0f} FPS)")

    p95_timings = {}
    for key in timings[0].keys():
        p95_timings[key] = sorted(t[key] for t in timings)[int(n * 0.95) - 1]
    print("\nP95 timings over", n, "runs:")
    total_time = sum(p95_timings.values())
    print(f"{'TOTAL':30s}: {total_time*1e-6:7.2f} ms  ({1e9/total_time:.0f} FPS)")


def main():
    img = test.get_image(ill_idx=0, mode='multi')
    
    cfg.awb_method = 3
    cfg.region_method = 'superpixels'
    run_one_cfg(cfg)
    cfg.awb_method = 3
    cfg.region_method = 'tiles'
    run_one_cfg(cfg)

    n_warmup = 30
    n_measure = 100

    rows = []

    for m in range(4):
        cfg.awb_method = m
        for r in ['tiles', 'superpixels']:
            if m == 0 and r == 'superpixels':
                continue  # Method 0 does not use regions
            cfg.region_method = r
            print(f"\n=== AWB Method {m}, Region Method {r} ===")
            print("Warm-up runs...")
            # Warm-up
            for _ in range(n_warmup):
                run_awb(img, cfg, cfg.awb_method)
        
            print("Measuring runs...")
            # Measurement
            timings = []
            for _ in range(n_measure):
                result = run_awb(img, cfg, cfg.awb_method)
                timings.append(result['timings'])
            
            # Timings statistics
            avg_timings = {}
            median_timings = {}
            p95_timings = {}
            for key in timings[0].keys():
                avg_timings[key] = sum(t[key] for t in timings) / n_measure
                median_timings[key] = sorted(t[key] for t in timings)[n_measure // 2]
                p95_timings[key] = sorted(t[key] for t in timings)[int(n_measure * 0.95) - 1]
            total_time_avg = sum(avg_timings.values())
            total_time_median = sum(median_timings.values())
            total_time_p95 = sum(p95_timings.values())
            rows.append((
                m,
                r,
                total_time_avg * 1e-6,
                total_time_median * 1e-6,
                total_time_p95 * 1e-6,
            ))
                
    print("\nSummary of timings (in ms):")
    print(f"{'Method':6s} {'Region':6s} {'Avg':>10s} {'Median':>10s} {'P95':>10s}")
    for row in rows:
        print(f"{row[0]:6d} {row[1][0]:6s} {row[2]:10.2f} {row[3]:10.2f} {row[4]:10.2f}")

if __name__ == "__main__":
    main()