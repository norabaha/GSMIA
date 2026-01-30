# GSMIA: General Spatial Multi-Illuminant Automatic White Balance

A comprehensive framework for **automatic white balance (AWB) correction** in images captured under various illuminants. This thesis project implements and compares multiple AWB methods using global and region-based approaches.

## Overview

GSMIA provides multiple computational methods for correcting color casts in images caused by different lighting conditions (tungsten, daylight, fluorescent, etc.). The framework includes:

- **4 distinct AWB methods** ranging from global to region-based approaches
- **Configurable processing pipeline** with support for different algorithms
- **Image capture module** for Raspberry Pi cameras
- **Evaluation and testing tools** for assessing correction quality

## Methods

- **Method 0**: Global statistics-based white balance correction
- **Method 1**: Enhanced global approach with preprocessing
- **Method 2**: Clustering-based regional correction
- **Method 3**: Super-pixel tiling with localized color correction

## Project Structure

```
src/               # Core AWB implementation
├── method0-3.py   # AWB algorithm implementations
├── utils/         # Helper utilities (clustering, ISP, smoothing)
└── methods.py     # Method orchestration

capture/           # Image capture and lightbox data
├── main.py        # Raspberry Pi camera capture script
└── lightbox/      # Captured images under different illuminants (A, D65, HZ, TL84, CWF)

test/              # Evaluation and testing
├── eval.ipynb     # Quantitative and qualitative evaluation
└── rask.py        # Additional testing utilities
```

## Usage

```python
from src.methods import run_awb
from src.utils.config import AWBConfig

# Load image and configure method
img = load_image(...)
cfg = AWBConfig(...)

# Run white balance correction
result = run_awb(img, cfg, method=3)
corrected_image = result['image']
```

## Requirements

- Python 3.7+
- NumPy, OpenCV
- Optional: Raspberry Pi with PiCamera2 (for image capture)

## Author

Nora Hanssen - Master thesis, [NTNU/2026]