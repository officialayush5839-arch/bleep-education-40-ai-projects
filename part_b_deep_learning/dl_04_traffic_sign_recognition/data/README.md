# GTSRB Dataset: German Traffic Sign Recognition Benchmark

## Overview
The German Traffic Sign Recognition Benchmark (GTSRB) is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011. It contains more than 50,000 images of 43 traffic sign classes captured under real-world driving conditions with varying illumination, occlusions, resolutions, and weather phenomena.

## Functional Categories and Class Taxonomy
The 43 classes are categorized into 5 primary operational categories:

### 1. Speed Limit Signs (8 classes)
- `Class 0`: Speed limit (20km/h)
- `Class 1`: Speed limit (30km/h)
- `Class 2`: Speed limit (50km/h)
- `Class 3`: Speed limit (60km/h)
- `Class 4`: Speed limit (70km/h)
- `Class 5`: Speed limit (80km/h)
- `Class 7`: Speed limit (100km/h)
- `Class 8`: Speed limit (120km/h)

### 2. Other Prohibitory Signs (4 classes)
- `Class 6`: End of speed limit (80km/h)
- `Class 9`: No passing
- `Class 10`: No passing for vehicles over 3.5 metric tons
- `Class 15`: No vehicles
- `Class 16`: Vehicles over 3.5 metric tons prohibited
- `Class 17`: No entry

### 3. Danger / Warning Signs (15 classes)
- `Class 11`: Right-of-way at the next intersection
- `Class 18`: General caution
- `Class 19`: Dangerous curve to the left
- `Class 20`: Dangerous curve to the right
- `Class 21`: Double curve
- `Class 22`: Bumpy road
- `Class 23`: Slippery road
- `Class 24`: Road narrows on the right
- `Class 25`: Road work
- `Class 26`: Traffic signals
- `Class 27`: Pedestrians
- `Class 28`: Children crossing
- `Class 29`: Bicycles crossing
- `Class 30`: Beware of ice/snow
- `Class 31`: Wild animals crossing

### 4. Mandatory Signs (8 classes)
- `Class 33`: Turn right ahead
- `Class 34`: Turn left ahead
- `Class 35`: Ahead only
- `Class 36`: Go straight or right
- `Class 37`: Go straight or left
- `Class 38`: Keep right
- `Class 39`: Keep left
- `Class 40`: Roundabout mandatory

### 5. Priority / Derestriction Signs (8 classes)
- `Class 12`: Priority road
- `Class 13`: Yield
- `Class 14`: Stop
- `Class 32`: End of all speed and passing limits
- `Class 41`: End of no passing
- `Class 42`: End of no passing for vehicles over 3.5 metric tons

## Preprocessing Requirements
Real-world signs exhibit extreme dynamic contrast variation: signs captured at dawn, in deep shadows under bridges, or against bright sun flare.
To address this:
1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** is applied to the **L** (luminance) channel in the **CIELAB** color space.
2. The image is resized to standard `(32, 32)` resolution.
3. Channels are normalized with ImageNet/standard statistics: `mean = [0.3403, 0.3121, 0.3214]`, `std = [0.2724, 0.2608, 0.2669]` or scaled to `[0, 1]`.

## Deterministic Synthetic Fallback Generator
For automated testing and zero-socket hermetic execution, `src/data_loader.py` includes a deterministic synthetic generator (`generate_synthetic_gtsrb_data`).
This generator programmatically draws:
- Circular red-bordered signs with numeric speed digits
- Triangular red-bordered warning signs with distinct internal patterns
- Circular blue mandatory signs with directional arrows
- Octagonal red Stop signs and triangular Yield signs
This ensures 100% offline test execution without requiring network downloads or Kaggle credentials.
