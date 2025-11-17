# 3D Gaussian Sphere Fitting for Antenna Patterns

A lightweight, fast, and robust Python tool that fits a **3D Gaussian-like spherical pattern** to real-world antenna radiation data using **spherical harmonics expansion**.  
Perfect for antenna engineers who want to convert measured (or simulated) far-field data into a smooth, compact, and visually appealing 3D model.

![example](output/example_pattern.png)  
_Example of a fitted multi-lobe ellipsoidal pattern (L = 10)_

## Key Features

- Pure NumPy + SciPy + Matplotlib → no heavy dependencies
- Reads simple CSV (theta_deg, phi_deg, gain_dB) → one-line data import
- High-order spherical harmonics fitting (up to L = 16, adjustable)
- Automatic numerical stabilization (clipping + rcond) to avoid SVD crash
- Interactive 3D visualization with Matplotlib
- One-click STL export → open in MeshLab, Blender, or send directly to 3D printer
- Only ~150 lines of clean, well-commented code

## Quick Start

```bash
# 1. Clone & enter
git clone https://github.com/yourname/3D_Gaussian_fitting.git
cd 3D_Gaussian_fitting

# 2. Install dependencies
pip install numpy scipy matplotlib pandas numpy-stl

# 3. Put your data as example_data.csv (see format below)

# 4. Run
python gaussian_sphere_fit.py
```
