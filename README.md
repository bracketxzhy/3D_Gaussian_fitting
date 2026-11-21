# 3D Gaussian Sphere Fitting for Antenna Patterns

A Python visualization prototype that fits a **3D Gaussian-like spherical pattern** to real-world antenna radiation data using **spherical harmonics expansion**.  
The implementation fits log-gain samples with spherical harmonics, multiplies by a Gaussian-shaped envelope, normalizes the radius, and optionally applies ellipsoid scaling. The surface is a visualization, not a calibrated electromagnetic solver.

**Scope:** antenna-pattern fitting, not 3D Gaussian Splatting or human reconstruction. No quantitative fit-accuracy or runtime benchmark is supplied.

| Component | Source |
| --- | --- |
| Spherical harmonics fitting and surface export | [gaussian_sphere_fit.py](gaussian_sphere_fit.py) |
| Parameters | [config.py](config.py) |
| Data loader | [input_data.py](input_data.py) |
| Example input | [example_data.csv](example_data.csv) |

## Key Features

- Pure NumPy + SciPy + Matplotlib → no heavy dependencies
- Reads simple CSV (theta_deg, phi_deg, gain_dB) → one-line data import
- High-order spherical harmonics fitting (up to L = 16, adjustable)
- Gain clipping before log conversion; NumPy least-squares fitting
- Interactive 3D visualization with Matplotlib
- One-click STL export → open in MeshLab, Blender, or send directly to 3D printer
- Small, configurable scientific-computing prototype

## Quick Start

```bash
# 1. Clone & enter
git clone https://github.com/bracketxzhy/3D_Gaussian_fitting.git
cd 3D_Gaussian_fitting

# 2. Install dependencies
pip install numpy "scipy<1.17" matplotlib pandas numpy-stl

# 3. Put your data as example_data.csv 

# 4. Run
python gaussian_sphere_fit.py
```

## Compatibility and interpretation

The code uses SciPy's legacy `scipy.special.sph_harm` API. The quick-start command constrains SciPy to versions before its removal; an API migration would require a separate code change. See the [SciPy API notice](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.special.sph_harm.html). Check the installed version if imports fail.

Configure angular units, gain representation, harmonic order, envelope, and output paths in [config.py](config.py) and inspect [input_data.py](input_data.py) before replacing the example CSV. Gaussian envelope, radius normalization, and ellipsoid scaling affect interpretation; exported STL coordinates do not preserve absolute antenna gain.
