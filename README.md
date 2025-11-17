# Gaussian Sphere Fitter  
**Spherical Harmonics Anisotropy + 3D Visualization + STL Export**

A lightweight Python tool to fit **3D polar radiation patterns** into a **smooth Gaussian sphere modulated by spherical harmonics**, ideal for:

- Antenna radiation modeling  
- Post-processing of EM simulations (CST / HFSS / FEKO)  
- 3D visualization and model export  

---

## Features

| Feature | Description |
|---------|-------------|
| **Spherical Harmonics Fitting** | Fits `log(gain)` using `scipy.special.sph_harm` |
| **Gaussian Base Sphere** | `exp(-θ²/2σθ² - φ²/2σφ²)` controls smoothness |
| **High-Resolution Grid** | Supports `N_GRID=400+` for ultra-smooth surfaces |
| **Ellipsoid Deformation** | Output can be stretched (X/Y/Z scaling) |
| **STL Export** | One-click 3D model export (`numpy-stl`) |
| **CSV Input** | Supports `theta_deg, phi_deg, gain_dB` format |

---

## Installation

```bash   ”“bash
pip install numpy scipy matplotlib pandas numpy-stl
