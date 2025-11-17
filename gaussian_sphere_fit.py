# gaussian_sphere_fit.py
import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 导入参数和数据
from config import *
from input_data import load_data

# ====================== 工具函数 ======================
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def generate_gaussian_sphere(theta_grid, phi_grid):
    phi_centered = phi_grid - np.pi
    r = np.exp(-0.5 * (theta_grid**2 / SIGMA_THETA**2 + phi_centered**2 / SIGMA_PHI**2))
    return r

def fit_spherical_harmonics(theta, phi, gain_db, l_max=L_MAX):
    gain_linear = 10**(gain_db / 10.0) if GAIN_IS_DB else gain_db.copy()
    log_gain = np.log(np.clip(gain_linear, 1e-15, None))
    n_samples = len(theta)
    n_coeffs = (l_max + 1)**2
    A = np.zeros((n_samples, n_coeffs))
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            A[:, idx] = np.real(sph_harm(m, l, phi, theta))
            idx += 1
    c, _, _, _ = np.linalg.lstsq(A, log_gain, rcond=None)
    return c

def reconstruct_from_sh(c, theta_grid, phi_grid, l_max=L_MAX):
    log_gain_rec = np.zeros_like(theta_grid)
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            log_gain_rec += c[idx] * np.real(sph_harm(m, l, phi_grid, theta_grid))
            idx += 1
    return log_gain_rec

def export_stl(X, Y, Z, filename):
    from stl import mesh
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    h, w = X.shape
    vertices = np.zeros((h*w, 3))
    vertices[:, 0] = X.flatten()
    vertices[:, 1] = Y.flatten()
    vertices[:, 2] = Z.flatten()
    faces = []
    for i in range(h-1):
        for j in range(w-1):
            v1 = i*w + j
            v2 = i*w + (j+1)
            v3 = (i+1)*w + (j+1)
            v4 = (i+1)*w + j
            faces.extend([[v1, v2, v3], [v1, v3, v4]])
    faces = np.array(faces)
    surface = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            surface.vectors[i][j] = vertices[f[j]]
    surface.save(filename)
    print(f"STL 已导出: {filename}")

# ====================== 主拟合函数 ======================
def fit_gaussian_sphere(theta_input, phi_input, gain_input):
    coeffs = fit_spherical_harmonics(theta_input, phi_input, gain_input)
    theta_grid = np.linspace(0, np.pi, N_GRID)
    phi_grid = np.linspace(0, 2*np.pi, N_GRID)
    THETA, PHI = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    r_base = generate_gaussian_sphere(THETA, PHI)
    log_aniso = reconstruct_from_sh(coeffs, THETA, PHI)
    aniso = np.exp(log_aniso)
    r_final = r_base * aniso
    r_final /= r_final.max()
    X, Y, Z = spherical_to_cartesian(r_final, THETA, PHI)
    # 椭球变形
    a, b, c = ELLIPSOID_SCALE
    X, Y, Z = X * a, Y * b, Z * c
    return X, Y, Z, coeffs

# ====================== 主程序 ======================
if __name__ == "__main__":
    # 1. 加载数据
    theta_in, phi_in, gain_in = load_data(csv_path="example_data.csv")

    # 2. 拟合
    X, Y, Z, coeffs = fit_gaussian_sphere(theta_in, phi_in, gain_in)

    # 3. 绘图
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', linewidth=0, antialiased=True, alpha=0.9, rstride=2, cstride=2)
    ax.set_title(f"Gaussian Sphere Fit (L={L_MAX}, N={N_GRID})", pad=20)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect(ELLIPSOID_SCALE)
    fig.colorbar(surf, shrink=0.6)
    plt.tight_layout()
    plt.show()

    # 4. 导出 STL
    if EXPORT_STL:
        export_stl(X, Y, Z, STL_FILENAME)

    print(f"拟合完成！球谐系数数量: {(L_MAX+1)**2}")