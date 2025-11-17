import numpy as np
from scipy.special import sph_harm
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ====================== 配置参数 ======================
L_MAX = 14         # 球谐函数最大阶数（控制各向异性复杂度）
GAIN_IS_DB = True  # 输入增益是否为 dB
SIGMA_THETA = np.pi / 7  # 高斯球在 theta 方向的宽度 (~30°)
SIGMA_PHI = np.pi / 5    # phi 方向宽度 (~45°)，可调
N_GRID = 400     # 重构高斯球的网格密度
# =====================================================

def spherical_to_cartesian(r, theta, phi):
    """球坐标转笛卡尔"""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def generate_gaussian_sphere(theta_grid, phi_grid):
    """生成基础高斯球径向函数 r(θ,φ) = exp(-θ²/2σθ² - φ²/2σφ²)"""
    r = np.exp(-0.5 * (theta_grid**2 / SIGMA_THETA**2 + 
                       (phi_grid % (2*np.pi) - np.pi)**2 / SIGMA_PHI**2))
    return r

def fit_spherical_harmonics(theta, phi, gain_db, l_max=L_MAX):
    """
    拟合 log(gain) = Σ c_lm * Y_lm(θ,φ)
    使用最小二乘，返回系数 c_lm (l=0..l_max, m=-l..l)
    """
    if GAIN_IS_DB:
        gain_linear = 10**(gain_db / 10.0)
    else:
        gain_linear = gain_db.copy()

    log_gain = np.log(np.clip(gain_linear, 1e-12, None))  # 避免 log(0)

    # 构建设计矩阵 A: 每行 [Y00, Y1-1, Y10, Y11, ...]
    n_samples = len(theta)
    n_coeffs = (l_max + 1)**2
    A = np.zeros((n_samples, n_coeffs))
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # scipy 的 sph_harm(m, l, phi, theta) 注意顺序！
            A[:, idx] = np.real(sph_harm(m, l, phi, theta))
            idx += 1

    # 最小二乘：min ||A c - log_gain||
    c, _ = np.linalg.lstsq(A, log_gain, rcond=None)[:2]
    return c

def reconstruct_from_sh(c, theta_grid, phi_grid, l_max=L_MAX):
    """从球谐系数重构 log(gain) 场"""
    log_gain_rec = np.zeros_like(theta_grid)
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            log_gain_rec += c[idx] * np.real(sph_harm(m, l, phi_grid, theta_grid))
            idx += 1
    return log_gain_rec

# ====================== 主程序 ======================
def fit_gaussian_sphere(theta_input, phi_input, gain_input):
    """
    输入：theta_input, phi_input (rad), gain_input (dB or linear)
    输出：
        - X, Y, Z: 高斯球三维点云 (N_GRID x N_GRID)
        - coeffs: 球谐拟合系数
        - theta_grid, phi_grid, r_gaussian: 用于调试
    """
    # Step 1: 拟合球谐函数到 log(gain)
    coeffs = fit_spherical_harmonics(theta_input, phi_input, gain_input, L_MAX)

    # Step 2: 生成高密度网格
    theta_grid = np.linspace(0, np.pi, N_GRID)
    phi_grid = np.linspace(0, 2*np.pi, N_GRID)
    THETA, PHI = np.meshgrid(theta_grid, phi_grid, indexing='ij')

    # Step 3: 重构各向异性调制 exp(Σ c_lm Y_lm)
    log_aniso = reconstruct_from_sh(coeffs, THETA, PHI, L_MAX)
    aniso = np.exp(log_aniso)

    # Step 4: 生成基础高斯球
    r_base = generate_gaussian_sphere(THETA, PHI)

    # Step 5: 最终高斯球：基础高斯 × 各向异性调制
    r_final = r_base * aniso
    r_final /= r_final.max()  # 归一化到 1

    # Step 6: 转为笛卡尔坐标
    X, Y, Z = spherical_to_cartesian(r_final, THETA, PHI)

    return X, Y, Z, coeffs, THETA, PHI, r_final

# ====================== 示例使用 ======================
if __name__ == "__main__":
    # ====================== 生成“带很多尖刺的椭球” ======================
    np.random.seed(123)

    # 1. 生成输入采样点（均匀分布在球面）
    n_samples = 3000
    theta_in = np.arccos(2 * np.random.rand(n_samples) - 1)  # 均匀 theta
    phi_in   = 2 * np.pi * np.random.rand(n_samples)

    # 2. 基础椭球：r = 1 / sqrt( (sinθ cosφ / a)^2 + (sinθ sinφ / b)^2 + (cosθ / c)^2 )
    a, b, c = 1.5, 1.5, 0.8  # X/Y 长，Z 短 → 扁椭球
    sin_theta = np.sin(theta_in)
    r_ellipsoid = 1.0 / np.sqrt(
        (sin_theta * np.cos(phi_in) / a)**2 +
        (sin_theta * np.sin(phi_in) / b)**2 +
        (np.cos(theta_in) / c)**2
    )

    # 3. 添加高阶尖刺（用随机球谐系数，l=10~20）
    L_MAX_SPIKE = 18
    n_coeffs = (L_MAX_SPIKE + 1)**2
    coeffs_spike = np.zeros(n_coeffs)
    idx = 0
    for l in range(L_MAX_SPIKE + 1):
        for m in range(-l, l + 1):
            if 10 <= l <= 20:  # 只用高阶产生尖刺
                coeffs_spike[idx] = np.random.randn() * (1.0 / (l + 1))  # 幅度随 l 衰减
            idx += 1

    # 重构尖刺调制
    spike_mod = np.zeros_like(theta_in)
    idx = 0
    for l in range(L_MAX_SPIKE + 1):
        for m in range(-l, l + 1):
            spike_mod += coeffs_spike[idx] * np.real(sph_harm(m, l, phi_in, theta_in))
            idx += 1
    spike_mod = 1 + 0.4 * spike_mod  # 尖刺高度 ±40%

    # 4. 合成方向图：椭球 × 尖刺
    gain_linear = r_ellipsoid * spike_mod
    gain_linear = np.clip(gain_linear, 1e-12, None)
    gain_db_in = 10 * np.log10(gain_linear)

    # ====================== 调用拟合函数 ======================
    X, Y, Z, coeffs_fit, THETA, PHI, R = fit_gaussian_sphere(theta_in, phi_in, gain_db_in)

    # ====================== 椭球变形（输出时再变形） ======================
    # 将拟合出的高斯球 → 椭球 + 尖刺
    X_final = X * a
    Y_final = Y * b
    Z_final = Z * c

    # ====================== 高质量绘图 ======================
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        X_final, Y_final, Z_final,
        cmap='magma', linewidth=0, antialiased=True,
        alpha=0.95, rstride=2, cstride=2
    )
    ax.set_title("带很多尖刺的椭球（高阶球谐拟合）", fontsize=16, pad=30)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect([a, b, c])  # 保持椭球比例
    fig.colorbar(surf, shrink=0.6)
    plt.tight_layout()
    plt.show()

    print("尖刺椭球生成完成！")
    print(f"拟合使用 L_MAX = {L_MAX}, 尖刺使用 l=10~20")