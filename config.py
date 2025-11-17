import numpy as np

# ====================== 拟合参数 ======================
L_MAX = 10                    # 球谐最大阶数（推荐 12~16）
N_GRID = 400                  # 重构网格密度（越高越光滑）
GAIN_IS_DB = True             # 输入增益是否为 dB

# 高斯球宽度（控制圆润度）
SIGMA_THETA = np.pi / 7       # ~25.7°
SIGMA_PHI   = np.pi / 5       # ~36°

# 可选：椭球变形比例（1,1,1）= 球
ELLIPSOID_SCALE = [1.0, 1.0, 1.0]  # [a_x, a_y, a_z]

# 输出设置
EXPORT_STL = True             # 是否导出 STL
STL_FILENAME = "output/gaussian_sphere.stl"