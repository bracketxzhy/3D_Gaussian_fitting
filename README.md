# 高斯球拟合器（Gaussian Sphere Fitter）  
**带球谐函数各向异性 + 三维可视化 + STL 导出**

一个轻量级 Python 工具，将 **3D 极坐标天线方向图** 拟合为 **光滑高斯球 + 球谐调制**，用于：

- 天线辐射模式建模  
- 电磁仿真后处理（CST / HFSS / FEKO）  
- 3D 可视化与模型导出  

---

## 特性

| 功能 | 说明 |
|------|------|
| **球谐拟合** | 使用 `scipy.special.sph_harm` 拟合 `log(gain)` |
| **高斯球基底** | `exp(-θ²/2σθ² - φ²/2σφ²)` 控制球形光滑度 |
| **精细网格** | 支持 `N_GRID=400+` 超光滑表面 |
| **椭球变形** | 可输出椭球（X/Y/Z 拉伸） |
| **STL 导出** | 一键导出 3D 模型（`numpy-stl`） |
| **CSV 输入** | 支持 `theta_deg, phi_deg, gain_dB` 格式 |

---

## 安装依赖

```bash
pip install numpy scipy matplotlib pandas numpy-stl
