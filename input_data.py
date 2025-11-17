# input_data.py
import numpy as np
import pandas as pd
from pathlib import Path

def load_data(csv_path="example_data.csv"):
    """
    仅从 CSV 文件加载数据（必须包含 theta_deg, phi_deg, gain_dB 三列）
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件未找到: {csv_path}")

    print(f"从 CSV 加载数据: {csv_path.name}")
    data = pd.read_csv(csv_path)

    # 必须包含这三列
    required_cols = ['theta_deg', 'phi_deg', 'gain_dB']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"CSV 必须包含列: {required_cols}")

    theta_rad = np.radians(data['theta_deg'].values.astype(float))
    phi_rad   = np.radians(data['phi_deg'].values.astype(float))
    gain_db   = data['gain_dB'].values.astype(float)

    print(f"成功加载 {len(theta_rad)} 个采样点")
    return theta_rad, phi_rad, gain_db