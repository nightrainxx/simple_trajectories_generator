"""
配置文件: 定义项目所需的所有路径和参数设置

包含:
- 输入/输出文件路径
- 环境参数设置
- 移动规则配置
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 输入文件路径
DEM_PATH = os.path.join(INPUT_DIR, "dem_30m_100km.tif")
SLOPE_PATH = os.path.join(INPUT_DIR, "slope_30m_100km.tif")
LANDCOVER_PATH = os.path.join(INPUT_DIR, "landcover_30m_100km.tif")

# 输出文件路径
SPEED_MAP_PATH = os.path.join(OUTPUT_DIR, "typical_speed_map.tif")
COST_MAP_PATH = os.path.join(OUTPUT_DIR, "cost_map.tif")
TRAJECTORY_PATH = os.path.join(OUTPUT_DIR, "trajectory.csv")

# 坡度分类参数
SLOPE_BINS = [-1, 5, 15, 30, 45, 91]  # 坡度分类边界值
SLOPE_LABELS = [0, 1, 2, 3, 4]  # 对应平地, 缓坡, 中坡, 陡坡, 峭壁

# 土地覆盖类型编码 (根据实际数据更新)
LANDCOVER_CODES = {
    'CROPLAND': 10,    # 耕地
    'FOREST': 20,      # 森林
    'GRASSLAND': 30,   # 草地
    'SHRUBLAND': 40,   # 灌木地
    'WETLAND': 50,     # 湿地
    'WATER': 60,       # 水体
    'TUNDRA': 70,      # 苔原
    'ARTIFICIAL': 80,  # 人工表面
    'BARELAND': 90,    # 裸地
    'SNOW': 100,       # 永久积雪/冰
    'NODATA': 255      # 无数据
}

# 速度规则配置 (单位: 米/秒)
SPEED_RULES = {
    # 耕地
    (10, 0): 40 / 3.6,  # 平地, 40 km/h
    (10, 1): 30 / 3.6,  # 缓坡, 30 km/h
    (10, 2): 20 / 3.6,  # 中坡, 20 km/h
    (10, 3): 10 / 3.6,  # 陡坡, 10 km/h
    (10, 4): 0.0,       # 峭壁, 不可通行
    
    # 森林
    (20, 0): 25 / 3.6,  # 平地, 25 km/h
    (20, 1): 15 / 3.6,  # 缓坡, 15 km/h
    (20, 2): 10 / 3.6,  # 中坡, 10 km/h
    (20, 3): 5 / 3.6,   # 陡坡, 5 km/h
    (20, 4): 0.0,       # 峭壁, 不可通行
    
    # 草地
    (30, 0): 35 / 3.6,  # 平地, 35 km/h
    (30, 1): 25 / 3.6,  # 缓坡, 25 km/h
    (30, 2): 15 / 3.6,  # 中坡, 15 km/h
    (30, 3): 8 / 3.6,   # 陡坡, 8 km/h
    (30, 4): 0.0,       # 峭壁, 不可通行
    
    # 灌木地
    (40, 0): 30 / 3.6,  # 平地, 30 km/h
    (40, 1): 20 / 3.6,  # 缓坡, 20 km/h
    (40, 2): 12 / 3.6,  # 中坡, 12 km/h
    (40, 3): 6 / 3.6,   # 陡坡, 6 km/h
    (40, 4): 0.0,       # 峭壁, 不可通行
    
    # 湿地
    (50, 0): 15 / 3.6,  # 平地, 15 km/h
    (50, 1): 10 / 3.6,  # 缓坡, 10 km/h
    (50, 2): 5 / 3.6,   # 中坡, 5 km/h
    (50, 3): 0.0,       # 陡坡, 不可通行
    (50, 4): 0.0,       # 峭壁, 不可通行
    
    # 水体 - 不可通行
    (60, 0): 0.0,
    (60, 1): 0.0,
    (60, 2): 0.0,
    (60, 3): 0.0,
    (60, 4): 0.0,
    
    # 苔原
    (70, 0): 20 / 3.6,  # 平地, 20 km/h
    (70, 1): 15 / 3.6,  # 缓坡, 15 km/h
    (70, 2): 8 / 3.6,   # 中坡, 8 km/h
    (70, 3): 4 / 3.6,   # 陡坡, 4 km/h
    (70, 4): 0.0,       # 峭壁, 不可通行
    
    # 人工表面
    (80, 0): 60 / 3.6,  # 平地, 60 km/h
    (80, 1): 50 / 3.6,  # 缓坡, 50 km/h
    (80, 2): 30 / 3.6,  # 中坡, 30 km/h
    (80, 3): 15 / 3.6,  # 陡坡, 15 km/h
    (80, 4): 0.0,       # 峭壁, 不可通行
    
    # 裸地
    (90, 0): 45 / 3.6,  # 平地, 45 km/h
    (90, 1): 35 / 3.6,  # 缓坡, 35 km/h
    (90, 2): 20 / 3.6,  # 中坡, 20 km/h
    (90, 3): 10 / 3.6,  # 陡坡, 10 km/h
    (90, 4): 0.0,       # 峭壁, 不可通行
    
    # 永久积雪/冰 - 不可通行
    (100, 0): 0.0,
    (100, 1): 0.0,
    (100, 2): 0.0,
    (100, 3): 0.0,
    (100, 4): 0.0,
    
    # 无数据 - 不可通行
    (255, 0): 0.0,
    (255, 1): 0.0,
    (255, 2): 0.0,
    (255, 3): 0.0,
    (255, 4): 0.0,
}

# 默认速度 (用于未定义的组合)
DEFAULT_SPEED_MPS = 5 / 3.6  # 5 km/h

# 轨迹生成参数
TARGET_LENGTH_RANGE = [80000, 120000]  # 目标轨迹长度范围(米)
DEFAULT_TARGET_SPEED = 30  # 默认目标平均速度(km/h) 