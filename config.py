"""
配置文件：定义所有项目参数和规则

包含:
- 文件路径配置
- 轨迹生成参数
- 地形规则
- 速度规则
- 坡向约束参数
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
INTERMEDIATE_DIR = os.path.join(OUTPUT_DIR, "intermediate")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 输入文件路径
DEM_PATH = os.path.join(INPUT_DIR, "dem_30m_100km.tif")
SLOPE_PATH = os.path.join(INPUT_DIR, "slope_30m_100km.tif")
ASPECT_PATH = os.path.join(INPUT_DIR, "aspect_30m_100km.tif")
LANDCOVER_PATH = os.path.join(INPUT_DIR, "landcover_30m_100km.tif")

# 输出文件路径
SPEED_MAP_PATH = os.path.join(INTERMEDIATE_DIR, "base_speed_map.tif")
COST_MAP_PATH = os.path.join(INTERMEDIATE_DIR, "cost_map.tif")
TRAJECTORY_PATH = os.path.join(OUTPUT_DIR, "trajectory.csv")

# 批量生成参数
NUM_TRAJECTORIES_TO_GENERATE = 500  # 要生成的轨迹总数
NUM_END_POINTS = 3                  # 要选择的固定终点数量
MIN_START_END_DISTANCE_METERS = 80000  # 起终点最小直线距离(米)

# 地物编码定义
URBAN_LANDCOVER_CODES = [1, 10]  # 城市/建成区编码
IMPASSABLE_LANDCOVER_CODES = [11]  # 不可通行区域编码(如水体)

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

# 基础速度规则 (单位: m/s)
BASE_SPEED_RULES = {
    1: 16.7,  # 60 km/h, 城市道路
    2: 11.1,  # 40 km/h, 乡村道路
    3: 8.3,   # 30 km/h, 土路
    4: 5.6,   # 20 km/h, 草地
    5: 4.2,   # 15 km/h, 灌木
    6: 2.8,   # 10 km/h, 森林
    7: 1.4,   # 5 km/h, 沙地
    8: 1.4,   # 5 km/h, 裸地
    9: 1.1,   # 4 km/h, 湿地
    10: 16.7, # 60 km/h, 建成区
    11: 0.0   # 0 km/h, 水体(不可通行)
}

# 坡度影响规则
SLOPE_SPEED_REDUCTION_FACTOR = 0.1  # 每增加1度坡度减少的速度比例

# 坡向约束参数
ASPECT_UPHILL_REDUCTION_K = 0.15    # 上坡影响因子
ASPECT_CROSS_REDUCTION_K = 0.2      # 横坡影响因子
MAX_CROSS_SLOPE_DEGREES = 30        # 最大允许横坡角度
MAX_BRAKING_SPEED_DOWNHILL = 8.3    # 下坡最大安全速度(30km/h)

# 速度规则 (landcover_code, slope_level) -> speed
SPEED_RULES = {
    # 平地 (0-5°)
    (1, 0): 16.7,  # 城市道路
    (2, 0): 11.1,  # 乡村道路
    (3, 0): 8.3,   # 土路
    (4, 0): 5.6,   # 草地
    (5, 0): 4.2,   # 灌木
    (6, 0): 2.8,   # 森林
    (7, 0): 1.4,   # 沙地
    (8, 0): 1.4,   # 裸地
    (9, 0): 1.1,   # 湿地
    (10, 0): 16.7, # 建成区
    (11, 0): 0.0,  # 水体
    
    # 缓坡 (5-15°)
    (1, 1): 13.9,  # 城市道路
    (2, 1): 8.3,   # 乡村道路
    (3, 1): 5.6,   # 土路
    (4, 1): 4.2,   # 草地
    (5, 1): 2.8,   # 灌木
    (6, 1): 1.9,   # 森林
    (7, 1): 1.1,   # 沙地
    (8, 1): 1.1,   # 裸地
    (9, 1): 0.8,   # 湿地
    (10, 1): 13.9, # 建成区
    (11, 1): 0.0,  # 水体
    
    # 中坡 (15-30°)
    (1, 2): 8.3,   # 城市道路
    (2, 2): 5.6,   # 乡村道路
    (3, 2): 4.2,   # 土路
    (4, 2): 2.8,   # 草地
    (5, 2): 1.9,   # 灌木
    (6, 2): 1.4,   # 森林
    (7, 2): 0.8,   # 沙地
    (8, 2): 0.8,   # 裸地
    (9, 2): 0.6,   # 湿地
    (10, 2): 8.3,  # 建成区
    (11, 2): 0.0,  # 水体
    
    # 陡坡 (30-45°)
    (1, 3): 4.2,   # 城市道路
    (2, 3): 2.8,   # 乡村道路
    (3, 3): 1.9,   # 土路
    (4, 3): 1.4,   # 草地
    (5, 3): 0.8,   # 灌木
    (6, 3): 0.6,   # 森林
    (7, 3): 0.3,   # 沙地
    (8, 3): 0.3,   # 裸地
    (9, 3): 0.3,   # 湿地
    (10, 3): 4.2,  # 建成区
    (11, 3): 0.0,  # 水体
    
    # 悬崖或无效值 (>45°)
    (1, 4): 0.0,   # 不可通行
    (2, 4): 0.0,
    (3, 4): 0.0,
    (4, 4): 0.0,
    (5, 4): 0.0,
    (6, 4): 0.0,
    (7, 4): 0.0,
    (8, 4): 0.0,
    (9, 4): 0.0,
    (10, 4): 0.0,
    (11, 4): 0.0
}

# 默认速度 (当没有匹配规则时使用)
DEFAULT_SPEED_MPS = 1.4  # 5 km/h

# 轨迹生成参数
TARGET_LENGTH_RANGE = [80000, 120000]  # 目标轨迹长度范围(米)
DEFAULT_TARGET_SPEED = 30  # 默认目标平均速度(km/h)

# 模拟参数
DT = 1.0  # 时间步长(秒)
MAX_ACCELERATION = 1.0  # 最大加速度(m/s^2)
MAX_DECELERATION = 2.0  # 最大减速度(m/s^2)
MAX_TURNING_RATE = 45.0  # 最大转向率(度/秒)

# 输出目录
BATCH_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "synthetic_batch_{}")
EVALUATION_DIR = os.path.join(OUTPUT_DIR, "evaluation_report_{}")

# 创建必要的目录
for directory in [INPUT_DIR, OUTPUT_DIR, INTERMEDIATE_DIR]:
    os.makedirs(directory, exist_ok=True) 