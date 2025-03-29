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
INVALID_AREA_PATH = os.path.join(INPUT_DIR, "invalid_area_mask.npy")

# 输出文件路径
SPEED_MAP_PATH = os.path.join(INTERMEDIATE_DIR, "base_speed_map.tif")
COST_MAP_PATH = os.path.join(INTERMEDIATE_DIR, "cost_map.tif")
TRAJECTORY_PATH = os.path.join(OUTPUT_DIR, "trajectory.csv")

# 批量生成参数
NUM_TRAJECTORIES_TO_GENERATE = 500  # 要生成的轨迹总数
NUM_END_POINTS = 4                  # 要选择的固定终点数量（4个战略位置）
MIN_START_END_DISTANCE_METERS = 60000  # 起终点最小直线距离(米)

# 地物编码定义
URBAN_LANDCOVER_CODES = [10]  # 城市/建成区编码
IMPASSABLE_LANDCOVER_CODES = [60]  # 不可通行区域编码(如水体)

# 坡度分类参数
SLOPE_BINS = [-1, 5, 15, 30, 45, 91]  # 坡度分类边界值
SLOPE_LABELS = [0, 1, 2, 3, 4]  # 对应平地, 缓坡, 中坡, 陡坡, 峭壁

# 土地覆盖类型编码 (根据实际数据更新)
LANDCOVER_CODES = {
    10: "耕地",
    20: "林地",
    30: "草地",
    40: "灌木地",
    50: "湿地",
    60: "水体",
    80: "建设用地",
    90: "裸地",
    255: "未分类"
}

# 不可通行的土地类型
IMPASSABLE_TYPES = [60]  # 水体

# 速度规则（土地类型, 坡度）-> 速度(米/秒)
SPEED_RULES = {
    # 耕地
    (10, 0): 18.1,  # 0-5°, 65km/h
    (10, 1): 16.7,  # 5-15°, 60km/h
    (10, 2): 11.1,  # 15-25°, 40km/h
    (10, 3): 5.6,   # 25-35°, 20km/h
    (10, 4): 0.0,   # >35°, 不可通行
    
    # 林地
    (20, 0): 16.7,  # 0-5°, 60km/h
    (20, 1): 13.9,  # 5-15°, 50km/h
    (20, 2): 8.3,   # 15-25°, 30km/h
    (20, 3): 5.6,   # 25-35°, 20km/h
    (20, 4): 0.0,   # >35°, 不可通行
    
    # 草地
    (30, 0): 19.4,  # 0-5°, 70km/h
    (30, 1): 16.7,  # 5-15°, 60km/h
    (30, 2): 13.9,  # 15-25°, 50km/h
    (30, 3): 11.1,  # 25-35°, 40km/h
    (30, 4): 0.0,   # >35°, 不可通行
    
    # 灌木地
    (40, 0): 18.1,  # 0-5°, 65km/h
    (40, 1): 16.7,  # 5-15°, 60km/h
    (40, 2): 11.1,  # 15-25°, 40km/h
    (40, 3): 5.6,   # 25-35°, 20km/h
    (40, 4): 0.0,   # >35°, 不可通行
    
    # 湿地
    (50, 0): 16.7,  # 0-5°, 60km/h
    (50, 1): 13.9,  # 5-15°, 50km/h
    (50, 2): 8.3,   # 15-25°, 30km/h
    (50, 3): 5.6,   # 25-35°, 20km/h
    (50, 4): 0.0,   # >35°, 不可通行
    
    # 水体
    (60, 0): 0.0,   # 不可通行
    (60, 1): 0.0,
    (60, 2): 0.0,
    (60, 3): 0.0,
    (60, 4): 0.0,
    
    # 建设用地
    (80, 0): 19.4,  # 0-5°, 70km/h
    (80, 1): 16.7,  # 5-15°, 60km/h
    (80, 2): 13.9,  # 15-25°, 50km/h
    (80, 3): 11.1,  # 25-35°, 40km/h
    (80, 4): 0.0,   # >35°, 不可通行
    
    # 裸地
    (90, 0): 18.1,  # 0-5°, 65km/h
    (90, 1): 16.7,  # 5-15°, 60km/h
    (90, 2): 13.9,  # 15-25°, 50km/h
    (90, 3): 8.3,   # 25-35°, 30km/h
    (90, 4): 0.0,   # >35°, 不可通行
    
    # 未分类
    (255, 0): 16.7,  # 默认 60km/h
    (255, 1): 13.9,  # 50km/h
    (255, 2): 8.3,   # 30km/h
    (255, 3): 5.6,   # 20km/h
    (255, 4): 0.0    # 不可通行
}

# 坡度分级（度）
SLOPE_LEVELS = [5, 15, 25, 35]

# 轨迹生成参数
TARGET_LENGTH_RANGE = [100000, 200000]  # 目标轨迹长度范围(米)
DEFAULT_TARGET_SPEED = 65  # 默认目标平均速度(km/h)
DEFAULT_SPEED_MPS = 18.1  # 默认速度（米/秒，约65km/h）

# 模拟参数
DT = 1.0  # 时间步长(秒)
MAX_ACCELERATION = 2.0  # 最大加速度(m/s^2)
MAX_DECELERATION = -3.0  # 最大减速度(m/s^2)
MAX_TURNING_RATE = 30.0  # 最大转向率(度/秒)

# 坡度影响参数
SLOPE_SPEED_REDUCTION_FACTOR = 0.1  # 坡度减速系数
ASPECT_UPHILL_REDUCTION_K = 0.04  # 上坡减速系数（降低影响）
ASPECT_CROSS_REDUCTION_K = 0.08  # 横坡减速系数（降低影响）
MAX_CROSS_SLOPE_DEGREES = 30.0  # 最大横坡角度
MAX_BRAKING_SPEED_DOWNHILL = 16.7  # 下坡最大制动速度（米/秒，约60km/h）

# 输出目录
BATCH_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "batch_trajectories_20_highspeed_65kmh")
EVALUATION_DIR = os.path.join(OUTPUT_DIR, "evaluation_report_{}")

# 地图尺寸
MAP_SIZE = (3333, 3333)

# 创建必要的目录
for directory in [INPUT_DIR, OUTPUT_DIR, INTERMEDIATE_DIR]:
    os.makedirs(directory, exist_ok=True) 