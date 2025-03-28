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
    (10, 0): 8.3,  # 0-5°, 30km/h
    (10, 1): 5.6,  # 5-15°, 20km/h
    (10, 2): 2.8,  # 15-25°, 10km/h
    (10, 3): 1.4,  # 25-35°, 5km/h
    (10, 4): 0.0,  # >35°, 不可通行
    
    # 林地
    (20, 0): 5.6,  # 0-5°, 20km/h
    (20, 1): 4.2,  # 5-15°, 15km/h
    (20, 2): 2.8,  # 15-25°, 10km/h
    (20, 3): 1.4,  # 25-35°, 5km/h
    (20, 4): 0.0,  # >35°, 不可通行
    
    # 草地
    (30, 0): 11.1,  # 0-5°, 40km/h
    (30, 1): 8.3,  # 5-15°, 30km/h
    (30, 2): 5.6,  # 15-25°, 20km/h
    (30, 3): 2.8,  # 25-35°, 10km/h
    (30, 4): 0.0,  # >35°, 不可通行
    
    # 灌木地
    (40, 0): 8.3,  # 0-5°, 30km/h
    (40, 1): 5.6,  # 5-15°, 20km/h
    (40, 2): 2.8,  # 15-25°, 10km/h
    (40, 3): 1.4,  # 25-35°, 5km/h
    (40, 4): 0.0,  # >35°, 不可通行
    
    # 湿地
    (50, 0): 8.3,  # 0-5°, 30km/h
    (50, 1): 5.6,  # 5-15°, 20km/h
    (50, 2): 2.8,  # 15-25°, 10km/h
    (50, 3): 1.4,  # 25-35°, 5km/h
    (50, 4): 0.0,  # >35°, 不可通行
    
    # 水体
    (60, 0): 0.0,  # 不可通行
    (60, 1): 0.0,
    (60, 2): 0.0,
    (60, 3): 0.0,
    (60, 4): 0.0,
    
    # 建设用地
    (80, 0): 13.9,  # 0-5°, 50km/h
    (80, 1): 11.1,  # 5-15°, 40km/h
    (80, 2): 8.3,  # 15-25°, 30km/h
    (80, 3): 5.6,  # 25-35°, 20km/h
    (80, 4): 0.0,  # >35°, 不可通行
    
    # 裸地
    (90, 0): 11.1,  # 0-5°, 40km/h
    (90, 1): 8.3,  # 5-15°, 30km/h
    (90, 2): 5.6,  # 15-25°, 20km/h
    (90, 3): 2.8,  # 25-35°, 10km/h
    (90, 4): 0.0,  # >35°, 不可通行
    
    # 未分类
    (255, 0): 5.6,  # 默认 20km/h
    (255, 1): 4.2,
    (255, 2): 2.8,
    (255, 3): 1.4,
    (255, 4): 0.0
}

# 坡度分级（度）
SLOPE_LEVELS = [5, 15, 25, 35]

# 轨迹生成参数
TARGET_LENGTH_RANGE = [80000, 120000]  # 目标轨迹长度范围(米)
DEFAULT_TARGET_SPEED = 30  # 默认目标平均速度(km/h)

# 模拟参数
DT = 1.0  # 时间步长(秒)
MAX_ACCELERATION = 1.0  # 最大加速度(m/s^2)
MAX_DECELERATION = -2.0  # 最大减速度(m/s^2)
MAX_TURNING_RATE = 30.0  # 最大转向率(度/秒)

# 输出目录
BATCH_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "synthetic_batch_{}")
EVALUATION_DIR = os.path.join(OUTPUT_DIR, "evaluation_report_{}")

# 创建必要的目录
for directory in [INPUT_DIR, OUTPUT_DIR, INTERMEDIATE_DIR]:
    os.makedirs(directory, exist_ok=True)

# 坡度影响因子
ASPECT_UPHILL_REDUCTION_K = 0.05  # 上坡减速系数
ASPECT_CROSS_REDUCTION_K = 0.01  # 横坡减速系数
MAX_CROSS_SLOPE_DEGREES = 30.0  # 最大横坡角度
MAX_BRAKING_SPEED_DOWNHILL = 5.6  # 下坡最大制动速度（米/秒） 