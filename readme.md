项目：基于环境约束的合成轨迹生成器 (简化版)

1. 项目概述与目标

目标: 开发一个工具，用于在给定的地理环境（DEM、坡度、土地覆盖）中，根据预设的移动规则，生成从指定起点到终点的合成轨迹。

核心要求:

- 环境感知: 生成的轨迹必须受地形坡度、土地覆盖类型的影响，体现在移动速度和路径选择上。
- 用户指定起终点: 用户能够指定轨迹的起始和结束位置（以像素坐标形式）。
- 可控轨迹长度: 生成的轨迹总长度应能控制在 80 公里至 120 公里范围内。
- 可控平均速度: 生成轨迹的平均速度应能大致控制在用户期望的范围内。

2. 安装与配置

2.1 环境要求
- Python 3.x
- conda虚拟环境: wargame

2.2 安装依赖
```bash
# 激活虚拟环境
conda activate wargame

# 安装依赖包
pip install -r requirements.txt
```

3. 数据准备

3.1 输入数据
请将以下数据文件放置在 `data/input` 目录下:

- dem_30m_100km.tif: 数字高程模型 (WGS 84, ~30m分辨率, 范围100x100km)
- slope_30m_100km.tif: 坡度数据 (单位：度, 与DEM对齐)
- landcover_30m_100km.tif: 土地覆盖数据 (11类分类编码, 与DEM对齐)

3.2 输出数据
程序会在 `data/output` 目录下生成以下文件:

- typical_speed_map.tif: 每个像素的典型行驶速度图 (m/s)
- cost_map.tif: 用于A*路径规划的通行成本图
- trajectory.csv: 生成的轨迹数据,包含时间戳和位置信息

4. 使用方法

4.1 基本用法
```bash
# 激活虚拟环境
conda activate wargame

# 运行程序
python main.py --start_row 100 --start_col 100 --end_row 200 --end_col 200 [--target_speed 30]
```

4.2 参数说明
- start_row: 起点行号
- start_col: 起点列号
- end_row: 终点行号
- end_col: 终点列号
- target_speed: 目标平均速度(km/h),默认30

5. 项目结构
```
.
├── config.py           # 配置文件
├── data/              # 数据目录
│   ├── input/        # 输入数据
│   └── output/       # 输出数据
├── environment.py     # 环境模块
├── main.py           # 主程序入口
├── map_generator.py   # 地图生成器
├── path_planner.py   # 路径规划器
├── readme.md         # 项目说明
├── requirements.txt   # 依赖包列表
├── trajectory_generator.py  # 轨迹生成器
└── utils.py          # 工具函数
```

6. 注意事项

- 确保输入数据文件存在且格式正确
- 起点和终点坐标必须在栅格范围内
- 起点和终点不能位于不可通行区域(如水体)
- 生成的轨迹长度应在80-120公里范围内
- 如果未找到满足要求的路径,程序会报错

7. 开发团队

- 开发者: [您的名字]
- 联系方式: [您的邮箱]

8. 版权声明

本项目仅用于学习和研究目的。未经授权,请勿用于商业用途。