"""
可视化战略轨迹

输入：
- landcover_30m_100km.tif：土地覆盖数据
- strategic_trajectories/*.csv：战略轨迹数据
- strategic_paths/*.csv：战略路径数据

输出：
- strategic_trajectories_visualization.png：战略轨迹可视化结果
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from matplotlib.patches import Patch
from config import OUTPUT_DIR

# 战略位置坐标
STRATEGIC_LOCATIONS = [
    (113, 481),     # 位置1: 建筑用地
    (2682, 1095),   # 位置2: 建筑用地
    (1427, 1812),   # 位置3: 建筑用地
    (2149, 2577)    # 位置4: 农田
]

# 战略位置名称
STRATEGIC_NAMES = [
    "战略位置1 (建筑用地)",
    "战略位置2 (建筑用地)",
    "战略位置3 (建筑用地)",
    "战略位置4 (农田)"
]

# 定义土地覆盖类型颜色方案
LANDCOVER_COLORS = {
    10: ('#0077BE', '水域'),      # 蓝色
    20: ('#80CCFF', '湿地'),      # 浅蓝色
    30: ('#90EE90', '草地'),      # 浅绿色
    40: ('#228B22', '灌木地'),    # 深绿色
    50: ('#CD5C5C', '建筑用地'),  # 红褐色
    60: ('#FFD700', '农田'),      # 金黄色
    80: ('#006400', '森林'),      # 深绿色
    90: ('#DEB887', '荒地'),      # 棕色
    255: ('#808080', '未分类')    # 灰色
}

def load_data():
    """加载所有数据"""
    print("加载数据...")
    
    # 加载地表覆盖数据
    with rasterio.open("data/input/landcover_30m_100km.tif") as src:
        landcover = src.read(1)
        print(f"\n地表覆盖数据信息:")
        print(f"形状: {landcover.shape}")
        print(f"类型: {landcover.dtype}")
        print(f"坐标系统: {src.crs}")
        print(f"无效值: {src.nodata}")
        
        # 数据统计
        unique, counts = np.unique(landcover, return_counts=True)
        print("\n地表类型分布:")
        for val, count in zip(unique, counts):
            percentage = count / landcover.size * 100
            name = LANDCOVER_COLORS.get(val, ('未知', '未知'))[1]
            print(f"类型 {val} ({name}): {count} 像素 ({percentage:.2f}%)")
    
    # 加载战略轨迹数据
    trajectories = []
    trajectories_dir = os.path.join(OUTPUT_DIR, "strategic_trajectories")
    if os.path.exists(trajectories_dir):
        for file in sorted(os.listdir(trajectories_dir)):
            if file.startswith("strategic_trajectory_") and file.endswith(".csv"):
                path = os.path.join(trajectories_dir, file)
                df = pd.read_csv(path)
                trajectories.append(df)
        print(f"加载了{len(trajectories)}条战略轨迹")
    else:
        print(f"警告：未找到战略轨迹目录 {trajectories_dir}")
    
    # 加载原始路径数据
    paths = []
    paths_dir = os.path.join(OUTPUT_DIR, "strategic_paths")
    if os.path.exists(paths_dir):
        for file in sorted(os.listdir(paths_dir)):
            if file.startswith("strategic_path_") and file.endswith(".csv"):
                path = os.path.join(paths_dir, file)
                df = pd.read_csv(path)
                paths.append(df)
        print(f"加载了{len(paths)}条战略路径")
    else:
        print(f"警告：未找到战略路径目录 {paths_dir}")
    
    return landcover, trajectories, paths

def create_rgb_image(landcover):
    """将地表覆盖数据转换为RGB图像"""
    # 创建一个RGB图像
    height, width = landcover.shape
    rgb_img = np.ones((height, width, 4), dtype=np.float32)  # RGBA

    # 为每种地表类型设置颜色
    for code, (color_hex, _) in LANDCOVER_COLORS.items():
        # 将十六进制颜色转换为RGB
        rgb = mcolors.hex2color(color_hex)
        
        # 找到该类型的所有像素
        mask = (landcover == code)
        
        # 设置RGB颜色 (前3个通道)
        for i in range(3):
            rgb_img[mask, i] = rgb[i]
    
    # 设置透明度 (第4个通道)
    rgb_img[:, :, 3] = 0.7  # 70% 不透明度
    
    return rgb_img

def main():
    """主函数"""
    # 加载数据
    landcover, trajectories, paths = load_data()
    
    if not trajectories:
        print("未找到战略轨迹数据")
        return
    
    # 创建RGB图像
    rgb_img = create_rgb_image(landcover)
    
    # 创建图形
    plt.figure(figsize=(16, 13))
    ax = plt.gca()
    
    # 绘制地表覆盖背景
    ax.imshow(rgb_img)
    
    # 绘制所有轨迹
    colors = ['blue', 'green', 'red', 'purple']  # 不同轨迹使用不同颜色
    for i, (traj, path) in enumerate(zip(trajectories, paths)):
        color_idx = i % len(colors)
        
        # 绘制原始路径（黑色虚线）
        ax.plot(path['col'], path['row'], 
                color='black', linestyle='--', linewidth=2.5, alpha=0.8)
        
        # 绘制平滑后的轨迹
        ax.plot(traj['col'], traj['row'], 
                color=colors[color_idx], linewidth=3.5, alpha=1.0,
                label=f"轨迹 {i+1}")
        
        # 标记起点
        ax.plot(traj['col'].iloc[0], traj['row'].iloc[0], 
                'o', color=colors[color_idx], markersize=12, markeredgecolor='black')
    
    # 标记战略位置
    for i, (row, col) in enumerate(STRATEGIC_LOCATIONS):
        ax.plot(col, row, '*', color='red', markersize=18, markeredgecolor='black')
        ax.text(col+20, row+20, STRATEGIC_NAMES[i], fontsize=14, color='black',
               bbox=dict(facecolor='white', alpha=0.7))
    
    # 添加图例
    legend_elements = []
    
    # 添加地表类型图例
    for code in sorted(LANDCOVER_COLORS.keys()):
        color_hex, name = LANDCOVER_COLORS[code]
        legend_elements.append(Patch(facecolor=color_hex, label=f"{name} ({code})"))
    
    # 添加轨迹图例
    legend_elements.extend([
        Patch(facecolor='black', label='原始路径'),
        Patch(facecolor='red', label='战略位置')
    ])
    
    # 放置图例
    plt.legend(handles=legend_elements, 
               loc='center left', 
               bbox_to_anchor=(1.02, 0.5),
               fontsize=12)
    
    # 添加标题和标签
    plt.title('战略轨迹规划结果', fontsize=18)
    plt.xlabel('东西方向 (像素)', fontsize=14)
    plt.ylabel('南北方向 (像素)', fontsize=14)
    
    # 显示网格
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 保存结果
    output_path = os.path.join(OUTPUT_DIR, 'strategic_trajectories_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存至: {output_path}")

if __name__ == "__main__":
    main() 