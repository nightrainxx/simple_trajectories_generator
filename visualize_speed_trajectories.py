"""
速度轨迹可视化脚本

输入文件:
- landcover_30m_100km.tif - 土地覆盖数据
- improved_trajectories/*.csv - 改进版轨迹数据
- improved_paths/*.csv - 改进版路径数据

输出文件:
- speed_visualization.png - 速度可视化结果图
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import glob
import rasterio
from matplotlib.collections import LineCollection

from config import DATA_DIR, OUTPUT_DIR

# 定义战略位置及其名称
STRATEGIC_LOCATIONS = [
    {"name": "军事基地", "coords": (113, 481)},
    {"name": "港口", "coords": (2682, 1095)},
    {"name": "机场", "coords": (1427, 1812)},
    {"name": "城市中心", "coords": (2149, 2577)}
]

# 定义土地覆盖类型的颜色方案 - 按照用户指定的配色
LANDCOVER_COLORS = {
    10: {'name': '水域', 'color': '#0077BE'},       # 蓝色
    20: {'name': '湿地', 'color': '#80CCFF'},       # 浅蓝色
    30: {'name': '草地', 'color': '#90EE90'},       # 浅绿色
    40: {'name': '灌木地', 'color': '#228B22'},     # 深绿色
    50: {'name': '建筑用地', 'color': '#CD5C5C'},   # 红褐色
    60: {'name': '农田', 'color': '#FFD700'},       # 金黄色
    80: {'name': '森林', 'color': '#006400'},       # 深绿色
    90: {'name': '荒地', 'color': '#DEB887'},       # 棕色
    255: {'name': '未分类', 'color': '#808080'}     # 灰色
}

def load_data():
    """加载数据"""
    # 加载土地覆盖数据
    landcover_path = os.path.join(DATA_DIR, "input", "landcover_30m_100km.tif")
    print(f"正在加载土地覆盖数据: {landcover_path}")
    
    with rasterio.open(landcover_path) as src:
        landcover_data = src.read(1)
        transform = src.transform
        crs = src.crs
        
    print(f"土地覆盖数据形状: {landcover_data.shape}")
    print(f"数据类型: {landcover_data.dtype}")
    print(f"坐标系统: {crs}")
    
    # 加载改进的轨迹数据
    trajectories = []
    trajectories_dir = os.path.join(OUTPUT_DIR, "improved_trajectories")
    if os.path.exists(trajectories_dir):
        trajectory_files = glob.glob(os.path.join(trajectories_dir, "*.csv"))
        for file in trajectory_files:
            try:
                traj = pd.read_csv(file)
                traj['file'] = os.path.basename(file)
                trajectories.append(traj)
            except Exception as e:
                print(f"无法读取轨迹文件 {file}: {e}")
        
        print(f"已加载 {len(trajectories)} 条改进轨迹")
    else:
        print(f"警告: 找不到轨迹目录 {trajectories_dir}")
    
    # 加载改进的路径数据
    paths = []
    paths_dir = os.path.join(OUTPUT_DIR, "improved_paths")
    if os.path.exists(paths_dir):
        path_files = glob.glob(os.path.join(paths_dir, "*.csv"))
        for file in path_files:
            try:
                path = pd.read_csv(file)
                path['file'] = os.path.basename(file)
                paths.append(path)
            except Exception as e:
                print(f"无法读取路径文件 {file}: {e}")
        
        print(f"已加载 {len(paths)} 条改进路径")
    else:
        print(f"警告: 找不到路径目录 {paths_dir}")
    
    return landcover_data, trajectories, paths

def create_rgb_image(landcover_data):
    """从土地覆盖数据创建RGB图像"""
    rows, cols = landcover_data.shape
    rgb_image = np.zeros((rows, cols, 4), dtype=np.uint8)
    
    # 创建颜色列表
    unique_values = np.unique(landcover_data)
    for value in unique_values:
        # 获取颜色
        if value in LANDCOVER_COLORS:
            color = LANDCOVER_COLORS[value]['color']
        else:
            print(f"警告: 未知土地覆盖类型 {value}，使用灰色")
            color = '#808080'  # 默认灰色
        
        # 将十六进制颜色转换为RGB值
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        
        # 创建掩码并应用颜色
        mask = (landcover_data == value)
        rgb_image[mask, 0] = r
        rgb_image[mask, 1] = g
        rgb_image[mask, 2] = b
        rgb_image[mask, 3] = 220  # 设置透明度为220/255
    
    return rgb_image

def plot_speed_trajectories(trajectories, ax):
    """绘制固定宽度的轨迹，使用颜色表示速度"""
    if not trajectories:
        return
    
    # 创建速度颜色映射
    speed_cmap = LinearSegmentedColormap.from_list(
        'speed_cmap', 
        [(0, 'blue'),      # 低速 - 蓝色
         (0.3, 'green'),   # 中低速 - 绿色
         (0.6, 'yellow'),  # 中速 - 黄色
         (0.8, 'orange'),  # 中高速 - 橙色
         (1.0, 'red')],    # 高速 - 红色
        N=256
    )
    
    # 为每条轨迹绘制线段集合
    for i, traj in enumerate(trajectories):
        # 提取坐标和速度数据
        points = np.array([traj['col'].values, traj['row'].values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        speeds = traj['speed'].values[:-1]  # 每段的速度取起点的速度
        
        # 找出最大和最小速度用于规范化
        speed_min = np.min(speeds)
        speed_max = np.max(speeds)
        
        # 如果速度都相同，避免除以零
        if speed_max == speed_min:
            norm_speeds = np.ones_like(speeds)
        else:
            # 规范化速度为0-1范围
            norm_speeds = (speeds - speed_min) / (speed_max - speed_min)
        
        # 创建带颜色映射的线段集合
        lc = LineCollection(segments, cmap=speed_cmap, norm=plt.Normalize(0, 1))
        lc.set_array(norm_speeds)
        lc.set_linewidth(2.5)  # 设置固定宽度
        
        # 添加到图中
        line = ax.add_collection(lc)
        
        # 记录速度范围
        print(f"轨迹 {i+1} 的速度范围: {speed_min:.2f} - {speed_max:.2f}")
    
    # 添加颜色条显示速度
    cbar = plt.colorbar(line, ax=ax)
    cbar.set_label('速度 (像素/秒)')
    
    return speed_min, speed_max

def plot_trajectory_stats(trajectories, ax):
    """绘制轨迹统计信息"""
    if not trajectories:
        return
    
    # 计算并显示每条轨迹的统计信息
    stats_text = []
    
    for i, traj in enumerate(trajectories):
        # 获取起点和终点
        start = (traj['row'].iloc[0], traj['col'].iloc[0])
        end = (traj['row'].iloc[-1], traj['col'].iloc[-1])
        
        # 计算总距离
        distances = np.sqrt(
            np.diff(traj['row'])**2 + np.diff(traj['col'])**2
        ) * 30  # 每像素30米
        total_distance = np.sum(distances)
        
        # 计算行程时间
        travel_time = np.max(traj['timestamp']) / 3600  # 转换为小时
        
        # 计算平均和最大速度
        avg_speed = total_distance / (travel_time * 1000) * 3600  # 转换为km/h
        max_speed = np.max(traj['speed']) * 30 * 3.6  # 像素/秒 * 30米/像素 * 3.6 = km/h
        
        # 添加信息到数组
        stats_text.append(
            f"轨迹 {i+1}:\n"
            f"  起点: ({start[0]:.0f}, {start[1]:.0f})\n"
            f"  终点: ({end[0]:.0f}, {end[1]:.0f})\n"
            f"  距离: {total_distance/1000:.2f} km\n"
            f"  时间: {travel_time:.2f} h\n"
            f"  平均速度: {avg_speed:.2f} km/h\n"
            f"  最大速度: {max_speed:.2f} km/h"
        )
    
    # 添加文本框
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.02, 0.02, '\n\n'.join(stats_text), transform=ax.transAxes, 
            fontsize=9, verticalalignment='bottom', bbox=props)

def main():
    """主函数"""
    # 加载数据
    landcover_data, trajectories, paths = load_data()
    
    # 检查是否有轨迹数据
    if not trajectories:
        print("错误: 没有找到轨迹数据，无法生成可视化")
        return
    
    # 创建RGB图像
    rgb_image = create_rgb_image(landcover_data)
    
    # 开始绘图
    plt.figure(figsize=(14, 12))
    ax = plt.subplot(111)
    
    # 显示RGB土地覆盖图像
    plt.imshow(np.transpose(rgb_image, (1, 0, 2)))
    
    # 绘制轨迹并标注速度
    speed_min, speed_max = plot_speed_trajectories(trajectories, ax)
    
    # 绘制原始路径
    if paths:
        for path in paths:
            plt.plot(path['col'], path['row'], '--', color='white', linewidth=0.8, alpha=0.6)
    
    # 标记战略位置
    for loc in STRATEGIC_LOCATIONS:
        plt.plot(loc['coords'][1], loc['coords'][0], 'o', color='black', markersize=10, markeredgecolor='white')
        plt.text(
            loc['coords'][1] + 30, loc['coords'][0], 
            loc['name'], 
            color='black', 
            fontsize=12, 
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
        )
    
    # 添加轨迹统计信息
    plot_trajectory_stats(trajectories, ax)
    
    # 创建图例
    patches = []
    
    # 土地覆盖类型图例
    unique_values = np.unique(landcover_data)
    for value in unique_values:
        if value in LANDCOVER_COLORS:
            color = LANDCOVER_COLORS[value]['color']
            name = LANDCOVER_COLORS[value]['name']
            patches.append(mpatches.Patch(color=color, label=f"{name} ({value})"))
    
    # 轨迹图例
    patches.append(plt.Line2D([0], [0], color='white', linestyle='--', linewidth=0.8, alpha=0.6, label='原始路径'))
    patches.append(plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='black', markeredgecolor='white', markersize=10, linestyle='None', label='战略位置'))
    
    # 添加图例
    plt.legend(
        handles=patches, 
        loc='upper right', 
        bbox_to_anchor=(1.0, 1.0), 
        fontsize=9
    )
    
    # 添加标题和轴标签
    plt.title('基于速度的战略轨迹可视化 (20%关键点比例)', fontsize=15)
    plt.xlabel('列 (东-西方向)')
    plt.ylabel('行 (南-北方向)')
    
    # 保存图像
    output_path = os.path.join(OUTPUT_DIR, "speed_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"速度轨迹可视化结果已保存到: {output_path}")
    
    # 关闭图像
    plt.close()

if __name__ == "__main__":
    main() 