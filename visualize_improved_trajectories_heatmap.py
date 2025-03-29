"""
改进轨迹热力图可视化脚本

输入文件:
- landcover_30m_100km.tif - 土地覆盖数据
- improved_trajectories/*.csv - 改进版轨迹数据
- improved_paths/*.csv - 改进版路径数据

输出文件:
- improved_heatmap_visualization.png - 改进版热力图可视化结果
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import glob
import rasterio
from scipy.ndimage import gaussian_filter

from config import DATA_DIR, OUTPUT_DIR

# 定义战略位置及其名称
STRATEGIC_LOCATIONS = [
    {"name": "军事基地", "coords": (113, 481)},
    {"name": "港口", "coords": (2682, 1095)},
    {"name": "机场", "coords": (1427, 1812)},
    {"name": "城市中心", "coords": (2149, 2577)}
]

# 定义土地覆盖类型的颜色方案
LANDCOVER_COLORS = {
    10: {'name': '水域', 'color': '#4169E1'},       # 蓝色
    20: {'name': '湿地', 'color': '#008080'},       # 青色
    30: {'name': '草地', 'color': '#90EE90'},       # 淡绿色
    40: {'name': '灌木丛', 'color': '#FFD700'},     # 金色
    50: {'name': '建成区', 'color': '#FF4500'},     # 橙红色
    60: {'name': '农田', 'color': '#FFFF00'},       # 黄色
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
    print(f"变换矩阵:\n{transform}")
    
    # 检查无效值
    invalid_mask = ~np.isfinite(landcover_data)
    if np.any(invalid_mask):
        print(f"警告: 发现 {np.sum(invalid_mask)} 个无效像素")
    
    # 统计土地覆盖类型的分布
    unique_values = np.unique(landcover_data)
    print(f"土地覆盖类型 (共 {len(unique_values)} 种): {unique_values}")
    
    total_pixels = landcover_data.size
    for value in unique_values:
        pixel_count = np.sum(landcover_data == value)
        percentage = (pixel_count / total_pixels) * 100
        
        if value in LANDCOVER_COLORS:
            type_name = LANDCOVER_COLORS[value]['name']
        else:
            type_name = f"未知类型 {value}"
            
        print(f"类型 {value} ({type_name}): {pixel_count} 像素 ({percentage:.2f}%)")
    
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
        rgb_image[mask, 3] = 200  # 设置透明度为200/255，使热力图可见
    
    return rgb_image

def create_heatmap(trajectories, shape):
    """从轨迹数据创建热力图"""
    if not trajectories:
        print("没有轨迹数据，无法创建热力图")
        return None
    
    # 创建空白热力图
    heatmap = np.zeros(shape, dtype=np.float32)
    
    # 统计每个像素的轨迹点计数
    total_points = 0
    for traj in trajectories:
        for _, row in traj.iterrows():
            x, y = int(row['row']), int(row['col'])
            if 0 <= x < shape[0] and 0 <= y < shape[1]:
                heatmap[x, y] += 1
                total_points += 1
    
    print(f"热力图总计 {total_points} 个点")
    
    # 应用高斯平滑
    sigma = 10  # 平滑程度，可调
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    
    # 标准化热力图值，使其在[0,1]范围内
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    return heatmap

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
        
        # 计算平均速度
        avg_speed = total_distance / (travel_time * 1000) * 3600  # 转换为km/h
        
        # 添加信息到数组
        stats_text.append(
            f"轨迹 {i+1}:\n"
            f"  起点: ({start[0]:.0f}, {start[1]:.0f})\n"
            f"  终点: ({end[0]:.0f}, {end[1]:.0f})\n"
            f"  距离: {total_distance/1000:.2f} km\n"
            f"  时间: {travel_time:.2f} h\n"
            f"  平均速度: {avg_speed:.2f} km/h"
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
    
    # 创建热力图
    heatmap = create_heatmap(trajectories, landcover_data.shape)
    
    # 开始绘图
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111)
    
    # 显示RGB土地覆盖图像
    plt.imshow(np.transpose(rgb_image, (1, 0, 2)))
    
    # 显示热力图
    if heatmap is not None:
        # 创建从蓝到红的热力图颜色映射
        heatmap_colors = LinearSegmentedColormap.from_list(
            'custom_heatmap', 
            [(0, (0, 0, 0, 0)),       # 低值为完全透明
             (0.3, (0, 0, 1, 0.7)),   # 低值为蓝色，70%不透明度
             (0.6, (1, 1, 0, 0.7)),   # 中值为黄色，70%不透明度
             (0.9, (1, 0, 0, 0.7)),   # 高值为红色，70%不透明度
             (1.0, (0.5, 0, 0, 0.9))], # 最高值为深红色，90%不透明度
            N=256
        )
        
        # 绘制热力图，使用转置使坐标系匹配
        plt.imshow(
            np.transpose(heatmap), 
            alpha=0.7, 
            cmap=heatmap_colors
        )
        
        # 添加颜色条
        cbar = plt.colorbar(shrink=0.8)
        cbar.set_label('轨迹密度')
    
    # 绘制原始路径
    if paths:
        for path in paths:
            plt.plot(path['col'], path['row'], 'w-', linewidth=1, alpha=0.3)
    
    # 标记战略位置
    for loc in STRATEGIC_LOCATIONS:
        plt.plot(loc['coords'][1], loc['coords'][0], 'ko', markersize=8)
        plt.text(
            loc['coords'][1] + 30, loc['coords'][0], 
            loc['name'], 
            color='black', 
            fontsize=12, 
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
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
    patches.append(plt.Line2D([0], [0], color='w', linewidth=1, alpha=0.5, label='原始路径'))
    patches.append(mpatches.Patch(color='blue', alpha=0.5, label='低密度区'))
    patches.append(mpatches.Patch(color='yellow', alpha=0.5, label='中密度区'))
    patches.append(mpatches.Patch(color='red', alpha=0.5, label='高密度区'))
    patches.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label='战略位置'))
    
    # 添加图例
    plt.legend(
        handles=patches, 
        loc='upper right', 
        bbox_to_anchor=(1.0, 1.0), 
        fontsize=9
    )
    
    # 添加标题和轴标签
    plt.title('改进战略轨迹热力图分析', fontsize=15)
    plt.xlabel('列 (东-西方向)')
    plt.ylabel('行 (南-北方向)')
    
    # 保存图像
    output_path = os.path.join(OUTPUT_DIR, "improved_heatmap_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"热力图可视化结果已保存到: {output_path}")
    
    # 关闭图像
    plt.close()

if __name__ == "__main__":
    main() 