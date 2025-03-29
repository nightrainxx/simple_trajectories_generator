"""
轨迹可视化模块：展示生成的轨迹

输入：
- landcover_30m_100km.tif：土地覆盖数据
- trajectories/*.csv：轨迹数据文件
- paths/*.csv：原始路径数据文件

输出：
- trajectories_visualization.png：可视化结果
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import rasterio
from config import OUTPUT_DIR
import plot_style  # 导入统一样式设置

def create_landcover_colormap():
    """创建土地覆盖类型的颜色映射"""
    colors = {
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
    return colors

def create_legend_elements():
    """创建图例元素"""
    colors = create_landcover_colormap()
    terrain_elements = [
        Patch(facecolor=color, label=label)
        for code, (color, label) in colors.items()
    ]
    
    # 添加轨迹相关的图例元素
    path_elements = [
        Patch(facecolor='gray', label='原始路径', linestyle='--'),
        Patch(facecolor='blue', label='平滑轨迹'),
        Patch(facecolor='green', label='起点'),
        Patch(facecolor='red', label='终点')
    ]
    
    return terrain_elements, path_elements

def load_trajectories():
    """加载所有轨迹数据"""
    trajectories = []
    trajectories_dir = os.path.join(OUTPUT_DIR, "trajectories")
    for file in sorted(os.listdir(trajectories_dir)):
        if file.startswith("trajectory_") and file.endswith(".csv"):
            path = os.path.join(trajectories_dir, file)
            df = pd.read_csv(path)
            trajectories.append(df)
    print(f"加载了{len(trajectories)}条轨迹")
    return trajectories

def load_original_paths():
    """加载所有原始路径数据"""
    paths = []
    paths_dir = os.path.join(OUTPUT_DIR, "paths")
    for file in sorted(os.listdir(paths_dir)):
        if file.startswith("path_") and file.endswith(".csv"):
            path = os.path.join(paths_dir, file)
            df = pd.read_csv(path)
            paths.append(df)
    print(f"加载了{len(paths)}条原始路径")
    return paths

def plot_trajectory_on_map(ax, trajectory_df, original_path_df, i):
    """在地图上绘制一条轨迹"""
    # 绘制原始路径（虚线）
    ax.plot(original_path_df['col'], original_path_df['row'], 
            color='black', linestyle='--', linewidth=2, alpha=0.8)
    
    # 绘制平滑后的轨迹（实线）
    ax.plot(trajectory_df['col'], trajectory_df['row'],
            color='blue', linewidth=3, alpha=1.0)
    
    # 标记起点和终点
    if i == 0:  # 只在第一条轨迹时添加图例标签
        ax.plot(trajectory_df['col'].iloc[0], trajectory_df['row'].iloc[0], 
                'go', markersize=10, label='起点')
        ax.plot(trajectory_df['col'].iloc[-1], trajectory_df['row'].iloc[-1], 
                'ro', markersize=10, label='终点')
    else:
        ax.plot(trajectory_df['col'].iloc[0], trajectory_df['row'].iloc[0], 
                'go', markersize=10)
        ax.plot(trajectory_df['col'].iloc[-1], trajectory_df['row'].iloc[-1], 
                'ro', markersize=10)

def plot_background(ax, landcover):
    """绘制地图背景"""
    # 打印地表覆盖数据的基本信息
    print("\n地表覆盖数据统计：")
    print(f"数据形状: {landcover.shape}")
    print(f"数据类型: {landcover.dtype}")
    unique, counts = np.unique(landcover, return_counts=True)
    for val, count in zip(unique, counts):
        percentage = count / landcover.size * 100
        print(f"类型 {val}: {count} 像素 ({percentage:.2f}%)")
    
    # 创建颜色映射
    colors = create_landcover_colormap()
    
    # 创建一个从0到255的颜色列表，默认为灰色
    color_list = ['#808080'] * 256
    
    # 为已知的地表类型代码设置对应的颜色
    for code, (color, _) in colors.items():
        color_list[code] = color
    
    cmap = ListedColormap(color_list)
    
    # 绘制地图
    im = ax.imshow(landcover, cmap=cmap, aspect='equal', origin='upper', alpha=0.7)
    return im

def main():
    """主函数"""
    # 加载数据
    print("加载数据...")
    with rasterio.open("data/input/landcover_30m_100km.tif") as src:
        landcover = src.read(1)
        print("\n栅格数据信息：")
        print(f"数据形状: {src.shape}")
        print(f"坐标系统: {src.crs}")
        print(f"变换矩阵: {src.transform}")
        print(f"波段数量: {src.count}")
        print(f"数据类型: {src.dtypes[0]}")
        
        # 检查并打印数据的有效范围
        print(f"有效数据范围: {src.bounds}")
        print(f"无效值: {src.nodata}")
    
    trajectories = load_trajectories()
    original_paths = load_original_paths()
    
    if not trajectories:
        print("未找到轨迹数据")
        return
        
    # 创建图形
    plt.figure(figsize=(15, 12))
    ax = plt.gca()
    
    # 绘制土地覆盖背景
    im = plot_background(ax, landcover)
    
    # 绘制所有轨迹
    for i, (traj, path) in enumerate(zip(trajectories, original_paths)):
        plot_trajectory_on_map(ax, traj, path, i)
    
    # 设置图形标题和标签
    ax.set_title('轨迹规划结果', fontsize=16)
    ax.set_xlabel('东西方向 (像素)', fontsize=16)
    ax.set_ylabel('南北方向 (像素)', fontsize=16)
    
    # 添加图例
    terrain_elements, path_elements = create_legend_elements()
    
    # 创建两个图例，确保它们都显示
    legend1 = ax.legend(handles=path_elements, title='轨迹元素',
                       loc='upper right', bbox_to_anchor=(1.3, 1.0),
                       fontsize=16)
    ax.add_artist(legend1)  # 保持第一个图例
    
    legend2 = ax.legend(handles=terrain_elements, title='地形类型',
                       loc='center right', bbox_to_anchor=(1.3, 0.5),
                       fontsize=16)
    ax.add_artist(legend2)  # 添加第二个图例
    
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 保存结果
    output_path = os.path.join(OUTPUT_DIR, "trajectories_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存至: {output_path}")

if __name__ == "__main__":
    main() 