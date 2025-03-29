"""
轨迹可视化脚本

用于可视化批量生成的轨迹
输入：
- landcover_30m_100km.tif：土地覆盖数据
- batch_trajectories/trajectories/*.csv：轨迹数据

输出：
- batch_trajectories/visualizations/：轨迹可视化图片
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from config import LANDCOVER_PATH, OUTPUT_DIR
import random
from tqdm import tqdm

# 土地覆盖类型颜色映射
LANDCOVER_COLORS = {
    10: (0.0, 0.5, 1.0),    # 水体 - 蓝色
    20: (0.0, 0.8, 0.8),    # 湿地 - 青色
    30: (0.5, 0.8, 0.0),    # 草地 - 浅绿色
    40: (0.0, 0.6, 0.0),    # 灌木 - 深绿色
    50: (0.6, 0.0, 0.0),    # 建筑 - 深红色
    60: (1.0, 0.8, 0.0),    # 农田 - 黄色
    80: (0.0, 0.3, 0.0),    # 森林 - 深绿色
    90: (0.8, 0.7, 0.6),    # 裸地 - 棕色
    255: (0.5, 0.5, 0.5)    # 未分类 - 灰色
}

def load_landcover():
    """加载土地覆盖数据"""
    print("加载土地覆盖数据...")
    with rasterio.open(LANDCOVER_PATH) as src:
        landcover = src.read(1)
        transform = src.transform
    return landcover, transform

def create_landcover_rgb(landcover):
    """从土地覆盖数据创建RGB图像"""
    print("创建土地覆盖RGB图像...")
    height, width = landcover.shape
    rgb = np.zeros((height, width, 4), dtype=np.float32)
    
    # 填充颜色
    for lc_type, color in LANDCOVER_COLORS.items():
        mask = (landcover == lc_type)
        rgb[mask, 0] = color[0]  # R
        rgb[mask, 1] = color[1]  # G
        rgb[mask, 2] = color[2]  # B
        rgb[mask, 3] = 0.5       # A - 半透明
    
    return rgb

def visualize_trajectories(num_trajectories=10, start_index=1):
    """可视化轨迹"""
    # 加载土地覆盖数据
    landcover, transform = load_landcover()
    rgb_image = create_landcover_rgb(landcover)
    
    # 创建输出目录
    output_dir = os.path.join(OUTPUT_DIR, "batch_trajectories", "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # 轨迹文件目录
    trajectories_dir = os.path.join(OUTPUT_DIR, "batch_trajectories", "trajectories")
    
    # 获取所有轨迹文件
    trajectory_files = [f for f in os.listdir(trajectories_dir) if f.endswith(".csv")]
    
    # 如果文件数量大于请求的数量，随机选择
    if len(trajectory_files) > num_trajectories:
        selected_files = random.sample(trajectory_files, num_trajectories)
    else:
        selected_files = trajectory_files
        print(f"警告：只找到{len(trajectory_files)}个轨迹文件，少于请求的{num_trajectories}个")
    
    # 为每条轨迹分配一种颜色
    cmap = plt.cm.jet
    colors = [cmap(i/len(selected_files)) for i in range(len(selected_files))]
    
    # 创建可视化
    print(f"可视化{len(selected_files)}条轨迹...")
    
    # 单个轨迹可视化
    for i, file_name in enumerate(tqdm(selected_files)):
        # 加载轨迹数据
        df = pd.read_csv(os.path.join(trajectories_dir, file_name))
        
        # 创建图形
        plt.figure(figsize=(12, 10))
        
        # 绘制背景
        plt.imshow(rgb_image)
        
        # 绘制轨迹
        plt.plot(df['col'], df['row'], '-', color=colors[i], linewidth=2, alpha=0.8)
        
        # 标记起点和终点
        plt.plot(df['col'].iloc[0], df['row'].iloc[0], 'o', color='white', markersize=8)
        plt.plot(df['col'].iloc[-1], df['row'].iloc[-1], 's', color='black', markersize=8)
        
        # 添加轨迹信息
        duration_hours = df['timestamp'].iloc[-1] / 3600
        distance_km = sum(
            np.sqrt(
                (df['row'].iloc[i+1] - df['row'].iloc[i])**2 +
                (df['col'].iloc[i+1] - df['col'].iloc[i])**2
            ) * 30  # 30米分辨率
            for i in range(len(df)-1)
        ) / 1000
        avg_speed = distance_km / duration_hours
        
        plt.title(f"轨迹 {file_name.split('_')[1].split('.')[0]}\n"
                 f"持续时间: {duration_hours:.2f}小时, 距离: {distance_km:.2f}公里, "
                 f"平均速度: {avg_speed:.2f}公里/小时", fontsize=12)
        
        # 保存图像
        output_path = os.path.join(output_dir, f"trajectory_{file_name.split('_')[1].split('.')[0]}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 综合可视化 - 在一张图上显示所有轨迹
    plt.figure(figsize=(16, 14))
    plt.imshow(rgb_image)
    
    # 绘制所有轨迹
    for i, file_name in enumerate(selected_files):
        df = pd.read_csv(os.path.join(trajectories_dir, file_name))
        plt.plot(df['col'], df['row'], '-', color=colors[i], linewidth=1.5, alpha=0.7, 
                 label=f"轨迹 {file_name.split('_')[1].split('.')[0]}")
        plt.plot(df['col'].iloc[0], df['row'].iloc[0], 'o', color=colors[i], markersize=6)
        plt.plot(df['col'].iloc[-1], df['row'].iloc[-1], 's', color=colors[i], markersize=6)
    
    plt.title(f"批量生成的{len(selected_files)}条轨迹可视化", fontsize=14)
    
    # 添加图例，放在图像外部
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # 保存综合可视化
    output_path = os.path.join(output_dir, f"combined_{num_trajectories}_trajectories.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化完成，结果保存在: {output_dir}")

def visualize_speeds(num_trajectories=5, start_index=1):
    """可视化轨迹速度分布"""
    # 加载土地覆盖数据
    landcover, transform = load_landcover()
    rgb_image = create_landcover_rgb(landcover)
    
    # 创建输出目录
    output_dir = os.path.join(OUTPUT_DIR, "batch_trajectories", "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # 轨迹文件目录
    trajectories_dir = os.path.join(OUTPUT_DIR, "batch_trajectories", "trajectories")
    
    # 获取所有轨迹文件
    trajectory_files = [f for f in os.listdir(trajectories_dir) if f.endswith(".csv")]
    
    # 如果文件数量大于请求的数量，随机选择
    if len(trajectory_files) > num_trajectories:
        selected_files = random.sample(trajectory_files, num_trajectories)
    else:
        selected_files = trajectory_files
        print(f"警告：只找到{len(trajectory_files)}个轨迹文件，少于请求的{num_trajectories}个")
    
    # 创建自定义颜色映射 - 从蓝色(慢)到红色(快)
    speed_cmap = LinearSegmentedColormap.from_list("speed_cmap", 
                                                [(0, 'blue'), (0.5, 'green'), (1, 'red')])
    
    # 创建可视化
    print(f"可视化{len(selected_files)}条轨迹的速度分布...")
    
    # 使用子图布局
    fig, axes = plt.subplots(len(selected_files), 1, figsize=(12, 6*len(selected_files)))
    if len(selected_files) == 1:
        axes = [axes]
    
    for i, (file_name, ax) in enumerate(zip(selected_files, axes)):
        # 加载轨迹数据
        df = pd.read_csv(os.path.join(trajectories_dir, file_name))
        
        # 找到速度的最小值和最大值
        min_speed = df['speed'].min()
        max_speed = df['speed'].max()
        
        # 标准化速度值到0-1范围用于着色
        norm_speeds = (df['speed'] - min_speed) / (max_speed - min_speed)
        
        # 在此子图上绘制背景
        ax.imshow(rgb_image)
        
        # 绘制轨迹，使用速度着色
        for j in range(len(df) - 1):
            x = [df['col'].iloc[j], df['col'].iloc[j+1]]
            y = [df['row'].iloc[j], df['row'].iloc[j+1]]
            ax.plot(x, y, '-', color=speed_cmap(norm_speeds.iloc[j]), linewidth=2)
        
        # 标记起点和终点
        ax.plot(df['col'].iloc[0], df['row'].iloc[0], 'o', color='white', markersize=8)
        ax.plot(df['col'].iloc[-1], df['row'].iloc[-1], 's', color='black', markersize=8)
        
        # 添加轨迹信息
        duration_hours = df['timestamp'].iloc[-1] / 3600
        distance_km = sum(
            np.sqrt(
                (df['row'].iloc[j+1] - df['row'].iloc[j])**2 +
                (df['col'].iloc[j+1] - df['col'].iloc[j])**2
            ) * 30  # 30米分辨率
            for j in range(len(df)-1)
        ) / 1000
        avg_speed = distance_km / duration_hours
        
        ax.set_title(f"轨迹 {file_name.split('_')[1].split('.')[0]} - 速度分布\n"
                   f"持续时间: {duration_hours:.2f}小时, 距离: {distance_km:.2f}公里, "
                   f"平均速度: {avg_speed:.2f}公里/小时\n"
                   f"速度范围: {min_speed*3.6:.2f}-{max_speed*3.6:.2f}公里/小时", fontsize=12)
    
    # 添加总标题
    fig.suptitle("轨迹速度分布可视化", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # 保存速度可视化
    output_path = os.path.join(output_dir, f"speed_distribution_{num_trajectories}_trajectories.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"速度可视化完成，结果保存在: {output_dir}")

def main():
    """主函数"""
    # 可视化10条随机轨迹
    visualize_trajectories(num_trajectories=10)
    
    # 可视化5条轨迹的速度分布
    visualize_speeds(num_trajectories=5)

if __name__ == "__main__":
    main() 