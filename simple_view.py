"""
简单的地图可视化脚本
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from config import OUTPUT_DIR

# 定义固定的颜色映射，每种地表类型使用不同且明显的颜色
COLORS = {
    10: '#FF0000',  # 红色 - 耕地
    20: '#00FF00',  # 绿色 - 林地
    30: '#0000FF',  # 蓝色 - 草地
    40: '#FFFF00',  # 黄色 - 灌木地
    50: '#FF00FF',  # 紫色 - 湿地
    60: '#00FFFF',  # 青色 - 水体
    80: '#FF8000',  # 橙色 - 建设用地
    90: '#8000FF',  # 紫蓝色 - 裸地
    255: '#808080'  # 灰色 - 未分类
}

# 地表类型名称
LANDCOVER_NAMES = {
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

def main():
    """主函数"""
    # 打开栅格文件
    print("加载地表覆盖数据...")
    with rasterio.open("data/input/landcover_30m_100km.tif") as src:
        landcover = src.read(1)
        print(f"数据形状: {landcover.shape}")
        print(f"数据类型: {landcover.dtype}")
        unique, counts = np.unique(landcover, return_counts=True)
        for val, count in zip(unique, counts):
            percentage = count / landcover.size * 100
            print(f"类型 {val} ({LANDCOVER_NAMES[val]}): {count} 像素 ({percentage:.2f}%)")
    
    # 创建一个RGB图像
    rgb_img = np.zeros((*landcover.shape, 3), dtype=np.uint8)
    
    # 为每个地表类型设置对应的RGB颜色
    for code, color_hex in COLORS.items():
        # 将十六进制颜色转换为RGB
        color_rgb = mcolors.hex2color(color_hex)
        # 找到该地表类型的所有像素
        mask = (landcover == code)
        # 设置颜色
        for i, c in enumerate(color_rgb):
            rgb_img[mask, i] = int(c * 255)
    
    # 加载轨迹数据
    trajectories = load_trajectories()
    original_paths = load_original_paths()
    
    # 创建图形
    plt.figure(figsize=(15, 12))
    
    # 显示地表覆盖
    plt.imshow(rgb_img)
    
    # 绘制所有轨迹
    for traj, path in zip(trajectories, original_paths):
        # 绘制原始路径（虚线）
        plt.plot(path['col'], path['row'], 'k--', linewidth=2)
        # 绘制平滑后的轨迹（实线）
        plt.plot(traj['col'], traj['row'], 'w-', linewidth=3)
        # 标记起点和终点
        plt.plot(traj['col'].iloc[0], traj['row'].iloc[0], 'go', markersize=10)
        plt.plot(traj['col'].iloc[-1], traj['row'].iloc[-1], 'ro', markersize=10)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = []
    
    # 添加地表类型图例
    for code in sorted(COLORS.keys()):
        if code in LANDCOVER_NAMES:
            legend_elements.append(
                Patch(facecolor=COLORS[code], 
                      label=f"{LANDCOVER_NAMES[code]} ({code})")
            )
    
    # 添加轨迹元素图例
    legend_elements.extend([
        Patch(facecolor='black', label='原始路径'),
        Patch(facecolor='white', label='平滑轨迹'),
        Patch(facecolor='green', label='起点'),
        Patch(facecolor='red', label='终点')
    ])
    
    # 使用一个图例
    plt.legend(handles=legend_elements, 
               loc='upper right',
               bbox_to_anchor=(1.3, 1.0),
               fontsize=12)
    
    plt.title('轨迹规划结果', fontsize=16)
    plt.xlabel('东西方向 (像素)', fontsize=14)
    plt.ylabel('南北方向 (像素)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 保存结果
    plt.savefig("simple_visualize.png", dpi=300, bbox_inches='tight')
    print("可视化结果已保存至: simple_visualize.png")

if __name__ == "__main__":
    main() 