"""
轨迹可视化模块：展示生成的轨迹

输入：
- landcover_30m_100km.tif：土地覆盖数据
- trajectories/*.csv：轨迹数据文件

输出：
- 轨迹可视化图像
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
import plot_style  # 导入统一样式设置
from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    LANDCOVER_PATH,
    LANDCOVER_CODES
)

def load_landcover():
    """加载土地覆盖数据"""
    with rasterio.open(LANDCOVER_PATH) as src:
        return src.read(1)

def create_landcover_colormap():
    """创建土地覆盖类型的颜色映射"""
    colors = {
        10: '#0077BE',    # 水域：蓝色
        20: '#80CCFF',    # 湿地：浅蓝色
        30: '#90EE90',    # 草地：浅绿色
        40: '#228B22',    # 灌木地：深绿色
        50: '#CD5C5C',    # 建筑用地：红褐色
        60: '#FFD700',    # 农田：金黄色
        80: '#006400',    # 森林：深绿色
        90: '#DEB887',    # 荒地：棕色
        255: '#808080'    # 未分类：灰色
    }
    
    # 创建颜色映射
    max_code = max(LANDCOVER_CODES.keys())
    cmap_colors = ['#000000'] * (max_code + 1)  # 默认黑色
    for code, color in colors.items():
        cmap_colors[code] = color
        
    return ListedColormap(cmap_colors)

def create_legend_elements():
    """创建图例元素"""
    from matplotlib.patches import Patch
    
    colors = {
        '水域': '#0077BE',
        '湿地': '#80CCFF',
        '草地': '#90EE90',
        '灌木地': '#228B22',
        '建筑用地': '#CD5C5C',
        '农田': '#FFD700',
        '森林': '#006400',
        '荒地': '#DEB887'
    }
    
    return [Patch(facecolor=color, label=name) for name, color in colors.items()]

def load_trajectories():
    """加载所有轨迹数据"""
    trajectories = []
    traj_dir = os.path.join(OUTPUT_DIR, "trajectories")
    
    for file in os.listdir(traj_dir):
        if file.startswith("trajectory_") and file.endswith(".csv"):
            path = os.path.join(traj_dir, file)
            df = pd.read_csv(path)
            trajectories.append(df)
            
    return trajectories

def plot_trajectory_on_map(landcover, trajectory, ax, title):
    """在地图上绘制单条轨迹"""
    # 绘制地形图
    cmap = create_landcover_colormap()
    im = ax.imshow(landcover, cmap=cmap)
    
    # 绘制轨迹
    points = trajectory[['row', 'col']].values
    speeds = trajectory['speed'].values
    
    # 绘制轨迹线
    ax.plot(points[:, 1], points[:, 0], 'w-', linewidth=0.8, alpha=0.6)
    
    # 绘制轨迹点，颜色表示速度
    scatter = ax.scatter(
        points[:, 1],
        points[:, 0],
        c=speeds,
        cmap='plasma',
        s=2,
        alpha=0.8
    )
    
    # 添加起点和终点标记
    ax.plot(points[0, 1], points[0, 0], 'g*', markersize=12, label='起点')
    ax.plot(points[-1, 1], points[-1, 0], 'r*', markersize=12, label='终点')
    
    # 添加地形图例
    legend_elements = create_legend_elements()
    terrain_legend = ax.legend(
        handles=legend_elements,
        title='地形类型',
        loc='center left',
        bbox_to_anchor=(1.02, 0.5)
    )
    ax.add_artist(terrain_legend)
    
    # 添加起终点图例
    path_legend = ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0)
    )
    ax.add_artist(path_legend)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('速度 (m/s)', fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    
    # 设置标题
    ax.set_title(title, pad=20)
    
    # 设置坐标轴标签
    ax.set_xlabel('列号', fontsize=16)
    ax.set_ylabel('行号', fontsize=16)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
def main():
    """主函数"""
    print("加载数据...")
    landcover = load_landcover()
    trajectories = load_trajectories()
    
    print(f"共加载了{len(trajectories)}条轨迹")
    
    # 创建图形
    n_trajectories = len(trajectories)
    n_cols = min(2, n_trajectories)
    n_rows = (n_trajectories + 1) // 2
    
    # 设置更大的图形尺寸以适应图例
    plt.figure(figsize=(20, 10 * n_rows))
    
    # 绘制每条轨迹
    for i, traj in enumerate(trajectories):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        # 计算统计信息
        duration = traj['timestamp'].max() / 3600  # 小时
        distance = sum(
            np.sqrt(
                (traj['row'].iloc[i+1] - traj['row'].iloc[i])**2 +
                (traj['col'].iloc[i+1] - traj['col'].iloc[i])**2
            ) * 30  # 30米分辨率
            for i in range(len(traj)-1)
        ) / 1000  # 公里
        avg_speed = distance / duration  # 公里/小时
        
        title = f"轨迹 {i+1}\n" \
                f"持续时间: {duration:.1f} 小时\n" \
                f"总距离: {distance:.1f} 公里\n" \
                f"平均速度: {avg_speed:.1f} 公里/小时"
                
        plot_trajectory_on_map(landcover, traj, ax, title)
        
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(OUTPUT_DIR, "trajectories_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n可视化结果已保存至: {output_path}")
    
    # 显示图像
    plt.show()
    
if __name__ == "__main__":
    main() 