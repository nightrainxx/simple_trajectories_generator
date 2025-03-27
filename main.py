"""
主程序模块: 实现轨迹生成的主要流程

包含:
- 命令行参数解析
- 环境数据加载
- 路径规划和轨迹生成
- 结果分析和保存
"""

import argparse
import numpy as np
import os
import rasterio
from typing import List, Tuple
import matplotlib.pyplot as plt

from environment import Environment
from path_planner import PathPlanner
from trajectory_generator import TrajectoryGenerator
from map_generator import MapGenerator
from utils import calculate_distance, haversine_distance

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='轨迹生成器')
    
    # 必需参数
    parser.add_argument(
        '--start_row',
        type=int,
        required=True,
        help='起点行号'
    )
    parser.add_argument(
        '--start_col',
        type=int,
        required=True,
        help='起点列号'
    )
    parser.add_argument(
        '--end_row',
        type=int,
        required=True,
        help='终点行号'
    )
    parser.add_argument(
        '--end_col',
        type=int,
        required=True,
        help='终点列号'
    )
    
    # 可选参数
    parser.add_argument(
        '--target_speed',
        type=float,
        default=None,
        help='目标速度(米/秒)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/output',
        help='输出目录'
    )
    
    return parser.parse_args()

def analyze_trajectory(
    trajectory: List[Tuple[int, int]],
    transform: rasterio.Affine
) -> dict:
    """
    分析轨迹特征

    参数:
        trajectory: 轨迹点列表
        transform: 仿射变换矩阵

    返回:
        特征字典
    """
    if not trajectory:
        return {
            'length': 0.0,
            'start': None,
            'end': None,
            'num_points': 0
        }
        
    # 计算轨迹长度
    length = 0.0
    for i in range(len(trajectory) - 1):
        length += calculate_distance(
            trajectory[i],
            trajectory[i + 1],
            transform
        )
        
    return {
        'length': length / 1000,  # 转换为千米
        'start': trajectory[0],
        'end': trajectory[-1],
        'num_points': len(trajectory)
    }

def analyze_path(
    path: List[Tuple[int, int]],
    transform: rasterio.Affine
) -> dict:
    """
    分析路径特征

    参数:
        path: 路径点列表
        transform: 仿射变换矩阵

    返回:
        特征字典
    """
    if not path:
        return {
            'length': 0.0,
            'start': None,
            'end': None,
            'num_points': 0
        }
        
    # 计算路径长度
    length = 0.0
    for i in range(len(path) - 1):
        length += calculate_distance(
            path[i],
            path[i + 1],
            transform
        )
        
    return {
        'length': length / 1000,  # 转换为千米
        'start': path[0],
        'end': path[-1],
        'num_points': len(path)
    }

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载环境数据
    print("加载环境数据...")
    env = Environment()
    env.load_data()
    
    # 生成代价地图
    print("生成代价地图...")
    map_gen = MapGenerator()
    cost_map = map_gen.generate_cost_map(
        env.dem,
        env.slope,
        env.landcover
    )
    
    # 可视化地图
    print("可视化地图...")
    map_gen.transform = env.transform
    map_gen.crs = env.crs
    map_gen.visualize_maps(args.output_dir)
    
    # 创建路径规划器
    print("规划路径...")
    planner = PathPlanner(
        cost_map=cost_map,
        transform=env.transform
    )
    
    # 规划路径
    start = (args.start_row, args.start_col)
    goal = (args.end_row, args.end_col)
    path = planner.find_path(start, goal)
    
    # 分析路径
    path_info = analyze_path(path, env.transform)
    print(f"路径长度: {path_info['length']:.2f} km")
    print(f"路径点数: {path_info['num_points']}")
    
    # 平滑路径
    smoothed_path = planner.smooth_path(path)
    
    # 创建轨迹生成器
    print("生成轨迹...")
    generator = TrajectoryGenerator(
        speed_map=map_gen.speed_map,
        transform=env.transform,
        target_speed=args.target_speed
    )
    
    # 生成轨迹
    trajectory, times = generator.generate_trajectory(smoothed_path)
    
    # 平滑轨迹
    smoothed_trajectory = generator.smooth_trajectory(trajectory)
    
    # 分析轨迹
    traj_info = analyze_trajectory(smoothed_trajectory, env.transform)
    print(f"轨迹长度: {traj_info['length']:.2f} km")
    print(f"轨迹点数: {traj_info['num_points']}")
    print(f"总时间: {times[-1]:.2f} 秒")
    print(f"平均速度: {(traj_info['length'] * 1000 / times[-1]):.2f} m/s")
    
    # 保存结果
    print("保存结果...")
    
    # 绘制路径和轨迹
    plt.figure(figsize=(10, 10))
    plt.imshow(cost_map, cmap='gray')
    
    # 绘制原始路径
    path_points = np.array(path)
    plt.plot(
        path_points[:, 1],
        path_points[:, 0],
        'r-',
        label='原始路径',
        linewidth=1
    )
    
    # 绘制平滑路径
    smoothed_path_points = np.array(smoothed_path)
    plt.plot(
        smoothed_path_points[:, 1],
        smoothed_path_points[:, 0],
        'g-',
        label='平滑路径',
        linewidth=1
    )
    
    # 绘制轨迹
    traj_points = np.array(smoothed_trajectory)
    plt.plot(
        traj_points[:, 1],
        traj_points[:, 0],
        'b.',
        label='轨迹点',
        markersize=1
    )
    
    plt.colorbar(label='代价')
    plt.legend()
    plt.title('路径和轨迹')
    plt.savefig(os.path.join(args.output_dir, 'trajectory.png'))
    plt.close()
    
    print("完成!")

if __name__ == '__main__':
    main() 