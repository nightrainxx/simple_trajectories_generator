"""
批量轨迹生成脚本

用于批量生成大量轨迹，并测量处理时间
"""

import os
import time
import random
import numpy as np
import pandas as pd
from typing import List, Tuple
from datetime import timedelta
import multiprocessing as mp
from tqdm import tqdm
import traceback
import concurrent.futures
import matplotlib.pyplot as plt

from improved_path_planner import ImprovedPathPlanner
from trajectory_generator import TrajectoryGenerator
from config import OUTPUT_DIR, NUM_TRAJECTORIES_TO_GENERATE, LANDCOVER_PATH, MIN_START_END_DISTANCE_METERS, INVALID_AREA_PATH, MAP_SIZE

import rasterio

def generate_random_points(num_points, map_size, invalid_mask=None, min_distance=50):
    """
    生成随机点，确保点不在无效区域内并且两点之间有最小距离
    
    参数:
    - num_points: 要生成的点数量
    - map_size: 地图大小 (height, width)
    - invalid_mask: 无效区域掩码 (True表示无效)
    - min_distance: 点之间的最小欧氏距离
    
    返回:
    - 随机点列表 [(x1,y1), (x2,y2), ...]
    """
    if invalid_mask is None:
        # 如果没有提供无效区域掩码，使用土地覆盖数据创建一个
        with rasterio.open(LANDCOVER_PATH) as src:
            landcover = src.read(1)
            # 水体、湿地和未分类区域被视为无效
            invalid_mask = np.isin(landcover, [10, 20, 255])
    
    points = []
    attempts = 0
    max_attempts = num_points * 100  # 最大尝试次数
    
    while len(points) < num_points and attempts < max_attempts:
        x = random.randint(0, map_size[1] - 1)
        y = random.randint(0, map_size[0] - 1)
        
        # 检查点是否在有效区域内
        if invalid_mask is not None and invalid_mask[y, x]:
            attempts += 1
            continue
        
        # 检查与已有点的最小距离
        too_close = False
        for px, py in points:
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < min_distance:
                too_close = True
                break
        
        if not too_close:
            points.append((x, y))
        
        attempts += 1
    
    if len(points) < num_points:
        print(f"警告: 只能生成 {len(points)}/{num_points} 个随机点 (尝试 {attempts} 次)")
    
    return points

def generate_single_trajectory(args):
    """生成单条轨迹"""
    trajectory_id, start_point, end_point, output_dir = args
    
    try:
        # 创建路径规划器和轨迹生成器
        planner = ImprovedPathPlanner()
        planner.load_cost_map()
        generator = TrajectoryGenerator()
        generator.load_data()
        
        # 生成路径
        success, path, cost = planner.find_path(start_point, end_point, key_point_ratio=0.2)
        if not success or path is None:
            return False, f"轨迹 {trajectory_id} 路径规划失败"
            
        # 平滑路径
        smoothed_path = planner.smooth_path(path)
        if smoothed_path is None:
            return False, f"轨迹 {trajectory_id} 路径平滑失败"
            
        # 生成轨迹
        trajectory_df, stats = generator.generate_trajectory(smoothed_path, trajectory_id)
        if trajectory_df is None:
            return False, f"轨迹 {trajectory_id} 轨迹生成失败"
            
        # 保存轨迹
        trajectory_path = os.path.join(output_dir, f'trajectory_{trajectory_id}.csv')
        trajectory_df.to_csv(trajectory_path, index=False)
        
        print(f"轨迹 {trajectory_id} 生成完成:")
        print(f"  持续时间: {stats['duration_hours']:.2f}小时")
        print(f"  总距离: {stats['distance_km']:.2f}公里")
        print(f"  平均速度: {stats['avg_speed']:.2f}公里/小时")
        
        return True, None
        
    except Exception as e:
        return False, f"轨迹 {trajectory_id} 生成出错: {str(e)}"

def main():
    """主函数"""
    # 开始计时
    start_time = time.time()
    print(f"开始生成 {NUM_TRAJECTORIES_TO_GENERATE} 条轨迹...")
    
    # 创建输出目录
    output_dir = os.path.join(OUTPUT_DIR, 'batch_trajectories_20_highspeed_65kmh/trajectories')
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化路径规划器以获取地图尺寸
    planner = ImprovedPathPlanner()
    planner.load_cost_map()
    rows, cols = planner.shape
    
    # 加载土地覆盖数据，用于选择有效起点和终点
    with rasterio.open(LANDCOVER_PATH) as src:
        landcover = src.read(1)
        map_size = landcover.shape
        # 水体、湿地和未分类区域被视为无效
        invalid_mask = np.isin(landcover, [10, 20, 255])
    
    # 生成随机起点和终点
    print("生成随机起点和终点...")
    
    # 计算实际地图的分辨率(米/像素)
    with rasterio.open(LANDCOVER_PATH) as src:
        # 获取仿射变换参数
        transform = src.transform
        # 计算像素尺寸(米/像素)
        pixel_width_m = abs(transform[0])
        pixel_height_m = abs(transform[4])
        avg_resolution = (pixel_width_m + pixel_height_m) / 2
    
    # 计算起点终点的最小欧氏距离(像素)
    min_distance_pixels = int(MIN_START_END_DISTANCE_METERS / avg_resolution)
    
    # 随机生成起点
    start_points = generate_random_points(NUM_TRAJECTORIES_TO_GENERATE, map_size, invalid_mask, min_distance=100)
    
    # 随机生成终点，确保与起点的欧氏距离超过最小限制
    end_points = []
    for start_point in start_points:
        attempts = 0
        valid_end = None
        
        while valid_end is None and attempts < 100:
            candidate = random.choice(generate_random_points(10, map_size, invalid_mask, min_distance=100))
            distance = np.sqrt((candidate[0] - start_point[0])**2 + (candidate[1] - start_point[1])**2)
            
            if distance >= min_distance_pixels:
                valid_end = candidate
            
            attempts += 1
        
        if valid_end:
            end_points.append(valid_end)
        else:
            # 如果找不到合适的终点，使用之前有效的终点
            end_points.append(random.choice(end_points) if end_points else (map_size[1]//2, map_size[0]//2))
    
    # 确保起点和终点配对的数量一致
    num_pairs = min(len(start_points), len(end_points))
    if num_pairs < NUM_TRAJECTORIES_TO_GENERATE:
        print(f"警告：只能生成 {num_pairs}/{NUM_TRAJECTORIES_TO_GENERATE} 对有效的起点-终点配对")
    
    # 准备任务参数
    tasks = []
    for i in range(NUM_TRAJECTORIES_TO_GENERATE):
        if i < len(start_points) and i < len(end_points):
            start = start_points[i]
            end = end_points[i]
            tasks.append((i, start, end, output_dir))
    
    # 使用多进程生成轨迹
    num_cores = min(mp.cpu_count(), 4)  # 最多使用4个核心
    print(f"使用 {num_cores} 个处理器并行生成轨迹...")
    
    start_time = time.time()
    
    with mp.Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(generate_single_trajectory, tasks), 
                          total=len(tasks), 
                          desc="生成轨迹进度"))
    
    # 统计结果
    success_count = sum(1 for success, _ in results if success)
    failed_count = len(results) - success_count
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_trajectory = total_time / len(tasks)
    estimated_time_500 = avg_time_per_trajectory * 500
    
    print("\n轨迹生成完成！")
    print(f"总耗时: {timedelta(seconds=int(total_time))}")
    print(f"成功生成: {success_count} 条轨迹")
    print(f"失败: {failed_count} 条轨迹")
    print(f"平均每条轨迹耗时: {avg_time_per_trajectory:.2f} 秒")
    print(f"估计生成500条轨迹需要: {timedelta(seconds=int(estimated_time_500))}")
    
    # 保存失败记录
    if failed_count > 0:
        error_log_path = os.path.join(output_dir, 'generation_errors.txt')
        with open(error_log_path, 'w', encoding='utf-8') as f:
            for success, error_msg in results:
                if not success:
                    f.write(f"{error_msg}\n")
        print(f"错误日志已保存至: {error_log_path}")
    
    # 生成报告
    report_path = os.path.join(os.path.dirname(output_dir), 'generation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("轨迹生成报告\n")
        f.write("============\n\n")
        f.write(f"总耗时: {timedelta(seconds=int(total_time))}\n")
        f.write(f"成功生成: {success_count} 条轨迹\n")
        f.write(f"失败: {failed_count} 条轨迹\n")
        f.write(f"平均每条轨迹耗时: {avg_time_per_trajectory:.2f} 秒\n")
        f.write(f"估计生成500条轨迹需要: {timedelta(seconds=int(estimated_time_500))}\n")
    
    print(f"详细报告已保存至: {report_path}")

if __name__ == "__main__":
    main() 