"""
轨迹生成模块: 根据路径和速度生成轨迹

包含:
- 轨迹点生成
- 速度控制
- 轨迹平滑
"""

import numpy as np
from typing import List, Tuple
import rasterio
from utils import calculate_distance

class TrajectoryGenerator:
    """轨迹生成器类: 根据路径和速度生成轨迹"""
    
    def __init__(
        self,
        speed_map: np.ndarray,
        transform: rasterio.Affine,
        target_speed: float = None,
        min_speed: float = 0.1
    ):
        """
        初始化轨迹生成器

        参数:
            speed_map: 速度图
            transform: rasterio的仿射变换矩阵
            target_speed: 目标速度(米/秒)
            min_speed: 最小速度(米/秒)
        """
        self.speed_map = speed_map
        self.transform = transform
        self.target_speed = target_speed
        self.min_speed = min_speed
        
    def generate_trajectory(
        self,
        path: List[Tuple[int, int]],
        dt: float = 1.0
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        生成轨迹和时间序列

        参数:
            path: 路径点列表
            dt: 时间步长(秒)

        返回:
            轨迹点列表和对应的时间序列
        """
        if not path:
            return [], []
            
        trajectory = [path[0]]  # 轨迹点列表
        times = [0.0]  # 时间序列
        current_time = 0.0
        
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]
            
            # 计算两点间距离(米)
            distance = calculate_distance(current, next_point, self.transform)
            
            # 获取当前点的速度
            current_speed = self.speed_map[current]
            
            # 如果指定了目标速度,使用较小值
            if self.target_speed is not None:
                current_speed = min(current_speed, self.target_speed)
                
            # 确保速度不小于最小值
            current_speed = max(current_speed, self.min_speed)
            
            # 计算移动到下一点需要的时间
            time_needed = distance / current_speed
            
            # 根据时间步长插值生成轨迹点
            num_steps = int(np.ceil(time_needed / dt))
            for step in range(1, num_steps + 1):
                t = step * dt / time_needed  # 插值系数
                if t > 1.0:  # 确保不超过终点
                    t = 1.0
                    
                # 线性插值计算轨迹点
                row = int(current[0] + t * (next_point[0] - current[0]))
                col = int(current[1] + t * (next_point[1] - current[1]))
                
                trajectory.append((row, col))
                current_time += dt
                times.append(current_time)
                
                if t >= 1.0:  # 已到达终点
                    break
                    
        return trajectory, times
        
    def smooth_trajectory(
        self,
        trajectory: List[Tuple[int, int]],
        window_size: int = 5
    ) -> List[Tuple[int, int]]:
        """
        使用移动平均平滑轨迹

        参数:
            trajectory: 原始轨迹
            window_size: 平滑窗口大小

        返回:
            平滑后的轨迹
        """
        if len(trajectory) <= window_size:
            return trajectory
            
        smoothed_trajectory = []
        half_window = window_size // 2
        
        # 保持起点和终点不变
        smoothed_trajectory.extend(trajectory[:half_window])
        
        # 对中间点进行平滑
        for i in range(half_window, len(trajectory) - half_window):
            window = trajectory[i - half_window:i + half_window + 1]
            row = int(np.mean([p[0] for p in window]))
            col = int(np.mean([p[1] for p in window]))
            smoothed_trajectory.append((row, col))
            
        # 保持终点不变
        smoothed_trajectory.extend(trajectory[-half_window:])
        
        return smoothed_trajectory 