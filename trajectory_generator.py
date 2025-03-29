"""
轨迹生成器模块：基于规则的运动模拟

输入：
- 基础速度图
- 坡度数据
- 坡向数据
- 路径点列表

输出：
- 轨迹点列表（包含时间戳、位置、速度、朝向）

处理流程：
1. 加载环境数据
2. 根据规则计算每个位置的速度约束
3. 模拟运动过程
4. 保存轨迹数据
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import rasterio
from config import (
    OUTPUT_DIR,
    SLOPE_PATH,
    ASPECT_PATH,
    DT,
    MAX_ACCELERATION,
    MAX_DECELERATION,
    MAX_TURNING_RATE,
    ASPECT_UPHILL_REDUCTION_K,
    ASPECT_CROSS_REDUCTION_K,
    MAX_CROSS_SLOPE_DEGREES,
    MAX_BRAKING_SPEED_DOWNHILL
)

class TrajectoryGenerator:
    """轨迹生成器类"""
    
    def __init__(self):
        """初始化轨迹生成器"""
        self.base_speed = None
        self.slope = None
        self.aspect = None
        self.transform = None
        self.shape = None
        
    def load_data(self) -> None:
        """加载环境数据"""
        print("加载环境数据...")
        
        # 加载基础速度图
        speed_path = os.path.join(OUTPUT_DIR, "intermediate", "base_speed_map.tif")
        with rasterio.open(speed_path) as src:
            self.base_speed = src.read(1)
            self.transform = src.transform
            self.shape = self.base_speed.shape
            
        # 加载坡度数据
        with rasterio.open(SLOPE_PATH) as src:
            self.slope = src.read(1)
            
        # 加载坡向数据
        with rasterio.open(ASPECT_PATH) as src:
            self.aspect = src.read(1)
            
    def calculate_slope_factors(
        self,
        pos: Tuple[int, int],
        heading: float
    ) -> Tuple[float, float, float]:
        """
        计算坡度影响因子
        
        参数:
            pos: 当前位置
            heading: 当前朝向（度，北为0，顺时针）
            
        返回:
            Tuple[float, float, float]: 上坡因子、横坡因子、下坡因子
        """
        # 获取当前位置的坡度和坡向
        slope_mag = self.slope[pos]
        slope_aspect = self.aspect[pos]
        
        # 计算车辆朝向与坡向的夹角
        angle_diff = abs(heading - slope_aspect)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
            
        # 计算上坡/下坡分量
        slope_along = slope_mag * np.cos(np.radians(angle_diff))
        
        # 计算横坡分量
        slope_cross = slope_mag * np.sin(np.radians(angle_diff))
        
        # 计算影响因子
        if slope_along > 0:  # 上坡
            uphill_factor = max(0.1, 1 - ASPECT_UPHILL_REDUCTION_K * slope_along)
            downhill_factor = 1.0
        else:  # 下坡
            uphill_factor = 1.0
            base_speed = self.base_speed[pos]
            if base_speed > 0:
                downhill_factor = min(1.0, MAX_BRAKING_SPEED_DOWNHILL / base_speed)
            else:
                downhill_factor = 0.1  # 基础速度为0时使用最小因子
            
        # 横坡影响
        cross_factor = max(0.05, 1 - ASPECT_CROSS_REDUCTION_K * slope_cross**2)
        if abs(slope_cross) > MAX_CROSS_SLOPE_DEGREES:
            cross_factor = 0.05  # 几乎停止
            
        return uphill_factor, cross_factor, downhill_factor
        
    def get_target_speed(
        self,
        pos: Tuple[int, int],
        heading: float
    ) -> float:
        """
        获取目标速度
        
        参数:
            pos: 当前位置
            heading: 当前朝向
            
        返回:
            float: 目标速度
        """
        # 获取基础速度
        base_speed = self.base_speed[pos]
        
        # 计算坡度影响
        uphill, cross, downhill = self.calculate_slope_factors(pos, heading)
        
        # 应用所有影响因子
        target_speed = base_speed * uphill * cross * downhill
        
        # 添加随机扰动（±10%）
        target_speed *= np.random.uniform(0.9, 1.1)
        
        return max(0.1, target_speed)  # 确保速度不会太小
        
    def interpolate_position(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int],
        ratio: float
    ) -> Tuple[float, float]:
        """
        线性插值计算位置
        
        参数:
            pos1: 起始位置
            pos2: 目标位置
            ratio: 插值比例
            
        返回:
            Tuple[float, float]: 插值后的位置
        """
        return (
            pos1[0] + (pos2[0] - pos1[0]) * ratio,
            pos1[1] + (pos2[1] - pos1[1]) * ratio
        )
        
    def calculate_heading(
        self,
        current_pos: Tuple[float, float],
        target_pos: Tuple[int, int]
    ) -> float:
        """
        计算朝向角度
        
        参数:
            current_pos: 当前位置
            target_pos: 目标位置
            
        返回:
            float: 朝向角度（度，北为0，顺时针）
        """
        dy = target_pos[0] - current_pos[0]
        dx = target_pos[1] - current_pos[1]
        angle = np.degrees(np.arctan2(-dx, dy))  # 使用-dx是因为我们要得到与y轴的夹角
        return angle % 360
        
    def generate_trajectory(
        self,
        path: List[Tuple[int, int]],
        trajectory_id: int
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        生成轨迹
        
        参数:
            path: 路径点列表
            trajectory_id: 轨迹ID
            
        返回:
            Tuple[pd.DataFrame, Dict]: 轨迹数据和统计信息
        """
        if len(path) < 2:
            raise ValueError("路径至少需要两个点")
            
        # 初始化状态
        current_pos = path[0]
        current_speed = 0.0
        path_index = 1
        time = 0.0
        
        # 初始化轨迹点列表
        trajectory = []
        
        print(f"\n生成轨迹 {trajectory_id}...")
        print(f"路径长度: {len(path)}个点")
        print(f"起点: {path[0]}")
        print(f"终点: {path[-1]}")
        
        # 主循环
        while path_index < len(path):
            # 获取目标点
            target_pos = path[path_index]
            
            # 计算朝向
            target_heading = self.calculate_heading(current_pos, target_pos)
            
            # 获取当前位置的整数坐标（用于查询地图）
            current_row = int(round(current_pos[0]))
            current_col = int(round(current_pos[1]))
            current_pos_int = (current_row, current_col)
            
            # 计算目标速度
            target_speed = self.get_target_speed(current_pos_int, target_heading)
            
            # 应用加速度限制
            speed_diff = target_speed - current_speed
            if speed_diff > 0:
                acceleration = min(speed_diff / DT, MAX_ACCELERATION)
            else:
                acceleration = max(speed_diff / DT, MAX_DECELERATION)
            
            # 更新速度
            current_speed = current_speed + acceleration * DT
            
            # 计算移动距离
            distance = current_speed * DT
            
            # 计算到目标点的距离
            to_target = np.sqrt(
                (target_pos[0] - current_pos[0])**2 +
                (target_pos[1] - current_pos[1])**2
            )
            
            # 如果可以到达目标点
            if distance >= to_target:
                current_pos = target_pos
                path_index += 1
            else:
                # 移动一步
                ratio = distance / to_target
                current_pos = self.interpolate_position(current_pos, target_pos, ratio)
                
            # 记录轨迹点
            trajectory.append({
                'trajectory_id': trajectory_id,
                'timestamp': time,
                'row': current_pos[0],
                'col': current_pos[1],
                'speed': current_speed,
                'heading': target_heading,
                'target_speed': target_speed,
                'acceleration': acceleration
            })
            
            # 更新时间
            time += DT
            
        # 创建DataFrame
        df = pd.DataFrame(trajectory)
        
        # 计算统计信息
        duration = time
        distance = sum(
            np.sqrt(
                (df['row'].iloc[i+1] - df['row'].iloc[i])**2 +
                (df['col'].iloc[i+1] - df['col'].iloc[i])**2
            ) * 30  # 30米分辨率
            for i in range(len(df)-1)
        )
        avg_speed = distance / duration
        
        # 创建统计信息字典
        stats_dict = {
            'duration_hours': duration/3600,
            'distance_km': distance/1000,
            'avg_speed': avg_speed*3.6
        }
        
        print(f"轨迹生成完成:")
        print(f"  持续时间: {stats_dict['duration_hours']:.2f}小时")
        print(f"  总距离: {stats_dict['distance_km']:.2f}公里")
        print(f"  平均速度: {stats_dict['avg_speed']:.2f}公里/小时")
        
        return df, stats_dict
        
    def save_trajectory(self, df: pd.DataFrame, trajectory_id: int) -> None:
        """
        保存轨迹数据
        
        参数:
            df: 轨迹数据
            trajectory_id: 轨迹ID
        """
        # 创建输出目录
        output_dir = os.path.join(OUTPUT_DIR, "trajectories")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存到CSV
        output_path = os.path.join(output_dir, f"trajectory_{trajectory_id}.csv")
        df.to_csv(output_path, index=False)
        print(f"轨迹数据已保存至: {output_path}")
        
def main():
    """主函数：测试轨迹生成器"""
    from path_planner import PathPlanner
    
    # 创建路径规划器
    planner = PathPlanner()
    planner.load_cost_map()
    
    # 测试路径规划
    start = (100, 100)
    goal = (200, 200)
    path, cost = planner.find_path(start, goal)
    
    if not path:
        print("未找到可行路径")
        return
        
    # 创建轨迹生成器
    generator = TrajectoryGenerator()
    generator.load_data()
    
    # 生成轨迹
    df, stats = generator.generate_trajectory(path, trajectory_id=0)
    
    # 保存轨迹
    generator.save_trajectory(df, trajectory_id=0)
    
if __name__ == "__main__":
    main() 