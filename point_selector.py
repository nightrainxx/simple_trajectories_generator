"""
点选择器模块：负责选择合适的起终点对

功能：
- 选择城市附近的终点
- 为每个终点选择合适的起点
- 确保起终点之间的距离满足要求
- 确保起终点在可通行区域
- 支持指定战略位置作为目标点
"""

import numpy as np
import rasterio
from typing import List, Tuple, Optional, Dict
import os
from config import (
    LANDCOVER_PATH,
    URBAN_LANDCOVER_CODES,
    IMPASSABLE_LANDCOVER_CODES,
    MIN_START_END_DISTANCE_METERS,
    NUM_END_POINTS
)
from map_generator import MapGenerator
import random
from environment import Environment

# 定义战略位置
STRATEGIC_LOCATIONS = [
    (481, 113),    # 位置1
    (1095, 2682),  # 位置2
    (1812, 1427),  # 位置3
    (2577, 2000)   # 位置4（调整到附近的可通行位置）
]

class PointSelector:
    """点选择器类：选择合适的起终点对"""
    
    def __init__(self):
        """初始化点选择器"""
        self.landcover = None
        self.transform = None
        self.crs = None
        self.pixel_size = None
        self.speed_map = None
        
    def load_landcover(self, landcover_path: str = LANDCOVER_PATH) -> None:
        """
        加载土地覆盖数据和速度图

        参数:
            landcover_path: 土地覆盖数据文件路径
        """
        with rasterio.open(landcover_path) as src:
            self.landcover = src.read(1)
            self.transform = src.transform
            self.crs = src.crs
            self.pixel_size = abs(self.transform[0])  # 假设像素是正方形的
            
        # 生成速度图
        map_gen = MapGenerator()
        map_gen.load_data()
        self.speed_map = map_gen.generate_base_speed_map()
        
    def pixel_to_meters(self, pixel_distance: float) -> float:
        """
        将像素距离转换为米

        参数:
            pixel_distance: 像素距离

        返回:
            距离(米)
        """
        return pixel_distance * self.pixel_size * 111000  # 1度约等于111km
        
    def find_urban_points(self) -> List[Tuple[int, int]]:
        """
        寻找城市区域的点

        返回:
            城市点列表 [(row, col), ...]
        """
        # 创建城市区域掩码
        urban_mask = np.zeros_like(self.landcover, dtype=bool)
        for code in URBAN_LANDCOVER_CODES:
            urban_mask |= (self.landcover == code)
            
        # 获取所有城市点的坐标
        urban_points = list(zip(*np.where(urban_mask)))
        
        # 如果没有找到城市点，抛出异常
        if not urban_points:
            raise ValueError("未找到任何城市区域")
            
        # 过滤掉不可通行的点
        urban_points = [
            point for point in urban_points 
            if self.is_valid_point(point[0], point[1])
        ]
        
        return urban_points
        
    def is_valid_point(self, row: int, col: int) -> bool:
        """
        检查点是否有效（在可通行区域内）

        参数:
            row: 行索引
            col: 列索引

        返回:
            点是否有效
        """
        if not (0 <= row < self.landcover.shape[0] and 0 <= col < self.landcover.shape[1]):
            return False
            
        # 检查土地覆盖类型
        if self.landcover[row, col] in IMPASSABLE_LANDCOVER_CODES:
            return False
            
        # 检查速度是否大于0
        if self.speed_map is not None and self.speed_map[row, col] <= 0:
            return False
            
        return True
        
    def calculate_distance(
        self,
        point1: Tuple[int, int],
        point2: Tuple[int, int]
    ) -> float:
        """
        计算两点之间的距离(米)

        参数:
            point1: 第一个点(row, col)
            point2: 第二个点(row, col)

        返回:
            距离(米)
        """
        row1, col1 = point1
        row2, col2 = point2
        pixel_distance = np.sqrt((row2 - row1)**2 + (col2 - col1)**2)
        return self.pixel_to_meters(pixel_distance)
        
    def select_start_points(
        self,
        goal: Tuple[int, int],
        num_points: int = 125  # 每个目标点生成125个起点，总共500条路径
    ) -> List[Tuple[int, int]]:
        """
        为目标点选择多个起点

        参数:
            goal: 目标点坐标
            num_points: 需要的起点数量

        返回:
            起点列表
        """
        if self.landcover is None:
            raise ValueError("请先加载土地覆盖数据")
            
        height, width = self.landcover.shape
        rng = np.random.default_rng()
        start_points = []
        
        print(f"为目标点 {goal} 选择起点...")
        print(f"地图大小: {height} x {width}")
        print(f"目标起点数量: {num_points}")
        
        # 在不同方向上寻找起点
        angles = np.linspace(0, 360, 32, endpoint=False)  # 32个方向
        points_per_direction = num_points // len(angles)
        print(f"每个方向的目标点数: {points_per_direction}")
        
        # 生成多个距离范围
        min_distance = MIN_START_END_DISTANCE_METERS / 111000 / self.pixel_size
        max_distance = min(height, width) / 2
        distance_ranges = [
            (min_distance, min_distance * 1.5),
            (min_distance * 1.5, min_distance * 2.0),
            (min_distance * 2.0, max_distance)
        ]
        
        for distance_range in distance_ranges:
            print(f"\n尝试距离范围: {distance_range[0]:.2f} - {distance_range[1]:.2f} 像素")
            distances = np.linspace(
                distance_range[0],
                distance_range[1],
                num=points_per_direction
            )
            
            for angle in angles:
                angle_rad = np.radians(angle)
                direction_points = 0
                
                for distance in distances:
                    # 计算候选起点
                    dx = distance * np.cos(angle_rad)
                    dy = distance * np.sin(angle_rad)
                    start_row = int(goal[0] + dy)
                    start_col = int(goal[1] + dx)
                    
                    # 检查起点是否有效
                    if self.is_valid_point(start_row, start_col):
                        start_points.append((start_row, start_col))
                        direction_points += 1
                        
                print(f"方向 {angle:.1f}°: 找到 {direction_points} 个有效起点")
                
                # 如果已经找到足够的点，就停止搜索
                if len(start_points) >= num_points:
                    break
                    
            # 如果已经找到足够的点，就停止搜索
            if len(start_points) >= num_points:
                break
                
        print(f"总共找到 {len(start_points)} 个有效起点\n")
        
        # 如果找到的点太多，随机选择所需数量的点
        if len(start_points) > num_points:
            start_points = rng.choice(start_points, size=num_points, replace=False).tolist()
            print(f"随机选择 {num_points} 个起点\n")
            
        return start_points
        
    def select_start_end_pairs(
        self,
        num_pairs: int = NUM_END_POINTS,
        use_strategic: bool = True
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        选择起终点对

        参数:
            num_pairs: 需要的起终点对数量
            use_strategic: 是否使用战略位置作为目标点

        返回:
            起终点对列表 [((start_row, start_col), (end_row, end_col)), ...]
        """
        if self.landcover is None:
            raise ValueError("请先加载土地覆盖数据")
            
        pairs = []
        
        if use_strategic:
            # 使用战略位置作为目标点
            for goal in STRATEGIC_LOCATIONS:
                if not self.is_valid_point(*goal):
                    print(f"警告：战略位置 {goal} 不可达")
                    continue
                    
                # 为每个目标点选择多个起点
                start_points = self.select_start_points(goal)
                pairs.extend([(start, goal) for start in start_points])
        else:
            # 使用原有的选择逻辑
            urban_points = self.find_urban_points()
            if not urban_points:
                raise ValueError("未找到任何可通行的城市点")
                
            rng = np.random.default_rng()
            selected_ends = rng.choice(len(urban_points), size=min(num_pairs, len(urban_points)), replace=False)
            end_points = [urban_points[i] for i in selected_ends]
            
            for end_point in end_points:
                start_points = self.select_start_points(end_point, num_points=num_pairs)
                pairs.extend([(start, end_point) for start in start_points])
                
        return pairs
        
def save_points_to_csv(start_end_pairs, output_file='start_end_pairs.csv'):
    """
    将起终点对保存到CSV文件
    
    参数:
        start_end_pairs: 包含起终点对的列表，每个元素是一个字典
        output_file: 输出CSV文件的路径
    """
    import csv
    import os
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # 写入CSV文件
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['start_row', 'start_col', 'end_row', 'end_col', 'distance'])
        writer.writeheader()
        for pair in start_end_pairs:
            writer.writerow({
                'start_row': pair['start'][0],
                'start_col': pair['start'][1],
                'end_row': pair['end'][0],
                'end_col': pair['end'][1],
                'distance': pair['distance']
            })
    print(f"\n起终点对数据已保存到: {output_file}")

def main():
    """主函数：测试点选择器"""
    # 加载环境数据
    print("加载土地覆盖数据和速度图...")
    env = Environment()
    env.load_data()
    
    # 战略位置列表
    strategic_points = [
        (481, 113),   # 位置1
        (1095, 2682), # 位置2
        (1812, 1427), # 位置3
        (2577, 2000)  # 位置4
    ]
    
    # 检查每个战略位置的有效性
    print("\n检查战略位置的有效性:\n")
    valid_points = []
    for i, point in enumerate(strategic_points, 1):
        row, col = point
        valid = is_valid_point(env, row, col)
        land_cover = env.landcover[row, col]
        base_speed = env.get_pixel_speed(row, col)
        
        print(f"位置 {i} ({point}):")
        print(f"  有效性: {'是' if valid else '否'}")
        print(f"  土地覆盖类型: {land_cover}")
        print(f"  基础速度: {base_speed:.1f} m/s\n")
        
        if valid:
            valid_points.append(point)
    
    # 显示地图信息
    print("地图信息:")
    print(f"  形状: {env.landcover.shape}")
    print(f"  有效土地类型: {sorted(list(set(env.landcover.flatten())))}")
    print(f"  不可通行土地类型: {IMPASSABLE_LANDCOVER_CODES}\n")
    
    # 为每个有效的目标点选择起点
    print("选择起终点对...")
    all_pairs = []
    for target_point in valid_points:
        print(f"为目标点 {target_point} 选择起点...")
        pairs = select_start_points(env, target_point)
        all_pairs.extend(pairs)
    
    # 显示结果统计
    print(f"\n找到 {len(all_pairs)} 对起终点:\n")
    for target_point in valid_points:
        count = sum(1 for pair in all_pairs if pair['end'] == target_point)
        print(f"目标点 {target_point}:")
        print(f"  起点数量: {count}\n")
    
    # 显示一些示例路径
    print("示例路径:")
    for i, pair in enumerate(all_pairs[:5], 1):
        print(f"第 {i} 对:")
        print(f"  起点: {pair['start']}")
        print(f"  终点: {pair['end']}")
        print(f"  距离: {pair['distance']:.2f} km")
    
    # 保存数据到CSV文件
    output_dir = 'data'
    save_points_to_csv(all_pairs, os.path.join(output_dir, 'start_end_pairs.csv'))

def is_valid_point(env, row, col):
    """
    检查给定点是否有效
    
    参数:
        env: Environment对象
        row: 行坐标
        col: 列坐标
    
    返回:
        bool: 点是否有效
    """
    # 检查是否在地图范围内
    if not (0 <= row < env.landcover.shape[0] and 0 <= col < env.landcover.shape[1]):
        return False
    
    # 检查土地类型是否可通行
    if env.landcover[row, col] in IMPASSABLE_LANDCOVER_CODES:
        return False
    
    # 检查基础速度是否大于0
    if env.get_pixel_speed(row, col) <= 0:
        return False
    
    return True

def select_start_points(env, target_point, num_points=125):
    """
    为给定目标点选择起点
    
    参数:
        env: Environment对象
        target_point: 目标点坐标 (row, col)
        num_points: 目标起点数量
    
    返回:
        pairs: 包含起终点对信息的列表，每个元素是一个字典
    """
    print(f"地图大小: {env.landcover.shape[0]} x {env.landcover.shape[1]}")
    print(f"目标起点数量: {num_points}")
    
    # 计算每个方向需要的点数
    num_angles = 32
    points_per_direction = max(3, num_points // num_angles)
    print(f"每个方向的目标点数: {points_per_direction}")
    
    # 生成角度列表
    angles = np.linspace(0, 360, num_angles, endpoint=False)
    
    # 计算最小和最大距离（以像素为单位）
    map_diagonal = np.sqrt(env.landcover.shape[0]**2 + env.landcover.shape[1]**2)
    min_distance = map_diagonal * 0.3
    max_distance = map_diagonal * 0.6
    
    # 定义距离范围
    distance_ranges = [
        (min_distance, min_distance * 1.5),
        (min_distance * 1.5, min_distance * 2),
        (min_distance * 2, min_distance * 1.14)
    ]
    
    valid_pairs = []
    
    # 将目标点从像素坐标转换为地理坐标
    target_lon, target_lat = rasterio.transform.xy(env.transform, target_point[0], target_point[1])
    
    # 对每个距离范围
    for d_min, d_max in distance_ranges:
        print(f"\n尝试距离范围: {d_min:.2f} - {d_max:.2f} 像素")
        
        # 对每个角度
        for angle in angles:
            # 在该方向上尝试不同的距离
            valid_points_in_direction = []
            
            for distance in np.linspace(d_min, d_max, points_per_direction * 2):
                # 计算候选起点
                dx = distance * np.cos(np.radians(angle))
                dy = distance * np.sin(np.radians(angle))
                
                start_row = int(target_point[0] + dy)
                start_col = int(target_point[1] + dx)
                
                # 检查点是否有效
                if is_valid_point(env, start_row, start_col):
                    # 将起点从像素坐标转换为地理坐标
                    start_lon, start_lat = rasterio.transform.xy(env.transform, start_row, start_col)
                    
                    # 计算地理距离（千米）
                    R = 6371  # 地球平均半径（千米）
                    dlat = np.radians(start_lat - target_lat)
                    dlon = np.radians(start_lon - target_lon)
                    a = np.sin(dlat/2)**2 + np.cos(np.radians(target_lat)) * np.cos(np.radians(start_lat)) * np.sin(dlon/2)**2
                    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                    distance = R * c
                    
                    valid_points_in_direction.append({
                        'start': (start_row, start_col),
                        'end': target_point,
                        'distance': distance
                    })
            
            # 如果找到的点数超过每个方向所需的点数，随机选择所需数量的点
            if len(valid_points_in_direction) > points_per_direction:
                valid_points_in_direction = random.sample(valid_points_in_direction, points_per_direction)
            
            valid_pairs.extend(valid_points_in_direction)
            print(f"方向 {angle:.1f}°: 找到 {len(valid_points_in_direction)} 个有效起点")
    
    print(f"总共找到 {len(valid_pairs)} 个有效起点")
    return valid_pairs

if __name__ == "__main__":
    main() 