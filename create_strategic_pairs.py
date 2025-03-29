"""
创建使用战略位置作为终点的起终点对

输入：
- 用户提供的战略位置坐标

输出：
- strategic_pairs.csv：新的起终点对数据文件
"""

import os
import numpy as np
import pandas as pd
import rasterio
from config import OUTPUT_DIR, MIN_START_END_DISTANCE_METERS

# 战略位置坐标（行，列）
STRATEGIC_LOCATIONS = [
    (113, 481),     # 位置1: 像素坐标: (481, 113), 建筑用地
    (2682, 1095),   # 位置2: 像素坐标: (1095, 2682), 建筑用地
    (1427, 1812),   # 位置3: 像素坐标: (1812, 1427), 建筑用地
    (2149, 2577)    # 位置4: 像素坐标: (2577, 2149), 农田
]

class StrategicPairGenerator:
    """战略起终点对生成器"""
    def __init__(self):
        """初始化"""
        self.dem = None
        self.slope = None
        self.landcover = None
        self.base_speed = None
        self.shape = None
        self.pixel_size = 30  # 默认像素大小（米）
        
    def load_data(self):
        """加载必要的数据"""
        print("加载环境数据...")
        
        # 加载DEM数据
        dem_path = "data/input/dem_30m_100km.tif"
        with rasterio.open(dem_path) as src:
            self.dem = src.read(1)
            self.shape = self.dem.shape
            
        # 加载坡度数据
        slope_path = "data/input/slope_30m_100km.tif"
        with rasterio.open(slope_path) as src:
            self.slope = src.read(1)
            
        # 加载土地覆盖数据
        landcover_path = "data/input/landcover_30m_100km.tif"
        with rasterio.open(landcover_path) as src:
            self.landcover = src.read(1)
            
        # 加载基础速度数据
        speed_path = os.path.join(OUTPUT_DIR, "intermediate", "base_speed_map.tif")
        try:
            with rasterio.open(speed_path) as src:
                self.base_speed = src.read(1)
        except:
            print("警告：无法加载基础速度图，将使用默认速度")
            self.base_speed = np.ones(self.shape) * 5.0  # 默认速度5 m/s
        
        print(f"数据加载完成，地图大小: {self.shape}")
        
    def is_valid_point(self, row, col):
        """检查点是否有效"""
        # 检查边界
        if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
            return False
            
        # 检查是否有效速度
        if self.base_speed is not None:
            if np.isnan(self.base_speed[row, col]) or self.base_speed[row, col] <= 0:
                return False
                
        return True
        
    def calculate_distance(self, point1, point2):
        """计算两点之间的距离（米）"""
        row1, col1 = point1
        row2, col2 = point2
        # 使用欧几里得距离 * 像素大小
        distance = np.sqrt((row1 - row2) ** 2 + (col1 - col2) ** 2) * self.pixel_size
        return distance
        
    def select_start_points(self, end_point, attempts_per_direction=12):
        """为终点选择起点"""
        start_points = []
        end_row, end_col = end_point
        
        # 计算搜索范围（像素）
        min_pixels = int(MIN_START_END_DISTANCE_METERS / self.pixel_size)
        max_pixels = int(min_pixels * 1.5)  # 设置最大距离为最小距离的1.5倍
        
        print(f"\n为终点{end_point}选择起点")
        print(f"搜索范围: {min_pixels}-{max_pixels}像素 ({MIN_START_END_DISTANCE_METERS/1000:.1f}-{MIN_START_END_DISTANCE_METERS*1.5/1000:.1f}公里)")
        
        # 在不同距离和角度上尝试选择起点
        distances = np.linspace(min_pixels, max_pixels, attempts_per_direction)
        angles = np.linspace(0, 360, 30)  # 每12度取一个点
        
        for distance in distances:
            for angle in angles:
                # 计算候选点坐标
                rad = np.radians(angle)
                row = int(end_row + distance * np.sin(rad))
                col = int(end_col + distance * np.cos(rad))
                
                # 检查点的有效性
                if self.is_valid_point(row, col):
                    actual_dist = self.calculate_distance((row, col), end_point)
                    if actual_dist >= MIN_START_END_DISTANCE_METERS:
                        start_points.append((row, col))
                        print(f"找到有效起点: ({row}, {col}), 距离: {actual_dist/1000:.1f}公里")
                        
        print(f"共找到{len(start_points)}个有效起点")
        return start_points
        
    def generate_pairs(self):
        """生成起终点对"""
        # 确保数据已加载
        if self.dem is None:
            self.load_data()
            
        pairs = []
        
        # 检查每个战略位置是否有效
        valid_locations = []
        for i, loc in enumerate(STRATEGIC_LOCATIONS):
            if self.is_valid_point(loc[0], loc[1]):
                valid_locations.append(loc)
                landcover_type = self.landcover[loc[0], loc[1]]
                print(f"战略位置 {i+1}: {loc}, 有效, 土地类型: {landcover_type}")
            else:
                print(f"战略位置 {i+1}: {loc}, 无效")
        
        # 为每个有效的战略位置生成起点
        for i, end_point in enumerate(valid_locations):
            start_points = self.select_start_points(end_point)
            if start_points:
                # 为每个终点随机选择125个起点（如果有这么多的话）
                num_points = min(125, len(start_points))
                selected_points = np.random.choice(len(start_points), num_points, replace=False)
                for idx in selected_points:
                    start_point = start_points[idx]
                    pairs.append((start_point, end_point))
        
        # 保存结果
        if pairs:
            self.save_pairs(pairs)
            print(f"\n总共生成了 {len(pairs)} 对战略起终点对")
        else:
            print("警告：未能生成任何有效的起终点对")
            
        return pairs
            
    def save_pairs(self, pairs):
        """保存起终点对数据"""
        # 创建DataFrame
        data = []
        for start, end in pairs:
            distance = self.calculate_distance(start, end)
            
            # 获取土地覆盖类型和基础速度
            start_landcover = self.landcover[start] if self.landcover is not None else -1
            end_landcover = self.landcover[end] if self.landcover is not None else -1
            
            start_speed = self.base_speed[start] if self.base_speed is not None else 5.0
            end_speed = self.base_speed[end] if self.base_speed is not None else 5.0
            
            data.append({
                'start_row': start[0],
                'start_col': start[1],
                'end_row': end[0],
                'end_col': end[1],
                'distance_meters': distance,
                'start_landcover': start_landcover,
                'end_landcover': end_landcover,
                'start_speed': start_speed,
                'end_speed': end_speed
            })
        
        df = pd.DataFrame(data)
        
        # 保存到CSV
        output_path = os.path.join(OUTPUT_DIR, "strategic_pairs.csv")
        df.to_csv(output_path, index=False)
        print(f"\n战略起终点对数据已保存至: {output_path}")
        
def main():
    """主函数"""
    # 创建目录
    os.makedirs(os.path.join(OUTPUT_DIR, "strategic_trajectories"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "strategic_paths"), exist_ok=True)
    
    # 创建生成器
    generator = StrategicPairGenerator()
    
    # 生成起终点对
    pairs = generator.generate_pairs()
    
    print(f"\n总共生成了 {len(pairs)} 对战略起终点对")
    
if __name__ == "__main__":
    main() 