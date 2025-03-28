"""
点选择器模块：负责选择合适的起终点对

输入：
- 土地覆盖数据
- 基础速度图
- 成本图

输出：
- 起终点对列表

处理流程：
1. 选择终点（靠近城市区域）
2. 为每个终点选择合适的起点
3. 验证并保存起终点对
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import rasterio
from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    LANDCOVER_PATH,
    URBAN_LANDCOVER_CODES,
    IMPASSABLE_LANDCOVER_CODES,
    NUM_END_POINTS,
    MIN_START_END_DISTANCE_METERS
)

class PointSelector:
    """点选择器类"""
    
    def __init__(self):
        """初始化点选择器"""
        self.landcover = None
        self.base_speed = None
        self.cost = None
        self.transform = None
        self.shape = None
        self.pixel_size = 30.0  # 固定栅格大小为30米
        
    def load_data(self) -> None:
        """加载数据"""
        print("加载环境数据...")
        
        # 加载土地覆盖数据
        with rasterio.open(LANDCOVER_PATH) as src:
            self.landcover = src.read(1)
            self.transform = src.transform
            self.shape = self.landcover.shape
            print(f"数据形状: {self.shape}")
            print(f"像素大小: {self.pixel_size:.2f}米")
            
        # 加载基础速度图
        speed_path = os.path.join(OUTPUT_DIR, "intermediate", "base_speed_map.tif")
        with rasterio.open(speed_path) as src:
            self.base_speed = src.read(1)
            
        # 加载成本图
        cost_path = os.path.join(OUTPUT_DIR, "intermediate", "cost_map.tif")
        with rasterio.open(cost_path) as src:
            self.cost = src.read(1)
            
    def is_valid_point(self, row: int, col: int) -> bool:
        """
        检查点是否有效
        
        参数:
            row: 行索引
            col: 列索引
            
        返回:
            bool: 是否为有效点
        """
        # 检查边界
        if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
            return False
            
        # 检查是否可通行
        if self.landcover[row, col] in IMPASSABLE_LANDCOVER_CODES:
            return False
            
        # 检查是否有基础速度
        if self.base_speed[row, col] <= 0:
            return False
            
        # 检查是否有有限成本
        if not np.isfinite(self.cost[row, col]):
            return False
            
        return True
        
    def calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """
        计算两点间的欧氏距离（米）
        
        参数:
            point1: 第一个点的(行,列)坐标
            point2: 第二个点的(行,列)坐标
            
        返回:
            float: 距离（米）
        """
        row1, col1 = point1
        row2, col2 = point2
        pixel_dist = np.sqrt((row2 - row1)**2 + (col2 - col1)**2)
        return pixel_dist * self.pixel_size
        
    def select_end_points(self) -> List[Tuple[int, int]]:
        """
        选择终点
        
        返回:
            List[Tuple[int, int]]: 终点列表，每个元素为(行,列)坐标
        """
        print("\n选择终点...")
        end_points = []
        
        # 找到所有城市区域的点
        urban_points = []
        for code in URBAN_LANDCOVER_CODES:
            points = np.argwhere(self.landcover == code)
            for point in points:
                if self.is_valid_point(point[0], point[1]):
                    urban_points.append(tuple(point))
                    
        if not urban_points:
            raise ValueError("未找到有效的城市区域点")
            
        # 随机选择指定数量的终点
        if len(urban_points) > NUM_END_POINTS:
            selected_indices = np.random.choice(len(urban_points), NUM_END_POINTS, replace=False)
            end_points = [urban_points[i] for i in selected_indices]
        else:
            end_points = urban_points
            
        print(f"已选择 {len(end_points)} 个终点")
        for i, point in enumerate(end_points):
            print(f"终点 {i+1}: {point}, 土地类型: {self.landcover[point]}, 基础速度: {self.base_speed[point]:.1f} m/s")
        return end_points
        
    def select_start_points(self, end_point: Tuple[int, int], num_attempts: int = 1000) -> List[Tuple[int, int]]:
        """
        为终点选择起点
        
        参数:
            end_point: 终点坐标(行,列)
            num_attempts: 尝试次数
            
        返回:
            List[Tuple[int, int]]: 起点列表
        """
        start_points = []
        end_row, end_col = end_point
        
        # 计算搜索范围（像素）
        min_pixels = int(MIN_START_END_DISTANCE_METERS / self.pixel_size)
        max_pixels = int(min_pixels * 1.5)  # 设置最大距离为最小距离的1.5倍
        
        print(f"\n为终点{end_point}选择起点")
        print(f"搜索范围: {min_pixels}-{max_pixels}像素 ({MIN_START_END_DISTANCE_METERS/1000:.1f}-{MIN_START_END_DISTANCE_METERS*1.5/1000:.1f}公里)")
        
        # 在不同距离和角度上尝试选择起点
        for distance in [min_pixels, (min_pixels + max_pixels)//2, max_pixels]:
            for angle in np.linspace(0, 360, 12):  # 每30度取一个点
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
        
    def select_start_end_pairs(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        选择起终点对
        
        返回:
            List[Tuple[Tuple[int, int], Tuple[int, int]]]: 起终点对列表
        """
        # 选择终点
        end_points = self.select_end_points()
        
        # 为每个终点选择起点
        pairs = []
        for end_point in end_points:
            start_points = self.select_start_points(end_point)
            if start_points:
                # 为每个终点随机选择一个起点
                start_point = start_points[np.random.randint(len(start_points))]
                pairs.append((start_point, end_point))
                
        # 保存结果
        self.save_pairs(pairs)
        
        return pairs
        
    def save_pairs(self, pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> None:
        """
        保存起终点对
        
        参数:
            pairs: 起终点对列表
        """
        # 创建DataFrame
        data = []
        for start, end in pairs:
            distance = self.calculate_distance(start, end)
            data.append({
                'start_row': start[0],
                'start_col': start[1],
                'end_row': end[0],
                'end_col': end[1],
                'distance_meters': distance,
                'start_landcover': self.landcover[start],
                'end_landcover': self.landcover[end],
                'start_speed': self.base_speed[start],
                'end_speed': self.base_speed[end]
            })
            
        df = pd.DataFrame(data)
        
        # 保存到CSV
        output_path = os.path.join(OUTPUT_DIR, "start_end_pairs.csv")
        df.to_csv(output_path, index=False)
        print(f"\n起终点对数据已保存至: {output_path}")
        
def main():
    """主函数：运行点选择流程"""
    # 创建点选择器
    selector = PointSelector()
    
    # 加载数据
    selector.load_data()
    
    # 选择起终点对
    pairs = selector.select_start_end_pairs()
    
    print("\n点选择完成！")
    
if __name__ == "__main__":
    main() 