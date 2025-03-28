"""
环境地图生成器模块：负责构建基础速度图和成本图

输入：
- 土地覆盖数据
- 坡度数据
- 坡向数据

输出：
- 基础速度图
- 成本图

处理流程：
1. 加载环境数据
2. 根据规则计算基础速度
3. 应用坡度影响生成成本图
"""

import os
import numpy as np
import rasterio
from typing import Tuple, Optional
from config import (
    INPUT_DIR,
    LANDCOVER_PATH,
    SLOPE_PATH,
    ASPECT_PATH,
    BASE_SPEED_RULES,
    SLOPE_SPEED_REDUCTION_FACTOR,
    IMPASSABLE_LANDCOVER_CODES,
    MAX_CROSS_SLOPE_DEGREES
)

class EnvironmentMapper:
    """环境地图生成器类"""
    
    def __init__(self):
        """初始化环境地图生成器"""
        self.landcover = None
        self.slope = None
        self.aspect = None
        self.transform = None
        self.crs = None
        self.base_speed_map = None
        self.cost_map = None
        
    def load_data(self) -> None:
        """加载环境数据"""
        print("加载环境数据...")
        
        # 加载土地覆盖数据
        with rasterio.open(LANDCOVER_PATH) as src:
            self.landcover = src.read(1)
            self.transform = src.transform
            self.crs = src.crs
            print(f"土地覆盖数据形状: {self.landcover.shape}")
            print(f"土地覆盖类型: {np.unique(self.landcover)}")
            
        # 加载坡度数据
        with rasterio.open(SLOPE_PATH) as src:
            self.slope = src.read(1)
            print(f"\n坡度数据统计:")
            print(f"最小值: {np.nanmin(self.slope):.2f}°")
            print(f"最大值: {np.nanmax(self.slope):.2f}°")
            print(f"平均值: {np.nanmean(self.slope):.2f}°")
            
        # 加载坡向数据
        with rasterio.open(ASPECT_PATH) as src:
            self.aspect = src.read(1)
            print(f"\n坡向数据范围: [{np.nanmin(self.aspect):.2f}°, {np.nanmax(self.aspect):.2f}°]")
            
    def calculate_base_speed_map(self) -> np.ndarray:
        """
        计算基础速度图
        
        返回:
            base_speed_map: 基础速度数组 (m/s)
        """
        print("\n计算基础速度图...")
        
        # 初始化基础速度图
        self.base_speed_map = np.zeros_like(self.landcover, dtype=np.float32)
        
        # 应用土地覆盖规则
        for code, speed in BASE_SPEED_RULES.items():
            mask = (self.landcover == code)
            self.base_speed_map[mask] = speed
            
        # 标记不可通行区域
        for code in IMPASSABLE_LANDCOVER_CODES:
            mask = (self.landcover == code)
            self.base_speed_map[mask] = 0
            
        print(f"基础速度范围: [{np.min(self.base_speed_map):.2f}, {np.max(self.base_speed_map):.2f}] m/s")
        
        return self.base_speed_map
        
    def calculate_cost_map(self) -> np.ndarray:
        """
        计算成本图
        
        返回:
            cost_map: 成本数组 (秒/米)
        """
        print("\n计算成本图...")
        
        if self.base_speed_map is None:
            self.calculate_base_speed_map()
            
        # 初始化成本图
        self.cost_map = np.full_like(self.base_speed_map, np.inf, dtype=np.float32)
        
        # 计算坡度影响因子
        slope_factor = np.exp(SLOPE_SPEED_REDUCTION_FACTOR * self.slope)
        
        # 计算有效速度（考虑坡度影响）
        valid_mask = (self.base_speed_map > 0)
        self.cost_map[valid_mask] = 1.0 / (self.base_speed_map[valid_mask] / slope_factor[valid_mask])
        
        # 标记不可通行区域（坡度过大）
        steep_mask = (self.slope > MAX_CROSS_SLOPE_DEGREES)
        self.cost_map[steep_mask] = np.inf
        
        # 统计信息
        finite_costs = self.cost_map[np.isfinite(self.cost_map)]
        print(f"成本范围: [{np.min(finite_costs):.2f}, {np.max(finite_costs):.2f}]")
        print(f"无穷大成本数量: {np.sum(np.isinf(self.cost_map))}")
        
        return self.cost_map
        
    def save_maps(self, output_dir: str) -> None:
        """
        保存地图

        参数:
            output_dir: 输出目录
        """
        if self.base_speed_map is None or self.cost_map is None:
            raise ValueError("请先计算速度图和成本图")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存基础速度图
        with rasterio.open(
            os.path.join(output_dir, 'base_speed_map.tif'),
            'w',
            driver='GTiff',
            height=self.base_speed_map.shape[0],
            width=self.base_speed_map.shape[1],
            count=1,
            dtype=self.base_speed_map.dtype,
            crs=self.crs,
            transform=self.transform
        ) as dst:
            dst.write(self.base_speed_map, 1)
            
        # 保存成本图
        with rasterio.open(
            os.path.join(output_dir, 'cost_map.tif'),
            'w',
            driver='GTiff',
            height=self.cost_map.shape[0],
            width=self.cost_map.shape[1],
            count=1,
            dtype=self.cost_map.dtype,
            crs=self.crs,
            transform=self.transform
        ) as dst:
            dst.write(self.cost_map, 1)
            
def main():
    """主函数：运行环境地图生成流程"""
    # 创建环境地图生成器
    mapper = EnvironmentMapper()
    
    # 加载数据
    mapper.load_data()
    
    # 计算基础速度图
    mapper.calculate_base_speed_map()
    
    # 计算成本图
    mapper.calculate_cost_map()
    
    # 保存结果
    output_dir = os.path.join("data", "output", "intermediate")
    mapper.save_maps(output_dir)
    
    print("\n环境地图生成完成！")
    
if __name__ == "__main__":
    main() 