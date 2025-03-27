"""
地图生成器模块: 负责生成速度图和成本图

包含:
- 速度图生成
- 成本图生成
- 地图可视化
"""

import numpy as np
import rasterio
from tqdm import tqdm
from typing import Tuple
import matplotlib.pyplot as plt
import os
from config import SPEED_RULES, DEFAULT_SPEED_MPS, SPEED_MAP_PATH, COST_MAP_PATH
from environment import Environment

class MapGenerator:
    """地图生成器类: 生成速度图和成本图"""
    
    def __init__(self):
        """初始化地图生成器"""
        self.speed_map = None
        self.cost_map = None
        
    def classify_slope(self, slope: float) -> int:
        """
        将坡度分类

        参数:
            slope: 坡度值(度)

        返回:
            坡度等级(0-4)
        """
        if slope < 0:  # 无效值
            return 4
        elif slope < 5:
            return 0  # 平地
        elif slope < 15:
            return 1  # 缓坡
        elif slope < 30:
            return 2  # 中坡
        elif slope < 45:
            return 3  # 陡坡
        else:
            return 4  # 悬崖
            
    def get_pixel_speed(
        self,
        dem: float,
        slope: float,
        landcover: int
    ) -> float:
        """
        计算像素点的速度

        参数:
            dem: 高程值(米)
            slope: 坡度值(度)
            landcover: 土地覆盖类型

        返回:
            速度值(米/秒)
        """
        # 处理无效值
        if np.isnan(dem) or np.isnan(slope):
            return 0.0
            
        # 获取坡度等级
        slope_level = self.classify_slope(slope)
        
        # 查找速度规则
        speed_key = (landcover, slope_level)
        if speed_key in SPEED_RULES:
            return SPEED_RULES[speed_key]
        else:
            return DEFAULT_SPEED_MPS  # 使用默认速度
        
    def generate_cost_map(
        self,
        dem: np.ndarray,
        slope: np.ndarray,
        landcover: np.ndarray,
        max_cost: float = 1e6
    ) -> np.ndarray:
        """
        生成成本图

        参数:
            dem: 高程图
            slope: 坡度图
            landcover: 土地覆盖图
            max_cost: 不可通行区域的成本值

        返回:
            成本图数组
        """
        height, width = dem.shape
        self.speed_map = np.zeros((height, width), dtype=np.float32)
        
        # 计算每个像素的速度
        for row in range(height):
            for col in range(width):
                self.speed_map[row, col] = self.get_pixel_speed(
                    dem[row, col],
                    slope[row, col],
                    landcover[row, col]
                )
                
        # 处理无效速度值
        self.speed_map = np.where(
            np.isnan(self.speed_map) | (self.speed_map <= 0),
            1e-6,  # 设置一个很小的正数
            self.speed_map
        )
        
        # 计算成本
        self.cost_map = np.where(
            self.speed_map > 1e-6,
            1.0 / self.speed_map,  # 成本为单位距离所需时间
            max_cost  # 不可通行区域设置较大成本
        )
        
        return self.cost_map
    
    def save_maps(self) -> None:
        """
        保存速度图和成本图
        """
        # 确保地图已生成
        if self.speed_map is None:
            self.generate_cost_map()
            
        # 保存速度图
        with rasterio.open(
            SPEED_MAP_PATH,
            'w',
            driver='GTiff',
            height=self.env.shape[0],
            width=self.env.shape[1],
            count=1,
            dtype=self.speed_map.dtype,
            crs=self.env.crs,
            transform=self.env.transform
        ) as dst:
            dst.write(self.speed_map, 1)
            
        # 保存成本图
        with rasterio.open(
            COST_MAP_PATH,
            'w',
            driver='GTiff',
            height=self.env.shape[0],
            width=self.env.shape[1],
            count=1,
            dtype=self.cost_map.dtype,
            crs=self.env.crs,
            transform=self.env.transform
        ) as dst:
            dst.write(self.cost_map, 1)
            
    def validate_maps(self) -> bool:
        """
        验证生成的地图是否有效

        返回:
            地图是否有效
        """
        if self.speed_map is None or self.cost_map is None:
            return False
            
        # 检查数值范围
        if (self.speed_map < 0).any():
            return False
            
        if (self.cost_map < 0).any():
            return False
            
        # 检查速度图和成本图的对应关系
        pixel_size = abs(self.env.transform[0])
        valid_mask = self.speed_map > 1e-6
        
        cost_check = np.allclose(
            self.cost_map[valid_mask],
            1.0 / self.speed_map[valid_mask]
        )
        
        return cost_check 

    def visualize_maps(self, output_dir: str) -> None:
        """
        可视化速度图和成本图

        参数:
            output_dir: 输出目录
        """
        if self.speed_map is None or self.cost_map is None:
            raise ValueError("请先生成速度图和成本图")
            
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 可视化速度图
        plt.figure(figsize=(12, 8))
        plt.imshow(self.speed_map, cmap='viridis')
        plt.colorbar(label='速度 (m/s)')
        plt.title('典型速度图')
        plt.savefig(os.path.join(output_dir, 'typical_speed_map.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 可视化成本图
        plt.figure(figsize=(12, 8))
        plt.imshow(self.cost_map, cmap='magma')
        plt.colorbar(label='成本 (s/m)')
        plt.title('通行成本图')
        plt.savefig(os.path.join(output_dir, 'cost_map.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存为GeoTIFF文件
        if hasattr(self, 'transform') and hasattr(self, 'crs'):
            # 保存速度图
            with rasterio.open(
                os.path.join(output_dir, 'typical_speed_map.tif'),
                'w',
                driver='GTiff',
                height=self.speed_map.shape[0],
                width=self.speed_map.shape[1],
                count=1,
                dtype=self.speed_map.dtype,
                crs=self.crs,
                transform=self.transform
            ) as dst:
                dst.write(self.speed_map, 1)
                
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