"""
地图生成器模块：负责生成速度图和成本图

功能：
- 生成基础速度图
- 应用坡向约束
- 生成成本图
- 可视化结果
"""

import numpy as np
import rasterio
from typing import Tuple, Optional
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from config import (
    SLOPE_PATH,
    ASPECT_PATH,
    LANDCOVER_PATH,
    SPEED_MAP_PATH,
    COST_MAP_PATH,
    LANDCOVER_CODES,
    SLOPE_SPEED_REDUCTION_FACTOR,
    ASPECT_UPHILL_REDUCTION_K,
    ASPECT_CROSS_REDUCTION_K,
    MAX_CROSS_SLOPE_DEGREES,
    MAX_BRAKING_SPEED_DOWNHILL,
    IMPASSABLE_LANDCOVER_CODES,
    DEFAULT_SPEED_MPS
)

# 定义土地覆盖类型的基础速度（m/s）
BASE_SPEED_RULES = {
    LANDCOVER_CODES['CROPLAND']: 8.3,     # 30 km/h
    LANDCOVER_CODES['FOREST']: 2.8,       # 10 km/h
    LANDCOVER_CODES['GRASSLAND']: 5.6,    # 20 km/h
    LANDCOVER_CODES['SHRUBLAND']: 4.2,    # 15 km/h
    LANDCOVER_CODES['WETLAND']: 1.1,      # 4 km/h
    LANDCOVER_CODES['WATER']: 0.0,        # 不可通行
    LANDCOVER_CODES['ARTIFICIAL']: 16.7,  # 60 km/h
    LANDCOVER_CODES['BARELAND']: 1.4,     # 5 km/h
    LANDCOVER_CODES['NODATA']: 0.0        # 不可通行
}

class MapGenerator:
    """地图生成器类：生成速度图和成本图"""
    
    def __init__(self):
        """初始化地图生成器"""
        self.landcover = None
        self.elevation = None
        self.slope = None
        self.speed_map = None
        self.transform = None
        self.crs = None
        self.base_speed_map = None
        self.cost_map = None
        self.base_speeds = {
            10: 1.4,  # 水域
            20: 1.4,  # 城市
            30: 1.4,  # 裸地
            40: 1.4,  # 灌木
            50: 1.4,  # 湿地
            60: 1.4,  # 草地
            70: 1.0,  # 农田
            80: 0.7,  # 森林
            90: 0.0   # 冰雪
        }
        
    def load_data(self):
        """加载环境数据"""
        # 加载土地覆盖数据
        landcover_path = "data/input/landcover_30m_100km.tif"
        with rasterio.open(landcover_path) as src:
            self.landcover = src.read(1)
            self.transform = src.transform
            self.crs = src.crs
            
        # 加载高程数据
        elevation_path = "data/input/dem_30m_100km.tif"
        with rasterio.open(elevation_path) as src:
            self.elevation = src.read(1)
            
        # 计算坡度
        self.slope = self._calculate_slope()
        
        # 生成速度地图
        self.speed_map = self._generate_speed_map()
        
        # 生成成本图
        self.cost_map = self._generate_cost_map()
        
    def _calculate_slope(self):
        """计算坡度（度）"""
        if self.elevation is None:
            raise ValueError("需要先加载高程数据")
        
        # 计算x和y方向的梯度
        dy, dx = np.gradient(self.elevation)
        
        # 计算坡度（度）
        slope = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))
        return slope

    def _generate_speed_map(self):
        """生成速度地图"""
        if self.landcover is None or self.slope is None:
            raise ValueError("需要先加载土地覆盖数据和计算坡度")
        
        # 初始化速度地图
        speed_map = np.zeros_like(self.landcover, dtype=float)
        
        # 根据土地覆盖类型设置基础速度
        for lc_type, base_speed in self.base_speeds.items():
            mask = (self.landcover == lc_type)
            speed_map[mask] = base_speed
        
        # 根据坡度调整速度
        # 坡度大于45度时速度为0
        speed_map[self.slope > 45] = 0
        
        # 坡度在0-45度之间时，速度随坡度增加而线性减小
        slope_factor = 1 - self.slope / 45
        slope_factor = np.clip(slope_factor, 0, 1)
        speed_map *= slope_factor
        
        return speed_map
        
    def generate_base_speed_map(self) -> np.ndarray:
        """
        生成基础速度图

        返回:
            基础速度图数组
        """
        if self.landcover is None:
            raise ValueError("请先加载环境数据")
            
        # 初始化速度图
        self.base_speed_map = np.full_like(self.landcover, DEFAULT_SPEED_MPS, dtype=np.float32)
        
        # 应用基础速度规则
        for landcover_code, speed in BASE_SPEED_RULES.items():
            self.base_speed_map[self.landcover == landcover_code] = speed
            
        # 处理不可通行区域
        for code in IMPASSABLE_LANDCOVER_CODES:
            self.base_speed_map[self.landcover == code] = 0.0
            
        return self.base_speed_map
        
    def calculate_slope_factors(
        self,
        heading_degrees: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算坡度影响因子

        参数:
            heading_degrees: 行进方向(度)

        返回:
            along_slope: 沿途坡度
            cross_slope: 横向坡度
        """
        if self.slope is None or self.speed_map is None:
            raise ValueError("请先加载环境数据")
            
        # 将行进方向转换为弧度
        heading_rad = np.radians(heading_degrees)
        aspect_rad = np.radians(self.speed_map)
        slope_rad = np.radians(self.slope)
        
        # 计算沿途坡度和横向坡度
        direction_diff = aspect_rad - heading_rad
        along_slope = np.abs(self.slope * np.cos(direction_diff))
        cross_slope = np.abs(self.slope * np.sin(direction_diff))
        
        return along_slope, cross_slope
        
    def apply_slope_aspect_constraints(
        self,
        heading_degrees: float
    ) -> np.ndarray:
        """
        应用坡向约束

        参数:
            heading_degrees: 行进方向(度)

        返回:
            调整后的速度图
        """
        if self.base_speed_map is None:
            raise ValueError("请先生成基础速度图")
            
        # 计算坡度影响
        along_slope, cross_slope = self.calculate_slope_factors(heading_degrees)
        
        # 计算上坡减速因子
        uphill_reduction = np.maximum(0.1, 1 - ASPECT_UPHILL_REDUCTION_K * along_slope)
        
        # 计算横坡减速因子
        cross_reduction = np.maximum(0.05, 1 - ASPECT_CROSS_REDUCTION_K * cross_slope**2)
        
        # 应用坡度约束
        speed_map = self.base_speed_map * uphill_reduction * cross_reduction
        
        # 应用下坡制动限制
        speed_map = np.minimum(speed_map, MAX_BRAKING_SPEED_DOWNHILL)
        
        # 应用横坡阈值限制
        speed_map[cross_slope > MAX_CROSS_SLOPE_DEGREES] = 0.0
        
        return speed_map
        
    def generate_cost_map(
        self,
        heading_degrees: float = 0.0,
        max_cost: float = 1e6
    ) -> np.ndarray:
        """
        生成成本图

        参数:
            heading_degrees: 行进方向(度)
            max_cost: 不可通行区域的成本值

        返回:
            成本图数组
        """
        # 生成基础速度图
        self.generate_base_speed_map()
        
        # 应用坡向约束
        speed_map = self.apply_slope_aspect_constraints(heading_degrees)
        
        # 计算成本图
        self.cost_map = np.where(
            speed_map > 1e-6,
            1.0 / speed_map,  # 成本为单位距离所需时间
            max_cost  # 不可通行区域设置较大成本
        )
        
        return self.cost_map
        
    def save_maps(self) -> None:
        """保存速度图和成本图"""
        if self.base_speed_map is None or self.cost_map is None:
            raise ValueError("请先生成速度图和成本图")
            
        # 创建输出目录
        os.makedirs(os.path.dirname(SPEED_MAP_PATH), exist_ok=True)
        
        # 保存基础速度图
        with rasterio.open(
            SPEED_MAP_PATH,
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
            COST_MAP_PATH,
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
            
    def visualize_maps(self, output_dir: str) -> None:
        """
        可视化地图

        参数:
            output_dir: 输出目录
        """
        if self.base_speed_map is None or self.cost_map is None:
            raise ValueError("请先生成速度图和成本图")
            
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置字体
        times_font = FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf')
        simsun_font = FontProperties(fname='/usr/share/fonts/truetype/custom/simsun.ttc')
        
        # 绘制基础速度图
        plt.figure(figsize=(12, 8))
        plt.imshow(self.base_speed_map, cmap='viridis')
        plt.colorbar(label='速度 (m/s)')
        plt.title('基础速度图', fontproperties=simsun_font, fontsize=16)
        plt.xticks(fontproperties=times_font, fontsize=16)
        plt.yticks(fontproperties=times_font, fontsize=16)
        plt.savefig(os.path.join(output_dir, 'base_speed_map.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制成本图
        plt.figure(figsize=(12, 8))
        plt.imshow(self.cost_map, cmap='magma')
        plt.colorbar(label='成本 (s/m)')
        plt.title('通行成本图', fontproperties=simsun_font, fontsize=16)
        plt.xticks(fontproperties=times_font, fontsize=16)
        plt.yticks(fontproperties=times_font, fontsize=16)
        plt.savefig(os.path.join(output_dir, 'cost_map.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _generate_cost_map(self):
        """生成成本图"""
        if self.speed_map is None:
            raise ValueError("需要先生成速度地图")
        
        # 成本是速度的倒数
        # 为了避免除以0，我们先将速度为0的地方设为一个很小的数
        speed_map = self.speed_map.copy()
        speed_map[speed_map == 0] = 1e-6
        
        cost_map = 1.0 / speed_map
        return cost_map

def main():
    """主函数：测试地图生成器"""
    # 创建地图生成器
    generator = MapGenerator()
    
    # 加载数据
    print("加载环境数据...")
    generator.load_data()
    
    # 生成地图
    print("生成速度图和成本图...")
    generator.generate_cost_map(heading_degrees=45.0)  # 测试45度方向的地图
    
    # 保存地图
    print("保存地图文件...")
    generator.save_maps()
    
    # 可视化结果
    print("生成可视化图表...")
    generator.visualize_maps(os.path.join("data", "output", "maps"))
    
    print("地图生成完成！")
    
if __name__ == "__main__":
    main() 