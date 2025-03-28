"""
环境模块: 负责加载和处理环境数据,定义移动规则

包含:
- 环境数据加载和预处理
- 坡度分类
- 速度规则应用
"""

import numpy as np
import rasterio
from typing import Tuple, Dict, Optional
from config import (
    DEM_PATH, SLOPE_PATH, LANDCOVER_PATH,
    SLOPE_BINS, SLOPE_LABELS, SPEED_RULES,
    DEFAULT_SPEED_MPS
)

class Environment:
    """环境类: 管理环境数据和移动规则"""
    
    def __init__(self):
        """初始化环境对象"""
        self.dem = None
        self.slope = None
        self.landcover = None
        self.transform = None
        self.crs = None
        self.shape = None
        self.cost_map = None
        self.width = None
        self.height = None
        
    def load_data(self) -> None:
        """
        加载环境数据(DEM、坡度、土地覆盖)
        """
        # 加载DEM
        with rasterio.open(DEM_PATH) as src:
            self.dem = src.read(1)
            self.transform = src.transform
            self.crs = src.crs
            self.shape = self.dem.shape
            self.width = src.width
            self.height = src.height
            
            # 处理DEM的NoData值
            self.dem = np.where(
                self.dem < -1e30,
                np.nan,
                self.dem
            )
            
        # 加载坡度
        with rasterio.open(SLOPE_PATH) as src:
            self.slope = src.read(1)
            
            # 处理坡度的NoData值
            self.slope = np.where(
                self.slope < -999,
                np.nan,
                self.slope
            )
            
            # 打印坡度数据的基本信息
            print("\n坡度数据信息:")
            print(f"  形状: {self.slope.shape}")
            print(f"  数据类型: {self.slope.dtype}")
            print(f"  最小值: {np.nanmin(self.slope):.2f}°")
            print(f"  最大值: {np.nanmax(self.slope):.2f}°")
            print(f"  平均值: {np.nanmean(self.slope):.2f}°")
            print(f"  中位数: {np.nanmedian(self.slope):.2f}°")
            print(f"  标准差: {np.nanstd(self.slope):.2f}°")
            print(f"  NaN数量: {np.sum(np.isnan(self.slope))}")
            print(f"  0值数量: {np.sum(self.slope == 0)}")
            print(f"  负值数量: {np.sum(self.slope < 0)}")
            print(f"  大于45度数量: {np.sum(self.slope > 45)}")
            
            # 打印坡度分布
            bins = [0, 5, 15, 30, 45, np.inf]
            labels = ['平地', '缓坡', '中坡', '陡坡', '峭壁']
            hist, _ = np.histogram(self.slope[~np.isnan(self.slope)], bins=bins)
            print("\n坡度分布:")
            for i, (count, label) in enumerate(zip(hist, labels)):
                print(f"  {label}: {count} 像素 ({count/self.slope.size*100:.2f}%)")
            
        # 加载土地覆盖
        with rasterio.open(LANDCOVER_PATH) as src:
            self.landcover = src.read(1)
            
        # 验证数据形状一致性
        assert self.dem.shape == self.slope.shape == self.landcover.shape, \
            "输入栅格数据形状不一致"
            
        # 生成成本图
        self.generate_cost_map()
            
    def generate_cost_map(self) -> None:
        """
        生成成本图
        """
        if not self.validate_data():
            raise ValueError("环境数据无效，无法生成成本图")
            
        # 初始化成本图
        self.cost_map = np.full(self.shape, np.inf, dtype=np.float32)
        
        # 遍历每个像素
        for row in range(self.height):
            for col in range(self.width):
                # 如果是水体，设置为无穷大
                if self.landcover[row, col] == 60:
                    continue
                    
                # 如果坡度大于45度，设置为无穷大
                if self.slope[row, col] > 45:
                    continue
                    
                # 获取基础速度
                speed = self.get_pixel_speed(row, col)
                
                # 如果速度大于0，计算成本
                if speed > 0:
                    # 成本 = 距离/速度
                    # 这里假设像素大小为30米
                    self.cost_map[row, col] = 30.0 / speed
                else:
                    # 如果速度为0，设置一个较大但有限的成本
                    self.cost_map[row, col] = 1000.0
                    
        print("成本图生成完成")
        print(f"成本范围: [{np.min(self.cost_map)}, {np.max(self.cost_map)}]")
        print(f"无穷大值数量: {np.sum(np.isinf(self.cost_map))}")

    def classify_slope(self, slope_value: float) -> int:
        """
        将坡度值分类为坡度等级

        参数:
            slope_value: 坡度值(度)

        返回:
            坡度等级(0-4)
        """
        if np.isnan(slope_value):
            print(f"警告: 遇到NaN坡度值")
            return len(SLOPE_LABELS) - 1  # 将NoData值归类为最高坡度等级(不可通行)
        
        # 打印坡度值和分类结果
        slope_class = np.digitize(slope_value, SLOPE_BINS) - 1
        print(f"坡度值: {slope_value:.2f}°, 分类等级: {slope_class}")
        return slope_class
    
    def get_speed(
        self,
        landcover_code: int,
        slope_label: int
    ) -> float:
        """
        根据土地覆盖类型和坡度等级获取移动速度

        参数:
            landcover_code: 土地覆盖类型编码
            slope_label: 坡度等级(0-4)

        返回:
            移动速度(米/秒)
        """
        return SPEED_RULES.get(
            (landcover_code, slope_label),
            DEFAULT_SPEED_MPS
        )
    
    def get_pixel_speed(self, row: int, col: int) -> float:
        """
        获取指定像素位置的移动速度

        参数:
            row: 像素行号
            col: 像素列号

        返回:
            移动速度(米/秒)
        """
        # 获取该位置的坡度和土地覆盖类型
        slope_value = self.slope[row, col]
        landcover_code = self.landcover[row, col]
        
        # 如果是水体，返回0速度
        if landcover_code == 60:
            return 0.0
        
        # 如果坡度大于45度，返回0速度
        if slope_value > 45:
            return 0.0
        
        # 分类坡度
        slope_label = self.classify_slope(slope_value)
        
        # 获取速度规则
        speed = SPEED_RULES.get((landcover_code, slope_label))
        
        # 如果没有对应的规则，使用默认速度
        if speed is None:
            print(f"警告: 位置({row}, {col})的土地类型{landcover_code}和坡度等级{slope_label}没有对应的速度规则，使用默认速度")
            speed = DEFAULT_SPEED_MPS
        
        return speed
    
    def validate_data(self) -> bool:
        """
        验证环境数据的有效性

        返回:
            数据是否有效
        """
        if any(x is None for x in [
            self.dem, self.slope, self.landcover,
            self.transform, self.crs
        ]):
            print("部分数据未加载")
            return False
            
        # 检查数据范围
        valid_slope = np.logical_or(
            np.isnan(self.slope),
            np.logical_and(
                self.slope >= 0,
                self.slope <= 90
            )
        ).all()
        
        if not valid_slope:
            print("坡度数据范围无效")
            return False
            
        # 检查土地覆盖编码
        unique_codes = np.unique(self.landcover)
        print(f"土地覆盖编码: {unique_codes}")
        
        # 检查每个编码是否有对应的速度规则
        for code in unique_codes:
            if not any((code, slope_label) in SPEED_RULES for slope_label in SLOPE_LABELS):
                print(f"土地覆盖编码 {code} 没有对应的速度规则")
                return False
            
        return True 