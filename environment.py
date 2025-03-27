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
            
        # 加载土地覆盖
        with rasterio.open(LANDCOVER_PATH) as src:
            self.landcover = src.read(1)
            
        # 验证数据形状一致性
        assert self.dem.shape == self.slope.shape == self.landcover.shape, \
            "输入栅格数据形状不一致"
            
    def classify_slope(self, slope_value: float) -> int:
        """
        将坡度值分类为坡度等级

        参数:
            slope_value: 坡度值(度)

        返回:
            坡度等级(0-4)
        """
        if np.isnan(slope_value):
            return len(SLOPE_LABELS) - 1  # 将NoData值归类为最高坡度等级(不可通行)
        return np.digitize(slope_value, SLOPE_BINS) - 1
    
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
    
    def get_pixel_speed(
        self,
        row: int,
        col: int
    ) -> float:
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
        
        # 如果是NoData区域,返回0速度(不可通行)
        if np.isnan(slope_value) or np.isnan(self.dem[row, col]):
            return 0.0
            
        # 分类坡度
        slope_label = self.classify_slope(slope_value)
        
        # 获取速度
        return self.get_speed(landcover_code, slope_label)
    
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
            return False
            
        # 检查土地覆盖编码是否在合理范围内
        valid_codes = set(code for code, _ in SPEED_RULES.keys())
        if not all(code in valid_codes for code in np.unique(self.landcover)):
            return False
            
        return True 