"""
工具函数模块: 提供项目中需要的通用功能函数

包含:
- 坐标转换函数
- 距离计算函数
- 数据验证函数
"""

import numpy as np
import rasterio
from typing import Tuple, Union, List, Optional

def pixel_to_geo(
    transform: rasterio.Affine,
    row: int,
    col: int
) -> Tuple[float, float]:
    """
    将像素坐标转换为地理坐标

    参数:
        transform: rasterio的仿射变换矩阵
        row: 像素行号
        col: 像素列号

    返回:
        (lon, lat): 经度和纬度
    """
    lon, lat = transform * (col + 0.5, row + 0.5)
    return lon, lat

def geo_to_pixel(
    transform: rasterio.Affine,
    lon: float,
    lat: float
) -> Tuple[int, int]:
    """
    将地理坐标转换为像素坐标

    参数:
        transform: rasterio的仿射变换矩阵
        lon: 经度
        lat: 纬度

    返回:
        (row, col): 像素行号和列号
    """
    col, row = ~transform * (lon, lat)
    return int(row), int(col)

def haversine_distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float]
) -> float:
    """
    计算两个经纬度点之间的距离(米)

    参数:
        point1: 第一个点的坐标 (lon1, lat1)
        point2: 第二个点的坐标 (lon2, lat2)

    返回:
        两点间的距离(米)
    """
    # 地球平均半径(米)
    R = 6371000
    
    # 将经纬度转换为弧度
    lon1, lat1 = np.radians(point1)
    lon2, lat2 = np.radians(point2)
    
    # 计算经纬度差
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Haversine公式
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def calculate_distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    transform: Optional[rasterio.Affine] = None
) -> float:
    """
    计算两点间的距离

    参数:
        point1: 第一个点的坐标 (x1, y1)
        point2: 第二个点的坐标 (x2, y2)
        transform: rasterio的仿射变换矩阵,用于将像素坐标转换为地理坐标

    返回:
        两点间的距离(米)
    """
    if transform is not None:
        # 将像素坐标转换为地理坐标
        geo1 = pixel_to_geo(transform, point1[0], point1[1])
        geo2 = pixel_to_geo(transform, point2[0], point2[1])
        return haversine_distance(geo1, geo2)
    else:
        # 直接计算欧几里得距离
        dx = point2[1] - point1[1]
        dy = point2[0] - point1[0]
        return np.sqrt(dx*dx + dy*dy)

def validate_point(
    point: Tuple[int, int],
    shape: Tuple[int, int]
) -> bool:
    """
    验证点坐标是否在栅格范围内

    参数:
        point: 点坐标 (row, col)
        shape: 栅格形状 (height, width)

    返回:
        是否有效
    """
    row, col = point
    height, width = shape
    return (0 <= row < height) and (0 <= col < width)

def get_neighbors(
    point: Tuple[int, int],
    shape: Tuple[int, int],
    diagonal: bool = True
) -> List[Tuple[int, int]]:
    """
    获取栅格中某点的邻居像素坐标

    参数:
        point: 中心点坐标 (row, col)
        shape: 栅格形状 (height, width)
        diagonal: 是否包含对角线方向的邻居

    返回:
        邻居坐标列表
    """
    row, col = point
    height, width = shape
    neighbors = []

    # 上下左右四个方向
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1)
    ]
    
    # 如果包含对角线方向
    if diagonal:
        directions.extend([
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ])

    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if validate_point((new_row, new_col), shape):
            neighbors.append((new_row, new_col))

    return neighbors 