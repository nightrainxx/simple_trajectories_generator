"""
基础地图生成脚本

用于生成基础速度地图、成本地图等
输入：
- landcover_30m_100km.tif：土地覆盖数据
- slope.tif：坡度数据
- aspect.tif：坡向数据

输出：
- intermediate/base_speed_map.tif：基础速度地图
- intermediate/cost_map.tif：成本地图
"""

import os
import numpy as np
import rasterio
from config import (
    INPUT_DIR, 
    OUTPUT_DIR, 
    INTERMEDIATE_DIR,
    LANDCOVER_PATH,
    SLOPE_PATH,
    ASPECT_PATH,
    SPEED_RULES,
    SLOPE_LEVELS,
    IMPASSABLE_TYPES
)

def classify_slope(slope):
    """将坡度值分类为不同等级"""
    level = 0
    for threshold in SLOPE_LEVELS:
        if slope > threshold:
            level += 1
    return level

def generate_base_speed_map():
    """生成基础速度地图"""
    print("生成基础速度地图...")
    
    # 读取土地覆盖和坡度数据
    with rasterio.open(LANDCOVER_PATH) as src:
        landcover = src.read(1)
        meta = src.meta.copy()
    
    with rasterio.open(SLOPE_PATH) as src:
        slope = src.read(1)
    
    # 初始化速度图
    speed_map = np.zeros_like(landcover, dtype=np.float32)
    
    # 计算每个像素的速度
    for r in range(landcover.shape[0]):
        for c in range(landcover.shape[1]):
            lc_type = landcover[r, c]
            slope_val = slope[r, c]
            slope_level = classify_slope(slope_val)
            
            # 获取速度规则
            if (lc_type, slope_level) in SPEED_RULES:
                speed_map[r, c] = SPEED_RULES[(lc_type, slope_level)]
            else:
                # 默认使用未分类的规则
                speed_map[r, c] = SPEED_RULES.get((255, slope_level), 0.0)
    
    # 设置不可通行区域
    for lc_type in IMPASSABLE_TYPES:
        speed_map[landcover == lc_type] = 0.0
    
    # 保存速度图
    meta.update({
        'dtype': 'float32',
        'count': 1
    })
    
    output_path = os.path.join(INTERMEDIATE_DIR, "base_speed_map.tif")
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(speed_map, 1)
    
    print(f"基础速度地图已保存至: {output_path}")
    
    return speed_map

def generate_cost_map(speed_map):
    """根据速度地图生成成本地图"""
    print("生成成本地图...")
    
    # 读取元数据
    with rasterio.open(LANDCOVER_PATH) as src:
        meta = src.meta.copy()
    
    # 计算成本 (成本 = 1/速度)
    cost_map = np.zeros_like(speed_map)
    
    # 对可通行区域计算成本
    mask = speed_map > 0
    cost_map[mask] = 1.0 / speed_map[mask]
    
    # 设置不可通行区域为无穷大
    cost_map[~mask] = np.inf
    
    # 保存成本图
    meta.update({
        'dtype': 'float32',
        'count': 1
    })
    
    output_path = os.path.join(INTERMEDIATE_DIR, "cost_map.tif")
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(cost_map, 1)
    
    print(f"成本地图已保存至: {output_path}")
    
    return cost_map

def main():
    """主函数"""
    # 创建中间目录
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    
    # 生成基础速度地图
    speed_map = generate_base_speed_map()
    
    # 生成成本地图
    cost_map = generate_cost_map(speed_map)
    
    print("地图生成完成！")

if __name__ == "__main__":
    main() 