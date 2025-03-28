"""
从DEM数据计算坡度和坡向

输入:
- DEM数据文件 (data/input/dem_30m_100km.tif)

输出:
- 坡度数据文件 (data/input/slope_30m_100km.tif)
- 坡向数据文件 (data/input/aspect_30m_100km.tif)
"""

import os
import numpy as np
import rasterio
from terrain_analyzer import TerrainAnalyzer
from config import INPUT_DIR

def main():
    """主函数"""
    print("开始从DEM计算坡度和坡向...")
    
    # 输入输出文件路径
    dem_path = os.path.join(INPUT_DIR, "dem_30m_100km.tif")
    slope_path = os.path.join(INPUT_DIR, "slope_30m_100km.tif")
    aspect_path = os.path.join(INPUT_DIR, "aspect_30m_100km.tif")
    
    # 创建TerrainAnalyzer实例
    analyzer = TerrainAnalyzer()
    
    # 加载DEM数据
    print(f"\n读取DEM文件: {dem_path}")
    analyzer.load_dem(dem_path)
    
    # 计算坡度和坡向
    print("\n计算坡度和坡向...")
    slope, aspect = analyzer.calculate_slope_aspect()
    
    # 保存结果
    print("\n保存计算结果...")
    analyzer.save_results(slope_path, aspect_path)
    print(f"坡度数据已保存至: {slope_path}")
    print(f"坡向数据已保存至: {aspect_path}")
    
    print("\n计算完成!")

if __name__ == "__main__":
    main() 