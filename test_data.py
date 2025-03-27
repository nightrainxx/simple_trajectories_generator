"""
测试脚本: 检查输入数据文件的有效性
"""

import rasterio
import numpy as np
from config import DEM_PATH, SLOPE_PATH, LANDCOVER_PATH

def check_file(file_path: str) -> None:
    """检查栅格文件"""
    print(f"\n检查文件: {file_path}")
    try:
        with rasterio.open(file_path) as src:
            print(f"- 形状: {src.shape}")
            print(f"- 坐标系: {src.crs}")
            print(f"- 数据类型: {src.dtypes}")
            data = src.read(1)
            print(f"- 数值范围: [{np.nanmin(data)}, {np.nanmax(data)}]")
            print(f"- 是否有NoData: {np.isnan(data).any()}")
    except Exception as e:
        print(f"错误: {str(e)}")

def main():
    """主函数"""
    print("开始检查数据文件...")
    
    # 检查DEM
    check_file(DEM_PATH)
    
    # 检查坡度
    check_file(SLOPE_PATH)
    
    # 检查土地覆盖
    check_file(LANDCOVER_PATH)

if __name__ == '__main__':
    main() 