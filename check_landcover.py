"""
检查地表覆盖数据的脚本
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio

def create_landcover_colormap():
    """创建土地覆盖类型的颜色映射"""
    colors = {
        10: ('#0077BE', '水域'),      # 蓝色
        20: ('#80CCFF', '湿地'),      # 浅蓝色
        30: ('#90EE90', '草地'),      # 浅绿色
        40: ('#228B22', '灌木地'),    # 深绿色
        50: ('#CD5C5C', '建筑用地'),  # 红褐色
        60: ('#FFD700', '农田'),      # 金黄色
        80: ('#006400', '森林'),      # 深绿色
        90: ('#DEB887', '荒地'),      # 棕色
        255: ('#808080', '未分类')    # 灰色
    }
    return colors

def main():
    # 加载数据
    print("加载地表覆盖数据...")
    with rasterio.open("data/input/landcover_30m_100km.tif") as src:
        landcover = src.read(1)
        
        print(f"\n栅格数据基本信息:")
        print(f"数据形状: {src.shape}")
        print(f"坐标系统: {src.crs}")
        print(f"变换矩阵: {src.transform}")
        print(f"波段数量: {src.count}")
        print(f"数据类型: {src.dtypes[0]}")
        print(f"无效值: {src.nodata}")
        print(f"有效数据范围: {src.bounds}")
        
        # 检查数据分布
        unique, counts = np.unique(landcover, return_counts=True)
        print("\n地表类型分布:")
        for val, count in zip(unique, counts):
            percentage = count / landcover.size * 100
            print(f"类型代码 {val}: {count} 像素 ({percentage:.2f}%)")
        
        # 根据配置文件中的LANDCOVER_CODES补充名称
        try:
            from config import LANDCOVER_CODES
            print("\n地表类型名称:")
            for code in unique:
                if code in LANDCOVER_CODES:
                    print(f"类型代码 {code}: {LANDCOVER_CODES[code]}")
                else:
                    print(f"类型代码 {code}: 未知类型")
        except ImportError:
            print("无法导入配置文件中的LANDCOVER_CODES")
        
        # 创建一个简单地图查看数据
        colors = create_landcover_colormap()
        
        # 创建颜色映射
        color_list = ['#808080'] * 256  # 默认灰色
        for code, (color, _) in colors.items():
            color_list[code] = color
        cmap = ListedColormap(color_list)
        
        # 创建图形并显示
        plt.figure(figsize=(10, 8))
        plt.imshow(landcover, cmap=cmap, aspect='equal', origin='upper')
        plt.colorbar(label='地表类型代码')
        plt.title('地表覆盖类型分布')
        
        # 保存结果
        plt.savefig("landcover_check.png", dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至: landcover_check.png")
        
        # 输出前100个像素值看看数据内容
        print("\n数据前100个像素:")
        print(landcover.flatten()[:100])
        
        # 检查是否有异常值或错误值
        is_valid = (landcover >= 0) & (landcover <= 255)
        if not np.all(is_valid):
            invalid_count = np.sum(~is_valid)
            print(f"发现{invalid_count}个无效值!")
            print(f"无效值示例: {landcover[~is_valid][:10]}")

if __name__ == "__main__":
    main() 