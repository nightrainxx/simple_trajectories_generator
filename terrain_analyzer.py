"""
地形分析器模块：负责计算和保存坡度、坡向数据

功能：
- 从DEM计算坡度大小
- 从DEM计算坡向
- 保存计算结果
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
from typing import Tuple, Optional
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from config import DEM_PATH, SLOPE_PATH, ASPECT_PATH

class TerrainAnalyzer:
    """地形分析器类：处理DEM数据，计算坡度和坡向"""
    
    def __init__(self):
        """初始化地形分析器"""
        self.dem = None
        self.slope = None
        self.aspect = None
        self.transform = None
        self.crs = None
        
    def load_dem(self, dem_path: str = DEM_PATH) -> None:
        """
        加载DEM数据

        参数:
            dem_path: DEM文件路径
        """
        with rasterio.open(dem_path) as src:
            self.dem = src.read(1)
            self.transform = src.transform
            self.crs = src.crs
            
    def calculate_slope_aspect(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算坡度和坡向

        返回:
            slope: 坡度数组(度)
            aspect: 坡向数组(度，北为0，顺时针)
        """
        if self.dem is None:
            raise ValueError("请先加载DEM数据")
            
        # 获取像素大小
        pixel_size_x = abs(self.transform[0])
        pixel_size_y = abs(self.transform[4])
        
        # 计算梯度
        dy, dx = np.gradient(self.dem, pixel_size_y, pixel_size_x)
        
        # 计算坡度(度)
        self.slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        
        # 计算坡向(度)
        self.aspect = np.degrees(np.arctan2(-dx, dy))
        # 转换为地理坡向(北为0，顺时针)
        self.aspect = 90.0 - self.aspect
        self.aspect = np.where(self.aspect < 0, self.aspect + 360, self.aspect)
        
        return self.slope, self.aspect
        
    def save_results(
        self,
        slope_path: str = SLOPE_PATH,
        aspect_path: str = ASPECT_PATH
    ) -> None:
        """
        保存计算结果

        参数:
            slope_path: 坡度文件保存路径
            aspect_path: 坡向文件保存路径
        """
        if self.slope is None or self.aspect is None:
            raise ValueError("请先计算坡度和坡向")
            
        # 保存坡度
        with rasterio.open(
            slope_path,
            'w',
            driver='GTiff',
            height=self.slope.shape[0],
            width=self.slope.shape[1],
            count=1,
            dtype=self.slope.dtype,
            crs=self.crs,
            transform=self.transform
        ) as dst:
            dst.write(self.slope, 1)
            
        # 保存坡向
        with rasterio.open(
            aspect_path,
            'w',
            driver='GTiff',
            height=self.aspect.shape[0],
            width=self.aspect.shape[1],
            count=1,
            dtype=self.aspect.dtype,
            crs=self.crs,
            transform=self.transform
        ) as dst:
            dst.write(self.aspect, 1)
            
    def visualize_terrain(self, output_dir: str) -> None:
        """
        可视化地形分析结果

        参数:
            output_dir: 输出目录
        """
        if self.dem is None or self.slope is None or self.aspect is None:
            raise ValueError("请先完成地形分析")
            
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置字体
        times_font = FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf')
        simsun_font = FontProperties(fname='/usr/share/fonts/truetype/custom/simsun.ttc')
        
        # 绘制DEM
        plt.figure(figsize=(12, 8))
        plt.imshow(self.dem, cmap='terrain')
        plt.colorbar(label='高程 (m)')
        plt.title('数字高程模型 (DEM)', fontproperties=simsun_font, fontsize=16)
        plt.xticks(fontproperties=times_font, fontsize=16)
        plt.yticks(fontproperties=times_font, fontsize=16)
        plt.savefig(os.path.join(output_dir, 'dem_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制坡度
        plt.figure(figsize=(12, 8))
        plt.imshow(self.slope, cmap='YlOrRd')
        plt.colorbar(label='坡度 (°)')
        plt.title('坡度分布图', fontproperties=simsun_font, fontsize=16)
        plt.xticks(fontproperties=times_font, fontsize=16)
        plt.yticks(fontproperties=times_font, fontsize=16)
        plt.savefig(os.path.join(output_dir, 'slope_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制坡向
        plt.figure(figsize=(12, 8))
        plt.imshow(self.aspect, cmap='hsv')
        plt.colorbar(label='坡向 (°)')
        plt.title('坡向分布图', fontproperties=simsun_font, fontsize=16)
        plt.xticks(fontproperties=times_font, fontsize=16)
        plt.yticks(fontproperties=times_font, fontsize=16)
        plt.savefig(os.path.join(output_dir, 'aspect_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
def main():
    """主函数：运行地形分析流程"""
    # 创建地形分析器
    analyzer = TerrainAnalyzer()
    
    # 加载DEM
    print("加载DEM数据...")
    analyzer.load_dem()
    
    # 计算坡度和坡向
    print("计算坡度和坡向...")
    analyzer.calculate_slope_aspect()
    
    # 保存结果
    print("保存计算结果...")
    analyzer.save_results()
    
    # 可视化结果
    print("生成可视化图表...")
    analyzer.visualize_terrain(os.path.join("data", "output", "terrain_analysis"))
    
    print("地形分析完成！")
    
if __name__ == "__main__":
    main() 