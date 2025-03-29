"""
地图生成器模块：负责生成速度图和成本图

功能：
- 生成基础速度图
- 应用坡向约束
- 生成成本图
- 可视化结果
"""

import os
import sys
import numpy as np
from osgeo import gdal, gdalconst
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from config import (
    DEM_PATH,
    SLOPE_PATH,
    ASPECT_PATH,
    LANDCOVER_PATH,
    SPEED_MAP_PATH,
    COST_MAP_PATH,
    LANDCOVER_CODES,
    SPEED_RULES,
    SLOPE_BINS,
    SLOPE_LABELS,
    SLOPE_SPEED_REDUCTION_FACTOR,
    ASPECT_UPHILL_REDUCTION_K,
    ASPECT_CROSS_REDUCTION_K,
    MAX_CROSS_SLOPE_DEGREES,
    MAX_BRAKING_SPEED_DOWNHILL,
    IMPASSABLE_LANDCOVER_CODES,
    DEFAULT_SPEED_MPS
)

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
        self.width = None
        self.height = None
        
    @staticmethod
    def check_file_exists(file_path):
        """检查文件是否存在"""
        if not os.path.exists(file_path):
            print(f"错误：文件不存在: {file_path}")
            return False
        if not os.path.isfile(file_path):
            print(f"错误：路径不是文件: {file_path}")
            return False
        return True
        
    def load_raster(self, file_path, description):
        """加载栅格数据"""
        if not self.check_file_exists(file_path):
            print(f"警告：{description}文件不存在: {file_path}")
            return None, None, None
            
        print(f"加载{description}: {file_path}")
        ds = gdal.Open(file_path)
        if ds is None:
            print(f"警告：无法打开{description}文件")
            return None, None, None
            
        try:
            data = ds.GetRasterBand(1).ReadAsArray()
            if data is None:
                print(f"警告：无法读取{description}数据")
                return None, None, None
                
            # 转换为float32类型
            data = data.astype(np.float32)
            transform = ds.GetGeoTransform()
            projection = ds.GetProjection()
            
            print(f"{description}加载成功")
            print(f"数据形状: {data.shape}")
            print(f"数据范围: {np.min(data):.2f} - {np.max(data):.2f}")
            print(f"数据类型: {data.dtype}")
            
            return data, transform, projection
        except Exception as e:
            print(f"警告：读取{description}数据时出错: {str(e)}")
            return None, None, None
        finally:
            ds = None
            
    def load_data(self):
        """加载所有输入数据"""
        print("加载环境数据...")
        
        # 加载DEM
        print("\n=== 加载DEM数据 ===")
        self.dem, self.transform, self.projection = self.load_raster(DEM_PATH, "DEM数据")
        if self.dem is None:
            raise ValueError("无法加载DEM数据")
            
        # 加载坡度
        print("\n=== 加载坡度数据 ===")
        self.slope, _, _ = self.load_raster(SLOPE_PATH, "坡度数据")
        if self.slope is None:
            print("坡度文件不存在或无法读取，正在计算坡度...")
            self.calculate_slope()
            if self.slope is None:
                raise ValueError("坡度计算失败")
            print(f"坡度范围: {np.min(self.slope):.2f} - {np.max(self.slope):.2f} 度")
            
        # 加载坡向
        print("\n=== 加载坡向数据 ===")
        self.aspect, _, _ = self.load_raster(ASPECT_PATH, "坡向数据")
        if self.aspect is None:
            print("坡向文件不存在或无法读取，正在计算坡向...")
            self.calculate_aspect()
            if self.aspect is None:
                raise ValueError("坡向计算失败")
            print(f"坡向范围: {np.min(self.aspect):.2f} - {np.max(self.aspect):.2f} 度")
            
        # 加载土地覆盖
        print("\n=== 加载土地覆盖数据 ===")
        self.landcover, _, _ = self.load_raster(LANDCOVER_PATH, "土地覆盖数据")
        if self.landcover is None:
            raise ValueError("无法加载土地覆盖数据")
            
        # 验证数据加载
        if self.aspect is None:
            raise ValueError("坡向数据未正确加载或计算")
        if self.slope is None:
            raise ValueError("坡度数据未正确加载或计算")
            
        print("\n=== 数据加载完成 ===")
        print(f"DEM形状: {self.dem.shape}")
        print(f"坡度形状: {self.slope.shape}")
        print(f"坡向形状: {self.aspect.shape}")
        print(f"土地覆盖形状: {self.landcover.shape}")
        
        # 检查数据一致性
        shapes = {
            "DEM": self.dem.shape,
            "坡度": self.slope.shape,
            "坡向": self.aspect.shape,
            "土地覆盖": self.landcover.shape
        }
        if len(set(shapes.values())) > 1:
            print("错误：数据形状不一致:")
            for name, shape in shapes.items():
                print(f"{name}: {shape}")
            raise ValueError("数据形状不一致")
        
    def _calculate_slope(self):
        """计算坡度（度）"""
        if self.elevation is None:
            raise ValueError("需要先加载高程数据")
        
        # 计算x和y方向的梯度
        dy, dx = np.gradient(self.elevation)
        
        # 计算坡度（度）
        slope = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))
        return slope
        
    def _get_slope_class(self, slope_value: float) -> int:
        """获取坡度等级"""
        if np.isnan(slope_value):
            return len(SLOPE_LABELS) - 1
        return np.digitize(slope_value, SLOPE_BINS) - 1
        
    def generate_base_speed_map(self):
        """生成基础速度图"""
        print("生成基础速度图...")
        
        # 获取坡度分类
        slope_classes = self.classify_slope()
        
        # 初始化速度图
        self.base_speed_map = np.zeros_like(self.dem, dtype=np.float32)
        
        # 遍历每个像素
        rows, cols = self.landcover.shape
        for i in range(rows):
            for j in range(cols):
                landcover_code = int(self.landcover[i, j])  # 确保土地覆盖代码是整数
                slope_class = slope_classes[i, j]
                speed = self.get_speed(landcover_code, slope_class)
                self.base_speed_map[i, j] = speed
                
        print("基础速度图生成完成")
        print(f"速度范围: {np.min(self.base_speed_map):.2f} - {np.max(self.base_speed_map):.2f} m/s")
        
        # 统计各土地类型的速度分布
        unique_landcover = np.unique(self.landcover)
        print("\n土地类型速度统计：")
        for code in unique_landcover:
            mask = self.landcover == code
            if np.any(mask):
                speeds = self.base_speed_map[mask]
                print(f"土地类型 {int(code)}: "
                      f"最小速度 = {np.min(speeds):.2f} m/s, "
                      f"最大速度 = {np.max(speeds):.2f} m/s, "
                      f"平均速度 = {np.mean(speeds):.2f} m/s")
                
        return self.base_speed_map
        
    def apply_slope_aspect_constraints(self, heading_degrees):
        """应用坡度和坡向约束"""
        print(f"应用坡度和坡向约束 (航向: {heading_degrees}°)...")
        
        # 将航向角转换为弧度
        heading_rad = np.radians(heading_degrees)
        aspect_rad = np.radians(self.aspect)
        
        # 计算相对方位角（弧度）
        relative_aspect = aspect_rad - heading_rad
        
        # 计算上/下坡分量
        uphill_component = np.cos(relative_aspect) * self.slope
        
        # 计算横坡分量
        cross_slope = np.abs(np.sin(relative_aspect) * self.slope)
        
        # 获取基础速度图
        speed_map = self.generate_base_speed_map()
        
        # 应用上坡减速
        uphill_mask = uphill_component > 0
        speed_map[uphill_mask] *= np.exp(-ASPECT_UPHILL_REDUCTION_K * uphill_component[uphill_mask])
        
        # 应用下坡限速
        downhill_mask = uphill_component < 0
        max_downhill_speed = MAX_BRAKING_SPEED_DOWNHILL * np.exp(SLOPE_SPEED_REDUCTION_FACTOR * uphill_component[downhill_mask])
        speed_map[downhill_mask] = np.minimum(speed_map[downhill_mask], max_downhill_speed)
        
        # 应用横坡减速
        cross_slope_factor = np.exp(-ASPECT_CROSS_REDUCTION_K * cross_slope)
        speed_map *= cross_slope_factor
        
        # 处理过陡横坡
        speed_map[cross_slope > MAX_CROSS_SLOPE_DEGREES] = 0.0
        
        print("坡度和坡向约束应用完成")
        print(f"考虑地形后的速度范围: {np.min(speed_map):.2f} - {np.max(speed_map):.2f} m/s")
        
        # 统计各土地类型的速度分布
        unique_landcover = np.unique(self.landcover)
        print("\n考虑地形后的土地类型速度统计：")
        for code in unique_landcover:
            mask = self.landcover == code
            if np.any(mask):
                speeds = speed_map[mask]
                print(f"土地类型 {int(code)}: "
                      f"最小速度 = {np.min(speeds):.2f} m/s, "
                      f"最大速度 = {np.max(speeds):.2f} m/s, "
                      f"平均速度 = {np.mean(speeds):.2f} m/s")
                      
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
        print("\n=== 生成成本图 ===")
        
        # 生成基础速度图
        if self.base_speed_map is None:
            self.generate_base_speed_map()
        
        # 应用坡向约束得到实际速度图
        speed_map = self.apply_slope_aspect_constraints(heading_degrees)
        
        # 打印速度图的统计信息
        print("\n速度图统计:")
        print(f"最小速度: {np.min(speed_map):.2f} m/s")
        print(f"最大速度: {np.max(speed_map):.2f} m/s")
        print(f"平均速度: {np.mean(speed_map):.2f} m/s")
        print(f"零速度像素数: {np.sum(speed_map == 0)}")
        print(f"极低速度(<0.1 m/s)像素数: {np.sum(speed_map < 0.1)}")
        
        # 设置一个极小的速度阈值，避免除零
        min_speed = 0.1  # 0.1 m/s，相当于0.36 km/h
        
        # 计算成本图
        # 1. 对于不可通行区域（速度为0），设置为最大成本
        # 2. 对于极低速度区域（<0.1 m/s），设置为最大成本的一半
        # 3. 对于正常区域，成本为1.0/速度
        self.cost_map = np.where(
            speed_map <= 0.0,  # 不可通行区域
            max_cost,
            np.where(
                speed_map < min_speed,  # 极低速度区域
                max_cost * 0.5,
                1.0 / speed_map  # 正常区域
            )
        )
        
        # 打印成本图的统计信息
        print("\n成本图统计:")
        print(f"最小成本: {np.min(self.cost_map):.2f} s/m")
        print(f"最大成本: {np.max(self.cost_map):.2f} s/m")
        print(f"平均成本: {np.mean(self.cost_map):.2f} s/m")
        print(f"无穷大成本像素数: {np.sum(np.isinf(self.cost_map))}")
        print(f"NaN成本像素数: {np.sum(np.isnan(self.cost_map))}")
        print(f"最大成本像素数: {np.sum(self.cost_map >= max_cost)}")
        
        return self.cost_map
        
    def save_maps(self) -> None:
        """保存速度图和成本图"""
        if self.base_speed_map is None or self.cost_map is None:
            raise ValueError("请先生成速度图和成本图")
            
        # 创建输出目录
        os.makedirs(os.path.dirname(SPEED_MAP_PATH), exist_ok=True)
        
        # 保存基础速度图
        driver = gdal.GetDriverByName('GTiff')
        outds = driver.Create(
            SPEED_MAP_PATH, 
            self.base_speed_map.shape[1], 
            self.base_speed_map.shape[0], 
            1, 
            gdal.GDT_Float32
        )
        if outds is not None:
            outds.SetGeoTransform(self.transform)
            outds.SetProjection(self.projection)
            outband = outds.GetRasterBand(1)
            outband.WriteArray(self.base_speed_map)
            outband.FlushCache()
            outds = None
            print(f"基础速度图已保存到: {SPEED_MAP_PATH}")
            
        # 保存成本图
        outds = driver.Create(
            COST_MAP_PATH, 
            self.cost_map.shape[1], 
            self.cost_map.shape[0], 
            1, 
            gdal.GDT_Float32
        )
        if outds is not None:
            outds.SetGeoTransform(self.transform)
            outds.SetProjection(self.projection)
            outband = outds.GetRasterBand(1)
            outband.WriteArray(self.cost_map)
            outband.FlushCache()
            outds = None
            print(f"成本图已保存到: {COST_MAP_PATH}")
            
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
        
        # 绘制基础速度图
        plt.figure(figsize=(12, 8))
        plt.imshow(self.base_speed_map, cmap='viridis')
        plt.colorbar(label='速度 (m/s)')
        plt.title('基础速度图')
        plt.savefig(os.path.join(output_dir, 'base_speed_map.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制成本图
        plt.figure(figsize=(12, 8))
        plt.imshow(self.cost_map, cmap='magma')
        plt.colorbar(label='成本 (s/m)')
        plt.title('通行成本图')
        plt.savefig(os.path.join(output_dir, 'cost_map.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def classify_slope(self):
        """将坡度分类"""
        print("对坡度进行分类...")
        
        # 确保坡度数据存在
        if self.slope is None:
            raise ValueError("坡度数据未加载")
            
        # 使用numpy.digitize进行分类
        slope_classes = np.zeros_like(self.slope, dtype=np.int32)
        for i in range(len(SLOPE_BINS) - 1):
            mask = (self.slope >= SLOPE_BINS[i]) & (self.slope < SLOPE_BINS[i + 1])
            slope_classes[mask] = i
            
        # 处理超出最大坡度的情况
        slope_classes[self.slope >= SLOPE_BINS[-1]] = len(SLOPE_BINS) - 2
        
        # 统计各坡度等级的分布
        print("\n坡度分类统计：")
        total_pixels = slope_classes.size
        for i in range(len(SLOPE_BINS) - 1):
            mask = slope_classes == i
            count = np.sum(mask)
            percentage = count / total_pixels * 100
            if i == 0:
                range_str = f"{SLOPE_BINS[i]}-{SLOPE_BINS[i+1]}°"
            else:
                range_str = f"{SLOPE_BINS[i]}-{SLOPE_BINS[i+1]}°"
            print(f"坡度等级 {i} ({range_str}): {count} 像素 ({percentage:.2f}%)")
            
        return slope_classes

    def get_speed(self, landcover_code: int, slope_class: int) -> float:
        """
        获取指定土地覆盖类型和坡度等级的速度值
        
        参数:
            landcover_code: 土地覆盖类型代码
            slope_class: 坡度等级
            
        返回:
            速度值 (m/s)
        """
        # 确保输入参数是整数
        landcover_code = int(landcover_code)
        slope_class = int(slope_class)
        
        # 检查土地覆盖类型是否有效
        if landcover_code not in LANDCOVER_CODES:
            print(f"警告：未知的土地覆盖类型代码 {landcover_code}，使用默认速度")
            return DEFAULT_SPEED_MPS
            
        # 如果是不可通行的土地类型，返回0速度
        if landcover_code in IMPASSABLE_LANDCOVER_CODES:
            return 0.0
            
        # 如果坡度等级无效，返回0速度
        if slope_class < 0 or slope_class >= len(SLOPE_LABELS):
            return 0.0
            
        # 从速度规则中获取速度值
        speed = SPEED_RULES.get((landcover_code, slope_class))
        if speed is None:
            print(f"警告：未找到土地类型 {landcover_code} 和坡度等级 {slope_class} 的速度规则，使用默认速度")
            return DEFAULT_SPEED_MPS
            
        return speed

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