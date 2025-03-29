import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from map_generator import MapGenerator
from path_planner import PathPlanner
from matplotlib.font_manager import FontProperties
from typing import List, Tuple, Optional
import os
from config import OUTPUT_DIR

class PathVisualizer:
    """路径可视化器"""
    
    def __init__(self, env: MapGenerator):
        """
        初始化可视化器
        
        参数:
            env: MapGenerator对象，包含地图数据
        """
        self.env = env
        
        # 设置Times New Roman字体
        self.times_font = FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf')
        # 设置宋体
        self.simsun_font = FontProperties(fname='/usr/share/fonts/truetype/custom/simsun.ttc')
        
        # 设置matplotlib的默认字体
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.unicode_minus'] = False
        
    def visualize_paths(
        self,
        paths: List[List[Tuple[int, int]]],
        start_points: List[Tuple[int, int]],
        target_points: List[Tuple[int, int]],
        save_path: str
    ) -> None:
        """
        可视化多条路径
        
        参数:
            paths: 路径列表，每个路径是一个坐标点列表
            start_points: 起点列表
            target_points: 目标点列表
            save_path: 保存路径
        """
        # 创建图形和子图
        plt.figure(figsize=(15, 12))
        
        # 绘制成本图作为背景
        plt.imshow(self.env.cost_map, cmap='YlOrRd', alpha=0.6)
        
        # 绘制所有路径
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        for i, path in enumerate(paths):
            if path:
                path = np.array(path)
                plt.plot(path[:, 1], path[:, 0], 
                        color=colors[i % len(colors)], 
                        linewidth=2, 
                        label=f'路径 {i+1}',
                        alpha=0.8)
        
        # 绘制起点和终点
        for i, (start, target) in enumerate(zip(start_points, target_points)):
            plt.scatter(start[1], start[0], 
                       c='g', marker='^', s=150,
                       label='起点' if i == 0 else '')
            plt.scatter(target[1], target[0], 
                       c='r', marker='*', s=200,
                       label='目标点' if i == 0 else '')
        
        # 设置标题和标签
        plt.title('路径规划结果（背景为成本图）', fontproperties=self.simsun_font)
        plt.xlabel('东西方向（像素）', fontproperties=self.simsun_font)
        plt.ylabel('南北方向（像素）', fontproperties=self.simsun_font)
        
        # 设置图例
        plt.legend(prop=self.simsun_font, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # 设置刻度字体
        plt.xticks(fontproperties=self.times_font)
        plt.yticks(fontproperties=self.times_font)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """主函数：测试可视化效果"""
    # 1. 加载环境数据
    print("1. 加载环境数据...")
    env = MapGenerator()
    env.load_data()
    env.generate_cost_map()
    
    # 2. 创建路径规划器
    print("2. 创建路径规划器...")
    planner = PathPlanner(env)
    
    # 3. 创建可视化器
    print("3. 创建可视化器...")
    visualizer = PathVisualizer(env)
    
    # 4. 读取轨迹数据
    print("4. 读取轨迹数据...")
    trajectories_dir = os.path.join(OUTPUT_DIR, "batch_trajectories_20_highspeed_65kmh/trajectories")
    trajectory_files = sorted([f for f in os.listdir(trajectories_dir) 
                             if f.startswith("trajectory_") and f.endswith(".csv")])
    
    # 随机选择两条轨迹进行可视化
    selected_files = np.random.choice(trajectory_files, 2, replace=False)
    print(f"\n选择了两条轨迹进行可视化：")
    
    all_paths = []
    start_points = []
    end_points = []
    
    for file in selected_files:
        print(f"处理轨迹: {file}")
        df = pd.read_csv(os.path.join(trajectories_dir, file))
        path = list(zip(df['row'], df['col']))
        all_paths.append(path)
        start_points.append(path[0])
        end_points.append(path[-1])
        print(f"  起点: {path[0]}")
        print(f"  终点: {path[-1]}")
        print(f"  轨迹长度: {len(path)}个点")
    
    # 5. 保存可视化结果
    print("\n5. 保存可视化结果...")
    output_path = os.path.join(OUTPUT_DIR, "batch_trajectories_20_highspeed_65kmh/visualization.png")
    visualizer.visualize_paths(
        all_paths,
        start_points,
        end_points,
        output_path
    )
    print(f"可视化结果已保存到: {output_path}")

if __name__ == "__main__":
    main() 