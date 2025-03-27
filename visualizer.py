import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from map_generator import MapGenerator
from path_planner import PathPlanner
from matplotlib.font_manager import FontProperties
from typing import List, Tuple, Optional

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
    
    # 4. 读取起终点对数据
    print("4. 读取起终点对数据...")
    pairs_df = pd.read_csv('data/start_end_pairs.csv')
    
    # 获取所有不同的目标点
    target_points = pairs_df[['end_row', 'end_col']].drop_duplicates().values
    print(f"\n找到{len(target_points)}个目标点，每个目标点选择2条路径进行展示...")
    
    all_paths = []
    start_points = []
    target_points_used = []
    
    # 为每个目标点规划2条路径
    for i, target in enumerate(target_points, 1):
        print(f"\n处理目标点 {i}/{len(target_points)}: ({target[0]}, {target[1]})")
        
        # 获取该目标点对应的所有起点
        target_pairs = pairs_df[
            (pairs_df['end_row'] == target[0]) & 
            (pairs_df['end_col'] == target[1])
        ]
        
        # 随机选择2个起点
        selected_pairs = target_pairs.sample(n=min(2, len(target_pairs)))
        
        for j, (_, pair) in enumerate(selected_pairs.iterrows(), 1):
            start = (int(pair['start_row']), int(pair['start_col']))
            goal = (int(pair['end_row']), int(pair['end_col']))
            
            print(f"  规划路径 {j}/2: {start} -> {goal}")
            
            try:
                path = planner.find_path(start, goal)
                if path:
                    path = planner.smooth_path(path)
                    all_paths.append(path)
                    start_points.append(start)
                    target_points_used.append(goal)
                else:
                    print("    未找到有效路径")
            except ValueError as e:
                print(f"    错误: {e}")
    
    # 5. 保存可视化结果
    print("\n5. 保存可视化结果...")
    visualizer.visualize_paths(
        all_paths,
        start_points,
        target_points_used,
        'data/paths/path_visualization.png'
    )
    print("可视化结果已保存到: data/paths/path_visualization.png")

if __name__ == "__main__":
    main() 