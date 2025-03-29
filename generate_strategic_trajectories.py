"""
使用战略位置生成轨迹

输入：
- strategic_pairs.csv：战略起终点对数据
- cost_map.tif：成本图

输出：
- strategic_trajectories/*.csv：战略轨迹数据文件
- strategic_paths/*.csv：战略路径数据文件
"""

import os
import pandas as pd
import numpy as np
from path_planner import PathPlanner
from trajectory_generator import TrajectoryGenerator
from config import OUTPUT_DIR

def main():
    """主函数"""
    # 创建输出目录
    trajectories_dir = os.path.join(OUTPUT_DIR, "strategic_trajectories")
    paths_dir = os.path.join(OUTPUT_DIR, "strategic_paths")
    os.makedirs(trajectories_dir, exist_ok=True)
    os.makedirs(paths_dir, exist_ok=True)
    
    # 加载战略起终点对数据
    pairs_path = os.path.join(OUTPUT_DIR, "strategic_pairs.csv")
    if not os.path.exists(pairs_path):
        print(f"错误：找不到战略起终点对数据文件 {pairs_path}")
        print("请先运行 create_strategic_pairs.py 生成战略起终点对")
        return
        
    pairs_df = pd.read_csv(pairs_path)
    print(f"开始生成战略轨迹...")
    print(f"加载了{len(pairs_df)}对战略起终点")
    
    # 创建路径规划器和轨迹生成器
    planner = PathPlanner()
    generator = TrajectoryGenerator()
    generator.load_data()
    
    # 记录成功和失败的数量
    success_count = 0
    fail_count = 0
    
    # 为每对起终点生成轨迹
    for i, row in pairs_df.iterrows():
        print(f"\n处理第{i}对战略起终点:")
        start = (int(row['start_row']), int(row['start_col']))
        goal = (int(row['end_row']), int(row['end_col']))
        print(f"  起点: {start}")
        print(f"  终点: {goal}")
        
        try:
            # 规划路径
            original_path, smoothed_path, cost = planner.find_path(start, goal)
            
            if smoothed_path:
                print(f"  找到路径，长度: {len(smoothed_path)}个点")
                print(f"  路径总成本: {cost:.2f}")
                
                # 保存原始路径
                original_path_df = pd.DataFrame(original_path, columns=['row', 'col'])
                original_path_df.to_csv(
                    os.path.join(paths_dir, f"strategic_path_{i}.csv"),
                    index=False
                )
                
                # 生成轨迹
                print(f"\n生成战略轨迹 {i}...")
                df = generator.generate_trajectory(smoothed_path, i)
                
                # 保存轨迹到特定目录
                output_path = os.path.join(trajectories_dir, f"strategic_trajectory_{i}.csv")
                df.to_csv(output_path, index=False)
                print(f"战略轨迹数据已保存至: {output_path}")
                
                success_count += 1
            else:
                print("  未找到有效路径")
                fail_count += 1
                
        except Exception as e:
            print(f"  生成失败: {str(e)}")
            fail_count += 1
            
    print(f"\n战略轨迹生成完成:")
    print(f"  成功: {success_count}条")
    print(f"  失败: {fail_count}条")
    
if __name__ == "__main__":
    main() 