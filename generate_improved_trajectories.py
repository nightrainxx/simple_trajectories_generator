"""
使用改进的路径规划器生成战略轨迹

输入:
- strategic_pairs.csv - 战略点对数据
- cost_map.tif - 成本图数据

输出:
- improved_trajectories/*.csv - 改进版轨迹数据
- improved_paths/*.csv - 改进版路径数据
"""

import os
import pandas as pd
import numpy as np
from improved_path_planner import ImprovedPathPlanner
from trajectory_generator import TrajectoryGenerator
from config import OUTPUT_DIR

def main():
    """主函数"""
    print("使用改进的路径规划器生成战略轨迹...")
    
    # 创建输出目录
    improved_trajectories_dir = os.path.join(OUTPUT_DIR, "improved_trajectories")
    improved_paths_dir = os.path.join(OUTPUT_DIR, "improved_paths")
    
    os.makedirs(improved_trajectories_dir, exist_ok=True)
    os.makedirs(improved_paths_dir, exist_ok=True)
    
    # 检查战略点对文件是否存在
    strategic_pairs_path = os.path.join(OUTPUT_DIR, "strategic_pairs.csv")
    if not os.path.exists(strategic_pairs_path):
        print(f"错误: 找不到战略点对文件 {strategic_pairs_path}")
        return
    
    # 加载战略点对数据
    pairs_df = pd.read_csv(strategic_pairs_path)
    print(f"加载了 {len(pairs_df)} 个战略点对")
    
    # 初始化改进的路径规划器
    planner = ImprovedPathPlanner()
    planner.load_cost_map()
    
    # 初始化轨迹生成器
    trajectory_generator = TrajectoryGenerator()
    trajectory_generator.load_data()
    
    # 成功和失败计数
    success_count = 0
    failure_count = 0
    
    # 为每个点对生成轨迹
    for i, row in pairs_df.iterrows():
        pair_id = i
        start_x, start_y = int(row['start_row']), int(row['start_col'])
        end_x, end_y = int(row['end_row']), int(row['end_col'])
        
        print(f"\n处理战略点对 {pair_id}: ({start_x}, {start_y}) -> ({end_x}, {end_y})")
        
        try:
            # 使用改进的路径规划器寻找路径
            original_path, smoothed_path, path_cost = planner.find_path(
                start=(start_x, start_y),
                goal=(end_x, end_y)
            )
            
            if not original_path:
                print(f"未能为点对 {pair_id} 找到路径")
                failure_count += 1
                continue
            
            # 将路径转换为DataFrame并保存
            path_df = pd.DataFrame({
                'row': [p[0] for p in smoothed_path],
                'col': [p[1] for p in smoothed_path]
            })
            
            path_file = os.path.join(improved_paths_dir, f"improved_path_{pair_id}.csv")
            path_df.to_csv(path_file, index=False)
            print(f"保存路径到 {path_file}, {len(smoothed_path)} 个点, 总成本: {path_cost:.2f}")
            
            # 生成轨迹
            try:
                trajectory = trajectory_generator.generate_trajectory(smoothed_path, trajectory_id=pair_id)
                
                # 计算总距离和时间
                total_distance_m = sum(
                    np.sqrt(
                        (trajectory['row'].iloc[i+1] - trajectory['row'].iloc[i])**2 +
                        (trajectory['col'].iloc[i+1] - trajectory['col'].iloc[i])**2
                    ) * 30  # 30米分辨率
                    for i in range(len(trajectory)-1)
                )
                total_duration_h = trajectory['timestamp'].iloc[-1] / 3600
                avg_speed_kmh = (total_distance_m / 1000) / total_duration_h
                
                # 保存轨迹
                traj_file = os.path.join(improved_trajectories_dir, f"improved_trajectory_{pair_id}.csv")
                trajectory.to_csv(traj_file, index=False)
                
                print(f"保存轨迹到 {traj_file}")
                print(f"轨迹时长: {total_duration_h:.2f} 小时")
                print(f"总距离: {total_distance_m/1000:.2f} 公里")
                print(f"平均速度: {avg_speed_kmh:.2f} 公里/小时")
                
                success_count += 1
                
            except Exception as e:
                print(f"生成轨迹时出错: {e}")
                failure_count += 1
                
        except ValueError as e:
            print(f"路径规划错误: {e}")
            failure_count += 1
        except Exception as e:
            print(f"未知错误: {e}")
            failure_count += 1
    
    print("\n轨迹生成完成")
    print(f"成功: {success_count}")
    print(f"失败: {failure_count}")

if __name__ == "__main__":
    main() 