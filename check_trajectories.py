"""
检查轨迹起点和终点位置
"""
import os
import pandas as pd
import numpy as np
from config import OUTPUT_DIR

# 战略位置坐标
STRATEGIC_LOCATIONS = [
    (481, 113),    # 位置1: 建筑用地
    (1095, 2682),  # 位置2: 建筑用地
    (1812, 1427),  # 位置3: 建筑用地
    (2577, 2149)   # 位置4: 农田
]

def main():
    # 加载轨迹数据
    trajectories_dir = os.path.join(OUTPUT_DIR, "trajectories")
    trajectory_files = sorted([f for f in os.listdir(trajectories_dir) 
                              if f.startswith("trajectory_") and f.endswith(".csv")])
    
    print(f"找到{len(trajectory_files)}个轨迹文件\n")
    
    # 检查每个轨迹的起点和终点
    print("轨迹起点和终点位置:")
    for i, file in enumerate(trajectory_files):
        path = os.path.join(trajectories_dir, file)
        df = pd.read_csv(path)
        
        start_point = (df.iloc[0].col, df.iloc[0].row)
        end_point = (df.iloc[-1].col, df.iloc[-1].row)
        
        print(f"轨迹 {i+1} ({file}):")
        print(f"  起点: ({start_point[0]:.1f}, {start_point[1]:.1f})")
        print(f"  终点: ({end_point[0]:.1f}, {end_point[1]:.1f})")
        
        # 检查终点是否在战略位置附近
        distances = [np.sqrt((end_point[0]-loc[0])**2 + (end_point[1]-loc[1])**2) 
                    for loc in STRATEGIC_LOCATIONS]
        closest_idx = np.argmin(distances)
        min_distance = distances[closest_idx]
        
        if min_distance < 50:  # 50像素的距离阈值
            print(f"  终点接近战略位置 {closest_idx+1}, 距离: {min_distance:.1f}像素")
        else:
            print(f"  终点不在任何战略位置附近！最近的是位置 {closest_idx+1}, 距离: {min_distance:.1f}像素")
        
        print()
    
    # 加载原始路径数据
    print("\n原始路径起点和终点位置:")
    paths_dir = os.path.join(OUTPUT_DIR, "paths")
    path_files = sorted([f for f in os.listdir(paths_dir) 
                        if f.startswith("path_") and f.endswith(".csv")])
    
    for i, file in enumerate(path_files):
        path = os.path.join(paths_dir, file)
        df = pd.read_csv(path)
        
        start_point = (df.iloc[0].col, df.iloc[0].row)
        end_point = (df.iloc[-1].col, df.iloc[-1].row)
        
        print(f"路径 {i+1} ({file}):")
        print(f"  起点: ({start_point[0]:.1f}, {start_point[1]:.1f})")
        print(f"  终点: ({end_point[0]:.1f}, {end_point[1]:.1f})")
        
        # 检查终点是否在战略位置附近
        distances = [np.sqrt((end_point[0]-loc[0])**2 + (end_point[1]-loc[1])**2) 
                    for loc in STRATEGIC_LOCATIONS]
        closest_idx = np.argmin(distances)
        min_distance = distances[closest_idx]
        
        if min_distance < 50:  # 50像素的距离阈值
            print(f"  终点接近战略位置 {closest_idx+1}, 距离: {min_distance:.1f}像素")
        else:
            print(f"  终点不在任何战略位置附近！最近的是位置 {closest_idx+1}, 距离: {min_distance:.1f}像素")
        
        print()

if __name__ == "__main__":
    main() 