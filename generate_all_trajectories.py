"""
批量轨迹生成脚本

输入：
- start_end_pairs.csv：起终点对数据
- base_speed_map.tif：基础速度图
- slope_30m_100km.tif：坡度数据
- aspect_30m_100km.tif：坡向数据

输出：
- trajectories/trajectory_{id}.csv：轨迹数据文件
"""

import os
import pandas as pd
from path_planner import PathPlanner
from trajectory_generator import TrajectoryGenerator
from config import OUTPUT_DIR

def main():
    """主函数"""
    print("开始批量生成轨迹...")
    
    # 加载起终点对数据
    pairs_path = os.path.join(OUTPUT_DIR, "start_end_pairs.csv")
    if not os.path.exists(pairs_path):
        print(f"错误：找不到起终点对数据文件 {pairs_path}")
        return
        
    pairs_df = pd.read_csv(pairs_path)
    print(f"加载了{len(pairs_df)}对起终点")
    
    # 创建路径规划器
    planner = PathPlanner()
    planner.load_cost_map()
    
    # 创建轨迹生成器
    generator = TrajectoryGenerator()
    generator.load_data()
    
    # 为每对起终点生成轨迹
    success_count = 0
    for idx, row in pairs_df.iterrows():
        start = (int(row['start_row']), int(row['start_col']))
        end = (int(row['end_row']), int(row['end_col']))
        
        print(f"\n处理第{idx}对起终点:")
        print(f"  起点: {start}")
        print(f"  终点: {end}")
        
        # 规划路径
        path, cost = planner.find_path(start, end)
        if not path:
            print("  未找到可行路径，跳过")
            continue
            
        print(f"  找到路径，长度: {len(path)}个点")
        print(f"  路径总成本: {cost:.2f}")
        
        # 生成轨迹
        try:
            df = generator.generate_trajectory(path, trajectory_id=idx)
            generator.save_trajectory(df, trajectory_id=idx)
            success_count += 1
        except Exception as e:
            print(f"  生成轨迹时出错: {e}")
            continue
            
    print(f"\n轨迹生成完成:")
    print(f"  成功: {success_count}条")
    print(f"  失败: {len(pairs_df) - success_count}条")
    
if __name__ == "__main__":
    main() 