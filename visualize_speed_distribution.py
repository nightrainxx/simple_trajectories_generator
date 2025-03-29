#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轨迹速度分布可视化工具

输入参数:
    轨迹文件夹路径
    输出文件路径
    
输出结果:
    速度分布直方图
    速度统计数据

处理流程:
    读取轨迹数据
    分析速度分布
    生成可视化图表
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import argparse

# 尝试设置中文字体
try:
    font = FontProperties(fname=r'/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    plt.rcParams['font.family'] = ['WenQuanYi Micro Hei']
except:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def analyze_trajectory_speeds(trajectory_dir):
    """分析轨迹文件中的速度数据"""
    trajectory_files = glob.glob(os.path.join(trajectory_dir, '*.csv'))
    
    all_speeds = []
    trajectory_stats = []
    
    print(f"找到 {len(trajectory_files)} 个轨迹文件")
    
    for traj_file in trajectory_files:
        try:
            # 读取轨迹数据
            traj_data = pd.read_csv(traj_file)
            
            # 提取速度数据 (转换为km/h)
            speeds = traj_data['speed'] * 3.6  # m/s 转换为 km/h
            
            # 计算统计数据
            avg_speed = np.mean(speeds)
            max_speed = np.max(speeds)
            min_speed = np.min(speeds)
            std_speed = np.std(speeds)
            
            # 存储统计信息
            traj_id = os.path.basename(traj_file).split('_')[-1].split('.')[0]
            trajectory_stats.append({
                'trajectory_id': traj_id,
                'avg_speed': avg_speed,
                'max_speed': max_speed,
                'min_speed': min_speed,
                'std_speed': std_speed,
                'total_points': len(speeds)
            })
            
            # 添加到总体速度列表
            all_speeds.extend(speeds)
            
            print(f"处理文件 {traj_file}: 平均速度 {avg_speed:.2f} km/h")
            
        except Exception as e:
            print(f"处理文件 {traj_file} 时出错: {e}")
    
    return all_speeds, trajectory_stats

def visualize_speed_distribution(all_speeds, trajectory_stats, output_path):
    """创建速度分布可视化"""
    if not all_speeds or not trajectory_stats:
        print("没有找到有效的轨迹数据，无法生成可视化")
        return
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 设置子图
    plt.subplot(2, 2, 1)
    plt.hist(all_speeds, bins=30, alpha=0.7, color='blue')
    plt.title('速度分布直方图', fontsize=14)
    plt.xlabel('速度 (km/h)', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 速度箱线图
    plt.subplot(2, 2, 2)
    plt.boxplot(all_speeds, vert=False, widths=0.7)
    plt.title('速度箱线图', fontsize=14)
    plt.xlabel('速度 (km/h)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 轨迹平均速度条形图
    plt.subplot(2, 1, 2)
    stats_df = pd.DataFrame(trajectory_stats)
    
    if not stats_df.empty and 'avg_speed' in stats_df.columns:
        stats_df = stats_df.sort_values('avg_speed', ascending=False)
        
        traj_ids = stats_df['trajectory_id'].tolist()
        avg_speeds = stats_df['avg_speed'].tolist()
        
        bars = plt.bar(traj_ids, avg_speeds, alpha=0.7)
        plt.title('各轨迹平均速度', fontsize=14)
        plt.xlabel('轨迹ID', fontsize=12)
        plt.ylabel('平均速度 (km/h)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 在条形图上标注具体数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{height:.1f}', ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, "没有找到有效的轨迹平均速度数据", 
                 ha='center', va='center', fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"速度分布可视化已保存至: {output_path}")
    
    # 创建统计报告
    stats_df = pd.DataFrame(trajectory_stats)
    
    if not all_speeds:
        print("没有足够的数据生成统计报告")
        return
        
    summary_stats = {
        '总体平均速度': np.mean(all_speeds),
        '最高速度': np.max(all_speeds),
        '最低速度': np.min(all_speeds),
        '速度标准差': np.std(all_speeds),
        '总采样点数': len(all_speeds),
        '轨迹数量': len(trajectory_stats)
    }
    
    report_path = os.path.splitext(output_path)[0] + '_statistics.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("轨迹速度统计报告\n")
        f.write("=================\n\n")
        
        f.write("总体统计\n")
        f.write("-----------------\n")
        for key, value in summary_stats.items():
            f.write(f"{key}: {value:.2f}\n")
        
        f.write("\n各轨迹统计\n")
        f.write("-----------------\n")
        if not stats_df.empty:
            f.write(stats_df.to_string(index=False))
        else:
            f.write("没有有效的轨迹统计数据")
    
    print(f"速度统计报告已保存至: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='轨迹速度分布可视化工具')
    parser.add_argument('--trajectory_dir', type=str, required=True, 
                        help='包含轨迹CSV文件的文件夹路径')
    parser.add_argument('--output_path', type=str, default='speed_distribution.png',
                        help='输出图像文件路径')
    
    args = parser.parse_args()
    
    print(f"分析轨迹目录: {args.trajectory_dir}")
    
    # 分析轨迹速度
    all_speeds, trajectory_stats = analyze_trajectory_speeds(args.trajectory_dir)
    
    # 创建可视化
    visualize_speed_distribution(all_speeds, trajectory_stats, args.output_path)

if __name__ == '__main__':
    main()