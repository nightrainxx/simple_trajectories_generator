"""
路径规划器模块：使用A*算法进行路径规划

输入：
- 成本图
- 起点和终点坐标

输出：
- 路径点列表
- 路径总成本

处理流程：
1. 加载成本图
2. 使用A*算法找到最优路径
3. 对路径进行平滑处理
"""

import os
import numpy as np
from typing import List, Tuple, Optional
import heapq
from scipy.ndimage import gaussian_filter1d
import rasterio
from config import OUTPUT_DIR

class Node:
    """A*算法的节点类"""
    def __init__(self, pos: Tuple[int, int], g_cost: float, h_cost: float, parent=None):
        self.pos = pos
        self.g_cost = g_cost  # 从起点到当前点的成本
        self.h_cost = h_cost  # 从当前点到终点的估计成本
        self.f_cost = g_cost + h_cost  # 总成本
        self.parent = parent
        
    def __lt__(self, other):
        return self.f_cost < other.f_cost
        
class PathPlanner:
    """路径规划器类"""
    
    def __init__(self):
        """初始化路径规划器"""
        self.cost_map = None
        self.shape = None
        
    def load_cost_map(self) -> None:
        """加载成本图"""
        cost_path = os.path.join(OUTPUT_DIR, "intermediate", "cost_map.tif")
        with rasterio.open(cost_path) as src:
            self.cost_map = src.read(1)
            self.shape = self.cost_map.shape
            
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        计算启发式成本（曼哈顿距离）
        
        参数:
            pos: 当前位置
            goal: 目标位置
            
        返回:
            float: 启发式成本
        """
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        获取相邻节点
        
        参数:
            pos: 当前位置
            
        返回:
            List[Tuple[int, int]]: 相邻节点列表
        """
        row, col = pos
        neighbors = []
        
        # 8个方向的相邻点
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            new_row, new_col = row + dr, col + dc
            
            # 检查边界
            if not (0 <= new_row < self.shape[0] and 0 <= new_col < self.shape[1]):
                continue
                
            # 检查是否可通行
            if not np.isfinite(self.cost_map[new_row, new_col]):
                continue
                
            neighbors.append((new_row, new_col))
            
        return neighbors
        
    def get_path_cost(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        计算两个相邻点之间的路径成本
        
        参数:
            pos1: 第一个点
            pos2: 第二个点
            
        返回:
            float: 路径成本
        """
        # 对角线移动的成本是直线移动的√2倍
        is_diagonal = pos1[0] != pos2[0] and pos1[1] != pos2[1]
        distance = np.sqrt(2) if is_diagonal else 1.0
        
        # 使用两点的平均成本
        avg_cost = (self.cost_map[pos1] + self.cost_map[pos2]) / 2
        
        return distance * avg_cost
        
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Tuple[List[Tuple[int, int]], float]:
        """
        使用A*算法找到最优路径
        
        参数:
            start: 起点坐标
            goal: 终点坐标
            
        返回:
            Tuple[List[Tuple[int, int]], float]: 路径点列表和总成本
        """
        if self.cost_map is None:
            self.load_cost_map()
            
        # 检查起点和终点是否可通行
        if not np.isfinite(self.cost_map[start]) or not np.isfinite(self.cost_map[goal]):
            raise ValueError("起点或终点不可通行")
            
        # 初始化开放列表和关闭列表
        open_list = []
        closed_set = set()
        
        # 创建起点节点
        start_node = Node(start, 0, self.heuristic(start, goal))
        heapq.heappush(open_list, start_node)
        
        # 记录每个位置的最佳节点
        best_nodes = {start: start_node}
        
        print(f"\n开始寻路: 从{start}到{goal}")
        print(f"起点成本: {self.cost_map[start]:.2f}")
        print(f"终点成本: {self.cost_map[goal]:.2f}")
        
        # A*主循环
        while open_list:
            # 获取f成本最小的节点
            current = heapq.heappop(open_list)
            
            # 如果到达目标，构建路径并返回
            if current.pos == goal:
                path = []
                total_cost = current.g_cost
                while current:
                    path.append(current.pos)
                    current = current.parent
                path.reverse()
                
                # 平滑路径
                smoothed_path = self.smooth_path(path)
                
                print(f"找到路径！")
                print(f"路径长度: {len(smoothed_path)}个点")
                print(f"总成本: {total_cost:.2f}")
                
                return smoothed_path, total_cost
                
            # 将当前节点加入关闭列表
            closed_set.add(current.pos)
            
            # 检查相邻节点
            for neighbor_pos in self.get_neighbors(current.pos):
                # 如果节点已经在关闭列表中，跳过
                if neighbor_pos in closed_set:
                    continue
                    
                # 计算到相邻节点的成本
                g_cost = current.g_cost + self.get_path_cost(current.pos, neighbor_pos)
                
                # 如果这是一个新节点，或者找到了更好的路径
                if (neighbor_pos not in best_nodes or
                    g_cost < best_nodes[neighbor_pos].g_cost):
                    # 创建新节点
                    h_cost = self.heuristic(neighbor_pos, goal)
                    neighbor_node = Node(neighbor_pos, g_cost, h_cost, current)
                    
                    # 更新最佳节点记录
                    best_nodes[neighbor_pos] = neighbor_node
                    
                    # 添加到开放列表
                    heapq.heappush(open_list, neighbor_node)
                    
        print("未找到路径！")
        return [], float('inf')
        
    def smooth_path(
        self,
        path: List[Tuple[int, int]],
        sigma: float = 2.0
    ) -> List[Tuple[int, int]]:
        """
        使用高斯滤波平滑路径
        
        参数:
            path: 原始路径
            sigma: 高斯核标准差
            
        返回:
            List[Tuple[int, int]]: 平滑后的路径
        """
        if len(path) < 3:
            return path
            
        # 分离坐标
        rows = np.array([p[0] for p in path])
        cols = np.array([p[1] for p in path])
        
        # 应用高斯滤波
        smooth_rows = gaussian_filter1d(rows, sigma)
        smooth_cols = gaussian_filter1d(cols, sigma)
        
        # 四舍五入到整数并组合
        return list(zip(
            np.round(smooth_rows).astype(int),
            np.round(smooth_cols).astype(int)
        ))
        
def main():
    """主函数：测试路径规划器"""
    # 创建路径规划器
    planner = PathPlanner()
    
    # 加载成本图
    planner.load_cost_map()
    
    # 测试路径规划
    start = (100, 100)
    goal = (200, 200)
    path, cost = planner.find_path(start, goal)
    
    if path:
        print(f"\n路径规划成功:")
        print(f"路径长度: {len(path)}个点")
        print(f"总成本: {cost:.2f}")
    else:
        print("\n未找到可行路径")
        
if __name__ == "__main__":
    main() 