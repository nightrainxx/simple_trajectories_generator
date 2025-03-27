"""
路径规划模块: 使用A*算法实现最优路径规划

包含:
- A*算法实现
- 启发式函数
- 路径平滑和优化
"""

import numpy as np
import heapq
from typing import List, Tuple, Dict, Set, Optional
import rasterio
from utils import get_neighbors, calculate_distance
from map_generator import MapGenerator
from config import MAX_CROSS_SLOPE_DEGREES
from point_selector import PointSelector

class Node:
    """路径节点类"""
    def __init__(
        self,
        row: int,
        col: int,
        g_cost: float = 0,
        h_cost: float = 0,
        parent: Optional['Node'] = None
    ):
        self.row = row
        self.col = col
        self.g_cost = g_cost  # 从起点到当前点的实际成本
        self.h_cost = h_cost  # 从当前点到终点的估计成本
        self.f_cost = g_cost + h_cost  # 总成本
        self.parent = parent
        
    def __lt__(self, other: 'Node') -> bool:
        """优先队列比较函数"""
        return self.f_cost < other.f_cost
        
    def __eq__(self, other: object) -> bool:
        """相等性比较"""
        if not isinstance(other, Node):
            return NotImplemented
        return self.row == other.row and self.col == other.col
        
    def __hash__(self) -> int:
        """哈希函数，用于集合操作"""
        return hash((self.row, self.col))
        
class PathPlanner:
    """路径规划器类: 实现A*算法"""
    
    def __init__(
        self,
        env: MapGenerator
    ):
        """
        初始化路径规划器

        参数:
            env: MapGenerator对象，包含地形、速度和成本数据
        """
        self.env = env
        if self.env.cost_map is None:
            raise ValueError("请先生成成本图")
        
        # 预计算地图尺寸
        self.height, self.width = self.env.cost_map.shape
        
        # 设置搜索限制
        self.max_iterations = 100000  # 最大迭代次数
        
        # 预计算8个方向的偏移量和成本
        self.directions = [
            (-1, 0),   # 上
            (1, 0),    # 下
            (0, -1),   # 左
            (0, 1),    # 右
            (-1, -1),  # 左上
            (-1, 1),   # 右上
            (1, -1),   # 左下
            (1, 1)     # 右下
        ]
        self.direction_costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
        
        # 预计算成本阈值（用于剪枝）
        valid_costs = self.env.cost_map[self.env.cost_map < float('inf')]
        self.cost_threshold = np.percentile(valid_costs, 99)  # 使用99百分位数
        self.max_cost = self.cost_threshold * 2  # 设置最大成本为阈值的2倍
    
    def is_valid_point(self, row: int, col: int) -> bool:
        """检查点是否有效且成本合理"""
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
            
        cost = self.env.cost_map[row, col]
        return cost < self.max_cost
    
    def get_neighbors(self, node: Node) -> List[Tuple[int, int, float]]:
        """获取邻居节点（使用向量化操作）"""
        # 计算所有可能的邻居位置
        new_rows = node.row + np.array([d[0] for d in self.directions])
        new_cols = node.col + np.array([d[1] for d in self.directions])
        
        # 创建掩码
        valid_mask = (
            (new_rows >= 0) & (new_rows < self.height) &
            (new_cols >= 0) & (new_cols < self.width)
        )
        
        neighbors = []
        for i, (valid, dr, dc, base_cost) in enumerate(zip(
            valid_mask,
            new_rows,
            new_cols,
            self.direction_costs
        )):
            if valid and self.is_valid_point(dr, dc):
                cost = base_cost * self.env.cost_map[dr, dc]
                if cost < self.max_cost:  # 添加成本检查
                    neighbors.append((dr, dc, cost))
        
        return neighbors
    
    def heuristic(self, start: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """计算启发式成本（切比雪夫距离）"""
        dr = abs(start[0] - goal[0])
        dc = abs(start[1] - goal[1])
        return max(dr, dc)  # 使用切比雪夫距离
    
    def get_path_from_node(
        self,
        node: Node
    ) -> List[Tuple[int, int]]:
        """
        从终点节点回溯得到完整路径

        参数:
            node: 终点节点

        返回:
            路径点列表
        """
        path = []
        current = node
        
        while current is not None:
            path.append((current.row, current.col))
            current = current.parent
            
        return path[::-1]  # 反转列表,使其从起点开始
        
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """使用改进的A*算法寻找路径"""
        # 检查起点和终点是否可达
        if not self.is_valid_point(start[0], start[1]):
            raise ValueError("起点不可达")
        if not self.is_valid_point(goal[0], goal[1]):
            raise ValueError("终点不可达")
        
        # 计算直线距离作为参考
        direct_distance = np.sqrt(
            (goal[0] - start[0])**2 + 
            (goal[1] - start[1])**2
        )
        
        # 初始化起点
        start_node = Node(start[0], start[1], 0, self.heuristic(start, goal))
        
        # 使用优先队列和集合来优化搜索
        open_list = []
        heapq.heappush(open_list, start_node)
        closed_set = set()
        
        # 使用字典记录每个位置的g值，用于快速更新
        g_values = {(start[0], start[1]): 0}
        
        iterations = 0
        best_distance = float('inf')
        best_node = None
        
        while open_list and iterations < self.max_iterations:
            iterations += 1
            
            current = heapq.heappop(open_list)
            current_pos = (current.row, current.col)
            
            # 如果找到目标
            if current_pos == goal:
                path = []
                while current:
                    path.append((current.row, current.col))
                    current = current.parent
                return path[::-1]
            
            # 如果已经访问过，跳过
            if current_pos in closed_set:
                continue
            
            # 计算到目标的直线距离
            current_distance = np.sqrt(
                (goal[0] - current.row)**2 + 
                (goal[1] - current.col)**2
            )
            
            # 更新最佳距离
            if current_distance < best_distance:
                best_distance = current_distance
                best_node = current
            
            # 如果当前路径长度已经超过直线距离的2倍，考虑使用之前找到的最佳点
            if current.g_cost > 2 * direct_distance and best_node:
                path = []
                current = best_node
                while current:
                    path.append((current.row, current.col))
                    current = current.parent
                return path[::-1]
            
            closed_set.add(current_pos)
            
            # 检查所有邻居
            for next_row, next_col, cost in self.get_neighbors(current):
                next_pos = (next_row, next_col)
                
                # 如果已经访问过，跳过
                if next_pos in closed_set:
                    continue
                
                # 计算新的g值
                new_g = g_values[current_pos] + cost
                
                # 如果这个位置已经有更好的路径，跳过
                if next_pos in g_values and new_g >= g_values[next_pos]:
                    continue
                
                # 更新g值并添加到开放列表
                g_values[next_pos] = new_g
                h = self.heuristic(next_pos, goal)
                neighbor = Node(next_row, next_col, new_g, h, current)
                heapq.heappush(open_list, neighbor)
        
        # 如果达到最大迭代次数但找到了较好的路径，返回该路径
        if best_node and iterations >= self.max_iterations:
            print(f"    警告：达到最大迭代次数 {self.max_iterations}，返回次优路径")
            path = []
            current = best_node
            while current:
                path.append((current.row, current.col))
                current = current.parent
            return path[::-1]
        
        # 完全找不到路径
        print(f"    警告：达到最大迭代次数 {self.max_iterations}")
        return None
        
    def smooth_path(
        self,
        path: List[Tuple[int, int]],
        smoothing_factor: float = 0.5
    ) -> List[Tuple[int, int]]:
        """
        平滑路径

        参数:
            path: 原始路径点列表
            smoothing_factor: 平滑因子，范围[0, 1]

        返回:
            smoothed_path: 平滑后的路径点列表
        """
        if not path or len(path) <= 2:
            return path
        
        smoothed_path = [list(p) for p in path]
        change = True
        
        while change:
            change = False
            for i in range(1, len(smoothed_path) - 1):
                old_point = list(smoothed_path[i])
                
                # 向前一个点和后一个点的方向平滑
                smoothed_path[i][0] += smoothing_factor * (
                    smoothed_path[i-1][0] + smoothed_path[i+1][0] - 2 * smoothed_path[i][0]
                )
                smoothed_path[i][1] += smoothing_factor * (
                    smoothed_path[i-1][1] + smoothed_path[i+1][1] - 2 * smoothed_path[i][1]
                )
                
                # 如果点的位置改变了
                if old_point != smoothed_path[i]:
                    change = True
                    
                    # 检查新位置是否可行
                    new_row = int(round(smoothed_path[i][0]))
                    new_col = int(round(smoothed_path[i][1]))
                    
                    if not (0 <= new_row < self.env.cost_map.shape[0] and 
                           0 <= new_col < self.env.cost_map.shape[1] and
                           self.env.cost_map[new_row, new_col] < float('inf')):
                        # 如果新位置不可行，恢复原位置
                        smoothed_path[i] = old_point
        
        return [(int(round(p[0])), int(round(p[1]))) for p in smoothed_path]
        
def main():
    """主函数：测试路径规划器"""
    # 创建点选择器
    print("初始化点选择器...")
    selector = PointSelector()
    selector.load_landcover()
    
    # 选择起终点对
    print("选择起终点对...")
    pairs = selector.select_start_end_pairs(num_pairs=1)
    
    if not pairs:
        print("未找到有效的起终点对！")
        return
        
    start, goal = pairs[0]
    print(f"选择的起终点对:")
    print(f"  起点: {start}")
    print(f"  终点: {goal}")
    print(f"  直线距离: {selector.calculate_distance(start, goal)/1000:.2f} km")
    
    # 创建地图生成器
    print("\n初始化地图生成器...")
    map_gen = MapGenerator()
    
    # 加载环境数据
    print("加载环境数据...")
    map_gen.load_data()
    
    # 生成初始成本图
    print("生成成本图...")
    map_gen.generate_cost_map()
    
    # 创建路径规划器
    print("\n开始路径规划...")
    planner = PathPlanner(map_gen)
    
    try:
        path = planner.find_path(start, goal)
        
        if path:
            print(f"找到路径！路径长度: {len(path)} 个点")
            
            # 计算路径长度
            total_distance = 0
            for i in range(len(path)-1):
                total_distance += selector.calculate_distance(path[i], path[i+1])
            print(f"路径总长度: {total_distance/1000:.2f} km")
            
            # 平滑路径
            print("\n平滑路径...")
            smoothed_path = planner.smooth_path(path)
            print(f"平滑后路径长度: {len(smoothed_path)} 个点")
            
            # 计算平滑后路径长度
            total_distance = 0
            for i in range(len(smoothed_path)-1):
                total_distance += selector.calculate_distance(
                    smoothed_path[i],
                    smoothed_path[i+1]
                )
            print(f"平滑后路径总长度: {total_distance/1000:.2f} km")
        else:
            print("未找到可行路径！")
    except ValueError as e:
        print(f"错误: {e}")
        
if __name__ == "__main__":
    main() 