"""
路径规划模块: 使用A*算法实现最优路径规划

包含:
- A*算法实现
- 启发式函数
- 路径平滑和优化
"""

import numpy as np
import heapq
from typing import List, Tuple, Dict, Set
import rasterio
from utils import get_neighbors, calculate_distance

class Node:
    """路径节点类"""
    def __init__(
        self,
        position: Tuple[int, int],
        g_cost: float = float('inf'),
        h_cost: float = 0.0,
        parent: 'Node' = None
    ):
        self.position = position
        self.g_cost = g_cost  # 从起点到当前点的实际代价
        self.h_cost = h_cost  # 从当前点到终点的估计代价
        self.f_cost = g_cost + h_cost  # 总代价
        self.parent = parent
        
    def __lt__(self, other: 'Node') -> bool:
        """优先队列比较函数"""
        return self.f_cost < other.f_cost
        
class PathPlanner:
    """路径规划器类: 实现A*算法"""
    
    def __init__(
        self,
        cost_map: np.ndarray,
        transform: rasterio.Affine,
        max_cost: float = 1e6
    ):
        """
        初始化路径规划器

        参数:
            cost_map: 成本图
            transform: rasterio的仿射变换矩阵
            max_cost: 不可通行区域的成本值
        """
        self.cost_map = cost_map
        self.shape = cost_map.shape
        self.transform = transform
        self.max_cost = max_cost
        
    def heuristic(
        self,
        current: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> float:
        """
        计算启发式代价(欧几里得距离)

        参数:
            current: 当前位置
            goal: 目标位置

        返回:
            估计代价
        """
        # 使用实际距离作为启发式函数
        return calculate_distance(current, goal, self.transform)
        
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
            path.append(current.position)
            current = current.parent
            
        return path[::-1]  # 反转列表,使其从起点开始
        
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        使用A*算法寻找最优路径

        参数:
            start: 起点坐标
            goal: 终点坐标

        返回:
            路径点列表
        """
        # 检查起点和终点是否可达
        if (
            self.cost_map[start] >= self.max_cost or
            self.cost_map[goal] >= self.max_cost
        ):
            raise ValueError("起点或终点不可达")
            
        # 初始化起点
        start_node = Node(
            position=start,
            g_cost=0.0,
            h_cost=self.heuristic(start, goal)
        )
        
        # 初始化开放列表和关闭列表
        open_list: List[Node] = [start_node]  # 优先队列
        closed_set: Set[Tuple[int, int]] = set()  # 已访问节点集合
        node_dict: Dict[Tuple[int, int], Node] = {start: start_node}  # 节点字典
        
        while open_list:
            # 获取f值最小的节点
            current = heapq.heappop(open_list)
            
            # 如果到达目标,返回路径
            if current.position == goal:
                return self.get_path_from_node(current)
                
            # 将当前节点加入关闭列表
            closed_set.add(current.position)
            
            # 遍历相邻节点
            for next_pos in get_neighbors(current.position, self.shape):
                # 如果节点已在关闭列表中,跳过
                if next_pos in closed_set:
                    continue
                    
                # 如果节点不可通行,跳过
                if self.cost_map[next_pos] >= self.max_cost:
                    continue
                    
                # 计算移动代价
                move_cost = calculate_distance(
                    current.position,
                    next_pos,
                    self.transform
                ) * self.cost_map[next_pos]
                
                # 计算从起点经过当前节点到相邻节点的代价
                g_cost = current.g_cost + move_cost
                
                # 如果节点不在开放列表中,创建新节点
                if next_pos not in node_dict:
                    next_node = Node(
                        position=next_pos,
                        g_cost=g_cost,
                        h_cost=self.heuristic(next_pos, goal),
                        parent=current
                    )
                    node_dict[next_pos] = next_node
                    heapq.heappush(open_list, next_node)
                    continue
                    
                # 如果找到更好的路径,更新节点
                next_node = node_dict[next_pos]
                if g_cost < next_node.g_cost:
                    next_node.g_cost = g_cost
                    next_node.f_cost = g_cost + next_node.h_cost
                    next_node.parent = current
                    
                    # 重新加入优先队列
                    if next_node not in open_list:
                        heapq.heappush(open_list, next_node)
                        
        # 如果没有找到路径,返回空列表
        return []
        
    def smooth_path(
        self,
        path: List[Tuple[int, int]],
        window_size: int = 5
    ) -> List[Tuple[int, int]]:
        """
        使用移动平均平滑路径

        参数:
            path: 原始路径
            window_size: 平滑窗口大小

        返回:
            平滑后的路径
        """
        if len(path) <= window_size:
            return path
            
        smoothed_path = []
        half_window = window_size // 2
        
        # 保持起点和终点不变
        smoothed_path.extend(path[:half_window])
        
        # 对中间点进行平滑
        for i in range(half_window, len(path) - half_window):
            window = path[i - half_window:i + half_window + 1]
            row = int(np.mean([p[0] for p in window]))
            col = int(np.mean([p[1] for p in window]))
            
            # 检查平滑后的点是否可通行
            if self.cost_map[row, col] >= self.max_cost:
                smoothed_path.append(path[i])
            else:
                smoothed_path.append((row, col))
                
        # 保持终点不变
        smoothed_path.extend(path[-half_window:])
        
        return smoothed_path 