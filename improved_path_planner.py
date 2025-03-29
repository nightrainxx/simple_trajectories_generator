"""
改进版路径规划器：使用分层A*算法进行路径规划

改进内容：
1. 增大搜索半径，提高寻路效率
2. 实现分层规划：先粗粒度规划，再细粒度优化
3. 加入动态启发式权重

输入：
- 成本图
- 起点和终点坐标

输出：
- 路径点列表
- 路径总成本
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
        self.f_cost = g_cost + h_cost  # 总估计成本
        self.parent = parent
        
    def __lt__(self, other):
        return self.f_cost < other.f_cost
        
class ImprovedPathPlanner:
    """改进版路径规划器类"""
    
    def __init__(self):
        """初始化路径规划器"""
        self.cost_map = None
        self.shape = None
        self.coarse_map = None
        self.coarse_shape = None
        self.scale_factor = 8  # 粗粒度地图的缩放因子
        
    def load_cost_map(self) -> None:
        """加载成本图"""
        cost_path = os.path.join(OUTPUT_DIR, "intermediate", "cost_map.tif")
        with rasterio.open(cost_path) as src:
            self.cost_map = src.read(1)
            self.shape = self.cost_map.shape
            
        # 创建粗粒度地图
        self._create_coarse_map()
            
    def _create_coarse_map(self) -> None:
        """创建粗粒度成本地图"""
        if self.cost_map is None:
            raise ValueError("请先加载成本图")
            
        # 计算粗粒度地图的大小
        coarse_rows = self.shape[0] // self.scale_factor
        coarse_cols = self.shape[1] // self.scale_factor
        self.coarse_shape = (coarse_rows, coarse_cols)
        
        # 初始化粗粒度地图
        self.coarse_map = np.zeros(self.coarse_shape, dtype=np.float32)
        
        # 使用区域平均值填充粗粒度地图
        for r in range(coarse_rows):
            for c in range(coarse_cols):
                # 定义细粒度地图中的对应区域
                r_start = r * self.scale_factor
                r_end = min(r_start + self.scale_factor, self.shape[0])
                c_start = c * self.scale_factor
                c_end = min(c_start + self.scale_factor, self.shape[1])
                
                # 提取区域
                region = self.cost_map[r_start:r_end, c_start:c_end]
                
                # 计算区域的平均成本（忽略无穷大值）
                valid_region = region[np.isfinite(region)]
                if len(valid_region) > 0:
                    self.coarse_map[r, c] = np.mean(valid_region)
                else:
                    self.coarse_map[r, c] = np.inf
        
        print(f"创建了粗粒度地图，大小: {self.coarse_shape}")
            
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int], adaptive_weight: float = 1.0) -> float:
        """
        计算启发式成本（欧几里得距离）
        
        参数:
            pos: 当前位置
            goal: 目标位置
            adaptive_weight: 自适应权重
            
        返回:
            float: 启发式成本
        """
        # 使用欧几里得距离作为启发式函数
        distance = np.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)
        return adaptive_weight * distance
        
    def get_neighbors(self, pos: Tuple[int, int], is_coarse: bool = False) -> List[Tuple[int, int]]:
        """
        获取相邻节点
        
        参数:
            pos: 当前位置
            is_coarse: 是否在粗粒度地图上
            
        返回:
            List[Tuple[int, int]]: 相邻节点列表
        """
        row, col = pos
        neighbors = []
        
        # 使用的地图和形状
        current_map = self.coarse_map if is_coarse else self.cost_map
        current_shape = self.coarse_shape if is_coarse else self.shape
        
        # 搜索半径，粗粒度地图上使用更大的半径
        radius = 5 if is_coarse else 2
        
        # 在指定半径内搜索相邻点
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                # 跳过中心点
                if dr == 0 and dc == 0:
                    continue
                    
                new_row, new_col = row + dr, col + dc
                
                # 检查边界
                if not (0 <= new_row < current_shape[0] and 0 <= new_col < current_shape[1]):
                    continue
                    
                # 检查是否可通行
                if not np.isfinite(current_map[new_row, new_col]):
                    continue
                    
                neighbors.append((new_row, new_col))
                
        return neighbors
        
    def get_path_cost(self, pos1: Tuple[int, int], pos2: Tuple[int, int], is_coarse: bool = False) -> float:
        """
        计算两个相邻点之间的路径成本
        
        参数:
            pos1: 第一个点
            pos2: 第二个点
            is_coarse: 是否在粗粒度地图上
            
        返回:
            float: 路径成本
        """
        # 计算欧几里得距离
        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        
        # 使用的地图
        current_map = self.coarse_map if is_coarse else self.cost_map
        
        # 使用两点的平均成本
        avg_cost = (current_map[pos1] + current_map[pos2]) / 2
        
        # 添加局部变化因子，使相似成本区域也有细微差异
        local_variation = np.random.uniform(0.95, 1.05)
        
        return distance * avg_cost * local_variation
        
    def coarse_to_fine(self, coarse_pos: Tuple[int, int]) -> Tuple[int, int]:
        """将粗粒度位置转换为细粒度位置"""
        return (coarse_pos[0] * self.scale_factor, coarse_pos[1] * self.scale_factor)
        
    def fine_to_coarse(self, fine_pos: Tuple[int, int]) -> Tuple[int, int]:
        """将细粒度位置转换为粗粒度位置"""
        return (fine_pos[0] // self.scale_factor, fine_pos[1] // self.scale_factor)
        
    def a_star_search(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int], 
        is_coarse: bool = False,
        adaptive_weight: float = 1.0
    ) -> List[Tuple[int, int]]:
        """
        使用A*算法寻找路径
        
        参数:
            start: 起点
            goal: 终点
            is_coarse: 是否在粗粒度地图上
            adaptive_weight: 启发式的自适应权重
            
        返回:
            List[Tuple[int, int]]: 路径点列表
        """
        # 初始化开放列表和关闭列表
        open_list = []
        closed_set = set()
        
        # 创建起点节点
        start_node = Node(start, 0, self.heuristic(start, goal, adaptive_weight))
        heapq.heappush(open_list, start_node)
        
        # 记录每个位置的最佳节点
        best_nodes = {start: start_node}
        
        # A*主循环
        while open_list:
            # 获取f成本最小的节点
            current = heapq.heappop(open_list)
            
            # 如果到达目标，构建路径并返回
            if current.pos == goal:
                path = []
                while current:
                    path.append(current.pos)
                    current = current.parent
                path.reverse()
                return path
                
            # 将当前节点加入关闭列表
            closed_set.add(current.pos)
            
            # 检查相邻节点
            for neighbor_pos in self.get_neighbors(current.pos, is_coarse):
                # 如果节点已经在关闭列表中，跳过
                if neighbor_pos in closed_set:
                    continue
                    
                # 计算到相邻节点的成本
                g_cost = current.g_cost + self.get_path_cost(current.pos, neighbor_pos, is_coarse)
                
                # 如果这是一个新节点，或者找到了更好的路径
                if (neighbor_pos not in best_nodes or
                    g_cost < best_nodes[neighbor_pos].g_cost):
                    # 创建新节点
                    h_cost = self.heuristic(neighbor_pos, goal, adaptive_weight)
                    neighbor_node = Node(neighbor_pos, g_cost, h_cost, current)
                    
                    # 更新最佳节点记录
                    best_nodes[neighbor_pos] = neighbor_node
                    
                    # 添加到开放列表
                    heapq.heappush(open_list, neighbor_node)
                    
        # 未找到路径
        return []
        
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        key_point_ratio: float = 0.2  # 添加关键点比例参数
    ) -> Tuple[bool, List[Tuple[int, int]], float]:
        """
        使用分层A*算法找到最优路径
        
        参数:
            start: 起点坐标
            goal: 终点坐标
            key_point_ratio: 粗粒度路径中作为关键点的比例(0.0-1.0)
            
        返回:
            Tuple[bool, List[Tuple[int, int]], float]: 
            是否成功、平滑后的路径和总成本
        """
        if self.cost_map is None:
            self.load_cost_map()
            
        # 检查起点和终点是否可通行
        if not np.isfinite(self.cost_map[start]) or not np.isfinite(self.cost_map[goal]):
            print(f"起点或终点不可通行")
            return False, [], float('inf')
            
        print(f"\n开始分层寻路: 从{start}到{goal}")
        print(f"起点成本: {self.cost_map[start]:.2f}")
        print(f"终点成本: {self.cost_map[goal]:.2f}")
        
        # 第一步：在粗粒度地图上规划路径
        coarse_start = self.fine_to_coarse(start)
        coarse_goal = self.fine_to_coarse(goal)
        print(f"粗粒度规划: 从{coarse_start}到{coarse_goal}")
        
        coarse_path = self.a_star_search(
            coarse_start, 
            coarse_goal, 
            is_coarse=True,
            adaptive_weight=1.5  # 使用较大的权重使粗粒度搜索更快
        )
        
        if not coarse_path:
            print("在粗粒度地图上未找到路径")
            return False, [], float('inf')
            
        print(f"粗粒度路径长度: {len(coarse_path)}个点")
        
        # 计算粗粒度路径总成本
        coarse_cost = 0
        for i in range(len(coarse_path) - 1):
            coarse_cost += self.get_path_cost(coarse_path[i], coarse_path[i+1], is_coarse=True)
            
        print(f"粗粒度路径总成本: {coarse_cost:.2f}")
        
        # 第二步：在细粒度地图上优化路径
        # 从粗粒度路径创建关键点
        keypoints = [start]  # 起点
        
        # 添加粗粒度路径中的点，转换为细粒度
        # 根据key_point_ratio参数调整步长
        coarse_step = max(1, int(len(coarse_path) * (1 - key_point_ratio)))
        for i in range(coarse_step, len(coarse_path) - coarse_step, coarse_step):
            fine_point = self.coarse_to_fine(coarse_path[i])
            # 确保点在有效范围内
            fine_point = (
                min(max(0, fine_point[0]), self.shape[0] - 1),
                min(max(0, fine_point[1]), self.shape[1] - 1)
            )
            # 确保点是可通行的
            if np.isfinite(self.cost_map[fine_point]):
                keypoints.append(fine_point)
                
        keypoints.append(goal)  # 终点
        
        print(f"生成了{len(keypoints)}个关键点")
        
        # 在关键点之间进行细粒度路径规划
        fine_path = []
        total_cost = 0
        
        for i in range(len(keypoints) - 1):
            sub_start = keypoints[i]
            sub_goal = keypoints[i + 1]
            
            print(f"细粒度规划 {i+1}/{len(keypoints)-1}: 从{sub_start}到{sub_goal}")
            
            sub_path = self.a_star_search(
                sub_start, 
                sub_goal, 
                is_coarse=False,
                adaptive_weight=1.0  # 细粒度搜索使用标准权重
            )
            
            if not sub_path:
                print(f"在细粒度地图上未找到从{sub_start}到{sub_goal}的路径")
                # 尝试直接连接
                direct_path = self._connect_points(sub_start, sub_goal)
                if direct_path:
                    print("使用直接连接")
                    sub_path = direct_path
                    sub_cost = sum(
                        self.get_path_cost(direct_path[i], direct_path[i+1])
                        for i in range(len(direct_path) - 1)
                    )
                else:
                    # 如果还是失败，跳过这个子路径
                    continue
            else:
                # 计算子路径成本
                sub_cost = 0
                for j in range(len(sub_path) - 1):
                    sub_cost += self.get_path_cost(sub_path[j], sub_path[j+1])
                    
            # 添加到总路径中（避免重复添加连接点）
            if fine_path and sub_path:
                fine_path.extend(sub_path[1:])  # 去掉第一个点，因为它已经在前一段路径的末尾
            else:
                fine_path.extend(sub_path)
                
            total_cost += sub_cost
            
        if not fine_path:
            print("在细粒度地图上未找到完整路径")
            return False, [], float('inf')
            
        # 平滑路径
        smoothed_path = self.smooth_path(fine_path)
        
        print(f"找到路径！")
        print(f"原始路径长度: {len(fine_path)}个点")
        print(f"平滑后路径长度: {len(smoothed_path)}个点")
        print(f"总成本: {total_cost:.2f}")
        
        return True, smoothed_path, total_cost
        
    def _connect_points(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> List[Tuple[int, int]]:
        """在两点之间创建一条直线路径"""
        path = []
        
        # 计算点之间的距离
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance = max(abs(dx), abs(dy))
        
        # 如果距离过大，可能会导致路径穿过不可通行区域
        if distance > 100:
            return []
            
        # 使用Bresenham算法绘制线段
        for i in range(distance + 1):
            t = i / distance if distance > 0 else 0
            x = int(p1[0] + dx * t)
            y = int(p1[1] + dy * t)
            
            # 检查点是否可通行
            if 0 <= x < self.shape[0] and 0 <= y < self.shape[1]:
                if np.isfinite(self.cost_map[x, y]):
                    path.append((x, y))
                else:
                    # 如果遇到不可通行点，中断路径
                    return []
            else:
                # 如果超出边界，中断路径
                return []
                
        return path
        
    def smooth_path(
        self,
        path: List[Tuple[int, int]],
        sigma: float = 5.0,  # 增加sigma值使路径更平滑
        window_size: int = 7  # 添加滑动窗口大小参数
    ) -> List[Tuple[int, int]]:
        """
        使用高斯滤波平滑路径
        
        参数:
            path: 原始路径
            sigma: 高斯核标准差
            window_size: 滑动窗口大小
            
        返回:
            List[Tuple[int, int]]: 平滑后的路径
        """
        if len(path) < 5:
            return path
            
        # 分离坐标
        rows = np.array([p[0] for p in path])
        cols = np.array([p[1] for p in path])
        
        # 应用高斯滤波
        smooth_rows = gaussian_filter1d(rows, sigma)
        smooth_cols = gaussian_filter1d(cols, sigma)
        
        # 使用滑动窗口进一步平滑
        def moving_average(arr, window_size):
            weights = np.ones(window_size) / window_size
            return np.convolve(arr, weights, mode='valid')
        
        if len(smooth_rows) > window_size:
            smooth_rows = moving_average(smooth_rows, window_size)
            smooth_cols = moving_average(smooth_cols, window_size)
            
        # 确保平滑后的路径点仍然可通行
        smoothed_path = []
        for row, col in zip(smooth_rows, smooth_cols):
            r, c = int(round(row)), int(round(col))
            
            # 确保坐标在有效范围内
            r = min(max(0, r), self.shape[0] - 1)
            c = min(max(0, c), self.shape[1] - 1)
            
            # 只添加可通行的点
            if np.isfinite(self.cost_map[r, c]):
                smoothed_path.append((r, c))
                
        # 确保平滑后的路径至少包含起点和终点
        if len(smoothed_path) < 2:
            return [path[0], path[-1]]
            
        # 确保起点和终点不变
        if smoothed_path[0] != path[0]:
            smoothed_path.insert(0, path[0])
        if smoothed_path[-1] != path[-1]:
            smoothed_path.append(path[-1])
            
        return smoothed_path
        
def main():
    """测试函数"""
    # 创建路径规划器
    planner = ImprovedPathPlanner()
    planner.load_cost_map()
    
    # 测试寻路
    # 使用与之前相同的测试起点和终点
    start = (1914, 3284)
    goal = (113, 481)
    
    # 进行寻路
    success, path, cost = planner.find_path(start, goal)
    
    # 输出结果
    if success:
        print(f"\n寻路成功:")
        print(f"原始路径长度: {len(path)}个点")
        print(f"平滑后路径长度: {len(path)}个点")
        print(f"总成本: {cost:.2f}")
    else:
        print("寻路失败")
        
if __name__ == "__main__":
    main() 