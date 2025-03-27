"""
路径规划模块: 使用A*算法实现最优路径规划

包含:
- A*算法实现
- 启发式函数
- 路径平滑和优化
"""

import numpy as np
from typing import List, Tuple, Optional
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement

class PathPlanner:
    def __init__(self, env):
        self.env = env
        
    def is_valid_point(self, x: int, y: int) -> bool:
        """检查点是否有效"""
        if not (0 <= x < self.env.width and 0 <= y < self.env.height):
            return False
        return self.env.cost_map[x, y] < float('inf')
        
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
            
        # 处理成本图，将其转换为整数权重矩阵
        cost_map = self.env.cost_map.copy()
        
        # 将无穷大值替换为一个较大的有限值
        max_finite_cost = np.max(cost_map[np.isfinite(cost_map)])
        cost_map[np.isinf(cost_map)] = max_finite_cost * 10
        
        # 确保所有值都是有限的正数
        cost_map = np.clip(cost_map, 0, max_finite_cost * 10)
        
        # 将成本图缩放到合适的整数范围（1-1000）
        scale_factor = 1000.0 / max_finite_cost
        weights_matrix = np.round(cost_map * scale_factor).astype(np.int32)
        
        # 确保没有0权重（除非是障碍物）
        weights_matrix = np.maximum(weights_matrix, 1)
        
        # 将障碍物标记为0
        weights_matrix[self.env.cost_map >= max_finite_cost * 5] = 0
        
        # 打印调试信息
        print(f"权重矩阵类型: {weights_matrix.dtype}")
        print(f"权重范围: [{np.min(weights_matrix)}, {np.max(weights_matrix)}]")
        if np.isnan(weights_matrix).any():
            print("警告：权重矩阵中存在NaN值！")
        if np.isinf(weights_matrix).any():
            print("警告：权重矩阵中存在Inf值！")
            
        # 创建网格对象
        grid = Grid(matrix=weights_matrix.tolist())
        
        # 创建寻路器
        finder = AStarFinder(
            diagonal_movement=DiagonalMovement.always,
            weight=1,
            heuristic=lambda dx, dy: max(abs(dx), abs(dy))  # 切比雪夫距离
        )
        
        # 获取起点和终点
        start_node = grid.node(start[1], start[0])
        end_node = grid.node(goal[1], goal[0])
        
        # 寻找路径
        path, _ = finder.find_path(start_node, end_node, grid)
        
        # 如果找到路径，转换坐标格式
        if path:
            return [(node.y, node.x) for node in path]  # 转换为(row,col)格式
        
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