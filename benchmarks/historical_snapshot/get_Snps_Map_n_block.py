import math
import time
from multiprocessing import Pool, cpu_count
from collections import defaultdict, deque
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union


def process_chunk_optimized(chunk_data: List[Tuple[int, int, float]], col_indices: List[int],
                            hapcut2_1_line_count: int, a: float) -> Dict[Tuple[int, int], float]:
    """
    优化版本：处理一批稀疏矩阵数据，不保存权值为0的边
    """
    # 按行组织数据，使用列表存储
    row_data = defaultdict(list)
    for row_idx, col_idx, value in chunk_data:
        row_data[row_idx].append((col_idx, value))

    # 预分配权重字典
    local_weights = {}

    # 处理每一行
    for row_idx, col_value_pairs in row_data.items():
        # print(f"row_idx={row_idx}")
        num_nz = len(col_value_pairs)
        if num_nz < 2:
            continue

        # 按列索引排序（一次性排序）
        col_value_pairs.sort(key=lambda x: x[0])

        # 预计算归一化因子
        normalization_divisor = 1.0 / (num_nz - 1)

        max_nz = 0

        # 计算所有非零列对
        for idx1 in range(num_nz):
            col1, val1 = col_value_pairs[idx1]
            abs_val1 = abs(val1)
            # print(f"num_nz = {num_nz}")

            # 限制idx2不超过列表长度
            # end_idx = min(idx1 + 2, num_nz) # -1
            # end_idx = min(idx1 + 3, num_nz) # -2
            end_idx = num_nz # -N
            # end_idx = min(idx1 + max(2, (num_nz // 2)), num_nz) # -N/2
            # end_idx = min(idx1 + max(2, (num_nz // 3)), num_nz)  # -N/3
            # end_idx = min(idx1 + max(2, (num_nz // 4)), num_nz) # -N/4
            # end_idx = min(idx1 + max(2, (num_nz // 10)), num_nz) # -N/10
            for idx2 in range(idx1 + 1, end_idx):
                col2, val2 = col_value_pairs[idx2]
                abs_val2 = abs(val2)

                # 优化数学计算
                q1_comp = 1.0 - abs_val1
                q2_comp = 1.0 - abs_val2

                p1 = abs_val1 * abs_val2 + q1_comp * q2_comp
                p2 = abs_val1 * q2_comp + abs_val2 * q1_comp

                # 避免log10(0)的情况
                if p2 == 0:
                    weight = 10.0  # 使用大数值代替inf
                elif p1 == 0:
                    weight = -10.0
                else:
                    # 使用符号判断来决定log的参数顺序
                    if val1 * val2 > 0:
                        weight = math.log10(p1 / p2)
                    else:
                        weight = math.log10(p2 / p1)

                # 根据行索引是否超过第一个文件的最大行数来调整权重计算
                if 0 < hapcut2_1_line_count < row_idx + 1:
                    norm_weight = weight * normalization_divisor * a
                else:
                    norm_weight = weight * normalization_divisor

                # 过滤权值为0的边（考虑浮点数精度问题）
                if abs(norm_weight) < 1e-9:
                    continue

                # 确保使用有序的列索引对作为键
                pair_key = (min(col1, col2), max(col1, col2))
                if pair_key in local_weights:
                    local_weights[pair_key] += norm_weight
                else:
                    local_weights[pair_key] = norm_weight

    return local_weights


def compute_edge_weights_optimized(matrix_weight: List[Tuple[int, int, float]],
                                   pos_to_col_idx: Dict[int, int],
                                   hapcut2_1_line_count: int = 0,
                                   a: float = 1.0) -> Tuple[Optional[Dict[Tuple[int, int], float]],
Optional[Dict[int, int]],
Optional[Set[int]]]:
    """
    优化版本：计算边权重，不保存权值为0的边
    """
    print(f"开始处理矩阵数据，共 {len(matrix_weight)} 条记录")

    # 创建反向映射：列索引到位置
    col_idx_to_pos = {idx: pos for pos, idx in pos_to_col_idx.items()}
    col_indices = sorted(col_idx_to_pos.keys())

    if not col_indices:
        print("错误: 列映射为空")
        return None, None, None

    # 预过滤和转换数据类型（一次性完成）
    matrix_data = []
    for row_idx, col_idx, value in matrix_weight:
        float_val = float(value)
        if float_val != 0 and col_idx in col_idx_to_pos:
            matrix_data.append((row_idx, col_idx, float_val))

    print(f"过滤后共 {len(matrix_data)} 条非零记录")

    if not matrix_data:
        return {}, col_idx_to_pos, set(col_idx_to_pos.values())

    # 优化多进程设置
    num_processes = min(cpu_count(), 8)  # 限制最大进程数

    # 动态调整块大小
    optimal_chunk_size = max(1000, len(matrix_data) // (num_processes * 4))
    chunks = [matrix_data[i:i + optimal_chunk_size]
              for i in range(0, len(matrix_data), optimal_chunk_size)]

    print(f"将数据分为 {len(chunks)} 个块进行处理，每块约 {optimal_chunk_size} 条记录")

    # 并行处理，传递行数参数和调整参数a
    with Pool(processes=num_processes) as pool:
        # 使用partial传递额外参数
        from functools import partial
        process_func = partial(
            process_chunk_optimized,
            col_indices=col_indices,
            hapcut2_1_line_count=hapcut2_1_line_count,
            a=a
        )

        results = [pool.apply_async(process_func, (chunk,)) for chunk in chunks]

        # 汇总结果
        total_weights = {}
        for result in results:
            partial_dict = result.get()
            for key, val in partial_dict.items():
                # 再次过滤合并后可能为0的边
                if abs(val) < 1e-9:
                    continue
                if key in total_weights:
                    total_weights[key] += val
                    # 合并后如果变为0则移除
                    if abs(total_weights[key]) < 1e-9:
                        del total_weights[key]
                else:
                    total_weights[key] = val

    print(f"处理完成，共计算出 {len(total_weights)} 对列之间的非零权重")

    # 转换为位置-位置的边权重
    edge_weights = {}
    all_positions = set(col_idx_to_pos.values())

    # 批量转换边权重，再次过滤
    for (i, j), weight in total_weights.items():
        if abs(weight) < 1e-9:
            continue
        pos1 = col_idx_to_pos[i]
        pos2 = col_idx_to_pos[j]
        edge_weights[(pos1, pos2)] = -weight

    return edge_weights, col_idx_to_pos, all_positions


class UnionFindOptimized:
    """
    优化的并查集实现
    """

    def __init__(self, nodes: Set[int]):
        # 创建节点到索引的映射
        self.nodes = sorted(nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes)}

        n = len(self.nodes)
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, node: int) -> int:
        idx = self.node_to_idx[node]
        if self.parent[idx] != idx:
            self.parent[idx] = self.find(self.idx_to_node[self.parent[idx]])
        return self.idx_to_node[self.parent[idx]]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return

        px_idx, py_idx = self.node_to_idx[px], self.node_to_idx[py]
        if self.rank[px_idx] < self.rank[py_idx]:
            self.parent[px_idx] = py_idx
        elif self.rank[px_idx] > self.rank[py_idx]:
            self.parent[py_idx] = px_idx
        else:
            self.parent[py_idx] = px_idx
            self.rank[px_idx] += 1


def find_connected_components_ultra_optimized(edge_weights: Dict[Tuple[int, int], float],
                                              all_positions: Set[int]) -> List[
    Tuple[Set[int], List[Tuple[int, int, float]]]]:
    """
    超级优化版本：使用并查集查找连通分量
    """
    if not edge_weights:
        return []

    # 初始化并查集
    uf = UnionFindOptimized(all_positions)

    # 合并所有边连接的节点
    for (pos1, pos2), _ in edge_weights.items():
        uf.union(pos1, pos2)

    # 收集连通分量
    components_dict = defaultdict(set)
    for node in all_positions:
        root = uf.find(node)
        components_dict[root].add(node)

    # 构建结果，过滤单节点分量
    components_data = []
    single_node_components = 0

    for component in components_dict.values():
        # 过滤掉只有一个节点的连通分量
        if len(component) == 1:
            single_node_components += 1
            continue

        # 高效收集该分量的所有边
        component_edges = []
        for (pos1, pos2), weight in edge_weights.items():
            if pos1 in component and pos2 in component:
                component_edges.append((pos1, pos2, weight))

        components_data.append((component, component_edges))
        print(f"连通分量 {len(components_data)}: {len(component)} 个节点, {len(component_edges)} 条边")

    print(f"\n总共发现 {len(components_dict)} 个连通分量，其中 {single_node_components} 个单节点分量被过滤")
    print(f"保留了 {len(components_data)} 个有效连通分量（至少包含2个节点）")

    return components_data


def find_connected_components_bfs_optimized(edge_weights: Dict[Tuple[int, int], float],
                                            all_positions: Set[int]) -> List[
    Tuple[Set[int], List[Tuple[int, int, float]]]]:
    """
    超级优化的BFS版本
    """
    # 创建节点索引映射
    nodes = sorted(all_positions)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # 使用邻接列表（基于数组）
    n = len(nodes)
    adjacency = [[] for _ in range(n)]

    # 构建邻接表和边字典（只保留非零权重边）
    edge_dict = {}
    for (pos1, pos2), weight in edge_weights.items():
        if abs(weight) < 1e-9:  # 再次确认过滤
            continue
        if pos1 in node_to_idx and pos2 in node_to_idx:
            idx1, idx2 = node_to_idx[pos1], node_to_idx[pos2]
            adjacency[idx1].append(idx2)
            adjacency[idx2].append(idx1)
            edge_dict[(min(pos1, pos2), max(pos1, pos2))] = weight

    # 去重邻接表
    for i in range(n):
        adjacency[i] = list(set(adjacency[i]))

    # 找到连通分量
    visited = [False] * n
    components_data = []
    single_node_components = 0

    for start_idx in range(n):
        if visited[start_idx]:
            continue

        # BFS
        component_indices = []
        queue = deque([start_idx])
        visited[start_idx] = True

        while queue:
            current_idx = queue.popleft()
            component_indices.append(current_idx)

            for neighbor_idx in adjacency[current_idx]:
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    queue.append(neighbor_idx)

        # 过滤单节点分量
        if len(component_indices) == 1:
            single_node_components += 1
            continue

        # 转换回节点值
        component = {nodes[idx] for idx in component_indices}

        # 收集边
        component_edges = []
        for idx1 in component_indices:
            node1 = nodes[idx1]
            for idx2 in adjacency[idx1]:
                node2 = nodes[idx2]
                if node1 < node2 and node2 in component:  # 避免重复边
                    edge_key = (node1, node2)
                    if edge_key in edge_dict:
                        component_edges.append((node1, node2, edge_dict[edge_key]))

        components_data.append((component, component_edges))
        print(f"连通分量 {len(components_data)}: {len(component)} 个节点, {len(component_edges)} 条边")

    print(f"\n总共发现 {n - sum(visited) + len(components_data) + single_node_components} 个连通分量，"
          f"其中 {single_node_components} 个单节点分量被过滤")
    print(f"保留了 {len(components_data)} 个有效连通分量（至少包含2个节点）")

    return components_data


def main_optimized(matrix_weight: List[Tuple[int, int, float]],
                   pos_to_col_idx: Dict[int, int],
                   return_networkx: bool = False,
                   use_union_find: bool = True,
                   min_component_size: int = 2,
                   hapcut2_1_line_count: int = 0,
                   a: float = 1.0) -> Union[List[Tuple[Set[int], List[Tuple[int, int, float]]]], List]:
    """
    优化版主函数：从内存数据构建图并找到连通分量
    """
    # 边权重计算
    compute_weights_start_time = time.time()

    edge_weights, col_idx_to_pos, all_positions = compute_edge_weights_optimized(
        matrix_weight,
        pos_to_col_idx,
        hapcut2_1_line_count=hapcut2_1_line_count,
        a=a
    )
    if edge_weights is None:
        print("计算边权重失败")
        return None

    compute_weights_end_time = time.time()
    compute_weights_time = compute_weights_end_time - compute_weights_start_time
    print(f"\n========== 边权重计算耗时: {compute_weights_time:.2f} 秒 ==========\n")

    # 查找连通分量
    find_components_start_time = time.time()

    if use_union_find:
        print("使用优化的并查集算法查找连通分量...")
        components_data = find_connected_components_ultra_optimized(edge_weights, all_positions)
    else:
        print("使用超级优化的BFS算法查找连通分量...")
        components_data = find_connected_components_bfs_optimized(edge_weights, all_positions)

    find_components_end_time = time.time()
    find_components_time = find_components_end_time - find_components_start_time
    print(f"========== 查找连通分量耗时: {find_components_time:.2f} 秒 ==========\n")

    # NetworkX格式转换（如果需要）
    if return_networkx:
        print("转换为NetworkX格式...")
        convert_start_time = time.time()

        components_graphs = []
        for nodes, edges in components_data:
            G = nx.Graph()
            G.add_nodes_from(nodes)
            # 添加边时再次过滤
            filtered_edges = [(u, v, w) for u, v, w in edges if abs(w) >= 1e-9]
            G.add_weighted_edges_from(filtered_edges)
            components_graphs.append(G)

        convert_time = time.time() - convert_start_time
        print(f"========== NetworkX转换耗时: {convert_time:.2f} 秒 ==========\n")

        total_time = compute_weights_time + find_components_time + convert_time
        print(f"========== 总耗时: {total_time:.2f} 秒 ==========")
        return components_graphs
    else:
        total_time = compute_weights_time + find_components_time
        print(f"========== 总耗时: {total_time:.2f} 秒 ==========")
        return components_data


# 向后兼容的函数名
def main(matrix_weight, pos_to_col_idx, return_networkx=False, use_union_find=False, min_component_size=2,
         hapcut2_1_line_count=0, a=1.0):
    """向后兼容的主函数，添加了行数参数和调整参数a"""
    return main_optimized(
        matrix_weight,
        pos_to_col_idx,
        return_networkx,
        use_union_find,
        min_component_size,
        hapcut2_1_line_count=hapcut2_1_line_count,
        a=a
    )


# 保留原始函数以确保兼容性
def compute_edge_weights(matrix_weight, pos_to_col_idx, hapcut2_1_line_count=0, a=1.0):
    """向后兼容的边权重计算函数，添加了行数参数和调整参数a"""
    return compute_edge_weights_optimized(matrix_weight, pos_to_col_idx, hapcut2_1_line_count, a)


def find_connected_components_union_find(edge_weights, all_positions):
    """向后兼容的并查集函数"""
    return find_connected_components_ultra_optimized(edge_weights, all_positions)


def find_connected_components_from_edges_optimized(edge_weights, all_positions):
    """向后兼容的BFS函数"""
    return find_connected_components_bfs_optimized(edge_weights, all_positions)


def process_chunk(chunk_data, col_indices, hapcut2_1_line_count=0, a=1.0):
    """向后兼容的数据块处理函数，添加了行数参数和调整参数a"""
    return process_chunk_optimized(chunk_data, col_indices, hapcut2_1_line_count, a)
