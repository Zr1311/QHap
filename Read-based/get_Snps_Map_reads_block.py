import numpy as np
from multiprocessing import Pool, cpu_count
import sys
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
from collections import defaultdict
from functools import partial
import networkx as nx
import time  # 导入时间模块用于计时

# 全局变量
snp_data = None  # 存储稀疏SNP数据
row_mapping = None  # 存储行索引到片段ID的映射


def create_sparse_matrix_from_triple(matrix_triple):
    """
    从matrix_triple列表创建稀疏矩阵
    matrix_triple: [(row_idx, col_idx, value), ...]
    """
    print(f"从triple列表创建稀疏矩阵...")
    try:
        if not matrix_triple:
            print("错误: matrix_triple为空")
            return None

        row_indices = []
        col_indices = []
        values = []

        for triple in matrix_triple:
            if len(triple) != 3:
                continue
            row_idx, col_idx, value = triple
            # 验证值是否为有效SNP值
            if value in (-1, 0, 1):
                row_indices.append(row_idx)
                col_indices.append(col_idx)
                values.append(value)

        if not row_indices:
            print("错误: 没有有效的数据")
            return None

        # 获取最大行索引和列索引
        max_row = max(row_indices)
        max_col = max(col_indices)

        # 创建COO格式的稀疏矩阵
        sparse_matrix = coo_matrix((values, (row_indices, col_indices)),
                                   shape=(max_row + 1, max_col + 1),
                                   dtype=np.int8)

        print(f"成功创建稀疏矩阵: {len(row_indices)}个非零元素，"
              f"{max_row + 1}行，{max_col + 1}列")

        return sparse_matrix

    except Exception as e:
        print(f"错误: 创建稀疏矩阵时出错: {e}")
        return None


def reverse_mapping(fragment_to_idx):
    """
    反转映射：从 fragment->idx 转换为 idx->fragment
    """
    print(f"反转映射关系...")
    try:
        idx_to_fragment = {}
        for fragment, idx in fragment_to_idx.items():
            idx_to_fragment[idx] = fragment

        print(f"成功创建反转映射: {len(idx_to_fragment)}个映射关系")
        return idx_to_fragment
    except Exception as e:
        print(f"错误: 反转映射时出错: {e}")
        return None


def compute_column_spans(sparse_matrix):
    """
    计算每列的跨度（第一个和最后一个非零元素的行索引）
    返回: 一个列表，其中每个元素是一个元组 (start_row, end_row)
    """
    print("计算每列的跨度...")
    # 转换为CSC格式以高效访问列
    csc_mat = sparse_matrix.tocsc()
    num_cols = csc_mat.shape[1]
    spans = []

    for j in range(num_cols):
        # 获取列中所有非零元素的行索引
        col_indices = csc_mat.getcol(j).nonzero()[0]
        if col_indices.size > 0:
            # 计算跨度：第一个非零元素和最后一个非零元素的行索引
            start_row = col_indices[0]
            end_row = col_indices[-1]
            spans.append((start_row, end_row))
        else:
            spans.append((None, None))  # 空列

    print(f"计算完成: {sum(1 for s in spans if s[0] is not None)} 个非空列")
    return spans


def initialize_globals_shared(data, mapping):
    """
    初始化全局变量，在每个子进程中调用
    """
    global snp_data, row_mapping
    snp_data = data
    row_mapping = mapping


def process_column(column_index, spans):
    """
    处理单个列，计算该列跨度内两两行之间的差异值
    """
    start_row, end_row = spans[column_index]
    if start_row is None:  # 空列
        return {}

    # 获取该列的所有非零行索引和对应的值
    col_data = snp_data.getcol(column_index).tocsr()
    non_zero_rows = col_data.nonzero()[0]
    values = col_data.data

    # 创建行索引到值的映射
    row_to_value = dict(zip(non_zero_rows, values))

    # 初始化差异计数
    diff_counts = defaultdict(int)

    # 遍历跨度内的所有行对
    for row1 in range(start_row, end_row + 1):
        val1 = row_to_value.get(row1)
        if val1 is None:  # 行1在该列没有非零值
            continue

        for row2 in range(row1 + 1, end_row + 1):
            val2 = row_to_value.get(row2)
            if val2 is None:  # 行2在该列没有非零值
                continue

            if val1 != val2:
                # 使用元组确保行对的顺序一致
                pair = (row1, row2) if row1 < row2 else (row2, row1)
                diff_counts[pair] += 1

    return diff_counts


def create_snp_graph(sparse_data, mapping, spans):
    """
    构建SNP图的边生成器
    """
    global snp_data, row_mapping
    snp_data = sparse_data.tocsc()  # 转换为CSC格式以高效列访问
    row_mapping = mapping

    num_cols = snp_data.shape[1]

    # 准备列索引列表进行并行处理
    column_indices = list(range(num_cols))

    pool_size = cpu_count()
    print(f"使用 {pool_size} 个进程进行计算")

    # 记录开始时间
    start_time = time.time()

    # 初始化进程池
    with Pool(processes=pool_size,
              initializer=initialize_globals_shared,
              initargs=(snp_data, row_mapping)) as pool:

        # 使用partial函数绑定spans参数
        process_func = partial(process_column, spans=spans)

        # 并行处理列
        results = pool.imap_unordered(process_func, column_indices, chunksize=100)

        # 合并所有列的结果
        edge_counts = defaultdict(int)
        for res in tqdm(results, total=num_cols, desc="处理列", unit="列"):
            for pair, count in res.items():
                edge_counts[pair] += count

    # 生成边
    edges = []
    for (row1, row2), weight in edge_counts.items():
        if weight > 0:
            # 使用行映射获取片段ID
            node1 = row_mapping.get(row1, str(row1))
            node2 = row_mapping.get(row2, str(row2))
            edges.append((node1, node2, weight))

    # 记录结束时间并计算耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"构建SNP图耗时: {elapsed_time:.2f} 秒")

    return edges


def build_graph(edges, all_nodes):
    """
    从边列表构建NetworkX图
    """
    G = nx.Graph()

    # 添加所有节点
    for node in all_nodes:
        G.add_node(str(node))  # 确保节点是字符串格式

    # 添加边
    for node1, node2, weight in edges:
        G.add_edge(str(node1), str(node2), weight=weight)

    return G


def find_connected_components(G):
    """找到所有连通分量"""
    return list(nx.connected_components(G))


def create_component_graphs(G, components, edges):
    """
    为每个连通分量创建子图（跳过单节点分量）
    返回: 一个字典，键是分量索引，值是包含边的列表
    """
    # 记录开始时间
    start_time = time.time()

    # 过滤掉单节点分量
    multi_node_components = [(i, comp) for i, comp in enumerate(components) if len(comp) > 1]
    single_node_count = len(components) - len(multi_node_components)

    if single_node_count > 0:
        print(f"跳过 {single_node_count} 个单节点分量")

    # 创建节点到组件的映射（只包含多节点分量）
    node_to_component = {}
    new_component_index = {}  # 原始索引到新索引的映射

    for new_idx, (orig_idx, component) in enumerate(multi_node_components):
        new_component_index[orig_idx] = new_idx
        for node in component:
            node_to_component[node] = new_idx

    # 初始化结果字典（只包含多节点分量）
    component_graphs = {}
    for new_idx, (orig_idx, component) in enumerate(multi_node_components):
        component_graphs[new_idx] = {
            'nodes': list(component),
            'edges': []
        }

    # 分配边到对应的组件
    for node1, node2, weight in edges:
        node1_str = str(node1)
        node2_str = str(node2)

        comp_idx = node_to_component.get(node1_str)

        # 只处理属于多节点分量的边
        if comp_idx is not None:
            component_graphs[comp_idx]['edges'].append((node1, node2, weight))

    # 记录结束时间并计算耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"创建分量子图耗时: {elapsed_time:.2f} 秒")

    return component_graphs


def print_statistics(components, component_graphs):
    """打印统计信息"""
    print(f"\n===== 统计信息 =====")

    # 计算单节点分量数量
    single_node_components = sum(1 for comp in components if len(comp) == 1)
    multi_node_components = len(components) - single_node_components

    print(f"原始分量总数: {len(components)}")
    print(f"- 多节点分量: {multi_node_components}")
    print(f"- 单节点分量: {single_node_components} (已跳过)")
    print(f"保存的分块数量: {len(component_graphs)}")

    total_nodes = sum(len(comp_info['nodes']) for comp_info in component_graphs.values())
    print(f"保存的分块中的节点总数: {total_nodes}")

    # 打印每个保存的分块的信息
    print("\n保存的分块详情:")
    for i, comp_info in component_graphs.items():
        print(f"Component {i}: {len(comp_info['nodes'])} nodes, {len(comp_info['edges'])} edges")


def main(matrix_triple, fragment_to_idx, all_nodes):
    """
    主函数：构建SNP图并进行分块（跳过单节点分量）

    参数:
    - matrix_triple: [(row_idx, col_idx, value), ...] 的列表
    - fragment_to_idx: {fragment: row_idx} 的字典
    - all_nodes: 所有节点的列表

    返回:
    - component_graphs: 字典，键是分量索引，值是包含'nodes'和'edges'的字典
                       （不包含单节点分量）
    """
    print("===== 第一步：创建稀疏矩阵 =====")
    sparse_data = create_sparse_matrix_from_triple(matrix_triple)

    if sparse_data is None or sparse_data.size == 0:
        print("SNP数据创建失败或为空")
        return {}

    print("===== 第二步：处理映射关系 =====")
    # 反转映射关系：从 fragment->idx 转换为 idx->fragment
    idx_to_fragment = reverse_mapping(fragment_to_idx)

    if idx_to_fragment is None:
        print("映射关系处理失败")
        return {}

    # 节点数等于SNP数据的行数
    num_nodes = sparse_data.shape[0]
    print(f"图的节点数: {num_nodes}")

    print("===== 第三步：计算列跨度 =====")
    spans = compute_column_spans(sparse_data)

    print("===== 第四步：构建SNP图 =====")
    edges = create_snp_graph(sparse_data, idx_to_fragment, spans)
    print(f"构建的图包含 {len(edges)} 条边")

    print("===== 第五步：构建NetworkX图 =====")
    G = build_graph(edges, all_nodes)
    print(f"NetworkX图包含 {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")

    print("===== 第六步：查找连通分量 =====")
    components = find_connected_components(G)
    print(f"找到 {len(components)} 个连通分量")

    print("===== 第七步：创建分量子图（跳过单节点分量）=====")
    component_graphs = create_component_graphs(G, components, edges)

    # 打印统计信息
    print_statistics(components, component_graphs)

    print("\n===== 处理完成 =====")
    return component_graphs


# 为了兼容性，保留原来的接口
def get_Snps_Map_reads_block(matrix_triple, row_mapping, all_nodes):
    """
    兼容性接口
    """
    return main(matrix_triple, row_mapping, all_nodes)