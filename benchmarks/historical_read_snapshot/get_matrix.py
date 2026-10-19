import random


def process_sparse_matrices(matrix, pos_map, init_haplotype):
    """
    从内存中的稀疏矩阵和初始化单倍型数据生成三元稀疏矩阵

    参数:
    matrix (list of tuples): 内存中的稀疏矩阵列表，每项格式为 (row_idx, col_idx, base)
    pos_map (dict): 位置到列索引的映射 {pos: col_idx}
    init_haplotype (dict): 包含 'positions', 'sequence1', 'sequence2'

    返回:
    list: 三元稀疏矩阵列表 [(row_idx, col_idx, value), ...]
             value 为 1 表示匹配单倍型1, -1 表示匹配单倍型2
    """
    # 构建列索引到位点的映射
    col_idx_to_pos = {col: pos for pos, col in pos_map.items()}

    # 从 init_haplotype 中提取数据
    positions = init_haplotype.get('positions', [])
    seq1 = init_haplotype.get('sequence1', [])
    seq2 = init_haplotype.get('sequence2', [])

    # 构建位点到各单倍型碱基的映射
    h1_map = {pos: base for pos, base in zip(positions, seq1)}
    h2_map = {pos: base for pos, base in zip(positions, seq2)}

    matrix_triple = []  # 修改为列表存储
    for row_idx, col_idx, base in matrix:
        # 获取对应位点
        pos = col_idx_to_pos.get(col_idx)
        if pos is None:
            continue
        # 检查位点是否在单倍型中
        if pos not in h1_map or pos not in h2_map:
            continue
        h1_base = h1_map[pos]
        h2_base = h2_map[pos]
        # 判断匹配并赋值
        if base == h1_base:
            matrix_triple.append((row_idx, col_idx, 1))  # 添加元组到列表
        elif base == h2_base:
            matrix_triple.append((row_idx, col_idx, -1))  # 添加元组到列表
        # 其他情况视为 0，不保存

    return matrix_triple  # 返回列表

# 示例调用
# matrix_triple = process_sparse_matrices(
#     matrix=[(0,1,'A'), (0,2,'G'), ...],
#     pos_map={100:1, 200:2, ...},
#     init_haplotype={
#         'positions':[100,200,...],
#         'sequence1':['A','T',...],
#         'sequence2':['G','C',...]
#     }
# )