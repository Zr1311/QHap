def process_sparse_matrices(matrix, pos_map, init_haplotype, correct_rates):
    """
    处理稀疏矩阵数据，生成加权三元稀疏矩阵

    参数:
    matrix (list): 稀疏矩阵数据 [(row_idx, col_idx, base), ...]
    pos_map (dict): 位置到列索引的映射 {pos: col_idx}
    init_haplotype (dict): 初始单倍型数据 {'positions': [...], 'sequence1': [...], 'sequence2': [...]}
    correct_rates (list): 正确率数据 [(row_idx, col_idx, rate), ...]

    返回:
    list: 加权三元稀疏矩阵数据 [(row_idx, col_idx, weighted_value), ...]
    """

    # 创建列索引到位置的反向映射
    col_idx_to_pos = {col_idx: pos for pos, col_idx in pos_map.items()}

    # 创建位点到单倍型值的映射
    h1_map = {}  # 位点到单倍型1值的映射
    h2_map = {}  # 位点到单倍型2值的映射

    positions = init_haplotype['positions']
    sequence1 = init_haplotype['sequence1']
    sequence2 = init_haplotype['sequence2']

    for i, pos in enumerate(positions):
        h1_map[pos] = sequence1[i]
        h2_map[pos] = sequence2[i]

    # 将SNP矩阵数据转换为字典格式
    snp_data = {}  # {(row_idx, col_idx): base}
    for row_idx, col_idx, base in matrix:
        snp_data[(row_idx, col_idx)] = base

    # 将正确率数据转换为字典格式
    correct_rate_data = {}  # {(row_idx, col_idx): rate}
    for row_idx, col_idx, rate in correct_rates:
        correct_rate_data[(row_idx, col_idx)] = rate

    # 处理并生成三元稀疏矩阵（只保留非零值）
    ternary_matrix = {}  # {(row_idx, col_idx): value}
    for (row_idx, col_idx), base in snp_data.items():
        # 获取对应的位点
        if col_idx not in col_idx_to_pos:
            continue

        pos = col_idx_to_pos[col_idx]

        # 检查该位点是否在单倍型中
        if pos not in h1_map or pos not in h2_map:
            continue

        h1_base = h1_map[pos]
        h2_base = h2_map[pos]

        # 确定基因型
        if base == h1_base:
            ternary_matrix[(row_idx, col_idx)] = 1  # 与单倍型1匹配
        elif base == h2_base:
            ternary_matrix[(row_idx, col_idx)] = -1  # 与单倍型2匹配
        # 不保存值为0的项

    # 生成加权三元矩阵（只保留非零值）
    weighted_matrix_data = []
    for (row_idx, col_idx), value in ternary_matrix.items():
        rate = correct_rate_data.get((row_idx, col_idx), 1.0)  # 如果没有正确率信息，默认为1.0
        weighted_value = value * rate
        # 只保存非零值
        if weighted_value != 0:
            weighted_matrix_data.append((row_idx, col_idx, weighted_value))

    return weighted_matrix_data