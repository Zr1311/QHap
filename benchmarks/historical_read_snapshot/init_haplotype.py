import random

def process_hapcut2_file(hapcut2_output_file):
    """
    从HapCUT2输出文件中提取第二列的值

    参数:
    hapcut2_output_file (str): HapCUT2输出文件路径

    返回:
    list: 排序后的读取名称列表
    """
    read_names = set()
    try:
        with open(hapcut2_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    read_names.add(parts[1])
        return sorted(read_names)
    except Exception as e:
        print(f"处理HapCUT2文件时出错: {e}")
        return []


def generate_haplotype(snp_dict, sorted_positions):
    """
    从SNP字典生成两条随机单倍型序列

    参数:
    snp_dict (dict): SNP字典，键为位置，值为参考和替代碱基
    sorted_positions (list): 排序后的位置列表

    返回:
    tuple: 两条单倍型序列列表 (seq1, seq2)
    """
    seq1, seq2 = [], []
    for pos in sorted_positions:
        bases = [snp_dict[pos]['ref'], snp_dict[pos]['alt']]
        random.shuffle(bases)
        seq1.append(bases[0])
        seq2.append(bases[1])
    return seq1, seq2


def generate_haplotype_data(vcf_file, matrix, pos_map, extractHAIRS_file=None):
    """
    从VCF文件及内存中的稀疏矩阵(matrix)和位置映射(pos_map)生成单倍型数据

    参数:
    vcf_file (str): 输入VCF文件路径
    matrix (list of tuples): 稀疏矩阵列表，每项格式为 (row_idx, col_idx, base)
    pos_map (dict): 位置到列索引的映射 {pos: col_idx}
    extractHAIRS_file (str, optional): HapCUT2输出文件路径

    返回:
    tuple:
        init_haplotype (dict): 包含 'positions', 'sequence1', 'sequence2'
        all_nodes (list): 节点列表（若提供HapCUT2输出，则为其结果，否则为positions）
    """
    # 反转 pos_map，构建列索引到位置的映射
    col_to_pos = {col: pos for pos, col in pos_map.items()}

    # 提取矩阵中所有唯一的列索引
    unique_cols = {col_idx for _, col_idx, _ in matrix}
    # 根据列索引获取对应的位置
    selected_positions = {col_to_pos[col] for col in unique_cols if col in col_to_pos}

    # 从VCF文件中读取SNP信息
    snp_dict = {}
    with open(vcf_file, 'r', encoding='utf-8') as vf:
        for line in vf:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            _, pos_str, _, ref_base, alt_base = parts[:5]
            pos = int(pos_str)
            if pos in selected_positions:
                snp_dict[pos] = {'ref': ref_base, 'alt': alt_base}

    # 排序位点
    sorted_positions = sorted(snp_dict.keys())

    # 生成单倍型序列
    seq1, seq2 = generate_haplotype(snp_dict, sorted_positions)

    # 构建 init_haplotype 字典
    init_haplotype = {
        'positions': sorted_positions,
        'sequence1': seq1,
        'sequence2': seq2
    }

    # 生成 all_nodes
    if extractHAIRS_file:
        nodes = process_hapcut2_file(extractHAIRS_file)
        all_nodes = nodes if nodes else sorted_positions
    else:
        all_nodes = sorted_positions

    return init_haplotype, all_nodes

# 示例调用
# init_haplotype, all_nodes = generate_haplotype_data(
#     vcf_file="chr6_snp_PASS_filter.vcf",
#     matrix=[(r, c, b) for r, c, b in ...],
#     pos_map={pos: idx for idx, pos in enumerate(sorted_positions)},
#     extractHAIRS_file="hapcut2_output_HIFI.txt"
# )
