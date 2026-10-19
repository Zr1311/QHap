import random
import pandas as pd


def read_sparse_matrix_from_memory(sparse_matrix_data, pos_to_col_idx):
    """
    从内存中的稀疏矩阵数据获取有效的位点位置

    参数:
    sparse_matrix_data (list): 稀疏矩阵数据 [(row_idx, col_idx, base), ...]
    pos_to_col_idx (dict): 位置到列索引的映射

    返回:
    set: 包含有效位点位置的集合
    """
    # 从稀疏矩阵数据中提取所有唯一的列索引
    column_indices = set()
    for row_idx, col_idx, base in sparse_matrix_data:
        column_indices.add(col_idx)

    # 根据列索引获取对应的位置
    # 创建列索引到位置的反向映射
    col_idx_to_pos = {col_idx: pos for pos, col_idx in pos_to_col_idx.items()}

    selected_positions = set()
    for col_idx in column_indices:
        if col_idx in col_idx_to_pos:
            selected_positions.add(col_idx_to_pos[col_idx])

    return selected_positions


def process_snp_data_from_memory(vcf_path, sparse_matrix_data, pos_to_col_idx):
    """
    从内存中处理SNP数据，生成单倍型数据（固定随机种子为42）

    参数:
    vcf_path (str): VCF文件路径
    sparse_matrix_data (list): 稀疏矩阵数据
    pos_to_col_idx (dict): 位置到列索引的映射

    返回:
    tuple: (init_haplotype, all_nodes)
    """
    # 固定随机种子为42，确保结果完全可复现
    random.seed(42)

    # 使用稀疏矩阵数据获取有效位点
    selected_positions = read_sparse_matrix_from_memory(sparse_matrix_data, pos_to_col_idx)

    snp_dict = {}

    # 读取VCF文件
    vcf_columns = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', 'default']
    vcf_df = pd.read_csv(vcf_path, sep='\t', comment='#', names=vcf_columns, encoding='utf-8')

    # 处理VCF数据
    for _, row in vcf_df.iterrows():
        pos = int(row['POS'])

        # 只处理矩阵中出现的位点
        if pos in selected_positions:
            ref_base = row['REF']
            alt_base = row['ALT']
            snp_dict[pos] = {'ref': ref_base, 'alt': alt_base}

    # 排序位置
    sorted_positions = sorted(snp_dict.keys())

    # 生成两条序列
    seq1 = []
    seq2 = []

    for pos in sorted_positions:
        bases = [snp_dict[pos]['ref'], snp_dict[pos]['alt']]
        random.shuffle(bases)  # 使用固定种子的随机打乱，结果可复现
        seq1.append(bases[0])  # 第一条序列
        seq2.append(bases[1])  # 第二条序列

    # 构建单倍型数据结构
    init_haplotype = {
        'positions': sorted_positions,
        'sequence1': seq1,
        'sequence2': seq2
    }

    # 返回位点信息
    all_nodes = sorted_positions

    return init_haplotype, all_nodes


def generate_haplotype_data(vcf_file, matrix_data, pos_map):
    """
    生成单倍型数据的主函数（固定随机种子为42）

    参数:
    vcf_file (str): VCF文件路径
    matrix_data (list): 稀疏矩阵数据 [(row_idx, col_idx, base), ...]
    pos_map (dict): 位置到列索引的映射

    返回:
    tuple: (init_haplotype, all_nodes)
        - init_haplotype: 包含位点、序列1、序列2的字典
        - all_nodes: 位点列表
    """
    init_haplotype, all_nodes = process_snp_data_from_memory(
        vcf_file,
        matrix_data,
        pos_map
    )

    return init_haplotype, all_nodes

# 使用示例：
# init_haplotype, all_nodes = generate_haplotype_data(vcf_file, matrix, pos_map)
