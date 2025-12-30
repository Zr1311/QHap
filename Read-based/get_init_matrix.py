import pandas as pd


def read_vcf_file(vcf_path):
    """读取VCF文件并提取SNP信息"""
    vcf_columns = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', 'default']
    vcf_df = pd.read_csv(vcf_path, sep='\t', comment='#', names=vcf_columns, encoding='utf-8')

    # 按POS排序并获取对应的REF和ALT
    sorted_pos = sorted(vcf_df['POS'].unique())
    pos_info_list = []
    for pos in sorted_pos:
        row = vcf_df[vcf_df['POS'] == pos].iloc[0]
        pos_info_list.append((row['REF'], row['ALT']))

    return sorted_pos, pos_info_list


def create_mapping(sorted_pos):
    """创建位置与索引的映射关系"""
    pos_to_col_idx = {pos: idx for idx, pos in enumerate(sorted_pos)}
    col_idx_to_pos = {idx: pos for idx, pos in enumerate(sorted_pos)}
    return pos_to_col_idx, col_idx_to_pos


def process_hapcut_file(hapcut_path, pos_info_list, sorted_pos, pos_to_col_idx):
    """处理Hapcut2输出文件，构建稀疏矩阵数据"""
    fragment_to_idx = {}
    idx_to_fragment = {}
    sparse_matrix_data = []
    row_idx = 0

    with open(hapcut_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            parts = line.strip().split()
            if not parts:
                continue

            fragment = parts[1]
            fragment_to_idx[fragment] = row_idx
            idx_to_fragment[row_idx] = fragment

            n = int(parts[0])
            blocks = parts[2: 2 + 2 * n]

            for i in range(n):
                k = int(blocks[2 * i])
                s = blocks[2 * i + 1]

                for j, c in enumerate(s):
                    pos_index = k + j
                    if pos_index > len(pos_info_list):
                        continue

                    ref, alt = pos_info_list[pos_index - 1]
                    base = ref if c == '0' else alt
                    pos = sorted_pos[pos_index - 1]
                    col_idx = pos_to_col_idx[pos]

                    # 只存储非缺失值
                    if base != '-':
                        sparse_matrix_data.append((row_idx, col_idx, base))

            row_idx += 1

    return {
        'sparse_matrix': sparse_matrix_data,
        'fragment_to_index': fragment_to_idx,
        'index_to_fragment': idx_to_fragment
    }


def build_snp_sparse_matrix(vcf_path, hapcut_path):
    """
    构建SNP稀疏矩阵并返回内存结果

    参数:
    vcf_path (str): VCF文件路径
    hapcut_path (str): Hapcut2输出文件路径

    返回:
    dict: 包含稀疏矩阵和映射信息的字典
    """
    # 读取VCF文件信息
    sorted_pos, pos_info_list = read_vcf_file(vcf_path)

    # 创建位置映射
    pos_to_col_idx, col_idx_to_pos = create_mapping(sorted_pos)

    # 处理hapcut文件
    processed = process_hapcut_file(hapcut_path, pos_info_list, sorted_pos, pos_to_col_idx)

    print(f"稀疏矩阵构建完成，共处理 {len(processed['sparse_matrix'])} 个有效位点")
    print(f"片段数量: {len(processed['fragment_to_index'])}")
    print(f"位置数量: {len(sorted_pos)}")

    # 汇总所有结果，返回格式与参考代码保持一致
    result = {
        'sorted_positions': sorted_pos,
        'position_info': pos_info_list,
        'position_to_column': pos_to_col_idx,
        'column_to_position': col_idx_to_pos,
        **processed
    }

    return result
