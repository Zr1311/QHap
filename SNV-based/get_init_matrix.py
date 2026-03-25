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


def process_phasing_file(phasing_path, pos_info_list, sorted_pos, pos_to_col_idx):
    """处理phasing输出文件，构建稀疏矩阵数据"""
    fragment_to_idx = {}
    idx_to_fragment = {}
    sparse_matrix_data = []
    sparse_correct_rate_data = []
    row_idx = 0

    with open(phasing_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            parts = line.strip().split()
            if not parts:
                continue

            fragment = parts[1]
            fragment_to_idx[fragment] = row_idx
            idx_to_fragment[row_idx] = fragment

            n = int(parts[0])
            blocks = parts[2: 2 + 2 * n]
            quality_str = parts[-1]

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

                    if base != '-':
                        sparse_matrix_data.append((row_idx, col_idx, base))

                    if j < len(quality_str):
                        quality = ord(quality_str[j]) - 33
                        error_rate = 10 ** (-quality / 10)
                        correct_rate = 1 - error_rate
                        if correct_rate > 0:
                            sparse_correct_rate_data.append((row_idx, col_idx, correct_rate))

            row_idx += 1

    return {
        'sparse_matrix': sparse_matrix_data,
        'sparse_correct_rate': sparse_correct_rate_data,
        'fragment_to_index': fragment_to_idx,
        'index_to_fragment': idx_to_fragment
    }


def build_snp_sparse_matrix(vcf_path, phasing_path):
    """构建单倍型稀疏矩阵并返回内存结果"""
    sorted_pos, pos_info_list = read_vcf_file(vcf_path)
    pos_to_col_idx, col_idx_to_pos = create_mapping(sorted_pos)
    processed = process_phasing_file(phasing_path, pos_info_list, sorted_pos, pos_to_col_idx)

    # 汇总所有结果
    result = {
        'sorted_positions': sorted_pos,
        'position_info': pos_info_list,
        'position_to_column': pos_to_col_idx,
        'column_to_position': col_idx_to_pos,
        **processed
    }
    return result