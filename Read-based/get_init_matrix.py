import pandas as pd

def read_vcf_file(vcf_path):
    vcf_columns = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', 'default']
    vcf_df = pd.read_csv(vcf_path, sep='\t', comment='#', names=vcf_columns, encoding='utf-8')
    sorted_pos = sorted(vcf_df['POS'].unique())
    pos_info_list = []
    for pos in sorted_pos:
        row = vcf_df[vcf_df['POS'] == pos].iloc[0]
        pos_info_list.append((row['REF'], row['ALT']))
    return sorted_pos, pos_info_list

def create_mapping(sorted_pos):
    pos_to_col_idx = {pos: idx for idx, pos in enumerate(sorted_pos)}
    col_idx_to_pos = {idx: pos for idx, pos in enumerate(sorted_pos)}
    return pos_to_col_idx, col_idx_to_pos

def process_fragment_file(fragment_path, pos_info_list, sorted_pos, pos_to_col_idx):
    fragment_to_idx = {}
    idx_to_fragment = {}
    sparse_matrix_data = []
    row_idx = 0

    with open(fragment_path, 'r', encoding='utf-8') as infile:
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
                    if base != '-':
                        sparse_matrix_data.append((row_idx, col_idx, base))
            row_idx += 1

    return {
        'sparse_matrix': sparse_matrix_data,
        'fragment_to_index': fragment_to_idx,
        'index_to_fragment': idx_to_fragment
    }

def build_snp_sparse_matrix(vcf_path, fragment_path):
    sorted_pos, pos_info_list = read_vcf_file(vcf_path)
    pos_to_col_idx, col_idx_to_pos = create_mapping(sorted_pos)
    processed = process_fragment_file(fragment_path, pos_info_list, sorted_pos, pos_to_col_idx)

    print("Sparse matrix constructed successfully, total valid loci processed:", len(processed['sparse_matrix']))
    print("Number of fragments:", len(processed['fragment_to_index']))
    print("Number of positions:", len(sorted_pos))

    result = {
        'sorted_positions': sorted_pos,
        'position_info': pos_info_list,
        'position_to_column': pos_to_col_idx,
        'column_to_position': col_idx_to_pos,
        **processed
    }
    return result