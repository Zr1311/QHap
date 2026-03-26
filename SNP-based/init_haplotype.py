import random
import pandas as pd

def read_sparse_matrix_from_memory(sparse_matrix_data, pos_to_col_idx):
    column_indices = set()
    for row_idx, col_idx, base in sparse_matrix_data:
        column_indices.add(col_idx)

    col_idx_to_pos = {col_idx: pos for pos, col_idx in pos_to_col_idx.items()}

    selected_positions = set()
    for col_idx in column_indices:
        if col_idx in col_idx_to_pos:
            selected_positions.add(col_idx_to_pos[col_idx])

    return selected_positions

def process_snp_data_from_memory(vcf_path, sparse_matrix_data, pos_to_col_idx):
    random.seed(42)

    selected_positions = read_sparse_matrix_from_memory(sparse_matrix_data, pos_to_col_idx)

    snp_dict = {}

    vcf_columns = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', 'default']
    vcf_df = pd.read_csv(vcf_path, sep='\t', comment='#', names=vcf_columns, encoding='utf-8')

    for _, row in vcf_df.iterrows():
        pos = int(row['POS'])

        if pos in selected_positions:
            ref_base = row['REF']
            alt_base = row['ALT']
            snp_dict[pos] = {'ref': ref_base, 'alt': alt_base}

    sorted_positions = sorted(snp_dict.keys())

    seq1 = []
    seq2 = []

    for pos in sorted_positions:
        bases = [snp_dict[pos]['ref'], snp_dict[pos]['alt']]
        random.shuffle(bases)
        seq1.append(bases[0])
        seq2.append(bases[1])

    init_haplotype = {
        'positions': sorted_positions,
        'sequence1': seq1,
        'sequence2': seq2
    }

    all_nodes = sorted_positions

    return init_haplotype, all_nodes

def generate_haplotype_data(vcf_file, matrix_data, pos_map):
    init_haplotype, all_nodes = process_snp_data_from_memory(
        vcf_file,
        matrix_data,
        pos_map
    )

    return init_haplotype, all_nodes