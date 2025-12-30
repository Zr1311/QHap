import random

def process_fragment_file(fragment_output_file):
    read_names = set()
    try:
        with open(fragment_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    read_names.add(parts[1])
        return sorted(read_names)
    except Exception as e:
        print(f"处理片段文件时出错: {e}")
        return []

def generate_haplotype(snp_dict, sorted_positions):
    seq1, seq2 = [], []
    for pos in sorted_positions:
        bases = [snp_dict[pos]['ref'], snp_dict[pos]['alt']]
        random.shuffle(bases)
        seq1.append(bases[0])
        seq2.append(bases[1])
    return seq1, seq2

def generate_haplotype_data(vcf_file, matrix, pos_map, fragment_file=None):
    col_to_pos = {col: pos for pos, col in pos_map.items()}
    unique_cols = {col_idx for _, col_idx, _ in matrix}
    selected_positions = {col_to_pos[col] for col in unique_cols if col in col_to_pos}

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

    sorted_positions = sorted(snp_dict.keys())
    seq1, seq2 = generate_haplotype(snp_dict, sorted_positions)

    init_haplotype = {
        'positions': sorted_positions,
        'sequence1': seq1,
        'sequence2': seq2
    }

    if fragment_file:
        nodes = process_fragment_file(fragment_file)
        all_nodes = nodes if nodes else sorted_positions
    else:
        all_nodes = sorted_positions

    return init_haplotype, all_nodes