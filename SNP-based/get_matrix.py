def process_sparse_matrices(matrix, pos_map, init_haplotype, correct_rates):
    col_idx_to_pos = {col_idx: pos for pos, col_idx in pos_map.items()}

    h1_map = {}
    h2_map = {}

    positions = init_haplotype['positions']
    sequence1 = init_haplotype['sequence1']
    sequence2 = init_haplotype['sequence2']

    for i, pos in enumerate(positions):
        h1_map[pos] = sequence1[i]
        h2_map[pos] = sequence2[i]

    snp_data = {}
    for row_idx, col_idx, base in matrix:
        snp_data[(row_idx, col_idx)] = base

    correct_rate_data = {}
    for row_idx, col_idx, rate in correct_rates:
        correct_rate_data[(row_idx, col_idx)] = rate

    ternary_matrix = {}
    for (row_idx, col_idx), base in snp_data.items():
        if col_idx not in col_idx_to_pos:
            continue

        pos = col_idx_to_pos[col_idx]

        if pos not in h1_map or pos not in h2_map:
            continue

        h1_base = h1_map[pos]
        h2_base = h2_map[pos]

        if base == h1_base:
            ternary_matrix[(row_idx, col_idx)] = 1
        elif base == h2_base:
            ternary_matrix[(row_idx, col_idx)] = -1

    weighted_matrix_data = []
    for (row_idx, col_idx), value in ternary_matrix.items():
        rate = correct_rate_data.get((row_idx, col_idx), 1.0)
        weighted_value = value * rate
        if weighted_value != 0:
            weighted_matrix_data.append((row_idx, col_idx, weighted_value))

    return weighted_matrix_data