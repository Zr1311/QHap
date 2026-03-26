import random

def process_sparse_matrices(matrix, pos_map, init_haplotype):
    col_idx_to_pos = {col: pos for pos, col in pos_map.items()}

    positions = init_haplotype.get('positions', [])
    seq1 = init_haplotype.get('sequence1', [])
    seq2 = init_haplotype.get('sequence2', [])

    h1_map = {pos: base for pos, base in zip(positions, seq1)}
    h2_map = {pos: base for pos, base in zip(positions, seq2)}

    matrix_triple = []
    for row_idx, col_idx, base in matrix:
        pos = col_idx_to_pos.get(col_idx)
        if pos is None:
            continue
        if pos not in h1_map or pos not in h2_map:
            continue
        h1_base = h1_map[pos]
        h2_base = h2_map[pos]
        if base == h1_base:
            matrix_triple.append((row_idx, col_idx, 1))
        elif base == h2_base:
            matrix_triple.append((row_idx, col_idx, -1))

    return matrix_triple