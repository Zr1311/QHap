#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy import sparse
import tabu_SB_gpu
import time
from collections import defaultdict

if not hasattr(np, 'int'):
    np.int = int

import tabu_SB_gpu
from scipy.sparse import coo_matrix
import torch


def read_vcf_file(vcf_file_path):
    headers = []
    data_lines = []

    with open(vcf_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('##'):
                headers.append(line)
            elif line.startswith('#'):
                headers.append(line)
            elif line:
                data_lines.append(line)

    return headers, data_lines


def parse_vcf_data_line(line):
    parts = line.split('\t')
    if len(parts) < 10:
        return None

    return {
        'chrom': parts[0],
        'pos': int(parts[1]),
        'id': parts[2],
        'ref': parts[3],
        'alt': parts[4],
        'qual': parts[5],
        'filter': parts[6],
        'info': parts[7],
        'format': parts[8],
        'sample': parts[9] if len(parts) > 9 else '',
        'other_samples': parts[10:] if len(parts) > 10 else []
    }


def process_single_block(component_graph, init_haplotype_data, sparse_matrix,
                         pos_map, fragment_map, block_number,
                         haplotype_output_dir):
    nodes = component_graph['nodes']
    edges = component_graph['edges']

    if not edges:
        return None

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n_v = len(nodes)

    row_indices = []
    col_indices = []
    data = []

    for edge in edges:
        if isinstance(edge, (list, tuple)) and len(edge) >= 3:
            node1, node2, weight = edge[0], edge[1], edge[2]
        else:
            continue

        if node1 in node_to_idx and node2 in node_to_idx:
            idx1 = node_to_idx[node1]
            idx2 = node_to_idx[node2]
            row_indices.extend([idx1, idx2])
            col_indices.extend([idx2, idx1])
            data.extend([-weight, -weight])

    if not data:
        return None

    device = 'cuda'
    G = torch.sparse_coo_tensor(
        torch.tensor([row_indices, col_indices], dtype=torch.long),
        torch.tensor(data, dtype=torch.float),
        size=(n_v, n_v)
    ).to(device)

    row_sum = torch.sparse.sum(G, dim=1).to_dense()
    xi = 1 / torch.abs(row_sum).max()

    G = G.to_sparse_csr()
    s = tabu_SB_gpu.SB(G, n_iter=10000, xi=xi, dt=1., batch_size=200, device=device)
    s.update_b()

    best_sample = torch.sign(s.x).clone()
    energy = -0.5 * torch.sum(G @ best_sample * best_sample, dim=0)
    cut = -0.5 * energy - 0.25 * G.sum()
    ind = cut.argmax()
    best_classification = best_sample[:, ind].cpu().numpy()

    max_cut_value = cut[ind].item()
    print(f"Block {block_number} max cut value: {max_cut_value}")

    subset1_fragments = [node for node, idx in node_to_idx.items() if best_classification[idx] == 1]
    subset2_fragments = [node for node, idx in node_to_idx.items() if best_classification[idx] == -1]

    return {'subset1': subset1_fragments, 'subset2': subset2_fragments}


def find_non_zero_column_range_csr_optimized(sparse_matrix, row_indices):
    if not isinstance(sparse_matrix, csr_matrix):
        sparse_matrix = sparse_matrix.tocsr()

    non_zero_cols = []
    for idx in row_indices:
        start = sparse_matrix.indptr[idx]
        end = sparse_matrix.indptr[idx + 1]
        non_zero_cols.extend(sparse_matrix.indices[start:end])

    if not non_zero_cols:
        return None

    return min(non_zero_cols), max(non_zero_cols)


def process_haplotype_for_classification(subset1_fragments, subset2_fragments,
                                         init_haplotype_data, sparse_matrix,
                                         pos_map, fragment_map, block_number,
                                         haplotype_output_dir):
    if 'positions' in init_haplotype_data:
        init_positions = init_haplotype_data['positions']
        init_seq1 = init_haplotype_data['sequence1']
        init_seq2 = init_haplotype_data['sequence2']
        init_cols = [str(p) for p in init_positions]
        init_line2 = list(init_seq1)
        init_line3 = list(init_seq2)
    else:
        raise ValueError("Unsupported init_haplotype_data format")

    init_line2_arr = np.array(init_line2, dtype='U1')
    init_line3_arr = np.array(init_line3, dtype='U1')
    init_cols_array = np.array(init_cols)

    subset1_indices = [fragment_map[f] for f in subset1_fragments if f in fragment_map]
    subset2_indices = [fragment_map[f] for f in subset2_fragments if f in fragment_map]

    if not subset1_indices or not subset2_indices:
        return None

    if not isinstance(sparse_matrix, csr_matrix):
        sparse_matrix = sparse_matrix.tocsr()

    all_indices = subset1_indices + subset2_indices
    col_range = find_non_zero_column_range_csr_optimized(sparse_matrix, all_indices)

    if col_range is None:
        return None

    first_non_zero_col, last_non_zero_col = col_range
    print(f"Block {block_number}: Valid column range [{first_non_zero_col}, {last_non_zero_col}]")

    valid_col_range = np.arange(first_non_zero_col, last_non_zero_col + 1)

    subset1_matrix_valid = sparse_matrix[subset1_indices][:, valid_col_range]
    subset2_matrix_valid = sparse_matrix[subset2_indices][:, valid_col_range]

    cnt1_valid = (subset1_matrix_valid == 1).sum(axis=0).A1
    cnt_neg1_valid = (subset1_matrix_valid == -1).sum(axis=0).A1
    cnt2_valid = (subset2_matrix_valid == 1).sum(axis=0).A1
    cnt_neg2_valid = (subset2_matrix_valid == -1).sum(axis=0).A1

    cnt1 = np.zeros(sparse_matrix.shape[1], dtype=int)
    cnt_neg1 = np.zeros(sparse_matrix.shape[1], dtype=int)
    cnt2 = np.zeros(sparse_matrix.shape[1], dtype=int)
    cnt_neg2 = np.zeros(sparse_matrix.shape[1], dtype=int)

    cnt1[valid_col_range] = cnt1_valid
    cnt_neg1[valid_col_range] = cnt_neg1_valid
    cnt2[valid_col_range] = cnt2_valid
    cnt_neg2[valid_col_range] = cnt_neg2_valid

    col_idx_to_pos = {idx: pos for pos, idx in pos_map.items()}
    valid_col_indices = []
    valid_init_indices = []
    for col_idx in valid_col_range:
        if col_idx in col_idx_to_pos:
            pos = str(col_idx_to_pos[col_idx])
            if pos in init_cols:
                valid_col_indices.append(col_idx)
                valid_init_indices.append(init_cols.index(pos))

    if not valid_col_indices:
        return None

    valid_col_indices_arr = np.array(valid_col_indices)
    valid_init_indices_arr = np.array(valid_init_indices)

    mask_remove_sub = ((cnt1[valid_col_indices_arr] + cnt_neg1[valid_col_indices_arr]) == 0) & \
                      ((cnt2[valid_col_indices_arr] + cnt_neg2[valid_col_indices_arr]) == 0)

    non_zero_condition = (cnt1[valid_col_indices_arr] > 0) & (cnt_neg1[valid_col_indices_arr] > 0) & \
                         (cnt2[valid_col_indices_arr] > 0) & (cnt_neg2[valid_col_indices_arr] > 0)
    same_count_mask = (cnt1[valid_col_indices_arr] == cnt_neg1[valid_col_indices_arr]) & \
                      (cnt2[valid_col_indices_arr] == cnt_neg2[valid_col_indices_arr]) & non_zero_condition
    mask_remove_sub |= same_count_mask
    keep_mask_sub = ~mask_remove_sub

    diff1 = np.abs(cnt1[valid_col_indices_arr] - cnt_neg1[valid_col_indices_arr])
    diff2 = np.abs(cnt2[valid_col_indices_arr] - cnt_neg2[valid_col_indices_arr])
    both_1_more = (cnt1[valid_col_indices_arr] > cnt_neg1[valid_col_indices_arr]) & \
                  (cnt2[valid_col_indices_arr] > cnt_neg2[valid_col_indices_arr])
    both_neg1_more = (cnt1[valid_col_indices_arr] < cnt_neg1[valid_col_indices_arr]) & \
                     (cnt2[valid_col_indices_arr] < cnt_neg2[valid_col_indices_arr])

    haplo2_selection = np.where(
        both_1_more,
        np.where(diff1 > diff2, init_line2_arr[valid_init_indices_arr], init_line3_arr[valid_init_indices_arr]),
        np.where(
            both_neg1_more,
            np.where(diff1 > diff2, init_line3_arr[valid_init_indices_arr], init_line2_arr[valid_init_indices_arr]),
            np.where(
                cnt1[valid_col_indices_arr] != cnt_neg1[valid_col_indices_arr],
                np.where(cnt1[valid_col_indices_arr] > cnt_neg1[valid_col_indices_arr],
                         init_line2_arr[valid_init_indices_arr],
                         init_line3_arr[valid_init_indices_arr]),
                np.where(cnt2[valid_col_indices_arr] > cnt_neg2[valid_col_indices_arr],
                         init_line3_arr[valid_init_indices_arr],
                         init_line2_arr[valid_init_indices_arr])
            )
        )
    )

    haplo3_selection = np.where(
        both_1_more,
        np.where(diff1 > diff2, init_line3_arr[valid_init_indices_arr], init_line2_arr[valid_init_indices_arr]),
        np.where(
            both_neg1_more,
            np.where(diff1 > diff2, init_line2_arr[valid_init_indices_arr], init_line3_arr[valid_init_indices_arr]),
            np.where(
                cnt2[valid_col_indices_arr] != cnt_neg2[valid_col_indices_arr],
                np.where(cnt2[valid_col_indices_arr] > cnt_neg2[valid_col_indices_arr],
                         init_line2_arr[valid_init_indices_arr],
                         init_line3_arr[valid_init_indices_arr]),
                np.where(cnt1[valid_col_indices_arr] > cnt_neg1[valid_col_indices_arr],
                         init_line3_arr[valid_init_indices_arr],
                         init_line2_arr[valid_init_indices_arr])
            )
        )
    )

    filtered_cols = init_cols_array[valid_init_indices_arr][keep_mask_sub]
    haplo2_filtered = haplo2_selection[keep_mask_sub]
    haplo3_filtered = haplo3_selection[keep_mask_sub]

    same_base_mask = haplo2_filtered == haplo3_filtered
    final_cols = filtered_cols[~same_base_mask]
    final_haplo2 = haplo2_filtered[~same_base_mask]
    final_haplo3 = haplo3_filtered[~same_base_mask]

    return {
        'positions': [int(p) for p in final_cols],
        'haplotype1': list(final_haplo2),
        'haplotype2': list(final_haplo3)
    }


def write_vcf_output(vcf_headers, vcf_data_lines, haplotype_results, output_file):
    pos_to_block = {}
    for block_num, result in haplotype_results.items():
        if result and 'positions' in result:
            positions = result['positions']
            if positions:
                ps_value = positions[0]
                for i, pos in enumerate(positions):
                    pos_to_block[pos] = {
                        'block': block_num,
                        'ps': ps_value,
                        'hap1': result['haplotype1'][i],
                        'hap2': result['haplotype2'][i]
                    }

    with open(output_file, 'w') as f:
        ps_format_exists = any('##FORMAT=<ID=PS' in h for h in vcf_headers)

        last_format_idx = -1
        for i, header in enumerate(vcf_headers):
            if header.startswith('##FORMAT='):
                last_format_idx = i

        for i, header in enumerate(vcf_headers):
            f.write(header + '\n')

            if i == last_format_idx and not ps_format_exists:
                f.write('##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">\n')
                ps_format_exists = True

        for line in vcf_data_lines:
            data = parse_vcf_data_line(line)
            if not data:
                f.write(line + '\n')
                continue

            pos = data['pos']

            if pos in pos_to_block:
                block_info = pos_to_block[pos]
                ps_value = block_info['ps']
                hap1_base = block_info['hap1']
                hap2_base = block_info['hap2']

                format_fields = data['format'].split(':')
                sample_values = data['sample'].split(':')

                ref = data['ref']
                alt = data['alt']

                if hap1_base == ref and hap2_base == alt:
                    phased_gt = '0|1'
                elif hap1_base == alt and hap2_base == ref:
                    phased_gt = '1|0'
                elif hap1_base == ref and hap2_base == ref:
                    phased_gt = '0|0'
                elif hap1_base == alt and hap2_base == alt:
                    phased_gt = '1|1'
                else:
                    original_gt = sample_values[0] if sample_values else '0/1'
                    phased_gt = original_gt.replace('/', '|')

                if sample_values:
                    sample_values[0] = phased_gt

                if 'PS' not in format_fields:
                    new_format = ':'.join(format_fields) + ':PS'
                    new_sample = ':'.join(sample_values) + ':' + str(ps_value)
                else:
                    new_format = ':'.join(format_fields)
                    ps_index = format_fields.index('PS')
                    if ps_index < len(sample_values):
                        sample_values[ps_index] = str(ps_value)
                    else:
                        sample_values.append(str(ps_value))
                    new_sample = ':'.join(sample_values)

                parts = [data['chrom'], str(data['pos']), data['id'], data['ref'],
                         data['alt'], data['qual'], data['filter'], data['info'],
                         new_format, new_sample]
                parts.extend(data['other_samples'])
                f.write('\t'.join(parts) + '\n')
            else:
                f.write(line + '\n')

    print(f"VCF results written to: {output_file}")


def process_blocks(components_graphs, init_haplotype, matrix_triple,
                   pos_map, fragment_map, haplotype_output_dir, vcf_file=None):
    total_start_time = time.time()

    if not os.path.exists(haplotype_output_dir):
        os.makedirs(haplotype_output_dir)

    print(f"components_graphs type: {type(components_graphs)}")
    print(f"components_graphs length: {len(components_graphs)}")

    print("\nBuilding sparse matrix...")
    if matrix_triple:
        rows, cols, values = zip(*matrix_triple)
        max_row = max(fragment_map.values()) + 1
        max_col = max(pos_map.values()) + 1
        sparse_matrix = csr_matrix((values, (rows, cols)), shape=(max_row, max_col))
    else:
        print("Warning: matrix_triple is empty")
        return

    print(f"Sparse matrix shape: {sparse_matrix.shape}")
    print(f"Processing {len(components_graphs)} blocks\n")

    max_cut_total_time = 0.0

    classification_results = []
    for i in range(len(components_graphs)):
        block_number = i + 1

        print(f"Classifying block {block_number}/{len(components_graphs)}")
        component_graph = components_graphs[i]

        if not isinstance(component_graph, dict):
            continue

        if 'nodes' not in component_graph or 'edges' not in component_graph:
            continue

        try:
            block_start_time = time.time()
            subsets = process_single_block(
                component_graph=component_graph,
                init_haplotype_data=init_haplotype,
                sparse_matrix=sparse_matrix,
                pos_map=pos_map,
                fragment_map=fragment_map,
                block_number=block_number,
                haplotype_output_dir=haplotype_output_dir
            )
            block_process_time = time.time() - block_start_time
            max_cut_total_time += block_process_time
            if subsets:
                classification_results.append({
                    'block_number': block_number,
                    'subset1': subsets['subset1'],
                    'subset2': subsets['subset2']
                })
            print(f"    Block {block_number} classification completed\n")
        except Exception as e:
            continue

    print(f"Total max cut time: {max_cut_total_time:.2f}s")

    print("Generating haplotypes...\n")
    haplotype_results = {}
    haplotype_process_total_time = 0.0

    for result in classification_results:
        block_number = result['block_number']
        try:
            start_time = time.time()
            haplotype_result = process_haplotype_for_classification(
                subset1_fragments=result['subset1'],
                subset2_fragments=result['subset2'],
                init_haplotype_data=init_haplotype,
                sparse_matrix=sparse_matrix,
                pos_map=pos_map,
                fragment_map=fragment_map,
                block_number=block_number,
                haplotype_output_dir=haplotype_output_dir
            )
            if haplotype_result:
                haplotype_results[block_number] = haplotype_result
            haplotype_process_total_time += time.time() - start_time
        except Exception as e:
            continue

    print(f"Total haplotype processing time: {haplotype_process_total_time:.2f}s")

    if vcf_file and os.path.exists(vcf_file):
        print("\nGenerating VCF output...")
        vcf_headers, vcf_data_lines = read_vcf_file(vcf_file)
        output_vcf_file = os.path.join(haplotype_output_dir, 'phased.vcf')
        write_vcf_output(vcf_headers, vcf_data_lines, haplotype_results, output_vcf_file)

    print(f"\nAll blocks processed, total time: {time.time() - total_start_time:.2f}s")
    print("Program finished.")