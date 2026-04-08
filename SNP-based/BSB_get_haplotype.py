#!/usr/bin/env python
# coding: utf-8

import warnings
import time
import os
import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix
import tabu_SB_gpu
from collections import defaultdict
from phasing_post_processing import apply_phasing_post_processing

if not hasattr(np, 'int'):
    np.int = int

max_cut_total_time = 0
update_weights_total_time = 0
update_haplotype_total_time = 0
post_processing_total_time = 0


def edges_to_sparse_matrix(edges_data, nodes_list, negate=True):
    node_map = {node: idx for idx, node in enumerate(nodes_list)}
    n_v = len(nodes_list)

    if not edges_data:
        return csr_matrix((n_v, n_v)), node_map

    edges_array = np.array(edges_data, dtype=object)
    nodes1 = edges_array[:, 0]
    nodes2 = edges_array[:, 1]
    weights = edges_array[:, 2].astype(float)

    row_indices = np.array([node_map[n] for n in nodes1])
    col_indices = np.array([node_map[n] for n in nodes2])

    row = np.concatenate([row_indices, col_indices])
    col = np.concatenate([col_indices, row_indices])
    data = np.concatenate([weights, weights])

    if negate:
        data = -data

    G = coo_matrix((data, (row, col)), shape=(n_v, n_v))
    return G.tocsr(), node_map


def scipy_to_torch_sparse(G_csr, device='cuda'):
    G_coo = G_csr.tocoo()
    indices = torch.LongTensor([G_coo.row, G_coo.col]).to(device)
    values = torch.FloatTensor(G_coo.data).to(device)
    shape = G_coo.shape
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def update_sparse_matrix_weights(G_csr, best_classification):
    global update_weights_total_time
    start_time = time.time()

    G_coo = G_csr.tocoo()
    class_row = best_classification[G_coo.row]
    class_col = best_classification[G_coo.col]
    cross_edges = class_row != class_col
    G_coo.data[cross_edges] = -G_coo.data[cross_edges]
    modified_count = np.sum(cross_edges) // 2
    G_modified = G_coo.tocsr()

    update_weights_total_time += time.time() - start_time
    return G_modified, modified_count


def get_smaller_subset_vectorized(classification):
    subset1_mask = classification == 1
    subset1_count = np.sum(subset1_mask)
    subset2_count = len(classification) - subset1_count

    if subset1_count <= subset2_count:
        return np.where(subset1_mask)[0]
    else:
        return np.where(~subset1_mask)[0]


def update_haplotype_for_positions(initial_haplotype, position_counts, positions):
    global update_haplotype_total_time
    start_time = time.time()

    pos_to_idx = {pos: idx for idx, pos in enumerate(initial_haplotype['positions'])}
    phasing_result = {}

    for pos in positions:
        if pos in pos_to_idx:
            idx = pos_to_idx[pos]
            seq1_val = initial_haplotype['sequence1'][idx]
            seq2_val = initial_haplotype['sequence2'][idx]

            if position_counts.get(pos, 0) % 2 == 1:
                phasing_result[pos] = (seq2_val, seq1_val)
            else:
                phasing_result[pos] = (seq1_val, seq2_val)

    update_haplotype_total_time += time.time() - start_time
    return phasing_result


def read_vcf_header(vcf_file):
    header_lines = []
    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header_lines.append(line.rstrip())
            else:
                break
    return header_lines


def parse_vcf_line(line):
    fields = line.strip().split('\t')
    return {
        'chrom': fields[0],
        'pos': int(fields[1]),
        'id': fields[2],
        'ref': fields[3],
        'alt': fields[4],
        'qual': fields[5],
        'filter': fields[6],
        'info': fields[7],
        'format': fields[8],
        'sample': fields[9]
    }


def modify_vcf_line(vcf_line_dict, phase_info, phase_set):
    format_fields = vcf_line_dict['format'].split(':')
    sample_values = vcf_line_dict['sample'].split(':')
    gt_index = format_fields.index('GT')
    gt_value = sample_values[gt_index]
    hap1_allele, hap2_allele = phase_info

    gt1 = '0' if hap1_allele == vcf_line_dict['ref'] else '1'
    gt2 = '0' if hap2_allele == vcf_line_dict['ref'] else '1'
    phased_gt = f"{gt1}|{gt2}"
    sample_values[gt_index] = phased_gt

    if 'PS' not in format_fields:
        format_fields.append('PS')
        sample_values.append(str(phase_set))
    else:
        ps_index = format_fields.index('PS')
        sample_values[ps_index] = str(phase_set)

    new_format = ':'.join(format_fields)
    new_sample = ':'.join(sample_values)

    fields = [
        vcf_line_dict['chrom'],
        str(vcf_line_dict['pos']),
        vcf_line_dict['id'],
        vcf_line_dict['ref'],
        vcf_line_dict['alt'],
        vcf_line_dict['qual'],
        vcf_line_dict['filter'],
        vcf_line_dict['info'],
        new_format,
        new_sample
    ]

    return '\t'.join(fields)


def save_phased_vcf(vcf_file, all_phasing_results, output_path):
    header_lines = read_vcf_header(vcf_file)
    ps_header = '##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">'
    if ps_header not in header_lines:
        for i in range(len(header_lines) - 1, -1, -1):
            if header_lines[i].startswith('##FORMAT='):
                header_lines.insert(i + 1, ps_header)
                break

    pos_to_phasing = {}
    for block_info in all_phasing_results:
        for pos in block_info['phasing']:
            pos_to_phasing[pos] = {
                'phase_info': block_info['phasing'][pos],
                'phase_set': block_info['phase_set']
            }

    with open(output_path, 'w') as out_f:
        for header_line in header_lines:
            out_f.write(header_line + '\n')

        with open(vcf_file, 'r') as in_f:
            for line in in_f:
                if line.startswith('#'):
                    continue

                vcf_dict = parse_vcf_line(line)
                pos = vcf_dict['pos']

                if pos in pos_to_phasing:
                    phasing_info = pos_to_phasing[pos]
                    modified_line = modify_vcf_line(
                        vcf_dict,
                        phasing_info['phase_info'],
                        phasing_info['phase_set']
                    )
                    out_f.write(modified_line + '\n')
                else:
                    out_f.write(line)


def process_blocks(components_data, initial_haplotype, output_dir, vcf_file,
                   fragment_reads_file=None, matrix_data=None, pos_map=None,
                   enable_post_processing=True):
    global max_cut_total_time, post_processing_total_time
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} for max-cut computation")
    total_position_counts = defaultdict(int)
    all_phasing_results = []

    for block_idx, (nodes, edges) in enumerate(components_data):
        print(f"\nProcessing connected component {block_idx + 1}/{len(components_data)}")
        block_positions = list(nodes)
        node_count = len(block_positions)
        phase_set = min(block_positions)
        block_counts = defaultdict(int)

        if len(block_positions) == 1 and not edges:
            print(f"Component {block_idx + 1} is an isolated node, skipping BSB")
        else:
            G_csr, node_map = edges_to_sparse_matrix(edges, block_positions)
            reverse_node_map = {idx: node for node, idx in node_map.items()}
            converged = False
            iteration = 0
            max_iterations = 15
            use_h_parameter = False
            consecutive_negative_cuts = 0

            if node_count < 5000:
                n_iter = 500
                sample_size = 100
            else:
                n_iter = 1000
                sample_size = 100

            while not converged and iteration < max_iterations:
                iteration += 1
                print(f"Iteration {iteration}/{max_iterations}")
                start_time = time.time()

                if G_csr.nnz == 0:
                    print(f"Component {block_idx + 1} has no valid edges, skipping")
                    converged = True
                    continue

                G = scipy_to_torch_sparse(G_csr, device)
                row_sum = torch.sparse.sum(G, dim=1).to_dense()
                xi = 1 / torch.abs(row_sum).max()
                G = G.to_sparse_csr()

                if use_h_parameter:
                    s = tabu_SB_gpu.SB(G, h=0.0004, n_iter=n_iter, xi=xi, dt=1., batch_size=sample_size, device=device)
                else:
                    s = tabu_SB_gpu.SB(G, n_iter=n_iter, xi=xi, dt=1., batch_size=sample_size, device=device)

                s.update_b()
                best_sample = torch.sign(s.x).clone()
                energy = -0.5 * torch.sum(G @ (best_sample) * best_sample, dim=0)
                cut = -0.5 * energy - 0.25 * G.sum()
                ind = cut.argmax()

                max_cut_value = cut[ind].item()
                best_classification = best_sample[:, ind].cpu().numpy()
                count_1 = np.sum(best_classification == 1)
                count_neg1 = np.sum(best_classification == -1)
                max_cut_total_time += time.time() - start_time

                if count_1 == 0 or count_neg1 == 0:
                    print(f"Component {block_idx + 1} converged")
                    converged = True
                elif max_cut_value < 0:
                    if not use_h_parameter:
                        consecutive_negative_cuts += 1
                        if consecutive_negative_cuts >= 3:
                            use_h_parameter = True
                            if node_count > 15000:
                                n_iter = 20000
                                sample_size = 1000
                            elif node_count > 9000:
                                n_iter = 10000
                                sample_size = 500
                            else:
                                n_iter = 5000
                                sample_size = 500
                            consecutive_negative_cuts = 0
                else:
                    if not use_h_parameter and consecutive_negative_cuts > 0:
                        consecutive_negative_cuts = 0

                    G_csr, modified_count = update_sparse_matrix_weights(G_csr, best_classification)
                    smaller_indices = get_smaller_subset_vectorized(best_classification)

                    for idx in smaller_indices:
                        node = reverse_node_map[idx]
                        block_counts[node] += 1

            if iteration >= max_iterations and not converged:
                print(f"Max iterations {max_iterations} reached, stopping")

        for pos, count in block_counts.items():
            total_position_counts[pos] += count

        block_phasing = update_haplotype_for_positions(
            initial_haplotype,
            total_position_counts,
            block_positions
        )

        all_phasing_results.append({
            'block_idx': block_idx,
            'phase_set': phase_set,
            'positions': block_positions,
            'phasing': block_phasing
        })

        print(f"Block {block_idx + 1} phasing completed, phase set: {phase_set}")

    print("\nAll components processed.")

    if enable_post_processing and fragment_reads_file and matrix_data and pos_map:
        initial_output_vcf_path = os.path.join(output_dir, '_phased_tmp.vcf')
        save_phased_vcf(vcf_file, all_phasing_results, initial_output_vcf_path)

        print("\nApplying post-processing...")
        post_start_time = time.time()
        final_vcf_path = os.path.join(output_dir, 'phased.vcf')
        try:
            post_stats = apply_phasing_post_processing(
                vcf_file,
                fragment_reads_file,
                initial_output_vcf_path,
                matrix_data,
                pos_map,
                output_dir
            )
            post_processing_total_time = time.time() - post_start_time
            improved_vcf_path = os.path.join(output_dir, 'phased_improved.vcf')
            if os.path.exists(improved_vcf_path):
                import shutil
                shutil.move(improved_vcf_path, final_vcf_path)
            else:
                import shutil
                shutil.move(initial_output_vcf_path, final_vcf_path)
        except Exception as e:
            import shutil
            shutil.move(initial_output_vcf_path, final_vcf_path)
        finally:
            for tmp in [initial_output_vcf_path,
                        os.path.join(output_dir, 'phased_improved.vcf')]:
                if os.path.exists(tmp):
                    os.remove(tmp)
    else:
        final_vcf_path = os.path.join(output_dir, 'phased.vcf')
        save_phased_vcf(vcf_file, all_phasing_results, final_vcf_path)
