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

# 兼容旧版本 NumPy
if not hasattr(np, 'int'):
    np.int = int

import tabu_SB_gpu
from scipy.sparse import coo_matrix
import torch


def read_vcf_file(vcf_file_path):
    """读取VCF文件，返回头部信息和数据"""
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
    """解析VCF数据行"""
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
        print(f"    区块 {block_number} 没有边，跳过处理")
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
            print(f"    警告：边格式不正确: {edge}")
            continue

        if node1 in node_to_idx and node2 in node_to_idx:
            idx1 = node_to_idx[node1]
            idx2 = node_to_idx[node2]
            row_indices.extend([idx1, idx2])
            col_indices.extend([idx2, idx1])
            data.extend([-weight, -weight])

    if not data:
        print(f"    区块 {block_number} 没有有效的边，跳过处理")
        return None

    # 构建稀疏张量
    device = 'cuda'
    G = torch.sparse_coo_tensor(
        torch.tensor([row_indices, col_indices], dtype=torch.long),
        torch.tensor(data, dtype=torch.float),
        size=(n_v, n_v)
    ).to(device)

    row_sum = torch.sparse.sum(G, dim=1).to_dense()
    xi = 1 / torch.abs(row_sum).max()

    # 调用 SB 类
    G = G.to_sparse_csr()
    # s = tabu_SB_gpu.SB(G, n_iter=n_v*7, xi=xi, dt=1., batch_size=1000, device=device)
    s = tabu_SB_gpu.SB(G, n_iter=10000, xi=xi, dt=1., batch_size=200, device=device)
    # s = tabu_SB_gpu.SB(G, n_iter=20000, xi=xi, dt=1., batch_size=1000, device=device)
    s.update_b()

    # 获取最佳分类结果
    best_sample = torch.sign(s.x).clone()
    energy = -0.5 * torch.sum(G @ best_sample * best_sample, dim=0)
    cut = -0.5 * energy - 0.25 * G.sum()
    ind = cut.argmax()
    best_classification = best_sample[:, ind].cpu().numpy()

    # 获取并打印最大割值
    max_cut_value = cut[ind].item()
    print(f"    区块 {block_number} 的最大割值为: {max_cut_value}")

    subset1_fragments = [node for node, idx in node_to_idx.items() if best_classification[idx] == 1]
    subset2_fragments = [node for node, idx in node_to_idx.items() if best_classification[idx] == -1]

    return {'subset1': subset1_fragments, 'subset2': subset2_fragments}


def find_non_zero_column_range_csr_optimized(sparse_matrix, row_indices):
    """
    利用CSR矩阵特性高效找到指定行的非零列范围
    """
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
    """处理单个分类的单体型"""

    # 从 init_haplotype_data 中获取数据
    if 'positions' in init_haplotype_data:
        init_positions = init_haplotype_data['positions']
        init_seq1 = init_haplotype_data['sequence1']
        init_seq2 = init_haplotype_data['sequence2']
        init_cols = [str(p) for p in init_positions]
        init_line2 = list(init_seq1)
        init_line3 = list(init_seq2)
    else:
        raise ValueError("init_haplotype_data 格式不支持")

    # 转换为numpy数组以提高访问速度
    init_line2_arr = np.array(init_line2, dtype='U1')
    init_line3_arr = np.array(init_line3, dtype='U1')
    init_cols_array = np.array(init_cols)

    # 转换fragment到行索引的映射
    subset1_indices = [fragment_map[f] for f in subset1_fragments if f in fragment_map]
    subset2_indices = [fragment_map[f] for f in subset2_fragments if f in fragment_map]

    if not subset1_indices or not subset2_indices:
        print(f"    子集索引为空，跳过该区块")
        return None

    # 转换为CSR格式
    if not isinstance(sparse_matrix, csr_matrix):
        sparse_matrix = sparse_matrix.tocsr()

    # 使用优化的CSR方法找到非零列范围
    all_indices = subset1_indices + subset2_indices
    col_range = find_non_zero_column_range_csr_optimized(sparse_matrix, all_indices)

    if col_range is None:
        print(f"    区块 {block_number} 没有非零元素，跳过处理")
        return None

    first_non_zero_col, last_non_zero_col = col_range
    print(f"    区块 {block_number}: 有效列范围 [{first_non_zero_col}, {last_non_zero_col}]，"
          f"共 {last_non_zero_col - first_non_zero_col + 1} 列需要处理")

    # 创建有效列的索引范围
    valid_col_range = np.arange(first_non_zero_col, last_non_zero_col + 1)

    # 只提取有效列范围内的子矩阵
    subset1_matrix_valid = sparse_matrix[subset1_indices][:, valid_col_range]
    subset2_matrix_valid = sparse_matrix[subset2_indices][:, valid_col_range]

    # 在有效范围内进行统计计数
    cnt1_valid = (subset1_matrix_valid == 1).sum(axis=0).A1
    cnt_neg1_valid = (subset1_matrix_valid == -1).sum(axis=0).A1
    cnt2_valid = (subset2_matrix_valid == 1).sum(axis=0).A1
    cnt_neg2_valid = (subset2_matrix_valid == -1).sum(axis=0).A1

    # 将有效范围的计数映射回完整的列索引空间
    cnt1 = np.zeros(sparse_matrix.shape[1], dtype=int)
    cnt_neg1 = np.zeros(sparse_matrix.shape[1], dtype=int)
    cnt2 = np.zeros(sparse_matrix.shape[1], dtype=int)
    cnt_neg2 = np.zeros(sparse_matrix.shape[1], dtype=int)

    cnt1[valid_col_range] = cnt1_valid
    cnt_neg1[valid_col_range] = cnt_neg1_valid
    cnt2[valid_col_range] = cnt2_valid
    cnt_neg2[valid_col_range] = cnt_neg2_valid

    # 构建有效列索引映射
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
        print(f"    区块 {block_number} 没有有效的位点映射，跳过处理")
        return None

    # 转换为NumPy数组
    valid_col_indices_arr = np.array(valid_col_indices)
    valid_init_indices_arr = np.array(valid_init_indices)

    # 筛选逻辑
    mask_remove_sub = ((cnt1[valid_col_indices_arr] + cnt_neg1[valid_col_indices_arr]) == 0) & \
                      ((cnt2[valid_col_indices_arr] + cnt_neg2[valid_col_indices_arr]) == 0)

    non_zero_condition = (cnt1[valid_col_indices_arr] > 0) & (cnt_neg1[valid_col_indices_arr] > 0) & \
                         (cnt2[valid_col_indices_arr] > 0) & (cnt_neg2[valid_col_indices_arr] > 0)
    same_count_mask = (cnt1[valid_col_indices_arr] == cnt_neg1[valid_col_indices_arr]) & \
                      (cnt2[valid_col_indices_arr] == cnt_neg2[valid_col_indices_arr]) & non_zero_condition
    mask_remove_sub |= same_count_mask
    keep_mask_sub = ~mask_remove_sub

    # 碱基选择逻辑
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

    # 应用保留掩码
    filtered_cols = init_cols_array[valid_init_indices_arr][keep_mask_sub]
    haplo2_filtered = haplo2_selection[keep_mask_sub]
    haplo3_filtered = haplo3_selection[keep_mask_sub]

    # 移除碱基相同的位点
    same_base_mask = haplo2_filtered == haplo3_filtered
    final_cols = filtered_cols[~same_base_mask]
    final_haplo2 = haplo2_filtered[~same_base_mask]
    final_haplo3 = haplo3_filtered[~same_base_mask]

    # 返回结果而不是写入文件
    return {
        'positions': [int(p) for p in final_cols],
        'haplotype1': list(final_haplo2),
        'haplotype2': list(final_haplo3)
    }


def write_vcf_output(vcf_headers, vcf_data_lines, haplotype_results, output_file):
    """将单体型结果写入VCF格式文件"""

    # 创建位点到区块映射
    pos_to_block = {}
    for block_num, result in haplotype_results.items():
        if result and 'positions' in result:
            positions = result['positions']
            if positions:  # 确保有位点
                ps_value = positions[0]  # 使用区块的第一个位点作为PS值
                for i, pos in enumerate(positions):
                    pos_to_block[pos] = {
                        'block': block_num,
                        'ps': ps_value,
                        'hap1': result['haplotype1'][i],
                        'hap2': result['haplotype2'][i]
                    }

    with open(output_file, 'w') as f:
        # 检查是否已有PS格式说明
        ps_format_exists = any('##FORMAT=<ID=PS' in h for h in vcf_headers)

        # 找到最后一个##FORMAT行的位置
        last_format_idx = -1
        for i, header in enumerate(vcf_headers):
            if header.startswith('##FORMAT='):
                last_format_idx = i

        # 处理头部信息
        for i, header in enumerate(vcf_headers):
            f.write(header + '\n')

            # 在最后一个##FORMAT行后添加PS格式说明（如果还没有）
            if i == last_format_idx and not ps_format_exists:
                f.write('##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">\n')
                ps_format_exists = True

        # 处理数据行
        for line in vcf_data_lines:
            data = parse_vcf_data_line(line)
            if not data:
                f.write(line + '\n')
                continue

            pos = data['pos']

            # 检查该位点是否在某个区块中
            if pos in pos_to_block:
                block_info = pos_to_block[pos]
                ps_value = block_info['ps']
                hap1_base = block_info['hap1']
                hap2_base = block_info['hap2']

                # 解析FORMAT和SAMPLE字段
                format_fields = data['format'].split(':')
                sample_values = data['sample'].split(':')

                # 确定phased genotype
                ref = data['ref']
                alt = data['alt']

                # 根据单体型结果确定基因型
                if hap1_base == ref and hap2_base == alt:
                    phased_gt = '0|1'
                elif hap1_base == alt and hap2_base == ref:
                    phased_gt = '1|0'
                elif hap1_base == ref and hap2_base == ref:
                    phased_gt = '0|0'
                elif hap1_base == alt and hap2_base == alt:
                    phased_gt = '1|1'
                else:
                    # 默认保持原样但加上phase
                    original_gt = sample_values[0] if sample_values else '0/1'
                    phased_gt = original_gt.replace('/', '|')

                # 替换GT值
                if sample_values:
                    sample_values[0] = phased_gt

                # 添加PS到FORMAT和样本值（在数据行中添加）
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

                # 重建行
                parts = [data['chrom'], str(data['pos']), data['id'], data['ref'],
                         data['alt'], data['qual'], data['filter'], data['info'],
                         new_format, new_sample]
                parts.extend(data['other_samples'])
                f.write('\t'.join(parts) + '\n')
            else:
                # 位点不在任何区块中，保持原样
                f.write(line + '\n')

    print(f"VCF格式结果已写入: {output_file}")


def process_blocks(components_graphs, init_haplotype, matrix_triple,
                   pos_map, fragment_map, haplotype_output_dir, vcf_file=None):
    """主处理函数：先分类所有区块，再统一生成单体型"""
    total_start_time = time.time()

    if not os.path.exists(haplotype_output_dir):
        os.makedirs(haplotype_output_dir)

    print(f"components_graphs 类型: {type(components_graphs)}")
    print(f"components_graphs 长度: {len(components_graphs)}")

    print("\n构建稀疏矩阵...")
    if matrix_triple:
        rows, cols, values = zip(*matrix_triple)
        max_row = max(fragment_map.values()) + 1
        max_col = max(pos_map.values()) + 1
        sparse_matrix = csr_matrix((values, (rows, cols)), shape=(max_row, max_col))
    else:
        print("警告：matrix_triple 为空")
        return

    print(f"稀疏矩阵形状: {sparse_matrix.shape}")
    print(f"处理 {len(components_graphs)} 个区块\n")

    # 统计最大割过程总时间
    max_cut_total_time = 0.0

    # 第一步：收集所有区块的分类结果
    classification_results = []
    for i in range(len(components_graphs)):
        block_number = i + 1

        print(f"分类区块 {block_number}/{len(components_graphs)}")
        component_graph = components_graphs[i]

        if not isinstance(component_graph, dict):
            print(f"警告：区块 {block_number} 的数据格式不正确")
            continue

        if 'nodes' not in component_graph or 'edges' not in component_graph:
            print(f"警告：区块 {block_number} 缺少 'nodes' 或 'edges' 键")
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
            print(f"    区块 {block_number} 分类完成\n")
        except Exception as e:
            print(f"    分类区块 {block_number} 时出错: {str(e)}")
            continue

    print(f"最大割过程总时间: {max_cut_total_time:.2f}秒")

    # 第二步：根据分类结果统一生成单体型
    print("开始统一生成单体型...\n")
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
            print(f"    生成区块 {block_number} 单体型时出错: {str(e)}")
            continue

    print(f"处理单体型的总时间: {haplotype_process_total_time:.2f}秒")

    # 第三步：如果提供了VCF文件，生成VCF格式输出
    if vcf_file and os.path.exists(vcf_file):
        print("\n生成VCF格式输出...")
        vcf_headers, vcf_data_lines = read_vcf_file(vcf_file)
        output_vcf_file = os.path.join(haplotype_output_dir, 'phased.vcf')
        write_vcf_output(vcf_headers, vcf_data_lines, haplotype_results, output_vcf_file)

    print(f"\n所有区块处理完成，总耗时: {time.time() - total_start_time:.2f}秒")
    print("程序结束。")