#!/usr/bin/env python
# coding: utf-8

import warnings
import time
import os
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix, csr_matrix
import tabu_SB_gpu
from collections import defaultdict

# 导入后处理模块
from phasing_post_processing import apply_phasing_post_processing
#!/usr/bin/env python3

import sys
import pysam
from collections import defaultdict
import random
import os
import time
import tempfile
import get_Snps_Map_n_block
import get_init_matrix
from init_haplotype import generate_haplotype_data
# 导入自定义模块
from get_matrix import process_sparse_matrices
from BSB_get_haplotype import process_blocks
# 导入合并模块
import merge_phasing_files

def save_connected_components(components_graphs, output_dir):
    """
    将连通分量保存到文件中
    在输出目录下创建connected_components目录，保存各个blockn.txt文件
    """
    # 创建保存连通分量的目录
    components_dir = os.path.join(output_dir, "connected_components")
    os.makedirs(components_dir, exist_ok=True)

    # 遍历每个连通分量并保存
    for i, graph in enumerate(components_graphs, 1):
        # 构建文件名：block1.txt, block2.txt, ...
        filename = f"block{i}.txt"
        filepath = os.path.join(components_dir, filename)

        # 写入文件
        with open(filepath, 'w') as f:
            # 写入表头
            f.write("Node1\tNode2\tWeight\n")

            # 写入边信息
            # 检查graph的类型，根据不同类型获取边信息
            if hasattr(graph, 'edges'):  # 如果是NetworkX图
                for u, v, data in graph.edges(data=True):
                    weight = data.get('weight', 0.0)
                    f.write(f"{u}\t{v}\t{weight:.15f}\n")
            else:  # 如果是元组形式 (nodes, edges)
                nodes, edges = graph
                for u, v, weight in edges:
                    f.write(f"{u}\t{v}\t{weight:.15f}\n")

        print(
            f"已保存连通分量 {i} 到 {filepath}，包含 {len(edges) if not hasattr(graph, 'edges') else graph.number_of_edges()} 条边")

    return components_dir

# 主函数
def main():
    # 解析命令行参数
    if len(sys.argv) < 3:
        sys.exit('Usage: python3 %s <vcf.tab> <fragment_reads.txt>\n'
                 'OR: python3 %s --porec <vcf.tab> <fragment_reads.txt> <porec_vcf.tab> <porec_fragment_reads.txt> a=<value>\n'
                 'OR: python3 %s --no-post <vcf.tab> <fragment_reads.txt>  # 禁用后处理'
                 % (sys.argv[0], sys.argv[0], sys.argv[0]))

    # 检查是否禁用后处理
    enable_post_processing = True
    if '--no-post' in sys.argv:
        enable_post_processing = False
        sys.argv.remove('--no-post')
        print("注意: 已禁用后处理")

    # 检查是否有 --porec 参数
    if sys.argv[1] == '--porec':
        # 需要合并两个文件，且包含a参数
        if len(sys.argv) != 7:
            sys.exit(
                'Usage: python3 %s --porec <vcf.tab> <fragment_reads.txt> <porec_vcf.tab> <porec_fragment_reads.txt> a=<value>'
                % sys.argv[0])

        vcf1_file = sys.argv[2]
        fragment_reads1_file = sys.argv[3]
        vcf2_file = sys.argv[4]
        fragment_reads2_file = sys.argv[5]

        # 解析a参数
        a_param = sys.argv[6]
        if not a_param.startswith('a='):
            sys.exit("错误: a参数格式不正确，应为a=<value>")

        try:
            a = float(a_param.split('=')[1])
            print(f"从命令行获取到a的值: {a}")
        except ValueError:
            sys.exit("错误: a的值必须是一个数字")

        # 设置输出目录（使用第一个VCF文件的目录）
        output_dir = os.path.dirname(vcf1_file)
        if not output_dir:
            output_dir = os.getcwd()

        print(f"检测到 --porec 参数，开始合并文件...")
        print(f"VCF文件1: {vcf1_file}")
        print(f"fragment reads文件1: {fragment_reads1_file}")
        print(f"VCF文件2: {vcf2_file}")
        print(f"fragment reads文件2: {fragment_reads2_file}")

        # 调用合并函数
        merge_start_time = time.time()
        try:
            # 创建临时输出文件路径
            merged_vcf_file = os.path.join(output_dir, "merged.tab")
            merged_phasing_file = os.path.join(output_dir, "merged_fragment_reads.txt")

            # 调用合并函数，获取第一个文件的行数
            success, fragment_reads_1_line_count = merge_phasing_files.merge_files(
                vcf1_file, fragment_reads1_file,
                vcf2_file, fragment_reads2_file,
                merged_vcf_file, merged_phasing_file
            )

            if not success:
                sys.exit("文件合并失败")

            merge_end_time = time.time()
            merge_time = merge_end_time - merge_start_time
            print(f"文件合并完成 - 耗时: {merge_time:.2f} 秒")
            print(f"第一个fragment reads文件包含 {fragment_reads_1_line_count} 行")

            # 使用合并后的文件继续处理
            vcf_file = merged_vcf_file
            fragment_reads_file = merged_phasing_file
            print(f"\n使用合并后的文件继续处理...")
            print(f"合并后VCF文件: {vcf_file}")
            print(f"合并后fragment reads文件: {fragment_reads_file}")

        except Exception as e:
            sys.exit(f"文件合并过程中出错: {str(e)}")

    else:
        # 正常模式，只处理单个文件对
        if len(sys.argv) != 3:
            sys.exit('Usage: python3 %s <vcf.tab> <fragment_reads.txt>' % sys.argv[0])

        vcf_file = sys.argv[1]
        fragment_reads_file = sys.argv[2]
        merge_time = 0
        fragment_reads_1_line_count = 0
        a = 1

    # 设置输出目录
    output_dir = os.path.join(os.path.dirname(vcf_file))
    print(f"output_dir = {output_dir}")
    print(f"后处理功能: {'启用' if enable_post_processing else '禁用'}")

    # 记录开始的时间
    start_time = time.time()

    # 用于存储各函数运行时间
    timing_results = []

    # 如果有合并操作，添加到时间统计中
    if merge_time > 0:
        timing_results.append(("合并VCF和fragment reads文件 (merge_files)", merge_time))

    # 初始化碱基矩阵和正确率矩阵
    func_start = time.time()
    result = get_init_matrix.build_snp_sparse_matrix(vcf_file, fragment_reads_file)
    matrix = result['sparse_matrix']
    correct_rates = result['sparse_correct_rate']
    pos_map = result['position_to_column']
    fragment_map = result['fragment_to_index']
    func_end = time.time()
    init_matrix_time = func_end - func_start
    timing_results.append(("初始化碱基矩阵 (build_snp_sparse_matrix)", init_matrix_time))
    print(f"已生成碱基矩阵 - 耗时: {init_matrix_time:.2f} 秒")

    # 初始化单体型、记录节点
    func_start = time.time()
    init_haplotype, all_nodes = generate_haplotype_data(vcf_file, matrix, pos_map)
    func_end = time.time()
    generate_haplotype_time = func_end - func_start
    timing_results.append(("初始化单体型 (generate_haplotype_data)", generate_haplotype_time))
    print(f"已生成初始化单体型 - 耗时: {generate_haplotype_time:.2f} 秒")

    # 保存矩阵
    func_start = time.time()
    matrix_weight = process_sparse_matrices(matrix, pos_map, init_haplotype, correct_rates)
    func_end = time.time()
    process_matrices_time = func_end - func_start
    timing_results.append(("处理稀疏矩阵 (process_sparse_matrices)", process_matrices_time))
    print(f"加权三元矩阵已生成 - 耗时: {process_matrices_time:.2f} 秒\n")

    # 获取SNP映射和块，传递第一个文件的行数
    func_start = time.time()
    print(f"*** 在第 {fragment_reads_1_line_count + 1} 行，开始应用权重乘数 {a} ***\n")
    components_graphs = get_Snps_Map_n_block.main(
        matrix_weight,
        pos_map,
        fragment_reads_1_line_count=fragment_reads_1_line_count,
        a=a
    )
    func_end = time.time()
    get_components_time = func_end - func_start
    timing_results.append(("获取SNP映射和块 (get_Snps_Map_n_block.main)", get_components_time))
    print(f"SNP映射和块处理完成 - 耗时: {get_components_time:.2f} 秒")

    # 保存连通分量到文件
    func_start = time.time()
    if components_graphs:
        components_dir = save_connected_components(components_graphs, output_dir)
        print(f"连通分量已保存到目录: {components_dir}")
    else:
        print("未找到连通分量，不进行保存")
    func_end = time.time()
    save_components_time = func_end - func_start
    timing_results.append(("保存连通分量 (save_connected_components)", save_components_time))
    print(f"连通分量保存完成 - 耗时: {save_components_time:.2f} 秒")

    # 处理块 - 传入所有必要参数以支持后处理
    func_start = time.time()
    phasing_output_dir = os.path.join(os.path.dirname(vcf_file), 'phasing_output')

    # 调用改进后的process_blocks函数
    process_blocks(
        components_graphs,
        init_haplotype,
        phasing_output_dir,
        vcf_file,
        fragment_reads_file=fragment_reads_file,
        matrix_data=matrix,
        pos_map=pos_map,
        enable_post_processing=enable_post_processing
    )

    func_end = time.time()
    process_blocks_time = func_end - func_start

    if enable_post_processing:
        timing_results.append(("处理块并生成phased VCF (含后处理)", process_blocks_time))
    else:
        timing_results.append(("处理块并生成phased VCF (process_blocks)", process_blocks_time))

    print(f"块处理和VCF输出完成 - 耗时: {process_blocks_time:.2f} 秒")

    end_time = time.time()
    total_run_time = end_time - start_time

    # 打印详细的时间统计
    print("\n" + "=" * 60)
    print("各函数运行时间统计:")
    print("=" * 60)
    for func_name, func_time in timing_results:
        percentage = (func_time / total_run_time) * 100
        print(f"{func_name:<45} {func_time:>8.2f} 秒 ({percentage:>5.1f}%)")
    print("-" * 60)
    print(f"{'总运行时间':<45} {total_run_time:>8.2f} 秒 (100.0%)")
    print("=" * 60)

    # 保存运行时间到文件
    run_time_file = os.path.join(os.path.dirname(vcf_file), 'run_time.txt')
    with open(run_time_file, 'w') as f:
        f.write(f"程序运行时间统计\n")
        f.write("=" * 60 + "\n")

        if '--porec' in sys.argv:
            f.write("输入文件（合并模式）:\n")
            f.write(f"  VCF文件1: {sys.argv[2]}\n")
            f.write(f"  fragment reads文件1: {sys.argv[3]}\n")
            f.write(f"  VCF文件2: {sys.argv[4]}\n")
            f.write(f"  fragment reads文件2: {sys.argv[5]}\n")
            f.write(f"  a值: {a}\n")
            f.write(f"  合并后VCF文件: {vcf_file}\n")
            f.write(f"  合并后fragment reads文件: {fragment_reads_file}\n")
            f.write(f"  第一个fragment reads文件行数: {fragment_reads_1_line_count}\n")
        else:
            f.write("输入文件（单文件模式）:\n")
            f.write(f"  VCF文件: {vcf_file}\n")
            f.write(f"  fragment reads文件: {fragment_reads_file}\n")
            f.write(f"  a值: {a}\n")

        f.write(f"  后处理: {'启用' if enable_post_processing else '禁用'}\n")
        f.write("-" * 60 + "\n")
        f.write("输出文件:\n")
        phasing_output_dir = os.path.join(output_dir, "phasing_output")
        f.write(f"  Phased VCF文件: {os.path.join(phasing_output_dir, 'phased.vcf')}\n")

        if enable_post_processing:
            f.write(f"  初始Phased VCF: {os.path.join(phasing_output_dir, 'phased_initial.vcf')}\n")
            f.write(f"  改进后Phased VCF: {os.path.join(phasing_output_dir, 'phased_improved.vcf')}\n")
            f.write(f"  后处理报告: {os.path.join(phasing_output_dir, 'post_processing_report.txt')}\n")

        f.write("-" * 60 + "\n")
        f.write(f"总运行时间: {total_run_time:.2f} 秒\n")
        f.write("=" * 60 + "\n")
        f.write("各函数运行时间明细:\n")
        f.write("-" * 60 + "\n")
        for func_name, func_time in timing_results:
            percentage = (func_time / total_run_time) * 100
            f.write(f"{func_name:<45} {func_time:>8.2f} 秒 ({percentage:>5.1f}%)\n")
        f.write("=" * 60 + "\n")
        f.write(f"记录时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")

    print(f"\n运行时间详情已保存到 {run_time_file}")
    print(f"最终Phased VCF文件已保存到 {os.path.join(phasing_output_dir, 'phased.vcf')}")

    if enable_post_processing:
        print(f"后处理报告已保存到 {os.path.join(phasing_output_dir, 'post_processing_report.txt')}")
        print("\n提示: 使用 --no-post 参数可以禁用后处理功能")

    if '--porec' in sys.argv:
        print(f"\n注意：合并后的中间文件已保存在:")
        print(f"  - {vcf_file}")
        print(f"  - {fragment_reads_file}")


if __name__ == '__main__':
    main()
# 兼容旧版本 NumPy
if not hasattr(np, 'int'):
    np.int = int

# 用于存储各部分累计时间的全局变量
max_cut_total_time = 0
update_weights_total_time = 0
update_haplotype_total_time = 0
post_processing_total_time = 0  # 新增后处理时间统计


def edges_to_sparse_matrix(edges_data, nodes_list, negate=True):
    """
    将边列表转换为稀疏矩阵格式，用于BSB算法。

    参数:
    - edges_data: 边数据列表，每项为 (node1, node2, weight)
    - nodes_list: 节点列表
    - negate: 是否对权重取负

    返回:
    - G: scipy.sparse.csr_matrix
    - node_map: 节点到索引的映射
    """
    node_map = {node: idx for idx, node in enumerate(nodes_list)}
    n_v = len(nodes_list)

    if not edges_data:
        # 处理没有边的情况（孤立节点）
        return csr_matrix((n_v, n_v)), node_map

    # 向量化处理
    edges_array = np.array(edges_data, dtype=object)
    nodes1 = edges_array[:, 0]
    nodes2 = edges_array[:, 1]
    weights = edges_array[:, 2].astype(float)

    # 使用向量化操作获取索引
    row_indices = np.array([node_map[n] for n in nodes1])
    col_indices = np.array([node_map[n] for n in nodes2])

    # 构建双向边
    row = np.concatenate([row_indices, col_indices])
    col = np.concatenate([col_indices, row_indices])
    data = np.concatenate([weights, weights])

    if negate:
        data = -data

    G = coo_matrix((data, (row, col)), shape=(n_v, n_v))
    return G.tocsr(), node_map


def scipy_to_torch_sparse(G_csr, device='cuda'):
    """
    将scipy稀疏矩阵转换为PyTorch稀疏张量并移动到GPU
    """
    G_coo = G_csr.tocoo()
    indices = torch.LongTensor([G_coo.row, G_coo.col]).to(device)
    values = torch.FloatTensor(G_coo.data).to(device)
    shape = G_coo.shape
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def update_sparse_matrix_weights(G_csr, best_classification):
    """
    根据分类结果更新稀疏矩阵的权重（优化版本）。

    参数:
    - G_csr: CSR格式的稀疏矩阵（会被直接修改）
    - best_classification: 分类结果

    返回:
    - modified_count: 修改的边数
    """
    global update_weights_total_time
    start_time = time.time()

    # 转换为COO格式以便修改
    G_coo = G_csr.tocoo()

    # 获取分类结果
    class_row = best_classification[G_coo.row]
    class_col = best_classification[G_coo.col]

    # 找出跨集合的边
    cross_edges = class_row != class_col

    # 反转跨集合边的权重
    G_coo.data[cross_edges] = -G_coo.data[cross_edges]

    modified_count = np.sum(cross_edges) // 2  # 除以2因为是无向图

    # 转换回CSR格式
    G_modified = G_coo.tocsr()

    update_weights_total_time += time.time() - start_time
    return G_modified, modified_count


def get_smaller_subset_vectorized(classification):
    """
    使用向量化操作获取较小的集合。

    参数:
    - classification: 分类结果数组

    返回:
    - smaller_indices: 较小集合的索引数组
    """
    subset1_mask = classification == 1
    subset1_count = np.sum(subset1_mask)
    subset2_count = len(classification) - subset1_mask.size

    if subset1_count <= subset2_count:
        return np.where(subset1_mask)[0]
    else:
        return np.where(~subset1_mask)[0]


def update_haplotype_for_positions(initial_haplotype, position_counts, positions):
    """
    根据计数结果更新特定位置的单倍型。
    返回phasing信息而不是完整的haplotype字典。
    """
    global update_haplotype_total_time
    start_time = time.time()

    # 创建位置到索引的映射
    pos_to_idx = {pos: idx for idx, pos in enumerate(initial_haplotype['positions'])}

    # 存储phasing结果
    phasing_result = {}

    for pos in positions:
        if pos in pos_to_idx:
            idx = pos_to_idx[pos]

            # 获取原始序列值
            seq1_val = initial_haplotype['sequence1'][idx]
            seq2_val = initial_haplotype['sequence2'][idx]

            # 如果该位置的计数是奇数，交换seq1和seq2
            if position_counts.get(pos, 0) % 2 == 1:
                phasing_result[pos] = (seq2_val, seq1_val)
            else:
                phasing_result[pos] = (seq1_val, seq2_val)

    update_haplotype_total_time += time.time() - start_time
    return phasing_result


def read_vcf_header(vcf_file):
    """
    读取VCF文件的头部信息
    """
    header_lines = []
    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header_lines.append(line.rstrip())
            else:
                break
    return header_lines


def parse_vcf_line(line):
    """
    解析VCF数据行
    """
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
    """
    修改VCF行，添加phasing信息

    参数:
    - vcf_line_dict: 解析后的VCF行字典
    - phase_info: (hap1_allele, hap2_allele) 元组
    - phase_set: phase set ID (block的第一个位点)
    """
    # 解析FORMAT和SAMPLE字段
    format_fields = vcf_line_dict['format'].split(':')
    sample_values = vcf_line_dict['sample'].split(':')

    # 获取GT字段的索引
    gt_index = format_fields.index('GT')
    gt_value = sample_values[gt_index]

    # 确定phased genotype
    hap1_allele, hap2_allele = phase_info

    # 构建phased GT
    if hap1_allele == vcf_line_dict['ref']:
        gt1 = '0'
    else:
        gt1 = '1'

    if hap2_allele == vcf_line_dict['ref']:
        gt2 = '0'
    else:
        gt2 = '1'

    phased_gt = f"{gt1}|{gt2}"

    # 更新GT值
    sample_values[gt_index] = phased_gt

    # 添加PS字段
    if 'PS' not in format_fields:
        format_fields.append('PS')
        sample_values.append(str(phase_set))
    else:
        ps_index = format_fields.index('PS')
        sample_values[ps_index] = str(phase_set)

    # 重建FORMAT和SAMPLE字段
    new_format = ':'.join(format_fields)
    new_sample = ':'.join(sample_values)

    # 构建新的VCF行
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
    """
    生成phased VCF文件

    参数:
    - vcf_file: 原始VCF文件路径
    - all_phasing_results: 包含所有block的phasing结果
    - output_path: 输出VCF文件路径
    """
    # 读取VCF头部
    header_lines = read_vcf_header(vcf_file)

    # 添加PS格式说明（如果不存在）
    ps_header = '##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">'
    if ps_header not in header_lines:
        # 在最后一个FORMAT行后添加PS说明
        for i in range(len(header_lines) - 1, -1, -1):
            if header_lines[i].startswith('##FORMAT='):
                header_lines.insert(i + 1, ps_header)
                break

    # 创建位置到phasing信息的映射
    pos_to_phasing = {}
    for block_info in all_phasing_results:
        for pos in block_info['phasing']:
            pos_to_phasing[pos] = {
                'phase_info': block_info['phasing'][pos],
                'phase_set': block_info['phase_set']
            }

    # 写入输出文件
    with open(output_path, 'w') as out_f:
        # 写入头部
        for header_line in header_lines:
            out_f.write(header_line + '\n')

        # 处理数据行
        with open(vcf_file, 'r') as in_f:
            for line in in_f:
                if line.startswith('#'):
                    continue

                vcf_dict = parse_vcf_line(line)
                pos = vcf_dict['pos']

                if pos in pos_to_phasing:
                    # 有phasing信息，修改行
                    phasing_info = pos_to_phasing[pos]
                    modified_line = modify_vcf_line(
                        vcf_dict,
                        phasing_info['phase_info'],
                        phasing_info['phase_set']
                    )
                    out_f.write(modified_line + '\n')
                else:
                    # 没有phasing信息，保持原样
                    out_f.write(line)


def process_blocks(components_data, initial_haplotype, output_dir, vcf_file,
                   fragment_reads_file=None, matrix_data=None, pos_map=None,
                   enable_post_processing=True):
    """
    处理连通分量，使用优化的数据结构，并生成phased VCF。
    增加后处理功能以提高准确率。

    参数:
    - components_data: 组件数据列表，每项包含 (nodes, edges) 元组
    - initial_haplotype: 初始单倍型数据
    - output_dir: 输出目录
    - vcf_file: 原始VCF文件路径
    - fragment_reads_file: fragment reads文件路径（用于后处理）
    - matrix_data: 稀疏矩阵数据（用于后处理）
    - pos_map: 位置映射（用于后处理）
    - enable_post_processing: 是否启用后处理（默认True）
    """
    global max_cut_total_time, post_processing_total_time

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # GPU设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device} 进行最大割计算")

    # 初始化总计数字典
    total_position_counts = defaultdict(int)

    # 存储所有block的phasing结果
    all_phasing_results = []

    # 处理每个连通分量
    for block_idx, (nodes, edges) in enumerate(components_data):
        print(f"\n处理连通分量 {block_idx + 1}/{len(components_data)}")

        # 获取当前block的所有位置（节点）
        block_positions = list(nodes)
        # 获取当前block的节点数量
        node_count = len(block_positions)
        print(f"  当前连通分量包含 {node_count} 个节点")

        # 该block的phase set ID（使用block中最小的位置）
        phase_set = min(block_positions)

        # 初始化当前block的计数
        block_counts = defaultdict(int)

        # 检查是否是孤立节点
        if len(block_positions) == 1 and not edges:
            print(f"  连通分量 {block_idx + 1} 是孤立节点，跳过BSB处理")
        else:
            # 初始转换为稀疏矩阵
            G_csr, node_map = edges_to_sparse_matrix(edges, block_positions)
            reverse_node_map = {idx: node for node, idx in node_map.items()}

            converged = False
            iteration = 0
            max_iterations = 15  # 设置最大迭代次数为15

            # 添加标志来跟踪是否需要使用h参数
            use_h_parameter = False

            # 添加计数器来跟踪连续负割值的次数
            consecutive_negative_cuts = 0

            # BSB算法参数
            if node_count < 5000:
                n_iter = 500
                sample_size = 100
            else:
                n_iter = 1000
                sample_size = 100

            # 修改循环条件，增加最大迭代次数限制
            while not converged and iteration < max_iterations:
                iteration += 1
                print(f"  迭代 {iteration}/{max_iterations}")

                # 运行GPU版本的最大割算法
                start_time = time.time()

                # 将scipy稀疏矩阵转换为PyTorch稀疏张量并移至GPU
                if G_csr.nnz == 0:
                    print(f"  连通分量 {block_idx + 1} 没有有效的边，跳过处理")
                    converged = True
                    continue

                # 转换为PyTorch稀疏张量
                G = scipy_to_torch_sparse(G_csr, device)

                print(f"G的数据类型: {G.dtype}")

                # 计算行和并确定xi参数
                row_sum = torch.sparse.sum(G, dim=1).to_dense()
                xi = 1 / torch.abs(row_sum).max()

                # 调用SB类进行GPU计算
                G = G.to_sparse_csr()

                # 根据use_h_parameter标志决定是否使用h参数
                if use_h_parameter:
                    print(f"    使用h参数 (h=0.0004), n_iter={n_iter}, sample_size={sample_size}")
                    s = tabu_SB_gpu.SB(G, h=0.0004, n_iter=n_iter, xi=xi, dt=1., batch_size=sample_size, device=device)
                else:
                    print(f"    未使用h参数, n_iter={n_iter}, sample_size={sample_size}")
                    s = tabu_SB_gpu.SB(G, n_iter=n_iter, xi=xi, dt=1., batch_size=sample_size, device=device)

                s.update_b()

                # 获取最佳分类结果
                best_sample = torch.sign(s.x).clone()

                energy = -0.5 * torch.sum(G @ (best_sample) * best_sample, dim=0)
                print(f"***energy_min = {energy.min()}***")
                cut = -0.5 * energy - 0.25 * G.sum()
                ind = cut.argmax()

                # 将结果移回CPU并转换为numpy数组
                max_cut_value = cut[ind].item()
                best_classification = best_sample[:, ind].cpu().numpy()

                # 统计1和-1的数量
                count_1 = np.sum(best_classification == 1)
                count_neg1 = np.sum(best_classification == -1)

                max_cut_total_time += time.time() - start_time

                print(f"    最大割值: {max_cut_value}")
                print(f"    分类结果计数: 1={count_1}, -1={count_neg1}")

                if count_1 == 0 or count_neg1 == 0:
                    print(f"  节点分类已收敛, 连通分量 {block_idx + 1} 已收敛")
                    converged = True
                elif max_cut_value < 0:
                    # 只有在尚未启用增强参数时，才进行连续负割值的判断
                    if not use_h_parameter:
                        # 增加连续负割值计数
                        consecutive_negative_cuts += 1
                        print(f"  割值<0, 连续负割值次数: {consecutive_negative_cuts}/3")

                        # 当连续三次负割值时启用h参数，并根据节点数量调整n_iter
                        if consecutive_negative_cuts >= 3:
                            use_h_parameter = True
                            # 根据节点数量设置不同的n_iter值
                            if node_count > 15000:
                                n_iter = 20000
                                sample_size = 1000
                            elif node_count > 9000:
                                n_iter = 10000
                                sample_size = 500
                            else:
                                n_iter = 5000
                                sample_size = 500
                            print(f"  连续三次负割值，启用增强参数")
                            print(
                                f"  参数已调整: use_h_parameter=True, n_iter={n_iter}, sample_size={sample_size}, 节点数={node_count}")
                            # 重置计数器，避免重复调整
                            consecutive_negative_cuts = 0
                    else:
                        # 已经启用了增强参数，只打印信息但不进行判断
                        print(f"  割值<0 (增强参数已启用)")
                else:
                    # 割值>=0的情况
                    # 只有在未启用增强参数时才重置连续负割值计数器
                    if not use_h_parameter and consecutive_negative_cuts > 0:
                        print(f"  割值>=0，重置连续负割值计数器")
                        consecutive_negative_cuts = 0

                    # 在CPU上更新稀疏矩阵权重
                    G_csr, modified_count = update_sparse_matrix_weights(G_csr, best_classification)
                    print(f"    已反转 {modified_count} 条跨集合边的权重")

                    # 获取较小集合的索引
                    smaller_indices = get_smaller_subset_vectorized(best_classification)

                    # 更新计数
                    for idx in smaller_indices:
                        node = reverse_node_map[idx]
                        block_counts[node] += 1

                    print(f"    较小集合包含 {len(smaller_indices)} 个节点")

            # 检查是否因为达到最大迭代次数而退出循环
            if iteration >= max_iterations and not converged:
                print(f"  已达到最大迭代次数 {max_iterations}，停止迭代")

        # 更新总计数
        for pos, count in block_counts.items():
            total_position_counts[pos] += count

        # 基于当前的累积计数获取当前block的phasing信息
        block_phasing = update_haplotype_for_positions(
            initial_haplotype,
            total_position_counts,
            block_positions
        )

        # 保存phasing结果
        all_phasing_results.append({
            'block_idx': block_idx,
            'phase_set': phase_set,
            'positions': block_positions,
            'phasing': block_phasing
        })

        print(f"  Block {block_idx + 1} phasing完成，phase set: {phase_set}")

    print("\n所有连通分量处理完成。")

    # 生成初始phased VCF文件
    initial_output_vcf_path = os.path.join(output_dir, 'phased_initial.vcf')
    save_phased_vcf(vcf_file, all_phasing_results, initial_output_vcf_path)
    print(f"\n初始Phased VCF文件已保存到: {initial_output_vcf_path}")

    # 应用后处理（如果启用）
    if enable_post_processing and fragment_reads_file and matrix_data and pos_map:
        print("\n" + "=" * 60)
        print("应用后处理以提高准确率...")
        print("=" * 60)

        post_start_time = time.time()

        try:
            # 调用后处理函数
            post_stats = apply_phasing_post_processing(
                vcf_file,
                fragment_reads_file,
                initial_output_vcf_path,
                matrix_data,
                pos_map,
                output_dir
            )

            post_processing_total_time = time.time() - post_start_time

            # 将最终改进的VCF文件复制为主输出文件
            improved_vcf_path = os.path.join(output_dir, 'phased_improved.vcf')
            final_vcf_path = os.path.join(output_dir, 'phased.vcf')

            if os.path.exists(improved_vcf_path):
                import shutil
                shutil.copy2(improved_vcf_path, final_vcf_path)
                print(f"\n最终Phased VCF文件已保存到: {final_vcf_path}")

                # 打印后处理统计
                print("\n后处理统计:")
                print(f"  - 检测到的相位切换: {post_stats['phase_switches_detected']}")
                print(f"  - 修正的位点数: {post_stats['positions_corrected']}")
                print(f"  - 平均置信度: {post_stats['average_confidence']:.3f}")
                print(f"  - 合并的blocks: {post_stats['blocks_merged']}")
                print(f"  - 低置信度位点: {post_stats['low_confidence_positions']}")

        except Exception as e:
            print(f"\n警告: 后处理过程出错: {str(e)}")
            print("使用初始phasing结果作为最终输出")

            # 如果后处理失败，使用初始结果
            import shutil
            final_vcf_path = os.path.join(output_dir, 'phased.vcf')
            shutil.copy2(initial_output_vcf_path, final_vcf_path)
    else:
        # 如果不进行后处理，直接使用初始结果
        import shutil
        final_vcf_path = os.path.join(output_dir, 'phased.vcf')
        shutil.copy2(initial_output_vcf_path, final_vcf_path)
        print(f"\nPhased VCF文件已保存到: {final_vcf_path}")

    # 打印各部分耗时
    print("\n===== 性能统计 =====")
    print(f"求解最大割总耗时: {max_cut_total_time:.4f} 秒")
    print(f"更新图权重总耗时: {update_weights_total_time:.4f} 秒")
    print(f"更新单倍型总耗时: {update_haplotype_total_time:.4f} 秒")
    if post_processing_total_time > 0:
        print(f"后处理总耗时: {post_processing_total_time:.4f} 秒")