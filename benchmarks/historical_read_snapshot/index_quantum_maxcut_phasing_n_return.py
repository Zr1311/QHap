#!/usr/bin/env python3

import sys
import pysam
from collections import defaultdict
import random
import os
import time
import pandas as pd
import get_init_matrix
from init_haplotype import generate_haplotype_data
# 导入自定义模块
from get_matrix import process_sparse_matrices
import get_Snps_Map_reads_block
from BSB_get_haplotype_and_combine import process_blocks
# 导入合并模块
import merge_vcf_extractHAIRS


# 新增：保存连通分量的函数（适配代码2的components_graphs结构）
def save_connected_components(components_graphs, output_dir):
    """
    将连通分量保存到文件中
    在输出目录下创建connected_components目录，保存各个blockn.txt文件
    """
    # 创建保存连通分量的目录
    # components_dir = os.path.join(output_dir, "connected_components")
    components_dir = output_dir
    os.makedirs(components_dir, exist_ok=True)

    # 遍历每个连通分量并保存（components_graphs是字典，值为每个分量的信息）
    for i, graph in enumerate(components_graphs.values(), 1):
        # 构建文件名：block1.txt, block2.txt, ...
        filename = f"block{i}.txt"
        filepath = os.path.join(components_dir, filename)

        # 写入文件
        with open(filepath, 'w') as f:
            # 写入表头
            f.write("Node1\tNode2\tWeight\n")

            # 写入边信息（代码2返回的graph包含'edges'键，值为边列表[(u, v, weight), ...]）
            edges = graph['edges']
            for u, v, weight in edges:
                f.write(f"{u}\t{v}\t{weight:.15f}\n")

        print(f"已保存连通分量 {i} 到 {filepath}，包含 {len(edges)} 条边")

    return components_dir


# 函数计时装饰器
def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"{func.__name__} 运行时间: {run_time:.2f} 秒")
        return result, run_time

    return wrapper


# 使用装饰器包装需要计时的函数
build_snp_sparse_matrix = time_function(get_init_matrix.build_snp_sparse_matrix)
generate_haplotype_data = time_function(generate_haplotype_data)
process_sparse_matrices = time_function(process_sparse_matrices)
main_get_Snps_Map_reads_block = time_function(get_Snps_Map_reads_block.main)
process_blocks = time_function(process_blocks)
merge_files = time_function(merge_vcf_extractHAIRS.merge_files)


def process_phasing(vcf_file, extractHAIRS_file, overall_start_time=None):
    """
    执行phasing处理的核心函数
    """
    matrix_file_dir = os.path.join(os.path.dirname(vcf_file))

    if overall_start_time is None:
        overall_start_time = time.time()

    function_times = {}

    # 初始化稀疏矩阵
    print("开始生成碱基矩阵...")
    result, matrix_time = build_snp_sparse_matrix(vcf_file, extractHAIRS_file)
    matrix = result['sparse_matrix']
    pos_map = result['position_to_column']
    fragment_map = result['fragment_to_index']
    function_times["build_snp_sparse_matrix"] = matrix_time

    # 初始化haplotype
    print("开始生成初始化单体型...")
    (init_haplotype, all_nodes), haplotype_time = generate_haplotype_data(vcf_file, matrix, pos_map, extractHAIRS_file)
    function_times["generate_haplotype_data"] = haplotype_time

    # 保存矩阵
    print("开始生成三元矩阵...")
    matrix_triple, matrix_process_time = process_sparse_matrices(matrix, pos_map, init_haplotype)
    function_times["process_sparse_matrices"] = matrix_process_time

    print("开始构建子图...")
    components_graphs, graph_time = main_get_Snps_Map_reads_block(matrix_triple, fragment_map, all_nodes)
    function_times["get_Snps_Map_reads_block.main"] = graph_time

    # 新增：保存连通分量（在获取components_graphs后调用）
    haplotype_output_dir = os.path.join(os.path.dirname(vcf_file), 'phasing_output')  # 确保目录已定义
    save_output_dir = os.path.join(os.path.dirname(vcf_file), 'connected_components')
    # print("开始保存连通分量...")
    # save_connected_components(components_graphs, save_output_dir)

    print("开始处理模块...")
    # 修改这里：传入vcf_file参数
    _, process_blocks_time = process_blocks(components_graphs, init_haplotype, matrix_triple, pos_map, fragment_map,
                                            haplotype_output_dir, vcf_file=vcf_file)
    function_times["process_blocks"] = process_blocks_time

    return function_times


# 主函数
def main():
    # 检查是否使用--nf参数（合并模式）
    if len(sys.argv) > 1 and sys.argv[1] == '--nf':
        # 合并模式
        if len(sys.argv) != 6:
            sys.exit(
                'Usage: python3 %s --nf <vcf1.tab> <extractHAIRS1.txt> <vcf2.tab> <extractHAIRS2.txt>' % (sys.argv[0]))

        vcf1_file = sys.argv[2]
        extractHAIRS1_file = sys.argv[3]
        vcf2_file = sys.argv[4]
        extractHAIRS2_file = sys.argv[5]

        # 记录开始的时间
        overall_start_time = time.time()
        function_times = {}

        # 调用合并函数
        print("=" * 60)
        print("开始合并VCF和extractHAIRS文件...")
        print("=" * 60)

        try:
            # 设置输出文件路径（在第一个文件的目录下）
            output_dir = os.path.dirname(vcf1_file)
            merged_vcf_file = os.path.join(output_dir, "merged.vcf")
            merged_extractHAIRS_file = os.path.join(output_dir, "merged_extractHAIRS.txt")

            # 执行合并
            (merged_vcf, merged_extractHAIRS), merge_time = merge_files(
                vcf1_file, extractHAIRS1_file,
                vcf2_file, extractHAIRS2_file,
                merged_vcf_file, merged_extractHAIRS_file
            )

            function_times["merge_files"] = merge_time

            print(f"\n合并完成:")
            print(f"  合并后的VCF文件: {merged_vcf}")
            print(f"  合并后的extractHAIRS文件: {merged_extractHAIRS}")
            print("=" * 60)
            print("\n开始处理合并后的文件...")
            print("=" * 60)

            # 处理合并后的文件
            phasing_times = process_phasing(merged_vcf, merged_extractHAIRS, overall_start_time)
            function_times.update(phasing_times)

        except Exception as e:
            print(f"合并文件时出错: {e}")
            sys.exit(1)

        # 计算总运行时间
        overall_end_time = time.time()
        overall_run_time = overall_end_time - overall_start_time

        # 保存运行时间
        run_time_file = os.path.join(output_dir, 'run_time.txt')
        with open(run_time_file, 'w') as f:
            f.write(f"从合并文件开始到程序结束，总运行时间为：{overall_run_time:.2f} 秒\n")
            f.write("\n各函数运行时间:\n")
            for func_name, time_taken in function_times.items():
                f.write(f"{func_name}: {time_taken:.2f} 秒\n")

        print(f"\n从合并文件开始到程序结束，总运行时间为：{overall_run_time:.2f} 秒")
        print(f"各函数运行时间已保存到 {run_time_file}")

        # 打印函数运行时间的汇总
        print("\n各函数运行时间汇总:")
        for func_name, time_taken in function_times.items():
            print(f"{func_name}: {time_taken:.2f} 秒")

    else:
        # 原始模式（单文件处理）
        if len(sys.argv) != 3:
            sys.exit(
                'Usage: python3 %s <vcf.tab> <extractHAIRS.txt>\n       或: python3 %s --nf <vcf1.tab> <extractHAIRS1.txt> <vcf2.tab> <extractHAIRS2.txt>' % (
                    sys.argv[0], sys.argv[0]))

        vcf_file = sys.argv[1]
        extractHAIRS_file = sys.argv[2]

        # 记录开始的时间
        overall_start_time = time.time()

        # 执行phasing处理
        function_times = process_phasing(vcf_file, extractHAIRS_file, overall_start_time)

        # 计算总运行时间
        overall_end_time = time.time()
        overall_run_time = overall_end_time - overall_start_time

        # 保存运行时间
        run_time_file = os.path.join(os.path.dirname(vcf_file), 'run_time.txt')
        with open(run_time_file, 'w') as f:
            f.write(f"从过滤开始到程序结束，总运行时间为：{overall_run_time:.2f} 秒\n")
            f.write("\n各函数运行时间:\n")
            for func_name, time_taken in function_times.items():
                f.write(f"{func_name}: {time_taken:.2f} 秒\n")

        print(f"\n从过滤开始到程序结束，总运行时间为：{overall_run_time:.2f} 秒")
        print(f"各函数运行时间已保存到 {run_time_file}")

        # 打印函数运行时间的汇总
        print("\n各函数运行时间汇总:")
        for func_name, time_taken in function_times.items():
            print(f"{func_name}: {time_taken:.2f} 秒")


if __name__ == '__main__':
    main()