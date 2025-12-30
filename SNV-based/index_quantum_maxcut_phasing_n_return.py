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