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
from get_matrix import process_sparse_matrices
import get_Snps_Map_reads_block
import merge_phasing_files


def save_connected_components(components_graphs, output_dir):
    components_dir = output_dir
    os.makedirs(components_dir, exist_ok=True)

    for i, graph in enumerate(components_graphs.values(), 1):
        filename = f"block{i}.txt"
        filepath = os.path.join(components_dir, filename)

        with open(filepath, 'w') as f:
            f.write("Node1\tNode2\tWeight\n")
            edges = graph['edges']
            for u, v, weight in edges:
                f.write(f"{u}\t{v}\t{weight:.15f}\n")

    return components_dir


def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        return result, run_time

    return wrapper


build_snp_sparse_matrix = time_function(get_init_matrix.build_snp_sparse_matrix)
generate_haplotype_data = time_function(generate_haplotype_data)
process_sparse_matrices = time_function(process_sparse_matrices)
main_get_Snps_Map_reads_block = time_function(get_Snps_Map_reads_block.main)
merge_files = time_function(merge_phasing_files.merge_files)


def process_phasing(vcf_file, fragment_reads_file, overall_start_time=None, use_mindquantum=False):
    matrix_file_dir = os.path.join(os.path.dirname(vcf_file))

    if overall_start_time is None:
        overall_start_time = time.time()

    function_times = {}

    # Choose the block processor based on solver selection
    if use_mindquantum:
        from mindquantum_BSB_get_haplotype_and_combine import process_blocks as _process_blocks
        print("[MindQuantum BSB mode enabled]")
    else:
        from BSB_get_haplotype_and_combine import process_blocks as _process_blocks

    _process_blocks_timed = time_function(_process_blocks)

    print("Generating base matrix...")
    result, matrix_time = build_snp_sparse_matrix(vcf_file, fragment_reads_file)
    matrix = result['sparse_matrix']
    pos_map = result['position_to_column']
    fragment_map = result['fragment_to_index']
    function_times["build_snp_sparse_matrix"] = matrix_time

    print("Generating initial haplotype...")
    (init_haplotype, all_nodes), haplotype_time = generate_haplotype_data(vcf_file, matrix, pos_map, fragment_reads_file)
    function_times["generate_haplotype_data"] = haplotype_time

    print("Generating triple matrix...")
    matrix_triple, matrix_process_time = process_sparse_matrices(matrix, pos_map, init_haplotype)
    function_times["process_sparse_matrices"] = matrix_process_time

    print("Building subgraphs...")
    components_graphs, graph_time = main_get_Snps_Map_reads_block(matrix_triple, fragment_map, all_nodes)
    function_times["get_Snps_Map_reads_block.main"] = graph_time

    haplotype_output_dir = os.path.join(os.path.dirname(vcf_file), 'phasing_output')
    save_output_dir = os.path.join(os.path.dirname(vcf_file), 'connected_components')

    print("Processing blocks...")
    _, process_blocks_time = _process_blocks_timed(
        components_graphs, init_haplotype, matrix_triple, pos_map, fragment_map,
        haplotype_output_dir, vcf_file=vcf_file
    )
    function_times["process_blocks"] = process_blocks_time

    return function_times


def main():
    # ── Parse --mindquantum flag ──
    use_mindquantum = False
    argv = list(sys.argv)
    if '--mindquantum' in argv:
        use_mindquantum = True
        argv.remove('--mindquantum')

    if len(argv) > 1 and argv[1] == '--porec':
        if len(argv) != 6:
            sys.exit(
                'Usage: python3 %s [--mindquantum] --porec <vcf1.tab> <fragment_reads1.txt> '
                '<vcf2.tab> <fragment_reads2.txt>' % (argv[0])
            )

        vcf1_file = argv[2]
        fragment_reads1_file = argv[3]
        vcf2_file = argv[4]
        fragment_reads2_file = argv[5]

        overall_start_time = time.time()
        function_times = {}

        print("Merging VCF and fragment reads files...")

        try:
            output_dir = os.path.dirname(vcf1_file)
            merged_vcf_file = os.path.join(output_dir, "merged.tab")
            merged_phasing_file = os.path.join(output_dir, "merged_fragment_reads.txt")

            merge_result, merge_time = merge_files(
                vcf1_file, fragment_reads1_file,
                vcf2_file, fragment_reads2_file,
                merged_vcf_file, merged_phasing_file
            )

            merged_vcf, merged_fragment_reads = merged_vcf_file, merged_phasing_file

            if not merge_result:
                sys.exit(1)

            function_times["merge_files"] = merge_time

            print("Processing merged files...")
            phasing_times = process_phasing(
                merged_vcf, merged_fragment_reads, overall_start_time,
                use_mindquantum=use_mindquantum
            )
            function_times.update(phasing_times)

        except Exception as e:
            sys.exit(1)

        overall_end_time = time.time()
        overall_run_time = overall_end_time - overall_start_time

        run_time_file = os.path.join(output_dir, 'run_time.txt')
        with open(run_time_file, 'w') as f:
            f.write(f"Total runtime from merging to completion: {overall_run_time:.2f} seconds\n")
            if use_mindquantum:
                f.write("Solver: MindQuantum BSB\n")
            f.write("\nFunction runtimes:\n")
            for func_name, time_taken in function_times.items():
                f.write(f"{func_name}: {time_taken:.2f} seconds\n")

    else:
        if len(argv) != 3:
            sys.exit(
                'Usage: python3 %s [--mindquantum] <vcf.tab> <fragment_reads.txt>\n'
                '       or: python3 %s [--mindquantum] --porec <vcf1.tab> <fragment_reads1.txt> '
                '<vcf2.tab> <fragment_reads2.txt>' % (argv[0], argv[0])
            )

        vcf_file = argv[1]
        fragment_reads_file = argv[2]

        overall_start_time = time.time()
        function_times = process_phasing(
            vcf_file, fragment_reads_file, overall_start_time,
            use_mindquantum=use_mindquantum
        )
        overall_end_time = time.time()
        overall_run_time = overall_end_time - overall_start_time

        run_time_file = os.path.join(os.path.dirname(vcf_file), 'run_time.txt')
        with open(run_time_file, 'w') as f:
            f.write(f"Total runtime from filtering to completion: {overall_run_time:.2f} seconds\n")
            if use_mindquantum:
                f.write("Solver: MindQuantum BSB\n")
            f.write("\nFunction runtimes:\n")
            for func_name, time_taken in function_times.items():
                f.write(f"{func_name}: {time_taken:.2f} seconds\n")


if __name__ == '__main__':
    main()
