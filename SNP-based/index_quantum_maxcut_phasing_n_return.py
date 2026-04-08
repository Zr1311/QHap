#!/usr/bin/env python3
import sys
import pysam
import os
import time
import get_Snps_Map_n_block
import get_init_matrix
from init_haplotype import generate_haplotype_data
from get_matrix import process_sparse_matrices
import merge_phasing_files


def save_connected_components(components_graphs, output_dir):
    components_dir = os.path.join(output_dir, "connected_components")
    os.makedirs(components_dir, exist_ok=True)
    for i, graph in enumerate(components_graphs, 1):
        filename = f"block{i}.txt"
        filepath = os.path.join(components_dir, filename)
        with open(filepath, 'w') as f:
            f.write("Node1\tNode2\tWeight\n")
            if hasattr(graph, 'edges'):
                for u, v, data in graph.edges(data=True):
                    weight = data.get('weight', 0.0)
                    f.write(f"{u}\t{v}\t{weight:.15f}\n")
            else:
                nodes, edges = graph
                for u, v, weight in edges:
                    f.write(f"{u}\t{v}\t{weight:.15f}\n")
    return components_dir


def main():
    if len(sys.argv) < 3:
        sys.exit(
            'Usage: python3 %s [--mindquantum] [--no-post] <vcf.tab> <fragment_reads.txt>\n'
            'OR: python3 %s [--mindquantum] [--no-post] --porec <vcf.tab> <fragment_reads.txt> '
            '<porec_vcf.tab> <porec_fragment_reads.txt> a=<value>' % (
                sys.argv[0], sys.argv[0]))

    # ── Parse flags ──
    argv = list(sys.argv)

    use_mindquantum = False
    if '--mindquantum' in argv:
        use_mindquantum = True
        argv.remove('--mindquantum')
        print("[MindQuantum BSB mode enabled]")

    enable_post_processing = True
    if '--no-post' in argv:
        enable_post_processing = False
        argv.remove('--no-post')
        print("Note: Post-processing disabled")

    # ── Select process_blocks implementation ──
    if use_mindquantum:
        from mindquantum_BSB_get_haplotype import process_blocks
    else:
        from BSB_get_haplotype import process_blocks

    if len(argv) > 1 and argv[1] == '--porec':
        if len(argv) != 7:
            sys.exit(
                'Usage: python3 %s [--mindquantum] --porec <vcf.tab> <fragment_reads.txt> '
                '<porec_vcf.tab> <porec_fragment_reads.txt> a=<value>' % argv[0])

        vcf1_file = argv[2]
        fragment_reads1_file = argv[3]
        vcf2_file = argv[4]
        fragment_reads2_file = argv[5]
        a_param = argv[6]

        if not a_param.startswith('a='):
            sys.exit("Error: Invalid a parameter format")
        try:
            a = float(a_param.split('=')[1])
            print(f"Using a value: {a}")
        except ValueError:
            sys.exit("Error: a must be a number")

        output_dir = os.path.dirname(vcf1_file) or os.getcwd()
        print("Detected --porec mode, merging files...")

        merge_start_time = time.time()
        merged_vcf_file = os.path.join(output_dir, "merged.tab")
        merged_phasing_file = os.path.join(output_dir, "merged_fragment_reads.txt")

        success, fragment_reads_1_line_count = merge_phasing_files.merge_files(
            vcf1_file, fragment_reads1_file,
            vcf2_file, fragment_reads2_file,
            merged_vcf_file, merged_phasing_file
        )

        if not success:
            sys.exit("File merging failed")

        merge_time = time.time() - merge_start_time
        vcf_file = merged_vcf_file
        fragment_reads_file = merged_phasing_file
    else:
        if len(argv) != 3:
            sys.exit('Usage: python3 %s [--mindquantum] [--no-post] <vcf.tab> <fragment_reads.txt>' % argv[0])
        vcf_file = argv[1]
        fragment_reads_file = argv[2]
        merge_time = 0
        fragment_reads_1_line_count = 0
        a = 1

    output_dir = os.path.dirname(vcf_file)
    print(f"Output directory: {output_dir}")
    print(f"Post-processing: {'Enabled' if enable_post_processing else 'Disabled'}")
    print(f"Solver: {'MindQuantum BSB' if use_mindquantum else 'tabu_SB_gpu'}")

    start_time = time.time()
    timing_results = []

    if merge_time > 0:
        timing_results.append(("Merge files", merge_time))

    func_start = time.time()
    result = get_init_matrix.build_snp_sparse_matrix(vcf_file, fragment_reads_file)
    matrix = result['sparse_matrix']
    correct_rates = result['sparse_correct_rate']
    pos_map = result['position_to_column']
    fragment_map = result['fragment_to_index']
    init_matrix_time = time.time() - func_start
    timing_results.append(("Initialize matrix", init_matrix_time))
    print("Matrix initialized")

    func_start = time.time()
    init_haplotype, all_nodes = generate_haplotype_data(vcf_file, matrix, pos_map)
    generate_haplotype_time = time.time() - func_start
    timing_results.append(("Initialize haplotype", generate_haplotype_time))
    print("Haplotype initialized")

    func_start = time.time()
    matrix_weight = process_sparse_matrices(matrix, pos_map, init_haplotype, correct_rates)
    process_matrices_time = time.time() - func_start
    timing_results.append(("Process matrices", process_matrices_time))
    print("Weight matrix generated")

    func_start = time.time()
    components_graphs = get_Snps_Map_n_block.main(
        matrix_weight, pos_map,
        fragment_reads_1_line_count=fragment_reads_1_line_count, a=a
    )
    get_components_time = time.time() - func_start
    timing_results.append(("Get SNP blocks", get_components_time))
    print("SNP mapping and blocks processed")

    func_start = time.time()
    phasing_output_dir = os.path.join(os.path.dirname(vcf_file), 'phasing_output')
    process_blocks(
        components_graphs, init_haplotype, phasing_output_dir, vcf_file,
        fragment_reads_file=fragment_reads_file, matrix_data=matrix,
        pos_map=pos_map, enable_post_processing=enable_post_processing
    )
    process_blocks_time = time.time() - func_start
    timing_results.append(("Process blocks", process_blocks_time))
    print("Block processing and VCF output completed")

    end_time = time.time()
    total_run_time = end_time - start_time

    run_time_file = os.path.join(os.path.dirname(vcf_file), 'run_time.txt')
    with open(run_time_file, 'w') as f:
        f.write(f"Runtime statistics\n")
        f.write("=" * 60 + "\n")
        f.write(f"Solver: {'MindQuantum BSB' if use_mindquantum else 'tabu_SB_gpu'}\n")
        if '--porec' in sys.argv:
            f.write("Input (merged mode):\n")
            f.write(
                f"  VCF1: {sys.argv[2]}\n  Reads1: {sys.argv[3]}\n  VCF2: {sys.argv[4]}\n  Reads2: {sys.argv[5]}\n  a: {a}\n")
            f.write(f"  Merged VCF: {vcf_file}\n  Merged reads: {fragment_reads_file}\n")
        else:
            f.write("Input:\n")
            f.write(f"  VCF: {vcf_file}\n  Reads: {fragment_reads_file}\n  a: {a}\n")
        f.write(f"  Post-processing: {'Enabled' if enable_post_processing else 'Disabled'}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total runtime: {total_run_time:.2f}s\n")
        f.write("=" * 60 + "\n")

    print(f"\nRuntime details saved to: {run_time_file}")
    print(f"Phased VCF saved to: {os.path.join(phasing_output_dir, 'phased.vcf')}")


if __name__ == '__main__':
    main()
