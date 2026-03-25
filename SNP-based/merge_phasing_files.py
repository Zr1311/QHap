#!/usr/bin/env python3
"""
合并phasing数据文件脚本
功能：合并两个phasing输出文件及其对应的VCF文件，保持位点索引对应关系
核心处理：当两个文件存在相同位点但ref/alt（参考碱基/变异碱基）不同时，
         根据质量值（QUAL）选择保留质量更高的记录，并同步更新phasing文件中的索引
"""

import sys
from pathlib import Path
import re


def parse_vcf_line(line):
    """解析VCF文件中的一行数据，提取关键信息"""
    parts = line.split('\t')
    if len(parts) < 6:
        return None

    return {
        'chrom': parts[0],
        'pos': int(parts[1]),
        'id': parts[2],
        'ref': parts[3],
        'alt': parts[4],
        'qual': float(parts[5]) if parts[5] != '.' else 0.0,
        'line': line
    }


def read_vcf_file_detailed(vcf_file):
    """详细读取VCF文件，提取位点信息、表头和完整记录"""
    positions = []
    header_lines = []
    data_records = []

    with open(vcf_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                header_lines.append(line)
            else:
                record = parse_vcf_line(line)
                if record:
                    positions.append(record['pos'])
                    data_records.append(record)

    return positions, header_lines, data_records


def read_phasing_file_detailed(phasing_file):
    """详细读取phasing输出文件，解析区块、reads及SNP数据"""
    blocks = []
    line_count = 0

    with open(phasing_file, 'r') as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    block_data = {
                        'block_id': parts[0],
                        'read_id': parts[1],
                        'snp_data': [],
                        'original_line': line
                    }

                    i = 2
                    quality_str = None

                    if len(parts) > 2 and (len(parts) - 2) % 2 == 1:
                        quality_str = parts[-1]
                        parts_to_process = parts[2:-1]
                    else:
                        parts_to_process = parts[2:]

                    quality_idx = 0
                    for j in range(0, len(parts_to_process), 2):
                        if j + 1 < len(parts_to_process):
                            snp_idx = int(parts_to_process[j])
                            allele_str = parts_to_process[j + 1]

                            if quality_str and quality_idx < len(quality_str):
                                qual_chars = quality_str[quality_idx:quality_idx + len(allele_str)]
                                quality_idx += len(allele_str)
                            else:
                                qual_chars = None

                            block_data['snp_data'].append((snp_idx, allele_str, qual_chars))

                    blocks.append(block_data)

    return blocks, line_count


def merge_vcf_with_conflict_resolution(vcf1_data, vcf2_data):
    """合并两个VCF文件的记录，处理同一位点的ref/alt冲突"""
    positions1, headers1, records1 = vcf1_data
    positions2, headers2, records2 = vcf2_data

    pos_to_record1 = {r['pos']: r for r in records1}
    pos_to_record2 = {r['pos']: r for r in records2}

    merged_records = {}
    removed_from_file1 = set()
    removed_from_file2 = set()

    all_positions = set(pos_to_record1.keys()) | set(pos_to_record2.keys())

    for pos in all_positions:
        rec1 = pos_to_record1.get(pos)
        rec2 = pos_to_record2.get(pos)

        if rec1 and rec2:
            if rec1['ref'] == rec2['ref'] and rec1['alt'] == rec2['alt']:
                if rec1['qual'] >= rec2['qual']:
                    merged_records[pos] = rec1
                else:
                    merged_records[pos] = rec2
            else:
                if rec1['qual'] >= rec2['qual']:
                    merged_records[pos] = rec1
                    removed_from_file2.add(pos)
                    print(
                        f"位置 {pos}: ref/alt不同，保留文件1 (质量={rec1['qual']:.2f})，删除文件2 (质量={rec2['qual']:.2f})")
                else:
                    merged_records[pos] = rec2
                    removed_from_file1.add(pos)
                    print(
                        f"位置 {pos}: ref/alt不同，保留文件2 (质量={rec2['qual']:.2f})，删除文件1 (质量={rec1['qual']:.2f})")
        elif rec1:
            merged_records[pos] = rec1
        elif rec2:
            merged_records[pos] = rec2

    sorted_positions = sorted(merged_records.keys())
    sorted_records = [merged_records[pos] for pos in sorted_positions]
    sorted_lines = [r['line'] for r in sorted_records]

    return sorted_positions, headers1, sorted_lines, removed_from_file1, removed_from_file2


def create_position_mapping_with_removals(old_positions1, old_positions2, new_positions,
                                          removed_from_file1, removed_from_file2):
    """创建旧VCF文件的位点索引到新合并文件索引的映射"""
    pos_to_new_idx = {pos: idx + 1 for idx, pos in enumerate(new_positions)}

    mapping1 = {}
    removed_indices1 = set()
    for old_idx, pos in enumerate(old_positions1, 1):
        if pos in removed_from_file1:
            removed_indices1.add(old_idx)
        elif pos in pos_to_new_idx:
            mapping1[old_idx] = pos_to_new_idx[pos]

    mapping2 = {}
    removed_indices2 = set()
    for old_idx, pos in enumerate(old_positions2, 1):
        if pos in removed_from_file2:
            removed_indices2.add(old_idx)
        elif pos in pos_to_new_idx:
            mapping2[old_idx] = pos_to_new_idx[pos]

    return mapping1, mapping2, removed_indices1, removed_indices2


def update_phasing_with_removals(blocks, index_mapping, removed_indices):
    """更新phasing文件中的SNP索引，删除被移除位点的碱基及对应质量值"""
    updated_blocks = []

    for block in blocks:
        new_block = block.copy()
        new_snp_data = []

        for snp_idx, allele_str, qual_str in block['snp_data']:
            current_segment = []
            current_qual = []
            current_start_idx = None

            for i, char in enumerate(allele_str):
                actual_idx = snp_idx + i

                if actual_idx not in removed_indices and actual_idx in index_mapping:
                    new_idx = index_mapping[actual_idx]

                    if current_start_idx is None:
                        current_start_idx = new_idx
                        current_segment = [char]
                        if qual_str and i < len(qual_str):
                            current_qual = [qual_str[i]]
                    elif new_idx == current_start_idx + len(current_segment):
                        current_segment.append(char)
                        if qual_str and i < len(qual_str):
                            current_qual.append(qual_str[i])
                    else:
                        if current_segment:
                            qual_part = ''.join(current_qual) if current_qual else None
                            new_snp_data.append((current_start_idx, ''.join(current_segment), qual_part))
                        current_start_idx = new_idx
                        current_segment = [char]
                        current_qual = [qual_str[i]] if qual_str and i < len(qual_str) else []
                else:
                    if current_segment:
                        qual_part = ''.join(current_qual) if current_qual else None
                        new_snp_data.append((current_start_idx, ''.join(current_segment), qual_part))
                        current_segment = []
                        current_qual = []
                        current_start_idx = None

            if current_segment:
                qual_part = ''.join(current_qual) if current_qual else None
                new_snp_data.append((current_start_idx, ''.join(current_segment), qual_part))

        new_block['snp_data'] = new_snp_data
        if new_snp_data:
            updated_blocks.append(new_block)

    return updated_blocks


def write_merged_vcf(output_file, headers, data_lines):
    """将合并后的VCF数据写入输出文件"""
    with open(output_file, 'w') as f:
        for header in headers:
            f.write(header + '\n')
        for line in data_lines:
            f.write(line + '\n')
    print(f"合并后的VCF文件已保存到: {output_file}")


def write_merged_phasing(output_file, blocks1, blocks2):
    """将合并后的phasing数据写入输出文件"""
    with open(output_file, 'w') as f:
        for block in blocks1:
            if block['snp_data']:
                actual_block_count = len(block['snp_data'])
                line_parts = [str(actual_block_count), block['read_id']]
                quality_parts = []

                for idx, allele_str, qual_str in block['snp_data']:
                    line_parts.extend([str(idx), allele_str])
                    if qual_str:
                        quality_parts.append(qual_str)

                if quality_parts:
                    line_parts.append(''.join(quality_parts))

                f.write(' '.join(line_parts) + '\n')

        for block in blocks2:
            if block['snp_data']:
                actual_block_count = len(block['snp_data'])
                line_parts = [str(actual_block_count), block['read_id']]
                quality_parts = []

                for idx, allele_str, qual_str in block['snp_data']:
                    line_parts.extend([str(idx), allele_str])
                    if qual_str:
                        quality_parts.append(qual_str)

                if quality_parts:
                    line_parts.append(''.join(quality_parts))

                f.write(' '.join(line_parts) + '\n')

    print(f"合并后的phasing文件已保存到: {output_file}")


def merge_files(vcf1_file, phasing_1_file, vcf2_file, phasing_2_file, output_vcf_file, output_phasing_file):
    """
    合并两个VCF文件和对应的phasing文件的主函数
    """
    try:
        print("开始合并文件...")

        # 1. 读取并解析两个VCF文件
        print("读取VCF文件...")
        vcf1_data = read_vcf_file_detailed(vcf1_file)
        vcf2_data = read_vcf_file_detailed(vcf2_file)

        positions1, _, records1 = vcf1_data
        positions2, _, records2 = vcf2_data

        print(f"VCF文件1包含 {len(positions1)} 个位点")
        print(f"VCF文件2包含 {len(positions2)} 个位点")

        # 2. 合并VCF文件，处理ref/alt冲突
        print("\n合并VCF文件，处理ref/alt冲突...")
        merged_positions, merged_headers, merged_lines, removed_from_file1, removed_from_file2 = \
            merge_vcf_with_conflict_resolution(vcf1_data, vcf2_data)

        print(f"合并后的VCF文件包含 {len(merged_positions)} 个位点")
        print(f"从文件1删除 {len(removed_from_file1)} 个冲突位点")
        print(f"从文件2删除 {len(removed_from_file2)} 个冲突位点")

        # 3. 创建旧索引到新索引的映射
        print("\n创建索引映射...")
        mapping1, mapping2, removed_indices1, removed_indices2 = \
            create_position_mapping_with_removals(positions1, positions2, merged_positions,
                                                  removed_from_file1, removed_from_file2)

        # 4. 读取并解析两个phasing文件
        print("读取phasing文件...")
        blocks1, phasing_1_line_count = read_phasing_file_detailed(phasing_1_file)
        blocks2, _ = read_phasing_file_detailed(phasing_2_file)

        print(f"phasing文件1包含 {len(blocks1)} 个blocks，共 {phasing_1_line_count} 行")
        print(f"phasing文件2包含 {len(blocks2)} 个blocks")

        # 5. 更新phasing文件的索引
        print("更新phasing文件索引，删除冲突位点...")
        updated_blocks1 = update_phasing_with_removals(blocks1, mapping1, removed_indices1)
        updated_blocks2 = update_phasing_with_removals(blocks2, mapping2, removed_indices2)

        print(f"更新后文件1包含 {len(updated_blocks1)} 个有效blocks")
        print(f"更新后文件2包含 {len(updated_blocks2)} 个有效blocks")

        # 6. 写入合并后的文件
        print("\n写入合并后的文件...")
        write_merged_vcf(output_vcf_file, merged_headers, merged_lines)
        write_merged_phasing(output_phasing_file, updated_blocks1, updated_blocks2)

        print("\n文件合并完成！")
        return True, phasing_1_line_count

    except Exception as e:
        print(f"合并过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0


def main():
    """独立运行时的主函数"""
    phasing_porec_file = r"/mnt/e/my_work/phasing/phasing_v11_GPU/phasing_v13_GPU_porec/data_WT_porec/data_WT_porec/mhc/data_snps/phasing_output_porec_filtered.txt"
    vcf_porec_file = r"/mnt/e/my_work/phasing/phasing_v11_GPU/phasing_v13_GPU_porec/data_WT_porec/data_WT_porec/mhc/data_snps/chr6_snp_filter_porec.vcf"
    phasing_wt_file = r"/mnt/e/my_work/phasing/phasing_v11_GPU/phasing_v13_GPU_porec/data_WT_porec/data_WT_porec/mhc/data_snps/phasing_output_WT.txt"
    vcf_wt_file = r"/mnt/e/my_work/phasing/phasing_v11_GPU/phasing_v13_GPU_porec/data_WT_porec/data_WT_porec/mhc/data_snps/chr6_snp_filter_WT.vcf"

    output_dir = Path(phasing_porec_file).parent
    merged_phasing_file = output_dir / "phasing_output_merged.txt"
    merged_vcf_file = output_dir / "chr6_snp_filter_merged.vcf"

    print("开始处理文件...")

    success, line_count = merge_files(
        vcf_porec_file, phasing_porec_file,
        vcf_wt_file, phasing_wt_file,
        str(merged_vcf_file), str(merged_phasing_file)
    )

    if success:
        print(f"\n处理完成！第一个phasing文件包含 {line_count} 行")
    else:
        print("\n处理失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()