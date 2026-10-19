#!/usr/bin/env python3
"""
合并phasing数据文件脚本
功能：合并两个hapcut2输出文件及其对应的VCF文件，保持位点索引对应关系
核心处理：当两个文件存在相同位点但ref/alt（参考碱基/变异碱基）不同时，
         根据质量值（QUAL）选择保留质量更高的记录，并同步更新hapcut2文件中的索引
"""

import sys
from pathlib import Path  # 用于路径处理，跨平台兼容
import re  # 正则表达式模块，本脚本预留用于可能的字符串匹配


def parse_vcf_line(line):
    """解析VCF文件中的一行数据，提取关键信息

    VCF文件格式说明：每一行以\t分隔，包含染色体、位置、ID、ref、alt、质量等信息
    标准格式为：#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	...

    参数：
        line: VCF文件中的一行字符串（非表头行）
    返回：
        字典，包含解析后的关键信息：
            'chrom': 染色体名称
            'pos': 位点位置（整数）
            'id': 位点ID
            'ref': 参考碱基
            'alt': 变异碱基
            'qual': 质量值（浮点数，.表示0.0）
            'line': 原始行字符串（用于后续写入文件）
        若解析失败（如字段数不足），返回None
    """
    parts = line.split('\t')  # 按制表符分割字段
    if len(parts) < 6:  # VCF数据行至少需要6个字段（至QUAL）
        return None

    return {
        'chrom': parts[0],  # 第0列：染色体名称
        'pos': int(parts[1]),  # 第1列：位点位置（转换为整数）
        'id': parts[2],  # 第2列：位点ID（通常为rs号或.）
        'ref': parts[3],  # 第3列：参考碱基（REF）
        'alt': parts[4],  # 第4列：变异碱基（ALT）
        'qual': float(parts[5]) if parts[5] != '.' else 0.0,  # 第5列：质量值（QUAL）
        'line': line  # 原始行（保留用于输出）
    }


def read_vcf_file_detailed(vcf_file):
    """详细读取VCF文件，提取位点信息、表头和完整记录

    参数：
        vcf_file: VCF文件路径（字符串）
    返回：
        元组 (positions, header_lines, data_records)
            positions: 位点位置列表（整数，按文件顺序）
            header_lines: 表头行列表（所有以#开头的行）
            data_records: 解析后的位点记录列表（每个元素为parse_vcf_line返回的字典）
    """
    positions = []  # 存储所有位点的位置（用于索引映射）
    header_lines = []  # 存储表头行（保留用于输出合并后的VCF）
    data_records = []  # 存储解析后的位点记录

    with open(vcf_file, 'r') as f:
        for line in f:
            line = line.strip()  # 去除首尾空白（包括换行符）
            if line.startswith('#'):
                # 表头行（如##fileformat=VCFv4.2、#CHROM等）
                header_lines.append(line)
            else:
                # 数据行：解析并存储
                record = parse_vcf_line(line)
                if record:  # 解析成功才添加
                    positions.append(record['pos'])
                    data_records.append(record)

    return positions, header_lines, data_records


def read_hapcut2_file_detailed(hapcut2_file):
    """详细读取hapcut2输出文件，解析区块、reads及SNP数据

    hapcut2输出格式说明（简化）：
    每行格式通常为：[block_id] [read_id] [snp_idx1] [allele1] [snp_idx2] [allele2] ... [quality_str]
    其中snp_idx对应VCF文件中的位点索引（从1开始），allele为该read在对应位点的碱基，
    quality_str为质量值字符串（可选，与allele长度对应）

    参数：
        hapcut2_file: hapcut2输出文件路径（字符串）
    返回：
        元组 (blocks, line_count)
            blocks: 列表，每个元素为一个区块字典
                'block_id': 区块ID（原始文件中的第一个字段）
                'read_id': 读段ID（原始文件中的第二个字段）
                'snp_data': 列表，每个元素为元组 (snp_idx, allele_str, quality_str)
                            snp_idx: SNP索引（整数）
                            allele_str: 该索引开始的连续碱基字符串
                            quality_str: 对应碱基的质量值字符串（与allele_str长度相同，可能为None）
                'original_line': 原始行字符串（用于调试或备份）
            line_count: 文件的行数
    """
    blocks = []  # 存储所有解析后的区块
    line_count = 0  # 统计文件行数

    with open(hapcut2_file, 'r') as f:
        for line in f:
            line_count += 1  # 计数行数
            line = line.strip()
            if line:  # 跳过空行
                parts = line.split()  # 按空格分割字段
                if len(parts) >= 3:  # 至少需要block_id、read_id、1对snp数据
                    block_data = {
                        'block_id': parts[0],
                        'read_id': parts[1],
                        'snp_data': [],  # 存储(索引, 碱基字符串, 质量字符串)元组
                        'original_line': line
                    }

                    # 解析SNP索引和碱基信息
                    i = 2  # 从第三个字段开始是SNP数据
                    quality_str = None  # 质量值字符串（可能不存在）

                    # 检查最后一个字段是否为质量值（SNP数据字段数为偶数，质量值为额外字段）
                    # 例如：若parts[2:]有3个元素（奇数），则最后一个为质量值
                    if len(parts) > 2 and (len(parts) - 2) % 2 == 1:
                        quality_str = parts[-1]  # 最后一个字段为质量值
                        parts_to_process = parts[2:-1]  # 处理除质量值外的SNP数据
                    else:
                        parts_to_process = parts[2:]  # 无质量值，全部为SNP数据

                    # 处理SNP索引和碱基对（每两个字段一组：索引+碱基）
                    quality_idx = 0  # 质量值字符串的当前位置
                    for j in range(0, len(parts_to_process), 2):
                        if j + 1 < len(parts_to_process):  # 确保有对应的碱基字段
                            snp_idx = int(parts_to_process[j])  # SNP索引（整数）
                            allele_str = parts_to_process[j + 1]  # 碱基字符串

                            # 提取对应的质量值（与碱基字符串长度相同）
                            if quality_str and quality_idx < len(quality_str):
                                # 从当前位置截取与碱基长度相同的质量值
                                qual_chars = quality_str[quality_idx:quality_idx + len(allele_str)]
                                quality_idx += len(allele_str)  # 更新质量值位置
                            else:
                                qual_chars = None  # 无质量值或已读完

                            block_data['snp_data'].append((snp_idx, allele_str, qual_chars))

                    blocks.append(block_data)

    return blocks, line_count


def merge_vcf_with_conflict_resolution(vcf1_data, vcf2_data):
    """合并两个VCF文件的记录，处理同一位点的ref/alt冲突（核心逻辑）

    冲突处理规则：
    1. 若两位点位置相同且ref/alt一致：保留质量值（QUAL）更高的记录
    2. 若两位点位置相同但ref/alt不同：保留质量值更高的记录，记录被删除的位点
    3. 仅在一个文件中存在的位点：直接保留

    参数：
        vcf1_data: 第一个VCF文件的数据（read_vcf_file_detailed的返回值）
        vcf2_data: 第二个VCF文件的数据（同上）
    返回：
        元组 (sorted_positions, headers, sorted_lines, removed_from_file1, removed_from_file2)
            sorted_positions: 合并后按位置排序的位点列表
            headers: 合并后的表头（使用第一个文件的表头）
            sorted_lines: 合并后按位置排序的原始数据行
            removed_from_file1: 从第一个文件删除的位点位置集合
            removed_from_file2: 从第二个文件删除的位点位置集合
    """
    positions1, headers1, records1 = vcf1_data
    positions2, headers2, records2 = vcf2_data

    # 创建位置到记录的映射（便于快速查找同一位点）
    pos_to_record1 = {r['pos']: r for r in records1}
    pos_to_record2 = {r['pos']: r for r in records2}

    # 合并记录，处理冲突
    merged_records = {}  # 存储合并后的记录（key: 位置，value: 记录）
    removed_from_file1 = set()  # 记录从文件1删除的位置
    removed_from_file2 = set()  # 记录从文件2删除的位置

    # 所有位点的并集（处理所有可能的位点）
    all_positions = set(pos_to_record1.keys()) | set(pos_to_record2.keys())

    for pos in all_positions:
        rec1 = pos_to_record1.get(pos)  # 文件1中的记录（可能为None）
        rec2 = pos_to_record2.get(pos)  # 文件2中的记录（可能为None）

        if rec1 and rec2:
            # 情况1：两位点位置相同
            if rec1['ref'] == rec2['ref'] and rec1['alt'] == rec2['alt']:
                # 1.1 ref/alt相同：保留质量更高的
                if rec1['qual'] >= rec2['qual']:
                    merged_records[pos] = rec1
                else:
                    merged_records[pos] = rec2
            else:
                # 1.2 ref/alt不同（冲突）：保留质量更高的，记录删除信息
                if rec1['qual'] >= rec2['qual']:
                    merged_records[pos] = rec1
                    removed_from_file2.add(pos)  # 文件2的该位点被删除
                    print(
                        f"位置 {pos}: ref/alt不同，保留文件1 (质量={rec1['qual']:.2f})，删除文件2 (质量={rec2['qual']:.2f})")
                else:
                    merged_records[pos] = rec2
                    removed_from_file1.add(pos)  # 文件1的该位点被删除
                    print(
                        f"位置 {pos}: ref/alt不同，保留文件2 (质量={rec2['qual']:.2f})，删除文件1 (质量={rec1['qual']:.2f})")
        elif rec1:
            # 情况2：仅文件1有该位点
            merged_records[pos] = rec1
        elif rec2:
            # 情况3：仅文件2有该位点
            merged_records[pos] = rec2

    # 按位置排序（确保VCF文件位点有序）
    sorted_positions = sorted(merged_records.keys())
    sorted_records = [merged_records[pos] for pos in sorted_positions]
    sorted_lines = [r['line'] for r in sorted_records]  # 提取原始行用于输出

    return sorted_positions, headers1, sorted_lines, removed_from_file1, removed_from_file2


def create_position_mapping_with_removals(old_positions1, old_positions2, new_positions,
                                          removed_from_file1, removed_from_file2):
    """创建旧VCF文件的位点索引到新合并文件索引的映射（考虑被删除的位点）

    说明：VCF文件中的位点索引从1开始，hapcut2文件中的snp_idx对应此索引。
          当某些位点被删除后，需重新映射索引以保持对应关系。

    参数：
        old_positions1: 第一个原VCF文件的位点位置列表
        old_positions2: 第二个原VCF文件的位点位置列表
        new_positions: 合并后VCF文件的位点位置列表（已排序）
        removed_from_file1: 从第一个文件删除的位点位置集合
        removed_from_file2: 从第二个文件删除的位点位置集合
    返回：
        元组 (mapping1, mapping2, removed_indices1, removed_indices2)
            mapping1: 文件1的旧索引到新索引的映射（key: 旧索引，value: 新索引）
            mapping2: 文件2的旧索引到新索引的映射（同上）
            removed_indices1: 文件1中被删除位点的旧索引集合
            removed_indices2: 文件2中被删除位点的旧索引集合
    """
    # 创建位置到新索引的映射（新索引从1开始，与合并后的VCF顺序一致）
    pos_to_new_idx = {pos: idx + 1 for idx, pos in enumerate(new_positions)}

    # 处理文件1的索引映射
    mapping1 = {}  # 旧索引 -> 新索引
    removed_indices1 = set()  # 被删除的旧索引
    for old_idx, pos in enumerate(old_positions1, 1):  # 旧索引从1开始
        if pos in removed_from_file1:
            # 该位置被删除，记录其旧索引
            removed_indices1.add(old_idx)
        elif pos in pos_to_new_idx:
            # 该位置保留，映射到新索引
            mapping1[old_idx] = pos_to_new_idx[pos]

    # 处理文件2的索引映射（逻辑同上）
    mapping2 = {}
    removed_indices2 = set()
    for old_idx, pos in enumerate(old_positions2, 1):
        if pos in removed_from_file2:
            removed_indices2.add(old_idx)
        elif pos in pos_to_new_idx:
            mapping2[old_idx] = pos_to_new_idx[pos]

    return mapping1, mapping2, removed_indices1, removed_indices2


def update_hapcut2_with_removals(blocks, index_mapping, removed_indices):
    """更新hapcut2文件中的SNP索引，删除被移除位点的碱基及对应质量值

    核心逻辑：
    1. 对于每个区块中的SNP数据，检查每个碱基对应的原始索引
    2. 若索引被删除：剔除该碱基及质量值，分割连续片段
    3. 若索引保留：映射到新索引，合并连续片段（保持连续索引的碱基为一个整体）

    参数：
        blocks: 原始hapcut2文件解析后的区块列表（read_hapcut2_file_detailed的返回）
        index_mapping: 旧索引到新索引的映射（create_position_mapping_with_removals返回）
        removed_indices: 被删除的旧索引集合（同上）
    返回：
        更新后的区块列表（仅保留有效数据的区块）
    """
    updated_blocks = []

    for block in blocks:
        new_block = block.copy()  # 复制原始区块信息
        new_snp_data = []  # 存储更新后的SNP数据

        # 处理每个SNP数据项（原始索引、碱基字符串、质量字符串）
        for snp_idx, allele_str, qual_str in block['snp_data']:
            # 用于临时存储连续的碱基片段（避免因删除索引导致的非连续）
            current_segment = []  # 当前连续片段的碱基
            current_qual = []  # 当前片段的质量值
            current_start_idx = None  # 当前片段的起始新索引

            # 逐个处理碱基（allele_str中的每个字符对应一个索引）
            for i, char in enumerate(allele_str):
                # 计算该碱基对应的原始索引（snp_idx为起始索引，i为偏移）
                actual_idx = snp_idx + i

                # 检查该索引是否有效（未被删除且存在映射）
                if actual_idx not in removed_indices and actual_idx in index_mapping:
                    new_idx = index_mapping[actual_idx]  # 映射到新索引

                    if current_start_idx is None:
                        # 开始新的连续片段
                        current_start_idx = new_idx
                        current_segment = [char]
                        # 同步添加质量值（若存在且索引有效）
                        if qual_str and i < len(qual_str):
                            current_qual = [qual_str[i]]
                    elif new_idx == current_start_idx + len(current_segment):
                        # 新索引与当前片段连续（+1），继续添加到当前片段
                        current_segment.append(char)
                        if qual_str and i < len(qual_str):
                            current_qual.append(qual_str[i])
                    else:
                        # 新索引与当前片段不连续，保存当前片段并开始新片段
                        if current_segment:
                            qual_part = ''.join(current_qual) if current_qual else None
                            new_snp_data.append((current_start_idx, ''.join(current_segment), qual_part))
                        current_start_idx = new_idx
                        current_segment = [char]
                        current_qual = [qual_str[i]] if qual_str and i < len(qual_str) else []
                else:
                    # 索引被删除，结束当前片段（若有）
                    if current_segment:
                        qual_part = ''.join(current_qual) if current_qual else None
                        new_snp_data.append((current_start_idx, ''.join(current_segment), qual_part))
                        current_segment = []
                        current_qual = []
                        current_start_idx = None

            # 保存最后一个未处理的连续片段
            if current_segment:
                qual_part = ''.join(current_qual) if current_qual else None
                new_snp_data.append((current_start_idx, ''.join(current_segment), qual_part))

        new_block['snp_data'] = new_snp_data
        if new_snp_data:  # 仅保留有有效数据的区块（过滤空区块）
            updated_blocks.append(new_block)

    return updated_blocks


def write_merged_vcf(output_file, headers, data_lines):
    """将合并后的VCF数据写入输出文件

    参数：
        output_file: 输出文件路径（Path对象或字符串）
        headers: 表头行列表
        data_lines: 数据行列表（已排序）
    """
    with open(output_file, 'w') as f:
        # 写入表头
        for header in headers:
            f.write(header + '\n')
        # 写入数据行
        for line in data_lines:
            f.write(line + '\n')
    print(f"合并后的VCF文件已保存到: {output_file}")


def write_merged_hapcut2(output_file, blocks1, blocks2):
    """将合并后的hapcut2数据写入输出文件

    格式说明：保持hapcut2原格式，每行包含区块ID、read ID、SNP索引-碱基对、质量值（可选）

    参数：
        output_file: 输出文件路径（Path对象或字符串）
        blocks1: 第一个hapcut2文件更新后的区块列表
        blocks2: 第二个hapcut2文件更新后的区块列表
    """
    with open(output_file, 'w') as f:
        # 写入第一个文件的区块
        for block in blocks1:
            if block['snp_data']:  # 仅写入有数据的区块
                # 实际区块数为SNP数据的长度（每个元素代表一个连续片段）
                actual_block_count = len(block['snp_data'])
                line_parts = [str(actual_block_count), block['read_id']]  # 前两个字段
                quality_parts = []  # 收集所有质量值片段

                # 拼接SNP索引和碱基对
                for idx, allele_str, qual_str in block['snp_data']:
                    line_parts.extend([str(idx), allele_str])
                    if qual_str:  # 若有质量值，收集起来
                        quality_parts.append(qual_str)

                # 若有质量值，添加到行尾（保持原格式）
                if quality_parts:
                    line_parts.append(''.join(quality_parts))

                # 写入该行
                f.write(' '.join(line_parts) + '\n')

        # 写入第二个文件的区块（逻辑同上）
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

    print(f"合并后的hapcut2文件已保存到: {output_file}")


def merge_files(vcf1_file, hapcut2_1_file, vcf2_file, hapcut2_2_file, output_vcf_file, output_hapcut2_file):
    """
    合并两个VCF文件和对应的hapcut2文件的主函数

    参数：
        vcf1_file: 第一个VCF文件路径
        hapcut2_1_file: 第一个hapcut2文件路径
        vcf2_file: 第二个VCF文件路径
        hapcut2_2_file: 第二个hapcut2文件路径
        output_vcf_file: 输出的合并后VCF文件路径
        output_hapcut2_file: 输出的合并后hapcut2文件路径

    返回：
        元组 (bool, int): 成功返回(True, 第一个文件的行数)，失败返回(False, 0)
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

        # 3. 创建旧索引到新索引的映射（考虑被删除的位点）
        print("\n创建索引映射...")
        mapping1, mapping2, removed_indices1, removed_indices2 = \
            create_position_mapping_with_removals(positions1, positions2, merged_positions,
                                                  removed_from_file1, removed_from_file2)

        # 4. 读取并解析两个hapcut2文件
        print("读取hapcut2文件...")
        blocks1, hapcut2_1_line_count = read_hapcut2_file_detailed(hapcut2_1_file)  # 获取第一个文件的行数
        blocks2, _ = read_hapcut2_file_detailed(hapcut2_2_file)

        print(f"hapcut2文件1包含 {len(blocks1)} 个blocks，共 {hapcut2_1_line_count} 行")
        print(f"hapcut2文件2包含 {len(blocks2)} 个blocks")

        # 5. 更新hapcut2文件的索引，删除冲突位点对应的碱基
        print("更新hapcut2文件索引，删除冲突位点...")
        updated_blocks1 = update_hapcut2_with_removals(blocks1, mapping1, removed_indices1)
        updated_blocks2 = update_hapcut2_with_removals(blocks2, mapping2, removed_indices2)

        print(f"更新后文件1包含 {len(updated_blocks1)} 个有效blocks")
        print(f"更新后文件2包含 {len(updated_blocks2)} 个有效blocks")

        # 6. 写入合并后的文件
        print("\n写入合并后的文件...")
        write_merged_vcf(output_vcf_file, merged_headers, merged_lines)
        write_merged_hapcut2(output_hapcut2_file, updated_blocks1, updated_blocks2)

        # 7. 验证结果（输出统计信息）
        print("\n验证结果:")
        print(f"原始VCF文件总位点数: {len(positions1) + len(positions2)}")
        print(f"合并后VCF文件位点数: {len(merged_positions)}")
        print(f"去重和冲突处理后减少的位点数: {len(positions1) + len(positions2) - len(merged_positions)}")

        # 显示部分被删除的位点（便于检查）
        if removed_from_file1:
            print(f"\n从文件1删除的位点位置（前5个）: {list(removed_from_file1)[:5]}")
        if removed_from_file2:
            print(f"从文件2删除的位点位置（前5个）: {list(removed_from_file2)[:5]}")

        print("\n文件合并完成！")
        return True, hapcut2_1_line_count  # 返回成功状态和第一个文件的行数

    except Exception as e:
        print(f"合并过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0


def main():
    """独立运行时的主函数：执行文件合并的完整流程"""
    # 输入文件路径（可根据实际情况修改）
    hapcut2_porec_file = r"/mnt/e/my_work/phasing/phasing_v11_GPU/phasing_v13_GPU_porec/data_WT_porec/data_WT_porec/mhc/data_snps/hapcut2_output_porec_filtered.txt"
    vcf_porec_file = r"/mnt/e/my_work/phasing/phasing_v11_GPU/phasing_v13_GPU_porec/data_WT_porec/data_WT_porec/mhc/data_snps/chr6_snp_filter_porec.vcf"
    hapcut2_wt_file = r"/mnt/e/my_work/phasing/phasing_v11_GPU/phasing_v13_GPU_porec/data_WT_porec/data_WT_porec/mhc/data_snps/hapcut2_output_WT.txt"
    vcf_wt_file = r"/mnt/e/my_work/phasing/phasing_v11_GPU/phasing_v13_GPU_porec/data_WT_porec/data_WT_porec/mhc/data_snps/chr6_snp_filter_WT.vcf"

    # 输出文件路径（与输入文件同目录）
    output_dir = Path(hapcut2_porec_file).parent  # 获取输入文件的父目录
    merged_hapcut2_file = output_dir / "hapcut2_output_merged.txt"  # 合并后的hapcut2文件
    merged_vcf_file = output_dir / "chr6_snp_filter_merged.vcf"  # 合并后的VCF文件

    print("开始处理文件...")

    # 调用合并函数
    success, line_count = merge_files(
        vcf_porec_file, hapcut2_porec_file,
        vcf_wt_file, hapcut2_wt_file,
        str(merged_vcf_file), str(merged_hapcut2_file)
    )

    if success:
        print(f"\n处理完成！第一个hapcut2文件包含 {line_count} 行")
    else:
        print("\n处理失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()
