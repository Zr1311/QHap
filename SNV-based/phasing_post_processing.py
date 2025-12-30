#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from collections import defaultdict, Counter, OrderedDict
from scipy.sparse import csr_matrix
import os
import warnings
from numba import jit, prange
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
from typing import Dict, List, Tuple, Set
import hashlib

warnings.filterwarnings('ignore')

# 设置全局随机种子确保一致性
np.random.seed(42)


class PhasingPostProcessorOptimized:
    """优化的后处理器，确保结果的确定性和可重复性"""

    def __init__(self, vcf_file, fragment_reads_file, phased_vcf_file, matrix_data, pos_map):
        """
        初始化后处理器

        参数:
        - vcf_file: 原始VCF文件路径
        - fragment_reads_file: fragment reads文件路径
        - phased_vcf_file: 初始phasing结果VCF文件路径
        - matrix_data: 稀疏矩阵数据
        - pos_map: 位置映射字典
        """
        self.vcf_file = vcf_file
        self.fragment_reads_file = fragment_reads_file
        self.phased_vcf_file = phased_vcf_file
        self.matrix_data = matrix_data
        self.pos_map = pos_map

        # 创建反向映射以加速查找 - 使用OrderedDict保证顺序
        self.idx_to_pos = OrderedDict((idx, pos) for pos, idx in sorted(pos_map.items()))

        # 顺序读取数据，避免并行带来的不确定性
        self.vcf_data = self._read_vcf_data_optimized()
        self.reads_data = self._parse_fragment_reads_optimized()
        self.phasing_result = self._read_phased_vcf_optimized()

        # 预处理reads数据为更高效的格式
        self._preprocess_reads_data()

        # 初始化统计数据
        self.position_stats = {}
        self._initialize_position_stats()

    def _read_vcf_data_optimized(self) -> Dict:
        """优化的VCF文件读取 - 使用OrderedDict保证顺序"""
        vcf_data = OrderedDict()
        with open(self.vcf_file, 'r') as f:
            # 跳过header行
            lines = [line for line in f if not line.startswith('#')]

        # 按位置排序处理
        parsed_lines = []
        for line in lines:
            fields = line.strip().split('\t')
            pos = int(fields[1])
            parsed_lines.append((pos, fields))

        # 排序确保顺序一致
        parsed_lines.sort(key=lambda x: x[0])

        for pos, fields in parsed_lines:
            vcf_data[pos] = {
                'chrom': fields[0],
                'ref': fields[3],
                'alt': fields[4],
                'qual': fields[5],
                'genotype': fields[9].split(':')[0] if len(fields) > 9 else '0/1'
            }
        return vcf_data

    def _parse_fragment_reads_optimized(self) -> List[Dict]:
        """优化的fragment reads文件解析 - 保持原始顺序"""
        reads_data = []

        with open(self.fragment_reads_file, 'r') as f:
            lines = f.readlines()

        # 批量处理 - 保持原始顺序
        for line_num, line in enumerate(lines):
            parts = line.strip().split()
            if not parts:
                continue

            fragment_id = parts[1]
            n_blocks = int(parts[0])
            blocks = parts[2:2 + 2 * n_blocks]
            quality_str = parts[-1] if len(parts) > 2 + 2 * n_blocks else ''

            # 使用列表保持顺序
            fragment_positions = []
            fragment_alleles = []
            fragment_qualities = []

            for i in range(n_blocks):
                start_pos_idx = int(blocks[2 * i]) - 1
                allele_str = blocks[2 * i + 1]

                for j, allele in enumerate(allele_str):
                    if allele in ['0', '1']:
                        pos_idx = start_pos_idx + j
                        # 使用预建的反向映射
                        actual_pos = self.idx_to_pos.get(pos_idx)

                        if actual_pos:
                            fragment_positions.append(actual_pos)
                            fragment_alleles.append(int(allele))
                            fragment_qualities.append(
                                ord(quality_str[j]) - 33 if j < len(quality_str) else 20
                            )

            if fragment_positions:
                reads_data.append({
                    'fragment_id': fragment_id,
                    'positions': np.array(fragment_positions, dtype=np.int32),
                    'alleles': np.array(fragment_alleles, dtype=np.int8),
                    'qualities': np.array(fragment_qualities, dtype=np.int8),
                    'line_num': line_num
                })

        return reads_data

    def _read_phased_vcf_optimized(self) -> Dict:
        """优化的phased VCF读取 - 使用OrderedDict"""
        phasing_result = OrderedDict()

        with open(self.phased_vcf_file, 'r') as f:
            lines = [line for line in f if not line.startswith('#')]

        # 解析并排序
        parsed_lines = []
        for line in lines:
            fields = line.strip().split('\t')
            pos = int(fields[1])
            parsed_lines.append((pos, fields))

        parsed_lines.sort(key=lambda x: x[0])

        for pos, fields in parsed_lines:
            format_fields = fields[8].split(':')
            sample_values = fields[9].split(':')

            # 使用字典查找代替列表index
            format_dict = {field: idx for idx, field in enumerate(format_fields)}

            if 'GT' in format_dict:
                gt_idx = format_dict['GT']
                gt = sample_values[gt_idx]

                if '|' in gt:  # phased
                    alleles = gt.split('|')
                    ps_idx = format_dict.get('PS', -1)
                    phase_set = int(sample_values[ps_idx]) if ps_idx >= 0 else 0

                    phasing_result[pos] = {
                        'hap1': int(alleles[0]),
                        'hap2': int(alleles[1]),
                        'phase_set': phase_set,
                        'original_gt': gt
                    }

        return phasing_result

    def _preprocess_reads_data(self):
        """预处理reads数据以加速后续查询 - 使用OrderedDict保证顺序"""
        # 创建位置到reads的索引 - 使用OrderedDict
        self.position_to_reads = defaultdict(list)
        for read_idx, read in enumerate(self.reads_data):
            for pos in read['positions']:
                self.position_to_reads[int(pos)].append(read_idx)

        # 对每个位置的reads列表排序以保证顺序
        for pos in self.position_to_reads:
            self.position_to_reads[pos].sort()

        # 创建read pairs索引用于phase switch检测
        self.read_pairs = defaultdict(list)
        for read_idx, read in enumerate(self.reads_data):
            positions = read['positions']
            for i in range(len(positions) - 1):
                pos1, pos2 = int(positions[i]), int(positions[i + 1])
                if read_idx not in self.read_pairs[(pos1, pos2)]:
                    self.read_pairs[(pos1, pos2)].append(read_idx)

        # 对每个pair的reads列表排序
        for key in self.read_pairs:
            self.read_pairs[key].sort()

    def _initialize_position_stats(self):
        """初始化位置统计数据"""
        for pos in sorted(self.phasing_result.keys()):
            self.position_stats[pos] = {
                'support_0': 0,
                'support_1': 0,
                'quality_0': [],
                'quality_1': [],
                'reads': [],
                'conflicts': 0
            }

    def calculate_read_support_optimized(self):
        """优化的reads支持计算 - 确保顺序一致"""
        # 按顺序处理每个位点
        for pos in sorted(self.position_to_reads.keys()):
            read_indices = self.position_to_reads[pos]
            if pos not in self.phasing_result:
                continue

            stats = self.position_stats[pos]

            for read_idx in sorted(read_indices):
                read = self.reads_data[read_idx]
                # 使用NumPy的where函数快速查找
                pos_idx = np.where(read['positions'] == pos)[0]

                if len(pos_idx) > 0:
                    idx = pos_idx[0]
                    allele = read['alleles'][idx]
                    quality = read['qualities'][idx]

                    if allele == 0:
                        stats['support_0'] += 1
                        stats['quality_0'].append(quality)
                    else:
                        stats['support_1'] += 1
                        stats['quality_1'].append(quality)

                    if read['fragment_id'] not in stats['reads']:
                        stats['reads'].append(read['fragment_id'])

    @staticmethod
    @jit(nopython=True)
    def _compute_phase_support(allele1, allele2, phase1_hap1, phase2_hap1, phase2_hap2):
        """使用Numba加速的相位支持计算"""
        if phase1_hap1 == allele1:
            expected_allele2 = phase2_hap1
        else:
            expected_allele2 = phase2_hap2

        return allele2 == expected_allele2

    def detect_phase_switches_optimized(self) -> List[Dict]:
        """优化的相位切换检测 - 确保顺序一致"""
        phase_switches = []

        # 按phase block组织位点
        blocks = defaultdict(list)
        for pos in sorted(self.phasing_result.keys()):
            phase_info = self.phasing_result[pos]
            blocks[phase_info['phase_set']].append(pos)

        # 顺序处理每个block
        for block_id in sorted(blocks.keys()):
            positions = sorted(blocks[block_id])

            for i in range(len(positions) - 1):
                pos1, pos2 = positions[i], positions[i + 1]

                # 使用预建的索引快速查找相关reads
                same_phase_support = 0
                switch_phase_support = 0

                # 获取连接这两个位点的reads
                connecting_reads = self.read_pairs.get((pos1, pos2), [])

                for read_idx in sorted(connecting_reads):
                    read = self.reads_data[read_idx]

                    # 使用NumPy快速查找
                    idx1 = np.where(read['positions'] == pos1)[0]
                    idx2 = np.where(read['positions'] == pos2)[0]

                    if len(idx1) > 0 and len(idx2) > 0:
                        allele1 = read['alleles'][idx1[0]]
                        allele2 = read['alleles'][idx2[0]]

                        phase1 = self.phasing_result[pos1]
                        phase2 = self.phasing_result[pos2]

                        # 检查是否支持当前phasing
                        if self._compute_phase_support(
                                allele1, allele2,
                                phase1['hap1'], phase2['hap1'], phase2['hap2']
                        ):
                            same_phase_support += 1
                        else:
                            switch_phase_support += 1

                # 评估是否为相位切换
                if switch_phase_support > same_phase_support * 1.5:
                    total_support = same_phase_support + switch_phase_support
                    if total_support > 0:
                        phase_switches.append({
                            'position': pos2,
                            'same_support': same_phase_support,
                            'switch_support': switch_phase_support,
                            'confidence': switch_phase_support / (total_support + 0.01)
                        })

        return phase_switches

    def correct_phase_switches(self, phase_switches: List[Dict]) -> List[int]:
        """修正检测到的相位切换错误"""
        # 按位置和置信度排序确保一致性
        sorted_switches = sorted(phase_switches,
                                 key=lambda x: (x['position'], -x['confidence']))

        high_confidence_switches = [s for s in sorted_switches if s['confidence'] > 0.7]
        corrected_positions = []

        for switch in high_confidence_switches:
            pos = switch['position']
            if pos in self.phasing_result:
                # 交换alleles
                phase_info = self.phasing_result[pos]
                phase_info['hap1'], phase_info['hap2'] = phase_info['hap2'], phase_info['hap1']
                corrected_positions.append(pos)

        return sorted(corrected_positions)

    def calculate_phasing_confidence_vectorized(self) -> Dict[int, float]:
        """向量化的置信度计算 - 确保顺序一致"""
        confidence_scores = OrderedDict()

        # 批量处理所有位点 - 排序
        positions = sorted(list(self.phasing_result.keys()))
        confidences = np.zeros(len(positions))

        for i, pos in enumerate(positions):
            if pos not in self.position_stats:
                confidences[i] = 0.5
                continue

            stats = self.position_stats[pos]
            total_support = stats['support_0'] + stats['support_1']

            if total_support == 0:
                confidences[i] = 0.5
                continue

            # 使用NumPy计算
            qual_0 = np.array(sorted(stats['quality_0'])) if stats['quality_0'] else np.array([0])
            qual_1 = np.array(sorted(stats['quality_1'])) if stats['quality_1'] else np.array([0])

            avg_qual_0 = np.mean(qual_0)
            avg_qual_1 = np.mean(qual_1)

            support_ratio = max(stats['support_0'], stats['support_1']) / total_support
            quality_factor = (avg_qual_0 + avg_qual_1) / 80

            confidences[i] = min(0.7 * support_ratio + 0.3 * quality_factor, 1.0)

        # 转换回有序字典
        for i, pos in enumerate(positions):
            confidence_scores[pos] = confidences[i]

        return confidence_scores

    def merge_phase_blocks_optimized(self) -> List[Tuple[int, int]]:
        """优化的phase block合并 - 确保顺序一致"""
        blocks = defaultdict(list)
        for pos in sorted(self.phasing_result.keys()):
            phase_info = self.phasing_result[pos]
            blocks[phase_info['phase_set']].append(pos)

        sorted_blocks = sorted(blocks.items(), key=lambda x: (min(x[1]), x[0]))
        merged_blocks = []

        # 预计算block间的连接
        block_connections = OrderedDict()
        for i in range(len(sorted_blocks) - 1):
            block1_id, block1_positions = sorted_blocks[i]
            block2_id, block2_positions = sorted_blocks[i + 1]

            block1_set = set(block1_positions)
            block2_set = set(block2_positions)

            # 快速查找连接reads - 按顺序处理
            connecting_reads = []
            for read_idx, read in enumerate(self.reads_data):
                positions_set = set(read['positions'].tolist())
                if positions_set & block1_set and positions_set & block2_set:
                    connecting_reads.append(read)

                if len(connecting_reads) >= 10:
                    break

            block_connections[(block1_id, block2_id)] = connecting_reads

        # 评估合并 - 按顺序处理
        for (block1_id, block2_id), connecting_reads in block_connections.items():
            if len(connecting_reads) >= 3:
                consistent_count = len(connecting_reads)

                if consistent_count >= 3:
                    merged_blocks.append((block1_id, block2_id))

                    # 更新phase set
                    block2_positions = sorted([pos for pos, info in self.phasing_result.items()
                                               if info['phase_set'] == block2_id])
                    for pos in block2_positions:
                        self.phasing_result[pos]['phase_set'] = block1_id

        return merged_blocks

    def filter_low_confidence_positions(self, confidence_scores: Dict[int, float],
                                        threshold: float = 0.3) -> List[int]:
        """过滤低置信度的位点 - 确保顺序一致"""
        filtered_positions = []

        for pos in sorted(confidence_scores.keys()):
            if confidence_scores[pos] < threshold:
                filtered_positions.append(pos)

        return filtered_positions

    def write_improved_vcf_optimized(self, output_file: str):
        """优化的VCF文件输出"""
        # 一次性读取所有header
        with open(self.phased_vcf_file, 'r') as f:
            lines = f.readlines()

        header_lines = []
        data_start_idx = 0

        for i, line in enumerate(lines):
            if line.startswith('#'):
                header_lines.append(line.rstrip())
            else:
                data_start_idx = i
                break

        # 添加处理信息
        processing_header = '##phasing_post_processing=True'
        if processing_header not in header_lines:
            for i in range(len(header_lines) - 1, -1, -1):
                if header_lines[i].startswith('##'):
                    header_lines.insert(i + 1, processing_header)
                    break

        # 批量写入
        with open(output_file, 'w') as out_f:
            # 写入header
            out_f.write('\n'.join(header_lines) + '\n')

            # 批量处理数据行
            output_lines = []
            for line in lines[data_start_idx:]:
                fields = line.strip().split('\t')
                pos = int(fields[1])

                if pos in self.phasing_result:
                    format_fields = fields[8].split(':')
                    sample_values = fields[9].split(':')

                    format_dict = {field: idx for idx, field in enumerate(format_fields)}

                    if 'GT' in format_dict:
                        gt_idx = format_dict['GT']
                        phase_info = self.phasing_result[pos]
                        sample_values[gt_idx] = f"{phase_info['hap1']}|{phase_info['hap2']}"

                        if 'PS' in format_dict:
                            ps_idx = format_dict['PS']
                            sample_values[ps_idx] = str(phase_info['phase_set'])

                    fields[9] = ':'.join(sample_values)
                    output_lines.append('\t'.join(fields))
                else:
                    output_lines.append(line.rstrip())

            out_f.write('\n'.join(output_lines) + '\n')

    def generate_statistics_report(self, output_dir: str):
        """生成统计报告 - 确保统计结果一致"""
        report_file = os.path.join(output_dir, 'post_processing_report.txt')

        # 预计算统计数据 - 使用排序
        blocks = defaultdict(list)
        for pos in sorted(self.phasing_result.keys()):
            phase_info = self.phasing_result[pos]
            blocks[phase_info['phase_set']].append(pos)

        block_sizes = sorted([len(b) for b in blocks.values()])
        confidence_scores = self.calculate_phasing_confidence_vectorized()
        confidence_array = np.array([confidence_scores[k] for k in sorted(confidence_scores.keys())])

        with open(report_file, 'w') as f:
            f.write("Phasing Post-Processing Report\n")
            f.write("=" * 60 + "\n\n")

            f.write("Basic Statistics:\n")
            f.write(f"Total phased positions: {len(self.phasing_result)}\n")
            f.write(f"Total reads: {len(self.reads_data)}\n")
            f.write(f"Number of phase blocks: {len(blocks)}\n")
            f.write(f"Average block size: {np.mean(block_sizes):.2f}\n")
            f.write(f"Largest block size: {np.max(block_sizes) if block_sizes else 0}\n")

            f.write(f"\nConfidence Statistics:\n")
            f.write(f"Average confidence: {np.mean(confidence_array):.3f}\n")
            f.write(f"Positions with confidence > 0.9: {np.sum(confidence_array > 0.9)}\n")
            f.write(f"Positions with confidence < 0.5: {np.sum(confidence_array < 0.5)}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("Post-processing completed successfully.\n")

    def run_post_processing(self) -> Dict:
        """运行完整的后处理流程（确定性版本）"""
        print("开始后处理（确定性版本）...")

        # 1. 计算reads支持
        print("  1. 计算reads支持情况...")
        self.calculate_read_support_optimized()

        # 2. 检测相位切换
        print("  2. 检测可能的相位切换错误...")
        phase_switches = self.detect_phase_switches_optimized()
        print(f"     检测到 {len(phase_switches)} 个可能的相位切换")

        # 3. 修正相位切换
        print("  3. 修正相位切换错误...")
        corrected_positions = self.correct_phase_switches(phase_switches)
        print(f"     修正了 {len(corrected_positions)} 个位点")

        # 4. 计算置信度分数
        print("  4. 计算phasing置信度...")
        confidence_scores = self.calculate_phasing_confidence_vectorized()
        avg_confidence = np.mean(list(confidence_scores.values()))
        print(f"     平均置信度: {avg_confidence:.3f}")

        # 5. 尝试合并phase blocks
        print("  5. 尝试合并phase blocks...")
        merged_blocks = self.merge_phase_blocks_optimized()
        print(f"     合并了 {len(merged_blocks)} 对blocks")

        # 6. 过滤低置信度位点
        print("  6. 过滤低置信度位点...")
        filtered_positions = self.filter_low_confidence_positions(confidence_scores)
        print(f"     标记了 {len(filtered_positions)} 个低置信度位点")

        print("后处理完成！")

        return {
            'phase_switches_detected': len(phase_switches),
            'positions_corrected': len(corrected_positions),
            'average_confidence': float(avg_confidence),
            'blocks_merged': len(merged_blocks),
            'low_confidence_positions': len(filtered_positions)
        }


def apply_phasing_post_processing_optimized(vcf_file, fragment_reads_file, phased_vcf_file,
                                            matrix_data, pos_map, output_dir):
    """
    应用优化的后处理来提高phasing准确率 - 确定性版本

    参数:
    - vcf_file: 原始VCF文件路径
    - fragment_reads_file: fragment reads文件路径
    - phased_vcf_file: 初始phasing结果文件路径
    - matrix_data: 稀疏矩阵数据
    - pos_map: 位置映射
    - output_dir: 输出目录

    返回:
    - 处理统计信息字典
    """

    # 创建优化的后处理器实例
    processor = PhasingPostProcessorOptimized(
        vcf_file,
        fragment_reads_file,
        phased_vcf_file,
        matrix_data,
        pos_map
    )

    # 运行后处理
    stats = processor.run_post_processing()

    # 输出改进后的VCF文件
    improved_vcf_file = os.path.join(output_dir, 'phased_improved.vcf')
    processor.write_improved_vcf_optimized(improved_vcf_file)

    # 生成统计报告
    processor.generate_statistics_report(output_dir)

    print(f"\n改进后的VCF文件已保存到: {improved_vcf_file}")
    print(f"统计报告已保存到: {os.path.join(output_dir, 'post_processing_report.txt')}")

    return stats


# 保留向后兼容性
def apply_phasing_post_processing(vcf_file, fragment_reads_file, phased_vcf_file,
                                  matrix_data, pos_map, output_dir):
    """向后兼容的接口，调用优化版本"""
    return apply_phasing_post_processing_optimized(
        vcf_file, fragment_reads_file, phased_vcf_file,
        matrix_data, pos_map, output_dir
    )