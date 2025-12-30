#!/usr/bin/env python3
"""
高度优化的 extractHAIRS 工具 Python 实现
使用多进程并行处理、内存池、批量I/O等优化技术
支持 Pore-C 数据的 fragment 合并（使用代码2的处理流程）
"""

import sys
import os
import argparse
import logging
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque
import numpy as np
import pysam
import pyfaidx
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial, lru_cache
import pickle
import mmap
import gc
from array import array
import queue
import threading
import struct
from dataclasses import dataclass
import numba
from numba import jit, njit
import ctypes

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 常量
BSIZE = 500
QV_OFFSET = 33
MIN_BASE_QUALITY = 13
MIN_MAPPING_QUALITY = 20
MAX_INSERT_SIZE = 1000
MIN_INSERT_SIZE = 0
BATCH_SIZE = 50000  # 增大批处理大小
WRITE_BUFFER_SIZE = 100000  # 增大写缓冲
CHUNK_SIZE = 1000000  # 染色体分块大小
POREC_BATCH_SIZE = 10000  # Pore-C批处理大小


@dataclass
class ProcessingConfig:
    """处理配置数据类"""
    min_base_quality: int
    min_mapping_quality: int
    max_insert_size: int
    min_insert_size: int
    qv_offset: int
    data_type: int
    new_format: bool
    single_reads: bool
    porec_mode: bool
    pe_only: bool
    use_supplementary: bool
    hom: bool
    indels: bool


class CompactVariant:
    """内存优化的变体表示"""
    __slots__ = ['pos', 'ref', 'alt', 'a1', 'a2', 'is_het', 'is_indel', 'id']

    def __init__(self, position: int, ref_allele: str, alt_allele: str,
                 genotype: str, variant_id: int):
        self.pos = position
        self.ref = ref_allele
        self.alt = alt_allele
        self.id = variant_id
        self.is_het = False
        self.is_indel = False
        self.a1 = ""
        self.a2 = ""

        # 快速解析基因型
        if '/' in genotype or '|' in genotype:
            sep = '/' if '/' in genotype else '|'
            gt = genotype.split(':')[0].split(sep)

            if len(gt) == 2:
                if (gt[0] == '0' and gt[1] == '1') or (gt[0] == '1' and gt[1] == '0'):
                    self.is_het = True
                    self.a1 = self.ref
                    self.a2 = self.alt.split(',')[0]
                    self.is_indel = len(self.a1) != len(self.a2)


class CompactFragment:
    """代码2的Fragment类 - 用于Pore-C模式"""
    __slots__ = ['read_id', 'variant_ids', 'alleles', 'qualities', 'paired',
                 'mate_position', 'insert_size', 'barcode']

    def __init__(self, read_id: str):
        self.read_id = read_id
        self.variant_ids = array('i')  # int array
        self.alleles = bytearray()  # byte array for '0' or '1'
        self.qualities = array('B')  # unsigned byte array
        self.paired = False
        self.mate_position = 0
        self.insert_size = 0
        self.barcode = None

    def add_variant(self, variant_id: int, allele: str, quality: int):
        """Add a variant efficiently"""
        self.variant_ids.append(variant_id)
        self.alleles.append(ord(allele))
        self.qualities.append(min(quality, 255))

    def merge_with(self, other: 'CompactFragment'):
        """Merge another fragment into this one for Pore-C mode"""
        # Create a dictionary to track existing variants
        variant_dict = {}

        # Add existing variants
        for i in range(len(self.variant_ids)):
            var_id = self.variant_ids[i]
            variant_dict[var_id] = (self.alleles[i], self.qualities[i])

        # Add/update variants from other fragment
        for i in range(len(other.variant_ids)):
            var_id = other.variant_ids[i]
            if var_id in variant_dict:
                # Keep the one with higher quality
                if other.qualities[i] > variant_dict[var_id][1]:
                    variant_dict[var_id] = (other.alleles[i], other.qualities[i])
            else:
                variant_dict[var_id] = (other.alleles[i], other.qualities[i])

        # Sort by variant ID and rebuild arrays
        sorted_variants = sorted(variant_dict.items())

        # Clear existing arrays
        self.variant_ids = array('i')
        self.alleles = bytearray()
        self.qualities = array('B')

        # Rebuild with sorted merged data
        for var_id, (allele, quality) in sorted_variants:
            self.variant_ids.append(var_id)
            self.alleles.append(allele)
            self.qualities.append(quality)

    def to_string_fast(self, varlist: List, data_type: int, new_format: bool, single_reads: bool,
                       qv_offset: int) -> str:
        """Optimized string conversion"""
        n_vars = len(self.variant_ids)
        if n_vars == 0:
            return ""

        if n_vars < 2 and not single_reads:
            return ""

        # Count blocks efficiently
        blocks = 1
        for i in range(n_vars - 1):
            if self.variant_ids[i + 1] - self.variant_ids[i] != 1:
                blocks += 1

        # Build output using list and join (faster than string concatenation)
        parts = [str(blocks), self.read_id]

        # Add data type info
        if data_type == 2 and self.barcode:  # 10X
            parts.extend(['2', self.barcode, '-1'])
        elif new_format:
            parts.extend([str(data_type), '-1', '-1'])

        # Add variant alleles efficiently
        parts.append(str(self.variant_ids[0] + 1))
        allele_str = [chr(self.alleles[0])]

        for i in range(1, n_vars):
            if self.variant_ids[i] - self.variant_ids[i - 1] == 1:
                allele_str.append(chr(self.alleles[i]))
            else:
                parts.append(''.join(allele_str))
                parts.append(str(self.variant_ids[i] + 1))
                allele_str = [chr(self.alleles[i])]

        parts.append(''.join(allele_str))

        # Add quality scores
        qual_str = ''.join(chr(q + qv_offset) for q in self.qualities)
        parts.append(qual_str)

        return ' '.join(parts)


class FastFragment:
    """超快fragment表示，使用预分配数组 - 用于非Pore-C模式"""
    __slots__ = ['id', 'vars', 'alleles', 'quals', 'n_vars', 'capacity', 'barcode']

    def __init__(self, read_id: str, initial_capacity: int = 16):
        self.id = read_id
        self.capacity = initial_capacity
        self.vars = np.empty(initial_capacity, dtype=np.int32)
        self.alleles = np.empty(initial_capacity, dtype=np.uint8)
        self.quals = np.empty(initial_capacity, dtype=np.uint8)
        self.n_vars = 0
        self.barcode = None

    def add_variant_fast(self, var_id: int, allele: int, quality: int):
        """快速添加变体"""
        if self.n_vars >= self.capacity:
            # 动态扩容
            new_capacity = self.capacity * 2
            new_vars = np.empty(new_capacity, dtype=np.int32)
            new_alleles = np.empty(new_capacity, dtype=np.uint8)
            new_quals = np.empty(new_capacity, dtype=np.uint8)

            new_vars[:self.n_vars] = self.vars[:self.n_vars]
            new_alleles[:self.n_vars] = self.alleles[:self.n_vars]
            new_quals[:self.n_vars] = self.quals[:self.n_vars]

            self.vars = new_vars
            self.alleles = new_alleles
            self.quals = new_quals
            self.capacity = new_capacity

        self.vars[self.n_vars] = var_id
        self.alleles[self.n_vars] = allele
        self.quals[self.n_vars] = min(quality, 255)
        self.n_vars += 1


@njit
def process_cigar_fast(cigar_ops: np.ndarray, cigar_lens: np.ndarray,
                       ref_start: int, read_len: int) -> Dict[int, int]:
    """使用Numba加速的CIGAR处理"""
    ref_to_read = {}
    read_pos = 0
    ref_pos = ref_start + 1

    for i in range(len(cigar_ops)):
        op = cigar_ops[i]
        length = cigar_lens[i]

        if op in [0, 7, 8]:  # M, =, X
            for j in range(length):
                ref_to_read[ref_pos + j] = read_pos + j
            read_pos += length
            ref_pos += length
        elif op == 1:  # I
            read_pos += length
        elif op == 2:  # D
            ref_pos += length
        elif op == 4:  # S
            read_pos += length

    return ref_to_read


class PorecProcessor:
    """独立的Pore-C处理器（使用代码2的逻辑）"""

    def __init__(self, config: ProcessingConfig, variants_data: bytes,
                 variant_index: Dict, variant_positions: Dict):
        """初始化Pore-C处理器"""
        self.config = config
        self.variants = pickle.loads(variants_data)
        self.variant_index = variant_index
        self.variant_positions = variant_positions

    def get_porec_fragment_id(self, read_name: str) -> str:
        """Extract Pore-C fragment ID from read name"""
        if ':' in read_name:
            return read_name.split(':')[0]
        return read_name

    @lru_cache(maxsize=10000)
    def get_variant_range(self, chrom: str, start: int, end: int) -> List[int]:
        """Cached variant range lookup"""
        if chrom not in self.variant_positions:
            return []

        positions = self.variant_positions[chrom]
        start_idx = np.searchsorted(positions, start)
        end_idx = np.searchsorted(positions, end, side='right')

        return self.variant_index[chrom][start_idx:end_idx]

    def extract_variants_from_read(self, read: pysam.AlignedSegment,
                                   chrom: str) -> Optional[CompactFragment]:
        """Extract variants from read (代码2的逻辑)"""
        if chrom not in self.variant_index:
            return None

        # Get fragment ID for Pore-C mode
        fragment_id = self.get_porec_fragment_id(read.query_name)
        fragment = CompactFragment(fragment_id)

        # Get 10X barcode if available
        if self.config.data_type == 2 and read.has_tag('BX'):
            fragment.barcode = read.get_tag('BX')

        # Get variants in read range
        read_start = read.reference_start + 1
        read_end = read.reference_end

        variant_indices = self.get_variant_range(chrom, read_start, read_end)

        if not variant_indices:
            return None

        # Process CIGAR operations
        read_seq = read.query_sequence
        read_qual = read.query_qualities if read.query_qualities else [self.config.min_base_quality] * len(read_seq)

        # Build position mapping from reference to read coordinates
        ref_to_read = {}
        read_pos = 0
        ref_pos = read.reference_start + 1

        for op, length in read.cigartuples:
            if op in [0, 7, 8]:  # M, =, X
                for i in range(length):
                    ref_to_read[ref_pos + i] = read_pos + i
                read_pos += length
                ref_pos += length
            elif op == 1:  # I
                read_pos += length
            elif op == 2:  # D
                ref_pos += length
            elif op == 4:  # S
                read_pos += length

        # Process variants
        for var_id in variant_indices:
            variant = self.variants[var_id]

            if variant.pos in ref_to_read:
                rpos = ref_to_read[variant.pos]

                if 0 <= rpos < len(read_seq):
                    base = read_seq[rpos]
                    quality = min(read_qual[rpos], read.mapping_quality)

                    if quality >= self.config.min_base_quality:
                        allele = None
                        if base == variant.a1:
                            allele = '0'
                        elif base == variant.a2:
                            allele = '1'

                        if allele:
                            fragment.add_variant(var_id, allele, quality)

        return fragment if len(fragment.variant_ids) > 0 else None

    def process_reads_batch(self, reads: List[pysam.AlignedSegment],
                            chrom: str) -> List[CompactFragment]:
        """Process a batch of reads for Pore-C mode"""
        # Collect fragments by ID for merging
        temp_fragments = {}

        for read in reads:
            fragment = self.extract_variants_from_read(read, chrom)
            if fragment:
                frag_id = fragment.read_id
                if frag_id in temp_fragments:
                    # Merge with existing fragment
                    temp_fragments[frag_id].merge_with(fragment)
                else:
                    temp_fragments[frag_id] = fragment

        # Filter fragments that have at least 2 variants after merging
        fragments = []
        for frag_id, fragment in temp_fragments.items():
            if len(fragment.variant_ids) >= 2:
                fragments.append(fragment)

        return fragments


class ParallelBamProcessor:
    """并行BAM处理器 - 用于非Pore-C模式"""

    def __init__(self, config: ProcessingConfig, variants_data: bytes,
                 variant_index: Dict, variant_positions: Dict):
        """初始化处理器"""
        self.config = config
        # 反序列化变体数据
        self.variants = pickle.loads(variants_data)
        self.variant_index = variant_index
        self.variant_positions = variant_positions

    def process_read_batch_vectorized(self, reads: List[pysam.AlignedSegment],
                                      chrom: str) -> List[str]:
        """向量化批处理reads - 非Pore-C模式"""
        if chrom not in self.variant_positions:
            return []

        results = []
        positions = self.variant_positions[chrom]

        # 批量处理reads
        for read in reads:
            if read.mapping_quality < self.config.min_mapping_quality:
                continue

            # 获取read覆盖的变体范围
            read_start = read.reference_start + 1
            read_end = read.reference_end

            # 使用二分查找快速定位变体
            start_idx = np.searchsorted(positions, read_start)
            end_idx = np.searchsorted(positions, read_end, side='right')

            if start_idx >= end_idx:
                continue

            # 提取fragment
            fragment = self._extract_fragment_optimized(read, chrom, start_idx, end_idx)

            if fragment and fragment.n_vars > 0:
                # 直接输出
                if fragment.n_vars >= 2 or (self.config.single_reads and fragment.n_vars >= 1):
                    output = self._format_fragment_fast(fragment)
                    if output:
                        results.append(output)

        return results

    def _extract_fragment_optimized(self, read: pysam.AlignedSegment,
                                    chrom: str, start_idx: int, end_idx: int) -> Optional[FastFragment]:
        """优化的fragment提取"""
        var_indices = self.variant_index[chrom][start_idx:end_idx]
        if not var_indices:
            return None

        fragment = FastFragment(read.query_name, initial_capacity=len(var_indices))

        # 10X barcode
        if self.config.data_type == 2 and read.has_tag('BX'):
            fragment.barcode = read.get_tag('BX')

        # 构建位置映射
        read_seq = read.query_sequence
        read_qual = read.query_qualities if read.query_qualities else [self.config.min_base_quality] * len(read_seq)

        # 转换CIGAR为numpy数组以使用Numba加速
        cigar = read.cigartuples
        if not cigar:
            return None

        cigar_ops = np.array([op for op, _ in cigar], dtype=np.int32)
        cigar_lens = np.array([length for _, length in cigar], dtype=np.int32)

        ref_to_read = process_cigar_fast(cigar_ops, cigar_lens, read.reference_start, len(read_seq))

        # 处理变体
        for var_id in var_indices:
            variant = self.variants[var_id]

            if variant.pos in ref_to_read:
                rpos = ref_to_read[variant.pos]

                if 0 <= rpos < len(read_seq):
                    base = read_seq[rpos]
                    quality = min(read_qual[rpos], read.mapping_quality)

                    if quality >= self.config.min_base_quality:
                        allele = -1
                        if base == variant.a1:
                            allele = 0
                        elif base == variant.a2:
                            allele = 1

                        if allele >= 0:
                            fragment.add_variant_fast(var_id, allele, quality)

        return fragment if fragment.n_vars > 0 else None

    def _format_fragment_fast(self, fragment: FastFragment) -> str:
        """快速格式化输出"""
        if fragment.n_vars < 2 and not self.config.single_reads:
            return ""

        # 计算块数
        blocks = 1
        for i in range(fragment.n_vars - 1):
            if fragment.vars[i + 1] - fragment.vars[i] != 1:
                blocks += 1

        # 构建输出
        parts = [str(blocks), fragment.id]

        # 添加数据类型信息
        if self.config.data_type == 2 and fragment.barcode:
            parts.extend(['2', fragment.barcode, '-1'])
        elif self.config.new_format:
            parts.extend([str(self.config.data_type), '-1', '-1'])

        # 添加变体
        parts.append(str(fragment.vars[0] + 1))

        allele_str = []
        allele_str.append(str(fragment.alleles[0]))

        for i in range(1, fragment.n_vars):
            if fragment.vars[i] - fragment.vars[i - 1] == 1:
                allele_str.append(str(fragment.alleles[i]))
            else:
                parts.append(''.join(allele_str))
                parts.append(str(fragment.vars[i] + 1))
                allele_str = [str(fragment.alleles[i])]

        parts.append(''.join(allele_str))

        # 添加质量分数
        qual_str = ''.join(chr(q + self.config.qv_offset) for q in fragment.quals[:fragment.n_vars])
        parts.append(qual_str)

        return ' '.join(parts)


def process_chromosome_chunk(args):
    """处理染色体块的工作函数 - 非Pore-C模式"""
    (bam_file, chrom, start, end, config_pickle,
     variants_pickle, var_index, var_positions) = args

    config = pickle.loads(config_pickle)
    processor = ParallelBamProcessor(config, variants_pickle, var_index, var_positions)

    results = []
    batch = []

    with pysam.AlignmentFile(bam_file, 'rb', threads=2) as bamfile:
        for read in bamfile.fetch(chrom, start, end):
            # 基础过滤
            if read.is_unmapped or read.is_secondary or read.is_qcfail or read.is_duplicate:
                continue

            if not config.use_supplementary and read.is_supplementary:
                continue

            batch.append(read)

            # 批量处理
            if len(batch) >= BATCH_SIZE:
                results.extend(processor.process_read_batch_vectorized(batch, chrom))
                batch = []

        # 处理剩余
        if batch:
            results.extend(processor.process_read_batch_vectorized(batch, chrom))

    return results


def process_porec_bam_region(bam_file: str, chrom: str, config: ProcessingConfig,
                             variants_data: bytes, variant_index: Dict,
                             variant_positions: Dict, output_queue: queue.Queue):
    """Pore-C模式的BAM区域处理（代码2的逻辑）"""
    processor = PorecProcessor(config, variants_data, variant_index, variant_positions)
    porec_batch = defaultdict(list)

    with pysam.AlignmentFile(bam_file, 'rb', threads=2) as bamfile:
        for read in bamfile.fetch(chrom):
            # Apply filters
            if read.is_unmapped or read.is_secondary or read.is_qcfail or read.is_duplicate:
                continue

            if not config.use_supplementary and read.is_supplementary:
                continue

            if read.mapping_quality < config.min_mapping_quality:
                continue

            # Collect reads by fragment ID
            frag_id = processor.get_porec_fragment_id(read.query_name)
            porec_batch[frag_id].append(read)

            # Process batch when it's full
            if len(porec_batch) >= POREC_BATCH_SIZE:
                # Process all collected Pore-C fragments
                all_reads = []
                for frag_reads in porec_batch.values():
                    all_reads.extend(frag_reads)

                fragments = processor.process_reads_batch(all_reads, chrom)
                for frag in fragments:
                    frag_str = frag.to_string_fast(processor.variants, config.data_type,
                                                   config.new_format, config.single_reads,
                                                   config.qv_offset)
                    if frag_str:
                        output_queue.put(frag_str)

                porec_batch.clear()

        # Process remaining batch
        if porec_batch:
            all_reads = []
            for frag_reads in porec_batch.values():
                all_reads.extend(frag_reads)

            fragments = processor.process_reads_batch(all_reads, chrom)
            for frag in fragments:
                frag_str = frag.to_string_fast(processor.variants, config.data_type,
                                               config.new_format, config.single_reads,
                                               config.qv_offset)
                if frag_str:
                    output_queue.put(frag_str)


class OptimizedExtractHAIRS:
    """主类 - 高度优化版本"""

    def __init__(self, args):
        self.args = args
        self.variants = []
        self.variant_map = {}
        self.variant_positions = {}
        self.config = self._build_config(args)
        self.output_queue = queue.Queue(maxsize=1000)
        self.write_thread = None
        self.stop_writing = threading.Event()

        # 设置进程池
        self.n_workers = args.threads

        # Pore-C模式提示
        if args.porec:
            logger.info("=" * 60)
            logger.info("Pore-C mode enabled!")
            logger.info("- Using Code2's independent Pore-C processing flow")
            logger.info("- Fragments with the same ID prefix (before ':') will be merged")
            logger.info("- Only the prefix before ':' will be saved in the output")
            logger.info("=" * 60)

    def _build_config(self, args) -> ProcessingConfig:
        """构建处理配置"""
        data_type = 0
        new_format = args.new_format
        single_reads = args.single_reads

        if args.hic:
            data_type = 1
            new_format = True
        elif args.tenx:
            data_type = 2
            new_format = True
            single_reads = True

        return ProcessingConfig(
            min_base_quality=args.mbq,
            min_mapping_quality=args.mmq,
            max_insert_size=args.maxIS if not args.hic else 40000000,
            min_insert_size=args.minIS,
            qv_offset=args.qvoffset,
            data_type=data_type,
            new_format=new_format,
            single_reads=single_reads,
            porec_mode=args.porec,
            pe_only=args.pe_only,
            use_supplementary=args.use_supplementary,
            hom=args.hom,
            indels=args.indels
        )

    def load_vcf_parallel(self, vcf_file: str):
        """并行加载VCF文件"""
        logger.info(f"Loading VCF file: {vcf_file}")

        # 第一遍：快速统计行数
        n_lines = sum(1 for line in open(vcf_file) if not line.startswith('#'))

        # 预分配数组
        self.variants = []
        variant_id = 0
        sample_col = self.args.sample_col - 1 if self.args.sample_col else 9

        # 使用mmap加速读取
        with open(vcf_file, 'r+b') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                for line in iter(mmapped_file.readline, b""):
                    if line.startswith(b'#'):
                        continue

                    fields = line.decode('utf-8').strip().split('\t')
                    if len(fields) < 10:
                        continue

                    variant = CompactVariant(
                        position=int(fields[1]),
                        ref_allele=fields[3],
                        alt_allele=fields[4],
                        genotype=fields[sample_col],
                        variant_id=variant_id
                    )

                    if variant.is_het or self.config.hom:
                        chrom = fields[0]
                        self.variants.append(variant)

                        if chrom not in self.variant_map:
                            self.variant_map[chrom] = []
                        self.variant_map[chrom].append(variant_id)
                        variant_id += 1

        # 构建numpy索引
        for chrom, var_indices in self.variant_map.items():
            positions = np.array([self.variants[i].pos for i in var_indices], dtype=np.int32)
            self.variant_positions[chrom] = positions

        logger.info(f"Loaded {len(self.variants)} variants")

    def writer_thread(self):
        """专用写入线程"""
        buffer = []

        while not self.stop_writing.is_set() or not self.output_queue.empty():
            try:
                fragment = self.output_queue.get(timeout=0.1)
                if fragment is None:  # Poison pill
                    break

                buffer.append(fragment)

                if len(buffer) >= 1000:  # Write in batches
                    self.output_file.write('\n'.join(buffer) + '\n')
                    buffer = []

            except queue.Empty:
                if buffer:  # Flush partial buffer
                    self.output_file.write('\n'.join(buffer) + '\n')
                    buffer = []

        # Final flush
        if buffer:
            self.output_file.write('\n'.join(buffer) + '\n')

    def process_bam_parallel_porec(self, bam_files: List[str]):
        """Pore-C模式的并行BAM处理（使用代码2的流程）"""
        # 序列化变体数据供子进程使用
        variants_pickle = pickle.dumps(self.variants)

        # 启动写入线程
        self.write_thread = threading.Thread(target=self.writer_thread)
        self.write_thread.start()

        try:
            for bam_file in bam_files:
                logger.info(f"Processing BAM file in Pore-C mode: {bam_file}")

                with pysam.AlignmentFile(bam_file, 'rb') as bamfile:
                    chromosomes = [chrom for chrom in bamfile.references
                                   if chrom in self.variant_map]

                # 使用线程池处理各个染色体
                with ThreadPoolExecutor(max_workers=min(self.n_workers, len(chromosomes))) as executor:
                    futures = []
                    for chrom in chromosomes:
                        future = executor.submit(
                            process_porec_bam_region,
                            bam_file, chrom, self.config, variants_pickle,
                            self.variant_map, self.variant_positions,
                            self.output_queue
                        )
                        futures.append(future)

                    # 等待完成
                    for i, future in enumerate(as_completed(futures), 1):
                        future.result()
                        logger.info(f"Completed {i}/{len(chromosomes)} chromosomes")

        finally:
            # 停止写入线程
            self.output_queue.put(None)  # Poison pill
            self.stop_writing.set()
            self.write_thread.join()

    def process_bam_parallel(self, bam_files: List[str]):
        """并行处理BAM文件 - 非Pore-C模式"""
        # 序列化变体数据供子进程使用
        variants_pickle = pickle.dumps(self.variants)
        config_pickle = pickle.dumps(self.config)

        # 创建任务队列
        tasks = []

        for bam_file in bam_files:
            logger.info(f"Processing BAM file: {bam_file}")

            with pysam.AlignmentFile(bam_file, 'rb') as bamfile:
                # 获取染色体和大小
                for chrom in bamfile.references:
                    if chrom not in self.variant_map:
                        continue

                    chrom_length = bamfile.get_reference_length(chrom)

                    # 将染色体分块以实现更细粒度的并行
                    n_chunks = max(1, chrom_length // CHUNK_SIZE)
                    chunk_size = chrom_length // n_chunks

                    for i in range(n_chunks):
                        start = i * chunk_size
                        end = min((i + 1) * chunk_size, chrom_length)

                        task = (bam_file, chrom, start, end, config_pickle,
                                variants_pickle, self.variant_map, self.variant_positions)
                        tasks.append(task)

        # 使用进程池并行处理
        logger.info(f"Processing {len(tasks)} chunks with {self.n_workers} workers")

        # 创建写入缓冲区
        write_buffer = []
        buffer_lock = threading.Lock()

        def write_thread():
            """后台写入线程"""
            while True:
                with buffer_lock:
                    if write_buffer:
                        batch = write_buffer[:1000]
                        write_buffer[:1000] = []
                    else:
                        batch = []

                if batch:
                    self.output_file.write('\n'.join(batch) + '\n')
                    self.output_file.flush()
                elif stop_writing.is_set():
                    break
                else:
                    threading.Event().wait(0.1)

        # 启动写入线程
        stop_writing = threading.Event()
        writer = threading.Thread(target=write_thread)
        writer.start()

        # 并行处理
        completed = 0
        total_tasks = len(tasks)

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(process_chromosome_chunk, task) for task in tasks]

            for future in as_completed(futures):
                try:
                    results = future.result()

                    # 添加到写入缓冲区
                    with buffer_lock:
                        write_buffer.extend(results)

                    completed += 1

                    # 进度日志
                    if completed % 10 == 0 or completed == total_tasks:
                        logger.info(f"Completed {completed}/{total_tasks} chunks")

                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")

        # 停止写入线程
        stop_writing.set()
        writer.join()

        # 最终刷新
        with buffer_lock:
            if write_buffer:
                self.output_file.write('\n'.join(write_buffer) + '\n')

        logger.info(f"All {completed} chunks processed successfully")

    def run(self):
        """主执行方法"""
        # 加载VCF
        self.load_vcf_parallel(self.args.vcf)

        # 打开输出文件
        if self.args.out:
            self.output_file = open(self.args.out, 'w', buffering=1048576)  # 1MB缓冲
        else:
            self.output_file = sys.stdout

        try:
            if self.args.porec:
                # Pore-C模式：使用代码2的独立处理流程
                self.process_bam_parallel_porec(self.args.bam)
            else:
                # 非Pore-C模式：使用原有的并行处理流程
                self.process_bam_parallel(self.args.bam)
        finally:
            if self.output_file != sys.stdout:
                self.output_file.close()

        logger.info(f"Extraction complete. Processed {len(self.variants)} variants")

        # Pore-C模式完成提示
        if self.args.porec:
            logger.info("=" * 60)
            logger.info("Pore-C processing complete!")
            logger.info("Fragment IDs have been truncated at ':' for merging")
            logger.info("=" * 60)


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description='高度优化的 extractHAIRS - 从BAM/CRAM文件提取单倍型信息读数'
    )

    # 必需参数
    parser.add_argument('--bam', nargs='+', required=True, help='BAM/CRAM文件')
    parser.add_argument('--vcf', '--VCF', required=True, help='VCF文件')

    # 可选参数
    parser.add_argument('--out', '-o', help='输出文件')
    parser.add_argument('--ref', '--reference', help='参考基因组')
    parser.add_argument('--region', help='处理区域')

    # 质量过滤
    parser.add_argument('--mbq', type=int, default=13, help='最小碱基质量')
    parser.add_argument('--mmq', type=int, default=20, help='最小比对质量')
    parser.add_argument('--qvoffset', type=int, default=33, help='质量值偏移')

    # 插入片段大小
    parser.add_argument('--maxIS', type=int, default=1000, help='最大插入片段大小')
    parser.add_argument('--minIS', type=int, default=0, help='最小插入片段大小')

    # 读取类型选项
    parser.add_argument('--pe_only', '--PEonly', action='store_true', help='仅使用配对端读数')
    parser.add_argument('--single_reads', action='store_true', help='输出单变体fragments')

    # 数据类型
    parser.add_argument('--hic', '--HiC', action='store_true', help='HiC模式')
    parser.add_argument('--tenx', '--10X', action='store_true', help='10X模式')
    parser.add_argument('--porec', '--PoreC', action='store_true', help='Pore-C模式')
    parser.add_argument('--pacbio', action='store_true', help='PacBio模式')
    parser.add_argument('--ont', '--ONT', action='store_true', help='ONT模式')

    # 其他选项
    parser.add_argument('--indels', action='store_true', help='包含indels')
    parser.add_argument('--hom', action='store_true', help='包含纯合变体')
    parser.add_argument('--new_format', '--nf', action='store_true', help='新格式')
    parser.add_argument('--use_supplementary', action='store_true', help='使用补充比对')
    parser.add_argument('--sample_col', type=int, default=10, help='VCF样本列')
    parser.add_argument('--threads', '-t', type=int,
                        default=min(mp.cpu_count(), 16),
                        help='线程数')

    args = parser.parse_args()

    # 验证输入文件
    for bam_file in args.bam:
        if not os.path.exists(bam_file):
            logger.error(f"BAM file not found: {bam_file}")
            sys.exit(1)

    if not os.path.exists(args.vcf):
        logger.error(f"VCF file not found: {args.vcf}")
        sys.exit(1)

    # 运行
    try:
        extractor = OptimizedExtractHAIRS(args)
        extractor.run()
    except KeyboardInterrupt:
        logger.info("Process interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()