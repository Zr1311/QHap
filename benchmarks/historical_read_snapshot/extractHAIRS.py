#!/usr/bin/env python3
"""
Optimized Python implementation of HapCUT2's extractHAIRS tool
High-performance extraction of haplotype-informative reads from sorted BAM/CRAM files
With Pore-C support for fragment merging
"""

"""
python3 extractHAIRS.py --bam chr6.bam --VCF chr6_snp_filter_porec.vcf --out hapcut2_output_porec_filtered.txt
"""

import sys
import os
import argparse
import logging
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional, Set, Generator
from collections import defaultdict, deque
import numpy as np
import pysam
import pyfaidx
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial, lru_cache
import bisect
import pickle
import mmap
import gc
from array import array
import queue
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BSIZE = 500  # Block size for interval mapping
QV_OFFSET = 33  # Quality value offset
MIN_BASE_QUALITY = 13  # Minimum base quality
MIN_MAPPING_QUALITY = 20  # Minimum mapping quality
MAX_INSERT_SIZE = 1000  # Maximum insert size for paired-end reads
MIN_INSERT_SIZE = 0  # Minimum insert size
MAXFRAG = 500000  # Maximum number of fragments in buffer
BATCH_SIZE = 10000  # Batch size for processing
WRITE_BUFFER_SIZE = 50000  # Write buffer size


# Optimized data structures
class Variant:
    """Optimized variant representation with __slots__ for memory efficiency"""
    __slots__ = ['chrom', 'position', 'ref_allele', 'alt_allele', 'genotype',
                 'variant_id', 'is_heterozygous', 'is_indel', 'allele1', 'allele2',
                 'depth', '_hash']

    def __init__(self, chrom: str, position: int, ref_allele: str, alt_allele: str,
                 genotype: str, variant_id: int):
        self.chrom = chrom
        self.position = position
        self.ref_allele = ref_allele
        self.alt_allele = alt_allele
        self.genotype = genotype
        self.variant_id = variant_id
        self.is_heterozygous = False
        self.is_indel = False
        self.allele1 = ""
        self.allele2 = ""
        self.depth = 0
        self._hash = None

        # Process variant after initialization
        self._process_genotype()

    def _process_genotype(self):
        """Process genotype information"""
        if '/' in self.genotype or '|' in self.genotype:
            sep = '/' if '/' in self.genotype else '|'
            gt_parts = self.genotype.split(':')[0].split(sep)

            if len(gt_parts) == 2:
                gt1, gt2 = gt_parts[0], gt_parts[1]

                # Check for heterozygous variant
                if (gt1 == '0' and gt2 == '1') or (gt1 == '1' and gt2 == '0'):
                    self.is_heterozygous = True
                    self.allele1 = self.ref_allele
                    self.allele2 = self.alt_allele.split(',')[0]
                elif (gt1 == '0' and gt2 == '2') or (gt1 == '2' and gt2 == '0'):
                    self.is_heterozygous = True
                    self.allele1 = self.ref_allele
                    self.allele2 = self.alt_allele.split(',')[1] if ',' in self.alt_allele else self.alt_allele

                # Check if it's an indel
                if self.is_heterozygous:
                    self.is_indel = len(self.allele1) != len(self.allele2)

        # Pre-compute hash for faster lookups
        self._hash = hash((self.chrom, self.position))

    def __hash__(self):
        return self._hash if self._hash else hash((self.chrom, self.position))


class CompactFragment:
    """Memory-efficient fragment representation using arrays"""
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

    def to_string_fast(self, varlist: List[Variant], data_type: int, new_format: bool) -> str:
        """Optimized string conversion"""
        n_vars = len(self.variant_ids)
        if n_vars == 0:
            return ""

        if n_vars < 2 and data_type == 0:  # SINGLE_READS check moved to caller
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
        qual_str = ''.join(chr(q + QV_OFFSET) for q in self.qualities)
        parts.append(qual_str)

        return ' '.join(parts)


class OptimizedExtractHAIRS:
    """Optimized main class for extracting haplotype-informative reads"""

    def __init__(self, args):
        self.args = args
        self.variants = []
        self.variant_map = {}  # chrom -> list of variant indices
        self.variant_positions = {}  # chrom -> numpy array of positions for fast search
        self.chromosome_blocks = {}  # chrom -> interval map
        self.reference = None
        self.output_queue = queue.Queue(maxsize=1000)
        self.write_thread = None
        self.stop_writing = threading.Event()

        # Pore-C mode specific
        self.porec_mode = args.porec
        self.porec_fragments = {}  # Store fragments by Pore-C ID for merging

        # Pre-allocated buffers
        self.fragment_buffer = deque(maxlen=WRITE_BUFFER_SIZE)

        # Set global parameters
        global MIN_BASE_QUALITY, MIN_MAPPING_QUALITY, MAX_INSERT_SIZE, MIN_INSERT_SIZE
        global DATA_TYPE, NEW_FORMAT, SINGLE_READS, QV_OFFSET

        MIN_BASE_QUALITY = args.mbq
        MIN_MAPPING_QUALITY = args.mmq
        MAX_INSERT_SIZE = args.maxIS
        MIN_INSERT_SIZE = args.minIS
        QV_OFFSET = args.qvoffset

        # Set data type
        self.data_type = 0
        self.new_format = args.new_format
        self.single_reads = args.single_reads

        if args.hic:
            self.data_type = 1
            self.new_format = True
            MAX_INSERT_SIZE = 40000000
        elif args.tenx:
            self.data_type = 2
            self.new_format = True
            self.single_reads = True
        # Note: Pore-C mode doesn't change data_type or format,
        # it only affects how fragments are merged

        # Initialize output file
        if args.out:
            self.output_file = open(args.out, 'w', buffering=65536)  # Large buffer
        else:
            self.output_file = sys.stdout

    def get_porec_fragment_id(self, read_name: str) -> str:
        """Extract Pore-C fragment ID from read name"""
        if self.porec_mode and ':' in read_name:
            return read_name.split(':')[0]
        return read_name

    def load_vcf_optimized(self, vcf_file: str) -> None:
        """Optimized VCF loading with memory mapping"""
        logger.info(f"Loading VCF file: {vcf_file}")

        variant_id = 0
        sample_col = self.args.sample_col - 1 if self.args.sample_col else 9

        # Use memory mapping for large files
        with open(vcf_file, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                for line in iter(mmapped_file.readline, b""):
                    if line.startswith(b'#'):
                        continue

                    line_str = line.decode('utf-8').strip()
                    fields = line_str.split('\t')

                    if len(fields) < 10:
                        continue

                    chrom = fields[0]
                    pos = int(fields[1])
                    ref = fields[3]
                    alt = fields[4]
                    genotype = fields[sample_col]

                    variant = Variant(
                        chrom=chrom,
                        position=pos,
                        ref_allele=ref,
                        alt_allele=alt,
                        genotype=genotype,
                        variant_id=variant_id
                    )

                    if variant.is_heterozygous or self.args.hom:
                        self.variants.append(variant)

                        if chrom not in self.variant_map:
                            self.variant_map[chrom] = []
                        self.variant_map[chrom].append(variant_id)
                        variant_id += 1

        # Convert position lists to numpy arrays for faster searching
        for chrom, var_indices in self.variant_map.items():
            positions = np.array([self.variants[i].position for i in var_indices], dtype=np.int32)
            self.variant_positions[chrom] = positions

        logger.info(f"Loaded {len(self.variants)} variants from {len(self.variant_map)} chromosomes")
        self.build_interval_maps_optimized()

    def build_interval_maps_optimized(self) -> None:
        """Optimized interval map building"""
        for chrom, var_indices in self.variant_map.items():
            if not var_indices:
                continue

            positions = self.variant_positions[chrom]
            max_pos = positions[-1]  # Already sorted
            num_blocks = (max_pos // BSIZE) + 2

            # Use numpy for faster array operations
            interval_map = np.full(num_blocks, -1, dtype=np.int32)

            for block in range(num_blocks):
                block_start = block * BSIZE
                idx = np.searchsorted(positions, block_start)

                if idx < len(var_indices):
                    interval_map[block] = var_indices[idx]
                elif idx > 0:
                    interval_map[block] = var_indices[idx - 1]

            self.chromosome_blocks[chrom] = interval_map

    def load_reference(self, ref_file: str) -> None:
        """Load reference genome if provided"""
        if ref_file and ref_file != "None":
            logger.info(f"Loading reference genome: {ref_file}")
            self.reference = pyfaidx.Fasta(ref_file)

    @lru_cache(maxsize=10000)
    def get_variant_range(self, chrom: str, start: int, end: int) -> List[int]:
        """Cached variant range lookup"""
        if chrom not in self.variant_positions:
            return []

        positions = self.variant_positions[chrom]
        start_idx = np.searchsorted(positions, start)
        end_idx = np.searchsorted(positions, end, side='right')

        return self.variant_map[chrom][start_idx:end_idx]

    def extract_variants_from_read_optimized(self, read: pysam.AlignedSegment,
                                             chrom: str) -> Optional[CompactFragment]:
        """Optimized variant extraction from read"""
        if chrom not in self.variant_map:
            return None

        # Get fragment ID for Pore-C mode
        if self.porec_mode:
            fragment_id = self.get_porec_fragment_id(read.query_name)
        else:
            fragment_id = read.query_name

        fragment = CompactFragment(fragment_id)

        # Get 10X barcode if available
        if self.data_type == 2 and read.has_tag('BX'):
            fragment.barcode = read.get_tag('BX')

        # Get variants in read range using optimized lookup
        read_start = read.reference_start + 1
        read_end = read.reference_end

        variant_indices = self.get_variant_range(chrom, read_start, read_end)

        if not variant_indices:
            return None

        # Process CIGAR operations efficiently
        read_seq = read.query_sequence
        read_qual = read.query_qualities if read.query_qualities else [MIN_BASE_QUALITY] * len(read_seq)

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

            if variant.position in ref_to_read:
                rpos = ref_to_read[variant.position]

                if 0 <= rpos < len(read_seq):
                    base = read_seq[rpos]
                    quality = min(read_qual[rpos], read.mapping_quality)

                    if quality >= MIN_BASE_QUALITY:
                        allele = None
                        if base == variant.allele1:
                            allele = '0'
                        elif base == variant.allele2:
                            allele = '1'

                        if allele:
                            fragment.add_variant(var_id, allele, quality)

        return fragment if len(fragment.variant_ids) > 0 else None

    def process_reads_batch(self, reads: List[pysam.AlignedSegment],
                            chrom: str) -> List[CompactFragment]:
        """Process a batch of reads efficiently"""
        fragments = []

        if self.porec_mode:
            # Pore-C mode: collect fragments by ID for merging
            temp_fragments = {}

            for read in reads:
                fragment = self.extract_variants_from_read_optimized(read, chrom)
                if fragment:
                    frag_id = fragment.read_id
                    if frag_id in temp_fragments:
                        # Merge with existing fragment
                        temp_fragments[frag_id].merge_with(fragment)
                    else:
                        temp_fragments[frag_id] = fragment

            # Filter fragments that have at least 2 variants after merging
            for frag_id, fragment in temp_fragments.items():
                if len(fragment.variant_ids) >= 2:
                    fragments.append(fragment)
        else:
            # Original mode
            for read in reads:
                fragment = self.extract_variants_from_read_optimized(read, chrom)
                if fragment:
                    if len(fragment.variant_ids) >= 2 or (self.single_reads and len(fragment.variant_ids) >= 1):
                        fragments.append(fragment)

        return fragments

    def writer_thread(self):
        """Dedicated thread for writing output"""
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

    def process_bam_region_parallel(self, bam_file: str, region: Optional[str] = None) -> None:
        """Process BAM region with parallel read processing"""
        paired_reads = {}
        read_batch = []

        # For Pore-C mode, we need to collect all reads with same fragment ID
        if self.porec_mode:
            porec_batch = defaultdict(list)

        with pysam.AlignmentFile(bam_file, 'rb', threads=4) as bamfile:
            if bam_file.endswith('.cram') and self.reference:
                bamfile.reference_filename = self.args.ref

            iterator = bamfile.fetch(region=region) if region else bamfile.fetch()

            current_chrom = None

            for read in iterator:
                # Apply filters
                if read.is_unmapped or read.is_secondary or read.is_qcfail or read.is_duplicate:
                    continue

                if not self.args.use_supplementary and read.is_supplementary:
                    continue

                if read.mapping_quality < MIN_MAPPING_QUALITY:
                    continue

                chrom = read.reference_name

                if self.porec_mode:
                    # Pore-C mode: collect reads by fragment ID
                    frag_id = self.get_porec_fragment_id(read.query_name)

                    # Process batch when chromosome changes or batch is full
                    if current_chrom != chrom or len(porec_batch) >= BATCH_SIZE:
                        if porec_batch and current_chrom:
                            # Process all collected Pore-C fragments
                            all_reads = []
                            for frag_reads in porec_batch.values():
                                all_reads.extend(frag_reads)

                            fragments = self.process_reads_batch(all_reads, current_chrom)
                            for frag in fragments:
                                frag_str = frag.to_string_fast(self.variants, self.data_type, self.new_format)
                                if frag_str:
                                    self.output_queue.put(frag_str)

                        porec_batch.clear()
                        current_chrom = chrom

                    porec_batch[frag_id].append(read)
                else:
                    # Original processing mode
                    # Process batch when chromosome changes or batch is full
                    if current_chrom != chrom or len(read_batch) >= BATCH_SIZE:
                        if read_batch and current_chrom:
                            fragments = self.process_reads_batch(read_batch, current_chrom)
                            for frag in fragments:
                                frag_str = frag.to_string_fast(self.variants, self.data_type, self.new_format)
                                if frag_str:
                                    self.output_queue.put(frag_str)

                        read_batch = []
                        current_chrom = chrom

                    # Handle paired-end reads efficiently
                    if read.is_paired and not read.mate_is_unmapped:
                        if read.query_name in paired_reads:
                            # Process pair
                            mate = paired_reads.pop(read.query_name)
                            read_batch.extend([mate, read])
                        else:
                            paired_reads[read.query_name] = read

                            # Periodically clean old unpaired reads
                            if len(paired_reads) > 5000:
                                current_pos = read.reference_start
                                to_remove = [qname for qname, r in paired_reads.items()
                                             if current_pos - r.reference_start > MAX_INSERT_SIZE]

                                for qname in to_remove:
                                    if not self.args.pe_only:
                                        read_batch.append(paired_reads.pop(qname))
                                    else:
                                        del paired_reads[qname]
                    else:
                        if not self.args.pe_only:
                            read_batch.append(read)

            # Process final batch
            if self.porec_mode:
                if porec_batch and current_chrom:
                    all_reads = []
                    for frag_reads in porec_batch.values():
                        all_reads.extend(frag_reads)

                    fragments = self.process_reads_batch(all_reads, current_chrom)
                    for frag in fragments:
                        frag_str = frag.to_string_fast(self.variants, self.data_type, self.new_format)
                        if frag_str:
                            self.output_queue.put(frag_str)
            else:
                if read_batch and current_chrom:
                    fragments = self.process_reads_batch(read_batch, current_chrom)
                    for frag in fragments:
                        frag_str = frag.to_string_fast(self.variants, self.data_type, self.new_format)
                        if frag_str:
                            self.output_queue.put(frag_str)

                # Process remaining unpaired reads
                if not self.args.pe_only:
                    for read in paired_reads.values():
                        fragment = self.extract_variants_from_read_optimized(read, read.reference_name)
                        if fragment and (len(fragment.variant_ids) >= 2 or
                                         (self.single_reads and len(fragment.variant_ids) >= 1)):
                            frag_str = fragment.to_string_fast(self.variants, self.data_type, self.new_format)
                            if frag_str:
                                self.output_queue.put(frag_str)

    def process_chromosome_worker(self, args: Tuple[str, str]) -> None:
        """Worker function for processing a single chromosome"""
        bam_file, chrom = args
        self.process_bam_region_parallel(bam_file, chrom)

    def process_bam_files(self, bam_files: List[str]) -> None:
        """Process multiple BAM files with optimized parallelization"""
        # Start writer thread
        self.write_thread = threading.Thread(target=self.writer_thread)
        self.write_thread.start()

        try:
            for bam_file in bam_files:
                logger.info(f"Processing BAM file: {bam_file}")
                if self.porec_mode:
                    logger.info("Pore-C mode enabled - merging reads with same fragment ID")

                if self.args.region:
                    # Process specific region
                    self.process_bam_region_parallel(bam_file, self.args.region)
                else:
                    # Get chromosomes that have variants
                    with pysam.AlignmentFile(bam_file, 'rb') as bamfile:
                        chromosomes = [chrom for chrom in bamfile.references
                                       if chrom in self.variant_map]

                    # Process chromosomes in parallel using thread pool
                    with ThreadPoolExecutor(max_workers=min(self.args.threads, len(chromosomes))) as executor:
                        tasks = [(bam_file, chrom) for chrom in chromosomes]
                        futures = [executor.submit(self.process_chromosome_worker, task)
                                   for task in tasks]

                        # Wait for completion with progress tracking
                        for i, future in enumerate(as_completed(futures), 1):
                            future.result()
                            logger.info(f"Completed {i}/{len(chromosomes)} chromosomes")

        finally:
            # Signal writer thread to stop
            self.output_queue.put(None)  # Poison pill
            self.stop_writing.set()
            self.write_thread.join()

    def run(self) -> None:
        """Main execution method"""
        # Load VCF with optimized method
        self.load_vcf_optimized(self.args.vcf)

        # Load reference if provided
        if self.args.ref:
            self.load_reference(self.args.ref)

        # Process BAM files
        self.process_bam_files(self.args.bam)

        # Close output file
        if self.output_file != sys.stdout:
            self.output_file.close()

        # Print statistics
        logger.info(f"Extraction complete. Processed {len(self.variants)} variants")

        # Force garbage collection
        gc.collect()


def validate_inputs(args):
    """Validate input files and arguments"""
    for bam_file in args.bam:
        if not os.path.exists(bam_file):
            logger.error(f"BAM file not found: {bam_file}")
            sys.exit(1)

        # Check for index file
        if not (os.path.exists(bam_file + '.bai') or os.path.exists(bam_file + '.csi')):
            logger.warning(f"Index file not found for {bam_file}, this may slow down processing")

    if not os.path.exists(args.vcf):
        logger.error(f"VCF file not found: {args.vcf}")
        sys.exit(1)

    if args.ref and not os.path.exists(args.ref):
        logger.error(f"Reference file not found: {args.ref}")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Optimized extraction of haplotype-informative reads from sorted BAM/CRAM files with Pore-C support',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('--bam', nargs='+', required=True,
                        help='Sorted BAM/CRAM file(s)')
    parser.add_argument('--vcf', '--VCF', required=True,
                        help='VCF file with variants')

    # Optional arguments
    parser.add_argument('--out', '-o', default=None,
                        help='Output file for fragments (default: stdout)')
    parser.add_argument('--ref', '--reference', default=None,
                        help='Reference genome FASTA file')
    parser.add_argument('--region', default=None,
                        help='Genomic region to process (chr:start-end)')

    # Quality filters
    parser.add_argument('--mbq', type=int, default=13,
                        help='Minimum base quality (default: 13)')
    parser.add_argument('--mmq', type=int, default=20,
                        help='Minimum mapping quality (default: 20)')
    parser.add_argument('--qvoffset', type=int, default=33,
                        help='Quality value offset (default: 33)')

    # Insert size filters
    parser.add_argument('--maxIS', type=int, default=1000,
                        help='Maximum insert size (default: 1000)')
    parser.add_argument('--minIS', type=int, default=0,
                        help='Minimum insert size (default: 0)')

    # Read type options
    parser.add_argument('--pe_only', '--PEonly', action='store_true',
                        help='Use only paired-end reads')
    parser.add_argument('--single_reads', action='store_true',
                        help='Output fragments with single variant')

    # Data type options
    parser.add_argument('--hic', '--HiC', action='store_true',
                        help='HiC mode (sets maxIS to 40MB)')
    parser.add_argument('--tenx', '--10X', action='store_true',
                        help='10X Genomics mode')
    parser.add_argument('--porec', '--PoreC', action='store_true',
                        help='Pore-C mode: merge reads with same fragment ID (before first colon)')
    parser.add_argument('--pacbio', action='store_true',
                        help='PacBio long reads mode')
    parser.add_argument('--ont', '--ONT', action='store_true',
                        help='Oxford Nanopore mode')

    # Other options
    parser.add_argument('--indels', action='store_true',
                        help='Include indels')
    parser.add_argument('--hom', action='store_true',
                        help='Include homozygous variants')
    parser.add_argument('--new_format', '--nf', action='store_true',
                        help='Use new output format')
    parser.add_argument('--use_supplementary', action='store_true',
                        help='Use supplementary alignments')
    parser.add_argument('--sample_col', type=int, default=10,
                        help='Sample column in VCF (default: 10)')
    parser.add_argument('--threads', '-t', type=int,
                        default=min(mp.cpu_count(), 8),
                        help='Number of threads to use (default: min(CPU count, 8))')

    args = parser.parse_args()

    # Validate inputs
    validate_inputs(args)

    # Run extraction
    try:
        extractor = OptimizedExtractHAIRS(args)
        extractor.run()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()