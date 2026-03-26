#!/usr/bin/env python3

import sys
from pathlib import Path


def parse_vcf_line(line):
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
                else:
                    merged_records[pos] = rec2
                    removed_from_file1.add(pos)
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
    with open(output_file, 'w') as f:
        for header in headers:
            f.write(header + '\n')
        for line in data_lines:
            f.write(line + '\n')
    print(f"Merged VCF saved to: {output_file}")


def write_merged_phasing(output_file, blocks1, blocks2):
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

    print(f"Merged phasing saved to: {output_file}")


def merge_files(vcf1_file, phasing_1_file, vcf2_file, phasing_2_file, output_vcf_file, output_phasing_file):
    try:
        print("Starting file merging...")
        print("Reading VCF files...")
        vcf1_data = read_vcf_file_detailed(vcf1_file)
        vcf2_data = read_vcf_file_detailed(vcf2_file)

        positions1, _, records1 = vcf1_data
        positions2, _, records2 = vcf2_data

        print("\nMerging VCF files...")
        merged_positions, merged_headers, merged_lines, removed_from_file1, removed_from_file2 = merge_vcf_with_conflict_resolution(vcf1_data, vcf2_data)

        print("\nCreating index mapping...")
        mapping1, mapping2, removed_indices1, removed_indices2 = create_position_mapping_with_removals(positions1, positions2, merged_positions, removed_from_file1, removed_from_file2)

        print("Reading phasing files...")
        blocks1, phasing_1_line_count = read_phasing_file_detailed(phasing_1_file)
        blocks2, _ = read_phasing_file_detailed(phasing_2_file)

        print("Updating phasing indices...")
        updated_blocks1 = update_phasing_with_removals(blocks1, mapping1, removed_indices1)
        updated_blocks2 = update_phasing_with_removals(blocks2, mapping2, removed_indices2)

        print("\nWriting merged files...")
        write_merged_vcf(output_vcf_file, merged_headers, merged_lines)
        write_merged_phasing(output_phasing_file, updated_blocks1, updated_blocks2)

        print("\nFile merging completed!")
        return True, phasing_1_line_count

    except Exception as e:
        print(f"Error during merging: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0


def main():
    phasing_porec_file = r"/mnt/e/my_work/phasing/phasing_v11_GPU/phasing_v13_GPU_porec/data_WT_porec/data_WT_porec/mhc/data_snps/phasing_output_porec_filtered.txt"
    vcf_porec_file = r"/mnt/e/my_work/phasing/phasing_v11_GPU/phasing_v13_GPU_porec/data_WT_porec/data_WT_porec/mhc/data_snps/chr6_snp_filter_porec.vcf"
    phasing_wt_file = r"/mnt/e/my_work/phasing/phasing_v11_GPU/phasing_v13_GPU_porec/data_WT_porec/data_WT_porec/mhc/data_snps/phasing_output_WT.txt"
    vcf_wt_file = r"/mnt/e/my_work/phasing/phasing_v11_GPU/phasing_v13_GPU_porec/data_WT_porec/data_WT_porec/mhc/data_snps/chr6_snp_filter_WT.vcf"

    output_dir = Path(phasing_porec_file).parent
    merged_phasing_file = output_dir / "phasing_output_merged.txt"
    merged_vcf_file = output_dir / "chr6_snp_filter_merged.vcf"

    print("Processing files...")

    success, line_count = merge_files(
        vcf_porec_file, phasing_porec_file,
        vcf_wt_file, phasing_wt_file,
        str(merged_vcf_file), str(merged_phasing_file)
    )

    if success:
        print(f"\nProcessing done! First phasing file has {line_count} lines")
    else:
        print("\nProcessing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()