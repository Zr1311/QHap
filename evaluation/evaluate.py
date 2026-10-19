#!/usr/bin/env python3
"""Evaluate QHap using the manuscript's retained SE/HE convention.

SE counts maximal disagreement runs, minimizing event count over each phase set's
two orientations, and divides the pooled count by unique evaluated sites minus one.
This is NOT the standard adjacent-transition switch metric. HE minimizes allele
disagreements separately per phase set. Incompatible allele sets stay in the legacy
denominator, as in the original scripts, and are reported explicitly. Run separately
for each chromosome; the historical scripts were chromosome-specific.
"""
from pathlib import Path
from collections import defaultdict
import argparse, json, gzip, contextlib, io

def calculate_hamming_error(block_fundation, block_haplotype, matching_type, site_values, error_site_counts):
    # 初始化相关变量
    hamming_error = 0
    hamming_error_site_values = []  # 用于记录产生hamming error的位点值

    # 遍历每个位点
    for i in range(len(block_fundation[0])):
        base_fundation_2 = block_fundation[1][i]
        base_fundation_3 = block_fundation[2][i]
        base_haplotype_2 = block_haplotype[1][i]
        base_haplotype_3 = block_haplotype[2][i]

        # 方式1: haplotype 2 与 fundation 2，haplotype 3 与 fundation 3 匹配
        if matching_type == 1:
            if base_haplotype_2 == base_fundation_2 or base_haplotype_2 == base_fundation_3:
                if base_haplotype_3 == base_fundation_2 or base_haplotype_3 == base_fundation_3:
                    if base_haplotype_2 != base_fundation_2 or base_haplotype_3 != base_fundation_3:
                        hamming_error += 1
                        hamming_error_site_values.append(site_values[i])
                else:
                    print(
                        f"位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}")
                    error_site_counts[site_values[i]] = error_site_counts.get(site_values[i], 0) + 1
            else:
                print(
                    f"位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}")
                error_site_counts[site_values[i]] = error_site_counts.get(site_values[i], 0) + 1

        # 方式2: haplotype 2 与 fundation 3，haplotype 3 与 fundation 2 匹配
        elif matching_type == 2:
            if base_haplotype_2 == base_fundation_2 or base_haplotype_2 == base_fundation_3:
                if base_haplotype_3 == base_fundation_2 or base_haplotype_3 == base_fundation_3:
                    if base_haplotype_2 != base_fundation_3 or base_haplotype_3 != base_fundation_2:
                        hamming_error += 1
                        hamming_error_site_values.append(site_values[i])
            else:
                print(
                    f"位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}")
                error_site_counts[site_values[i]] = error_site_counts.get(site_values[i], 0) + 1

    return hamming_error, len(block_fundation[0]), hamming_error_site_values

def calculate_switch_error(block_fundation, block_haplotype, matching_type, site_values, error_site_counts):
    # 初始化相关变量
    short_switch_error = 0
    long_switch_error = 0
    in_switch_error = False
    consecutive_error_count = 0
    total_sites = len(block_fundation[0])  # 假设位点信息行的长度等于数据行数
    short_switch_error_site_values = []  # 用于记录产生 short switch error 的位点值
    long_switch_error_site_values = []  # 用于记录产生 long switch error 的位点值

    # 遍历每个位点
    for i in range(total_sites):
        base_fundation_2 = block_fundation[1][i]
        base_fundation_3 = block_fundation[2][i]
        base_haplotype_2 = block_haplotype[1][i]
        base_haplotype_3 = block_haplotype[2][i]

        # 方式1: haplotype 2 与 fundation 2，haplotype 3 与 fundation 3 匹配
        if matching_type == 1:
            if base_haplotype_2 == base_fundation_2 or base_haplotype_2 == base_fundation_3:
                if base_haplotype_3 == base_fundation_2 or base_haplotype_3 == base_fundation_3:
                    if base_haplotype_2 != base_fundation_2 or base_haplotype_3 != base_fundation_3:
                        if not in_switch_error:
                            consecutive_error_count = 1
                            in_switch_error = True
                        else:
                            consecutive_error_count += 1
                    elif base_haplotype_2 == base_fundation_2 and base_haplotype_3 == base_fundation_3:
                        if in_switch_error:
                            if consecutive_error_count == 1:
                                short_switch_error += 1
                                short_switch_error_site_values.append(site_values[i - 1])
                            else:
                                long_switch_error += 1
                                long_switch_error_site_values.extend(site_values[i - consecutive_error_count:i])
                            consecutive_error_count = 0
                            in_switch_error = False
                else:
                    print(
                        f"位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}")
                    error_site_counts[site_values[i]] = error_site_counts.get(site_values[i], 0) + 1
            else:
                print(
                    f"位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}")
                error_site_counts[site_values[i]] = error_site_counts.get(site_values[i], 0) + 1

        # 方式2: haplotype 2 与 fundation 3，haplotype 3 与 fundation 2 匹配
        elif matching_type == 2:
            if base_haplotype_2 == base_fundation_2 or base_haplotype_2 == base_fundation_3:
                if base_haplotype_3 == base_fundation_2 or base_haplotype_3 == base_fundation_3:
                    if base_haplotype_2 != base_fundation_3 or base_haplotype_3 != base_fundation_2:
                        if not in_switch_error:
                            consecutive_error_count = 1
                            in_switch_error = True
                        else:
                            consecutive_error_count += 1
                    elif base_haplotype_2 == base_fundation_3 and base_haplotype_3 == base_fundation_2:
                        if in_switch_error:
                            if consecutive_error_count == 1:
                                short_switch_error += 1
                                short_switch_error_site_values.append(site_values[i - 1])
                            else:
                                long_switch_error += 1
                                long_switch_error_site_values.extend(site_values[i - consecutive_error_count:i])
                            consecutive_error_count = 0
                            in_switch_error = False
                # else:
                #     print(f"位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}")
            # else:
            #     print(f"位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}")

    # 处理最后一个连续错误
    if in_switch_error:
        if consecutive_error_count == 1:
            short_switch_error += 1
            short_switch_error_site_values.append(site_values[total_sites - 1])
        else:
            long_switch_error += 1
            long_switch_error_site_values.extend(site_values[total_sites - consecutive_error_count:total_sites])

    return short_switch_error, long_switch_error, total_sites, short_switch_error_site_values, long_switch_error_site_values

def read_vcf(path):
    result = {}
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.startswith('#'):
                continue
            fields = line.rstrip().split('\t')
            if len(fields) < 10:
                raise ValueError('A single-sample VCF with FORMAT and sample columns is required')
            if len(fields) > 10:
                raise ValueError('Select one sample before evaluation')
            fmt = dict(zip(fields[8].split(':'), fields[9].split(':')))
            gt = fmt.get('GT', '')
            if len(fields[3]) != 1 or len(fields[4]) != 1 or gt not in ('0|1', '1|0'):
                continue
            alleles = [fields[3], fields[4]]
            key = (fields[0], int(fields[1]))
            if key in result:
                raise ValueError(f'Duplicate truth or call coordinate: {key}')
            result[key] = (alleles[int(gt[0])], alleles[int(gt[2])], fmt.get('PS', '.'))
    return result


def evaluate(phased_path, truth_path):
    calls = read_vcf(phased_path)
    truth = read_vcf(truth_path)
    chromosomes = {key[0] for key in calls}
    if len(chromosomes) != 1:
        raise ValueError('Evaluate one chromosome at a time to preserve the original pooled denominator')
    blocks = defaultdict(list)
    unphased_ps = []
    incompatible = []
    for key, call in calls.items():
        if key not in truth:
            continue
        if call[2] == '.':
            unphased_ps.append(key)
            continue
        if set(call[:2]) != set(truth[key][:2]):
            incompatible.append(key)
        blocks[(key[0], call[2])].append((key[1], call, truth[key]))
    n = errors = events = 0
    block_results = []
    for key, rows in sorted(blocks.items()):
        rows.sort()
        positions = [str(row[0]) for row in rows]
        reference = [positions, [row[2][0] for row in rows], [row[2][1] for row in rows]]
        inferred = [positions, [row[1][0] for row in rows], [row[1][1] for row in rows]]
        with contextlib.redirect_stdout(io.StringIO()):
            he = [calculate_hamming_error(reference, inferred, orientation, positions, {})[0]
                  for orientation in (1, 2)]
            se = [sum(calculate_switch_error(reference, inferred, orientation, positions, {})[:2])
                  for orientation in (1, 2)]
        errors += min(he)
        events += min(se)
        n += len(rows)
        block_results.append(dict(chromosome=key[0], phase_set=key[1],
                                  evaluated_sites=len(rows), HE_errors=min(he), SE_events=min(se)))
    lengths = sorted([max(row[0] for row in rows)-min(row[0] for row in rows)
                      for rows in blocks.values() if len(rows)>1], reverse=True)
    cumulative = 0
    n50 = 0
    for length in lengths:
        cumulative += length
        if 2*cumulative >= sum(lengths):
            n50 = length
            break
    return dict(convention='QHap legacy event-based SE and block-oriented HE',
                evaluated_sites=n, HE_errors=errors, SE_events=events,
                HE_percent=100*errors/n if n else None,
                SE_percent=100*events/(n-1) if n>1 else None,
                incompatible_allele_sites=incompatible,
                missing_phase_set_sites=unphased_ps,
                evaluation_site_N50_bp=n50, blocks=block_results,
                note='N50 here is restricted to evaluated sites; the manuscript N50 uses all phased output sites. Percentages are unrounded; retain historical table precision separately.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phased', required=True, type=Path)
    parser.add_argument('--truth', required=True, type=Path)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    result = evaluate(args.phased, args.truth)
    text = json.dumps(result, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text+'\n', encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
