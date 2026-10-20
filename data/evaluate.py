#!/usr/bin/env python3
"""Evaluate one chromosome with the supplied QHap evaluation conventions.

Consolidates switch error_test/, hamming_error_rate_test/ and other_test/ from
the read-based implementation. Paths and chromosome are command-line arguments.
The three scoring kernels below retain the original arithmetic and tie rules.

SE (N-1) is the short/long disagreement-run metric in 3_long_short_se.py.
SE (N) also preserves 3_SwitchError_and _rate.py as a separate result.
Neither is redefined as the conventional adjacent-transition switch error.
HE chooses the better orientation independently for each phase set.
The three N50 calculations retain the original ascending accumulation order.
"""
import argparse
from collections import defaultdict
import contextlib
import gzip
import io
import json
from pathlib import Path
import sys


def open_text(path):
    path = Path(path)
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def read_phased(path, chromosome):
    """Keep the original first-GT / last-PS sample-field convention."""
    blocks = {}
    slash_count = pipe_count = 0
    with open_text(path) as source:
        for number, line in enumerate(source, 1):
            if line.startswith("#"):
                continue
            row = line.strip().split("\t")
            if len(row) < 5:
                continue
            if row[0].lower() != chromosome.lower():
                raise ValueError(f"Line {number}: evaluate one chromosome at a time ({chromosome})")
            sample = row[-1]
            if len(row) >= 10:
                slash_count += "/" in sample
                pipe_count += "|" in sample
            gt_fields = sample.split(":")
            genotype, phase_set = gt_fields[0], gt_fields[-1]
            if "/" in genotype:
                continue
            alleles = {"0|1": (row[3], row[4]), "1|0": (row[4], row[3]),
                       "1|1": (row[4], row[4]), "0|0": (row[3], row[3])}
            if genotype not in alleles:
                raise ValueError(f"Line {number}: unsupported phased diploid genotype {genotype!r}")
            block = blocks.setdefault(phase_set, [[], [], []])
            block[0].append(row[1])
            first, second = alleles[genotype]
            block[1].append(first)
            block[2].append(second)
    return blocks, slash_count, pipe_count


def read_truth(path, chromosome):
    records = defaultdict(list)
    # complement.py counts distinct values in column 2, including a POS header.
    # Retain that convention instead of silently substituting a variant count.
    column_two_values = set()
    with open_text(path) as source:
        for line in source:
            row = line.strip().split("\t")
            if len(row) > 1:
                column_two_values.add(row[1])
            if len(row) < 10:
                continue
            try:
                position = int(row[1])
            except ValueError:
                continue
            if row[0].lower() != chromosome.lower():
                raise ValueError("Use a truth VCF for the same chromosome and evaluation region")
            records[position].append(row)
    return records, len(column_two_values)


def match_blocks(blocks, truth, split_truth_gt):
    """Match unique-position, single-base truth entries as in stage 2 scripts."""
    matched = []
    for index, (phase_set, block) in enumerate(blocks.items(), 1):
        reference = [[], [], []]
        inferred = [[], [], []]
        for offset, position in enumerate(block[0]):
            try:
                hits = truth.get(int(position), [])
            except ValueError:
                continue
            if len(hits) != 1:
                continue
            row = hits[0]
            ref, alt = row[3].strip().upper(), row[4].strip().upper()
            if len(ref) != 1 or len(alt) != 1:
                continue
            gt = row[-1].strip().upper()
            if split_truth_gt:
                gt = gt.split(":")[0]
            if "|" not in gt:
                continue
            left, right = gt.split("|")
            if left not in {"0", "1"} or right not in {"0", "1"}:
                continue
            reference[0].append(position)
            reference[1].append(ref if left == "0" else alt)
            reference[2].append(ref if right == "0" else alt)
            for i in range(3):
                inferred[i].append(block[i][offset])
        matched.append({"id": f"block{index}.tab", "phase_set": phase_set,
                        "truth": reference, "phased": inferred})
    # The original comparison files on Windows are enumerated by filename.
    # Stable filename ordering also retains the N50 site-count tie order.
    return sorted(matched, key=lambda item: item["id"] + "_haplotype.tab")


def score_blocks(blocks, kind):
    kernels = {"switch_runs": calculate_switch_error_runs,
               "switch_events": calculate_switch_error_events,
               "hamming": calculate_hamming_error}
    kernel = kernels[kind]
    all_sites = set()
    diagnostics = {}
    result_blocks = []
    errors = short_errors = long_errors = 0
    for block in blocks:
        positions = block["truth"][0]
        all_sites.update(positions)
        # Preserve the diagnostic counters while keeping JSON output readable.
        with contextlib.redirect_stdout(io.StringIO()):
            first = kernel(block["truth"], block["phased"], 1, positions, diagnostics)
            second = kernel(block["truth"], block["phased"], 2, positions, diagnostics)
        first_count = sum(first[:2]) if kind == "switch_runs" else first[0]
        second_count = sum(second[:2]) if kind == "switch_runs" else second[0]
        orientation = 1 if first_count <= second_count else 2
        chosen = first if orientation == 1 else second
        count = min(first_count, second_count)
        errors += count
        denominator = max(len(positions) - 1, 1) if kind == "switch_runs" else len(positions)
        rate = count / denominator if denominator else 0
        details = {"block": block["id"], "phase_set": block["phase_set"],
                   "matched_sites": len(positions), "orientation": orientation,
                   "errors": count, "rate": rate, "rate_4dp": f"{rate:.4f}"}
        if kind == "switch_runs":
            short, long, _, short_sites, long_sites = chosen
            short_errors += short
            long_errors += long
            details.update(short_errors=short, long_errors=long,
                           short_rate=short / denominator, long_rate=long / denominator,
                           short_error_positions=short_sites, long_error_positions=long_sites)
        else:
            details["error_positions"] = chosen[2]
        result_blocks.append(details)
    total_sites = len(all_sites)
    denominator = max(total_sites - 1, 1) if kind == "switch_runs" else total_sites
    rate = errors / denominator if denominator else 0
    result = {"errors": errors, "evaluated_sites": total_sites, "denominator": denominator,
              "rate": rate, "rate_4dp": f"{rate:.4f}", "percent": rate * 100,
              "status": "evaluated" if total_sites else "no_matched_sites",
              "blocks": result_blocks, "incompatible_allele_diagnostic_counts": diagnostics,
              "always_error_sites": [site for site, count in diagnostics.items()
                                     if count == len(blocks)]}
    if kind == "switch_runs":
        result.update(short_errors=short_errors, long_errors=long_errors,
                      short_rate=short_errors / denominator,
                      long_rate=long_errors / denominator)
    return result


def legacy_n50(blocks, mode):
    values = []
    for block in blocks:
        sites = block["phased"][0]
        if mode == "bp" and len(sites) >= 2:
            values.append((int(sites[-1]) - int(sites[0]), len(sites), block["id"]))
        elif mode in {"sites", "site_bp"} and sites:
            first = int(sites[0]) if sites[0].isdigit() else 0
            last = int(sites[-1]) if sites[-1].isdigit() else 0
            values.append((len(sites), last - first + 1, block["id"]))
    ordered = sorted(values, key=lambda item: item[0])
    total = sum(item[0] for item in values)
    cumulative = used = selected_value = 0
    selected_block = None
    for amount, span_or_sites, block_id in ordered:
        cumulative += amount
        used += 1
        if cumulative >= total / 2:
            selected_value = span_or_sites if mode == "site_bp" else amount
            selected_block = block_id
            break
    return {"value": selected_value, "blocks": len(values), "total_weight": total,
            "blocks_to_half": used, "block_fraction": used / len(values) if values else None,
            "block_fraction_2dp_percent": f"{used / len(values):.2%}" if values else None,
            "threshold_block": selected_block, "order": "ascending"}


def evaluate(phased, truth_path, chromosome, truth_gt_mode="gt"):
    blocks, unphased, phased_count = read_phased(phased, chromosome)
    truth, truth_column_count = read_truth(truth_path, chromosome)
    se_blocks = match_blocks(blocks, truth, split_truth_gt=True)
    he_blocks = match_blocks(blocks, truth, split_truth_gt=(truth_gt_mode == "gt"))
    se = score_blocks(se_blocks, "switch_runs")
    he = score_blocks(he_blocks, "hamming")
    ratio = unphased / phased_count if phased_count else None
    # complement.py runs only when both the retained count and divisor are > 0.
    completeness = se["evaluated_sites"] / truth_column_count if se["evaluated_sites"] and truth_column_count else None
    return {
        "inputs": {"phased": str(phased), "truth": str(truth_path),
                   "chromosome": chromosome, "truth_gt_mode": truth_gt_mode},
        "switch_error_n_minus_one": se,
        "switch_error_n": score_blocks(se_blocks, "switch_events"),
        "hamming_error": he,
        "other_metrics": {
            "unphased_records": unphased, "phased_records": phased_count,
            "unphased_to_phased_ratio": ratio,
            "unphased_to_phased_ratio_4dp": f"{ratio:.4f}" if ratio is not None else None,
            "truth_column2_unique_values": truth_column_count,
            "truth_site_completeness_fraction": completeness,
            "n50_bp": legacy_n50(se_blocks, "bp"),
            "n50_sites": legacy_n50(se_blocks, "sites"),
            "n50_site_bp": legacy_n50(se_blocks, "site_bp"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phased", type=Path, required=True, help="QHap phased VCF, plain or .gz")
    parser.add_argument("--truth", type=Path, required=True, help="Matching region's phased SNP truth, plain or .gz")
    parser.add_argument("--chrom", required=True, help="Chromosome, e.g. chr6 or chr22")
    parser.add_argument("--output", type=Path, required=True, help="JSON output, including per-block results")
    parser.add_argument("--truth-gt-mode", choices=("legacy", "gt"), default="gt",
                        help="HE truth parser: gt (default) takes GT from GT:AD; legacy requires a GT-only sample value")
    args = parser.parse_args()
    if args.output.resolve() in {args.phased.resolve(), args.truth.resolve()}:
        parser.error("--output must differ from both input paths")
    report = evaluate(args.phased, args.truth, args.chrom, args.truth_gt_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for label, key in (("SE (N-1)", "switch_error_n_minus_one"), ("HE", "hamming_error")):
        metric = report[key]
        if metric["status"] == "no_matched_sites":
            print(f"{label}: no matched sites; the stored legacy zero rate is not an accuracy estimate")
        else:
            print(f"{label}: {metric['errors']} errors / {metric['denominator']}, {metric['percent']:.8f}%")
    print(f"Saved: {args.output}")


# Scoring arithmetic retained from the original read-based scripts.

def calculate_switch_error_runs(block_fundation, block_haplotype, matching_type, site_values, error_site_counts):
    short_switch_error = 0
    long_switch_error = 0
    in_switch_error = False
    consecutive_error_count = 0
    total_sites = len(block_fundation[0])
    short_switch_error_site_values = []
    long_switch_error_site_values = []
    for i in range(total_sites):
        base_fundation_2 = block_fundation[1][i]
        base_fundation_3 = block_fundation[2][i]
        base_haplotype_2 = block_haplotype[1][i]
        base_haplotype_3 = block_haplotype[2][i]
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
                    print(f'位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}')
                    error_site_counts[site_values[i]] = error_site_counts.get(site_values[i], 0) + 1
            else:
                print(f'位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}')
                error_site_counts[site_values[i]] = error_site_counts.get(site_values[i], 0) + 1
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
    if in_switch_error:
        if consecutive_error_count == 1:
            short_switch_error += 1
            short_switch_error_site_values.append(site_values[total_sites - 1])
        else:
            long_switch_error += 1
            long_switch_error_site_values.extend(site_values[total_sites - consecutive_error_count:total_sites])
    return (short_switch_error, long_switch_error, total_sites, short_switch_error_site_values, long_switch_error_site_values)


def calculate_switch_error_events(block_fundation, block_haplotype, matching_type, site_values, error_site_counts):
    switch_error = 0
    in_switch_error = False
    total_sites = len(block_fundation[0])
    switch_error_site_values = []
    for i in range(total_sites):
        base_fundation_2 = block_fundation[1][i]
        base_fundation_3 = block_fundation[2][i]
        base_haplotype_2 = block_haplotype[1][i]
        base_haplotype_3 = block_haplotype[2][i]
        if matching_type == 1:
            if base_haplotype_2 == base_fundation_2 or base_haplotype_2 == base_fundation_3:
                if base_haplotype_3 == base_fundation_2 or base_haplotype_3 == base_fundation_3:
                    if base_haplotype_2 != base_fundation_2 or base_haplotype_3 != base_fundation_3:
                        if not in_switch_error:
                            switch_error += 1
                            in_switch_error = True
                            switch_error_site_values.append(site_values[i])
                    elif base_haplotype_2 == base_fundation_2 and base_haplotype_3 == base_fundation_3:
                        in_switch_error = False
                else:
                    print(f'位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}')
                    error_site_counts[site_values[i]] = error_site_counts.get(site_values[i], 0) + 1
            else:
                print(f'位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}')
                error_site_counts[site_values[i]] = error_site_counts.get(site_values[i], 0) + 1
        elif matching_type == 2:
            if base_haplotype_2 == base_fundation_2 or base_haplotype_2 == base_fundation_3:
                if base_haplotype_3 == base_fundation_2 or base_haplotype_3 == base_fundation_3:
                    if base_haplotype_2 != base_fundation_3 or base_haplotype_3 != base_fundation_2:
                        if not in_switch_error:
                            switch_error += 1
                            in_switch_error = True
                            switch_error_site_values.append(site_values[i])
                    elif base_haplotype_2 == base_fundation_3 and base_haplotype_3 == base_fundation_2:
                        in_switch_error = False
    return (switch_error, total_sites, switch_error_site_values)


def calculate_hamming_error(block_fundation, block_haplotype, matching_type, site_values, error_site_counts):
    hamming_error = 0
    hamming_error_site_values = []
    for i in range(len(block_fundation[0])):
        base_fundation_2 = block_fundation[1][i]
        base_fundation_3 = block_fundation[2][i]
        base_haplotype_2 = block_haplotype[1][i]
        base_haplotype_3 = block_haplotype[2][i]
        if matching_type == 1:
            if base_haplotype_2 == base_fundation_2 or base_haplotype_2 == base_fundation_3:
                if base_haplotype_3 == base_fundation_2 or base_haplotype_3 == base_fundation_3:
                    if base_haplotype_2 != base_fundation_2 or base_haplotype_3 != base_fundation_3:
                        hamming_error += 1
                        hamming_error_site_values.append(site_values[i])
                else:
                    print(f'位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}')
                    error_site_counts[site_values[i]] = error_site_counts.get(site_values[i], 0) + 1
            else:
                print(f'位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}')
                error_site_counts[site_values[i]] = error_site_counts.get(site_values[i], 0) + 1
        elif matching_type == 2:
            if base_haplotype_2 == base_fundation_2 or base_haplotype_2 == base_fundation_3:
                if base_haplotype_3 == base_fundation_2 or base_haplotype_3 == base_fundation_3:
                    if base_haplotype_2 != base_fundation_3 or base_haplotype_3 != base_fundation_2:
                        hamming_error += 1
                        hamming_error_site_values.append(site_values[i])
            else:
                print(f'位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}')
                error_site_counts[site_values[i]] = error_site_counts.get(site_values[i], 0) + 1
    return (hamming_error, len(block_fundation[0]), hamming_error_site_values)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"Evaluation failed: {error}")
