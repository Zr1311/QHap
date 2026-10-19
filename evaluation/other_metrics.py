#!/usr/bin/env python3
"""Full-output PS spans and completeness; use truth restricted to the target region."""
from pathlib import Path
from collections import defaultdict
import argparse,gzip,json

def rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):continue
            a=line.rstrip().split('\t')
            if len(a)<10:raise ValueError('Normalize malformed legacy VCFs before analysis')
            yield a

def summarize(phased,truth):
    truth_rows=list(rows(truth))
    truth_sites={(a[0],int(a[1])) for a in truth_rows if len(a[3])==len(a[4])==1 and a[9].split(':')[0] in ['0|1','1|0']}
    chromosomes={a[0] for a in truth_rows}
    if len(chromosomes)!=1:raise ValueError('Provide region-specific truth on one chromosome')
    blocks=defaultdict(list);matched=set();pipe=slash=0
    for a in rows(phased):
        if a[0] not in chromosomes:raise ValueError('Call and truth chromosomes differ')
        pipe+=int('|' in a[-1]);slash+=int('/' in a[-1])
        fmt=dict(zip(a[8].split(':'),a[9].split(':')));gt=fmt.get('GT','')
        if gt in ['0|1','1|0'] and len(a[3])==len(a[4])==1:
            key=(a[0],int(a[1]))
            if key in truth_sites:matched.add(key)
            if fmt.get('PS','.')!='.':blocks[(a[0],fmt['PS'])].append(int(a[1]))
    spans=sorted(max(x)-min(x) for x in blocks.values() if len(x)>1)
    def weighted_median(lengths):
        acc=0
        for v in lengths:
            acc+=v
            if acc>=sum(lengths)/2:return v
        return 0
    return dict(phased_records=pipe,unphased_records=slash,
                phasing_completeness_percent=100*pipe/(pipe+slash) if pipe+slash else None,
                legacy_one_minus_unphased_over_phased_percent=100*(1-slash/pipe) if pipe else None,
                matched_truth_sites=len(matched),region_truth_SNPs=len(truth_sites),
                SNP_completeness_percent=100*len(matched)/len(truth_sites) if truth_sites else None,
                full_output_PS_blocks=len(blocks),
                legacy_full_output_N50_bp=weighted_median(spans),
                descending_full_output_N50_bp=weighted_median(spans[::-1]),
                note='The legacy phasing-completeness helper prints slash/pipe; the two derived completeness expressions round identically for the audited tables. Its truth-position helper does not skip headers, but the released filtered truth has none. Here truth records explicitly exclude headers; use MHC.phased_snps.vcf.gz for MHC completeness, not whole chr6. Legacy N50 used ascending spans; both directions are reported for exact-half tie transparency.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phased',required=True,type=Path);p.add_argument('--truth',required=True,type=Path);p.add_argument('--json',type=Path)
    a=p.parse_args();text=json.dumps(summarize(a.phased,a.truth),indent=2)
    if a.json:a.json.write_text(text+'\n',encoding='utf-8')
    else:print(text)
