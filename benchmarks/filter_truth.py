#!/usr/bin/env python3
"""Reproduce the supplied phased-SNP truth filter without changing legacy coordinates.

Legacy mode tests BED start <= VCF POS <= BED end, as in filtered_as_bed.py.
Standard mode tests BED start <= POS-1 < BED end. The legacy output is retained
for manuscript reproduction. No PASS filter or allele normalization is inferred.
"""
from pathlib import Path
from bisect import bisect_right
import argparse,gzip,json,hashlib

def open_text(path):
    return gzip.open(path,'rt',encoding='utf-8') if str(path).endswith('.gz') else open(path,encoding='utf-8')

def intervals(path,chromosome):
    ranges=[]
    with open_text(path) as f:
        for line in f:
            a=line.split()
            if len(a)>=3 and a[0]==chromosome:ranges.append((int(a[1]),int(a[2])))
    ranges.sort();starts=[a for a,b in ranges];maxends=[];end=-1
    for a,b in ranges:end=max(end,b);maxends.append(end)
    return starts,maxends

def keep_position(position,starts,maxends,mode):
    coordinate=position if mode=='legacy' else position-1
    i=bisect_right(starts,coordinate)-1
    return i>=0 and (coordinate<=maxends[i] if mode=='legacy' else coordinate<maxends[i])

def filter_truth(source,bed,chromosome,output,mode='legacy',start=None,end=None,keep_headers=False):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    starts,maxends=intervals(bed,chromosome);counts=dict(chromosome_records=0,length_one_biallelic_records=0,phased_heterozygous_records=0,retained=0)
    output.parent.mkdir(parents=True,exist_ok=True)
    with open_text(source) as inf,output.open('w',encoding='utf-8',newline='') as outf:
        for line in inf:
            if line.startswith('#'):
                if keep_headers:outf.write(line)
                continue
            a=line.rstrip().split('\t')
            if len(a)<10 or a[0]!=chromosome:continue
            counts['chromosome_records']+=1;pos=int(a[1])
            if len(a[3])!=1 or len(a[4])!=1:continue
            counts['length_one_biallelic_records']+=1
            if a[9].split(':')[0] not in ['0|1','1|0']:continue
            counts['phased_heterozygous_records']+=1
            if start is not None and pos<start:continue
            if end is not None and pos>end:continue
            if keep_position(pos,starts,maxends,mode):outf.write(line);counts['retained']+=1
    return dict(chromosome=chromosome,mode=mode,start=start,end=end,counts=counts,
                output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                transformations='select chromosome; length REF=ALT=1; GT 0|1 or 1|0; optional closed interval; BED mask; no added PASS filter')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for option in ['source','bed','output']:p.add_argument('--'+option,required=True,type=Path)
    p.add_argument('--chromosome',required=True);p.add_argument('--mode',choices=['legacy','standard'],default='legacy')
    p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--keep-headers',action='store_true')
    a=p.parse_args();print(json.dumps(filter_truth(a.source,a.bed,a.chromosome,a.output,a.mode,a.start,a.end,a.keep_headers),indent=2))
