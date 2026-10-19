#!/usr/bin/env python3
"""Write a coordinate-overlap BAM subset without filtering or downsampling."""
from pathlib import Path
import argparse,hashlib,json,pysam

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def subset(source,output,chromosome,start,end,threads=4):
    source,output=Path(source),Path(output)
    if output.exists() or Path(str(output)+'.bai').exists():raise FileExistsError(output)
    if start<1 or end<start or threads<1:raise ValueError('Use positive one-based closed coordinates and thread count')
    output.parent.mkdir(parents=True,exist_ok=True);count=0
    with pysam.AlignmentFile(source,'rb') as inf,pysam.AlignmentFile(output,'wb',template=inf,threads=threads) as outf:
        for record in inf.fetch(chromosome,start-1,end):
            outf.write(record);count+=1
    pysam.index(str(output))
    return dict(region=dict(chromosome=chromosome,start=start,end=end,coordinate_system='one-based closed'),
                selection='all overlapping alignments; no quality filter or downsampling',
                source_file=source.name,source_sha256=sha(source),alignment_records=count,
                output_file=output.name,output_sha256=sha(output),output_bytes=output.stat().st_size,
                index_sha256=sha(Path(str(output)+'.bai')),pysam=pysam.__version__,threads=threads)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bam',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--chromosome',default='chr6');p.add_argument('--start',type=int,default=28477797);p.add_argument('--end',type=int,default=33448354);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();print(json.dumps(subset(a.bam,a.output,a.chromosome,a.start,a.end,a.threads),indent=2))
