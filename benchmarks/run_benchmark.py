#!/usr/bin/env python3
"""Run one published benchmark in a fresh output directory; inputs stay unchanged."""
from pathlib import Path
import argparse,gzip,json,hashlib,shutil,subprocess,sys,os
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'benchmarks/HG002_CHM13v2'
def unpack(source,destination):
    with gzip.open(source,'rb') as r,destination.open('wb') as w:
        shutil.copyfileobj(r,w)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--platform',choices=['HiFi','ONT','CycloneSEQ'],required=True)
    p.add_argument('--region',choices=['MHC','chr22'],required=True)
    p.add_argument('--method',choices=['snp','read'],default='snp')
    p.add_argument('--backend',choices=['standard','fast','mindquantum'],default='standard')
    p.add_argument('--seed',type=int,default=20261019)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.method=='read' and args.backend=='fast':p.error('fast is an SNP-only backend')
    if args.method=='read' and args.region!='MHC':p.error('The packaged historical read benchmarks cover MHC; use the core CLI for other inputs')
    if args.output.exists():p.error('Use a new output directory; existing work is not overwritten')
    args.output.mkdir(parents=True)
    source=DATA/args.platform
    unpack(source/f'{args.region}.input.vcf.gz',args.output/'input.vcf')
    fragments=(DATA/'ReadBased'/args.platform/'MHC.fragments.txt.gz') if args.method=='read' else source/f'{args.region}.fragments.txt.gz'
    unpack(fragments,args.output/'fragments.txt')
    chrom='chr6' if args.region=='MHC' else 'chr22'
    unpack(DATA/'truth'/f'{chrom}.phased_snps.vcf.gz',args.output/'truth.vcf')
    script=ROOT/('SNP-based' if args.method=='snp' else 'Read-based')/'index_quantum_maxcut_phasing_n_return.py'
    flags=[]
    if args.backend=='mindquantum':flags.append('--mindquantum')
    elif args.backend=='fast':flags.append('--fast')
    if args.method=='snp':flags.append('--no-post')
    elif args.backend=='standard':
        flags+=['--steps','20000' if args.platform=='ONT' else '10000',
                '--samples','1000' if args.platform=='ONT' else '200']
    # Seed both Python initialization and PyTorch before loading the CLI.
    code=('import random,runpy,sys; import numpy as np; import torch; '
          f'random.seed(42); np.random.seed(666); torch.manual_seed({args.seed}); '
          f'sys.path.insert(0,{str(script.parent)!r}); '
          f'sys.argv={[str(script),*flags,str(args.output.resolve()/"input.vcf"),str(args.output.resolve()/"fragments.txt")]!r}; '
          f'runpy.run_path({str(script)!r},run_name="__main__")')
    with (args.output/'run.log').open('w') as log:
        subprocess.run([sys.executable,'-c',code],stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,str(ROOT/'evaluation/evaluate.py'),
                    '--phased',str(args.output/'phasing_output/phased.vcf'),
                    '--truth',str(args.output/'truth.vcf'),
                    '--json',str(args.output/'metrics.json')],check=True)
    (args.output/'configuration.json').write_text(json.dumps(vars(args),default=str,indent=2))
    print('Completed:',args.output)
if __name__=='__main__':main()
