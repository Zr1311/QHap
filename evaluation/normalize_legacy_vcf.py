#!/usr/bin/env python3
"""Join the exact ':.' continuation artifact in retained unphased VCF rows.

No genotype, phase-set value, allele, position, or phased row is changed.
The original file must be retained; use a new output path. Other malformed
records are rejected. This is a format repair, not a phasing algorithm.
"""
from pathlib import Path
import argparse,gzip,json

def normalize(source, destination):
    source,destination=Path(source),Path(destination)
    if destination.exists():raise FileExistsError(destination)
    opener=gzip.open if str(source).endswith('.gz') else open
    with opener(source,'rt',encoding='utf-8') as f:lines=f.read().splitlines()
    output=[];repairs=[]
    for number,line in enumerate(lines,1):
        if line==':.' and output and not output[-1].startswith('#'):
            fields=output[-1].split('\t')
            if len(fields)!=10 or '|' in fields[9].split(':')[0]:
                raise ValueError(f'Unexpected continuation at line {number}')
            output[-1]+=line;repairs.append(number)
        elif line.startswith('#') or len(line.split('\t'))==10:
            output.append(line)
        else:raise ValueError(f'Malformed VCF at line {number}')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text('\n'.join(output)+'\n',encoding='utf-8')
    return dict(joined_continuation_lines=repairs,genotypes_or_phase_values_changed=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(normalize(a.input,a.output),indent=2))
