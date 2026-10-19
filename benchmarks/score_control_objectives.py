"""Score final phase assignments on the unchanged original CycloneSEQ chr22 graph.

The signed cut is sum(w_uv for flipped_u != flipped_v), relative to the
released initial haplotype. Scores of reparameterized intermediate graphs
are deliberately not compared. Larger values improve this surrogate only.
"""
from pathlib import Path
import argparse,json,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent

def alleles(lines):
    result={}
    for line in lines:
        if line.startswith('#'):continue
        a=line.rstrip().split('\t')
        if len(a)<10:continue
        gt=a[9].split(':')[0]
        if gt not in ('0|1','1|0'):continue
        options=[a[3],a[4]]
        result[int(a[1])]=(options[int(gt[0])],options[int(gt[2])])
    return result

def score(folder):
    records=json.loads((folder/'results.json').read_text())
    graph=ROOT/'control_results/graphs/data_WY_chr22_fast.npz'
    with np.load(graph,allow_pickle=False) as ar:
        metadata=json.loads(str(ar['metadata']))
        positions=ar['initial_positions'];seq1=ar['initial_sequence1'];seq2=ar['initial_sequence2']
        index={int(pos):i for i,pos in enumerate(positions)}
        edges=[]
        for i in range(metadata['component_count']):
            uv=ar[f'edge_uv_{i}'];weights=ar[f'edge_w_{i}']
            left=np.array([index[int(p)] for p in uv[:,0]],dtype=np.int64)
            right=np.array([index[int(p)] for p in uv[:,1]],dtype=np.int64)
            edges.append((ar[f'nodes_{i}'].size,left,right,weights))
    output=[]
    zpath=folder/'completed_run_outputs.zip'
    z=zipfile.ZipFile(zpath) if zpath.exists() else None
    for record in records:
        if record['dataset']!='data_WY_chr22_fast':continue
        relative=f'{record["dataset"]}/{record["setting"]}_seed{record["seed"]}/phased.vcf'
        p=folder/relative
        if p.exists():calls=alleles(p.read_text().splitlines())
        elif z is not None:calls=alleles(z.read(relative).decode('utf-8-sig').splitlines())
        else:raise FileNotFoundError(p)
        flipped=np.zeros(len(positions),dtype=bool)
        for i,pos in enumerate(positions):
            pair=calls[int(pos)]
            if pair==(seq1[i],seq2[i]):continue
            if pair==(seq2[i],seq1[i]):flipped[i]=True
            else:raise ValueError(f'Allele set mismatch at {pos}')
        component_scores=[dict(nodes=int(n),cut=float(np.sum(w[flipped[l]!=flipped[r]],dtype=np.float64))) for n,l,r,w in edges]
        output.append(dict(dataset=record['dataset'],setting=record['setting'],seed=record['seed'],
            HE_percent=record['HE_percent'],HE_errors=record['HE_errors'],
            original_graph_cut=sum(c['cut'] for c in component_scores),
            largest_component_cut=max(component_scores,key=lambda x:x['nodes'])['cut'],components=component_scores))
    if z is not None:z.close()
    return dict(definition='Sum of original signed edge weights crossing final flip parity, relative to the released initial haplotype; undirected edges counted once',
        graph='data_WY_chr22_fast.npz',interpretation='Higher cut is a better graph-surrogate value, not a biological accuracy or global-optimality certificate',runs=output)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control_directory',type=Path,help='Unpacked controls, or released control_results with completed_run_outputs.zip')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=score(args.control_directory)
    args.output.write_text(json.dumps(result,indent=2))
    for r in result['runs']:print(r['setting'],r['seed'],r['HE_percent'],r['original_graph_cut'])
