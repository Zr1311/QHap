"""Load numeric, ordered versions of the retained diagnostic graph caches."""
from pathlib import Path
import hashlib,json
import numpy as np

def load_graph(name,input_vcf):
    folder=Path(__file__).resolve().parent/'control_results/graphs'
    manifest=json.loads((folder/'MANIFEST.json').read_text())
    record=manifest[name];path=folder/record['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
        raise ValueError('Control graph checksum mismatch: '+name)
    with np.load(path,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']))
        initial={key:archive['initial_'+key].tolist() for key in ['positions','sequence1','sequence2']}
        components=[]
        for i in range(metadata['component_count']):
            nodes=archive[f'nodes_{i}'].tolist()
            uv=archive[f'edge_uv_{i}'];weights=archive[f'edge_w_{i}']
            edges=list(zip(uv[:,0].tolist(),uv[:,1].tolist(),weights.tolist()))
            components.append((nodes,edges))
    metadata.update(initial=initial,components=components,input=str(input_vcf),graph_source='released exact graph cache; recorded build time is historical')
    return metadata
