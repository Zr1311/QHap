"""Minimal new phasing controls. Run under WSL; inputs remain read-only.
Graphs are constructed once with the supplied code. Torch seeds are paired.
No post-processing, read extraction, or historic table timings are substituted.
"""
from pathlib import Path
import sys, os, time, json, pickle, contextlib, re, resource, hashlib,argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output_directory',type=Path,help='New or resumable control output directory')
args=parser.parse_args()
RUN=Path(__file__).resolve().parent
PUBLIC_ROOT=RUN.parent
BASE=args.output_directory.resolve()/'sources'
CODE=RUN/'historical_snapshot'
sys.path.insert(0,str(CODE))
import numpy as np, pandas as pd, scipy, torch
import get_init_matrix, get_matrix, init_haplotype, get_Snps_Map_n_block as graph_builder
import BSB_get_haplotype as std, BSB_get_haplotype_fast as fast
OUT=args.output_directory.resolve()/'controls';OUT.mkdir(parents=True,exist_ok=True)
import gzip,shutil
for ds,platform,region in [('data_WY_MHC','CycloneSEQ','MHC'),('data_HIFI_chr22','HiFi','chr22'),('data_ONT_chr22','ONT','chr22'),('data_WY_chr22_fast','CycloneSEQ','chr22')]:
 d=BASE/'data/data_snps'/ds;d.mkdir(parents=True,exist_ok=True)
 chrom='chr6_MHC' if region=='MHC' else 'chr22'
 for src,dst in [(RUN/'HG002_CHM13v2'/platform/(region+'.input.vcf.gz'),d/(chrom+'_snp_filter.vcf')),(RUN/'HG002_CHM13v2'/platform/(region+'.fragments.txt.gz'),d/'hapcut2_output.txt')]:
  if not dst.exists():
   with gzip.open(src,'rb') as inf,dst.open('wb') as outf:shutil.copyfileobj(inf,outf)
for chromosome in ['chr6','chr22']:
 truth_path=BASE/(chromosome+'.truth.vcf')
 if not truth_path.exists():
  with gzip.open(RUN/f'HG002_CHM13v2/truth/{chromosome}.phased_snps.vcf.gz','rb') as inf,truth_path.open('wb') as outf:shutil.copyfileobj(inf,outf)
sys.path.insert(0,str(RUN/'scripts'))
# The Windows-independent metric implementation is executed below with actual snapshot paths.
import ast, collections
def functions(p):
 t=ast.parse(p.read_text(encoding='utf-8-sig'));ns={};exec(compile(ast.Module(body=[n for n in t.body if isinstance(n,(ast.FunctionDef,ast.Import,ast.ImportFrom))],type_ignores=[]),str(p),'exec'),ns);return ns
HE=functions(CODE/'hamming_error_rate_test/3_he_rate.py')['calculate_hamming_error']
SE=functions(CODE/'switch error_test/3_long_short_se.py')['calculate_switch_error']
def vcf(p):
 out={}
 for l in Path(p).open():
  if l.startswith('#'):continue
  a=l.rstrip().split('\t')
  if len(a)<10:continue
  g=a[9].split(':')[0]
  if len(a[3])!=1 or len(a[4])!=1 or g not in ['0|1','1|0']:continue
  aa=[a[3],a[4]]; fm=dict(zip(a[8].split(':'),a[9].split(':')))
  out[(a[0],int(a[1]))]=(aa[int(g[0])],aa[int(g[2])],fm.get('PS','.'))
 return out
truths={c:vcf(BASE/(c+'.truth.vcf')) for c in ['chr6','chr22']}
def evaluate(p,chrom='chr22'):
 truth=truths[chrom]
 blocks=collections.defaultdict(list)
 for key,x in vcf(p).items():
  if key in truth:blocks[(key[0],x[2])].append((key[1],x,truth[key]))
 n=he=se=0
 for k,rows in blocks.items():
  rows.sort();pos=[str(r[0]) for r in rows];b=[pos,[r[2][0] for r in rows],[r[2][1] for r in rows]];h=[pos,[r[1][0] for r in rows],[r[1][1] for r in rows]]
  he+=min(HE(b,h,1,pos,{})[0],HE(b,h,2,pos,{})[0]);se+=min(sum(SE(b,h,1,pos,{})[:2]),sum(SE(b,h,2,pos,{})[:2]));n+=len(rows)
 return dict(n=n,HE_errors=he,SE_events=se,HE_percent=100*he/n,SE_percent=100*se/(n-1))

def prepare(name):
 cached=OUT/(name+'.pickle')
 if cached.exists():return pickle.loads(cached.read_bytes())
 d=BASE/'data/data_snps'/name;inp=next(d.glob('*snp_filter.vcf'));fr=d/'hapcut2_output.txt'
 with (OUT/(name+'_graph.log')).open('w') as log,contextlib.redirect_stdout(log):
  t=time.perf_counter();x=get_init_matrix.build_snp_sparse_matrix(str(inp),str(fr));matrix=x['sparse_matrix'];pos=x['position_to_column'];q=x['sparse_correct_rate']
  initial,_=init_haplotype.generate_haplotype_data(str(inp),matrix,pos);w=get_matrix.process_sparse_matrices(matrix,pos,initial,q)
  tgraph=time.perf_counter();components=graph_builder.main(w,pos,hapcut2_1_line_count=0,a=1);gt=time.perf_counter()-tgraph
  data=dict(name=name,input=str(inp),initial=initial,components=components,graph_seconds=gt,prepare_seconds=time.perf_counter()-t,nonzero_observations=len(w),fragments=len(x['fragment_to_index']))
 cached.write_bytes(pickle.dumps(data,protocol=4))
 return data

def run_setting(data,setting,seed):
 sub=OUT/data['name']/f'{setting}_seed{seed}';sub.mkdir(parents=True,exist_ok=True)
 module=fast
 fn=module.process_blocks
 if setting in ['steps2','samples2','field']:
  import inspect
  src=inspect.getsource(fast.process_blocks)
  if setting=='steps2':src=src.replace('n_iter = 1000','n_iter = 2000').replace('n_iter = 500','n_iter = 1000')
  if setting=='samples2':src=src.replace('sample_size = 100','sample_size = 200')
  if setting=='field':src=src.replace('SB(G, n_iter=', 'SB(G, h=0.0004, n_iter=')
  ns=dict(fast.__dict__);exec(compile(src,setting+'_diagnostic','exec'),ns);fn=ns['process_blocks']
 np.random.seed(666);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 with (sub/'solver.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log):
  fn(data['components'],data['initial'],str(sub),data['input'],enable_post_processing=False)
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 text=(sub/'solver.log').read_text()
 metrics=evaluate(sub/'phased.vcf',chrom='chr22' if 'chr22' in data['name'] else 'chr6')
 return dict(dataset=data['name'],setting=setting,seed=seed,seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),escalations=text.count('连续三次负割值，启用增强参数'),solve_calls=len(re.findall('最大割值:',text)),**metrics)

if __name__=='__main__':
 torch.set_num_threads(1)
 env=dict(torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),device_memory=torch.cuda.get_device_properties(0).total_memory,seeds=[20261019,20261020,20261021],scope='phasing core only; cached fragments and graph; no postprocessing')
 (OUT/'environment.json').write_text(json.dumps(env,indent=2))
 records=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else [];stats=[]
 for name in ['data_WY_MHC','data_HIFI_chr22','data_ONT_chr22','data_WY_chr22_fast']:
  data=prepare(name);cs=data['components'];largest=max(cs,key=lambda x:len(x[0]))
  nn=len(largest[0]);ee=len(largest[1]);stats.append(dict(dataset=name,total_components=len(cs),total_nodes=sum(len(c[0]) for c in cs),total_edges=sum(len(c[1]) for c in cs),largest_nodes=nn,largest_edges=ee,csr_bytes_float32_int64=2*ee*(4+8)+(nn+1)*8,spin_and_momentum_bytes_S100=2*nn*100*4,graph_seconds=data['graph_seconds'],prepare_seconds=data['prepare_seconds'],nonzero_observations=data['nonzero_observations'],fragments=data['fragments']))
  print('GRAPH',json.dumps(stats[-1]),flush=True)
  (OUT/'graph_statistics.json').write_text(json.dumps(stats,indent=2))
  # Real existing graph sizes, with the same fast budget to isolate size from escalation.
  for seed in [20261019,20261020,20261021]:
   if any(r['dataset']==name and r['setting']=='fast' and r['seed']==seed for r in records):continue
   r=run_setting(data,'fast',seed);records.append(r);print('RUN',json.dumps(r),flush=True)
   (OUT/'results.json').write_text(json.dumps(records,indent=2))
  if name=='data_WY_chr22_fast':
   for setting in ['steps2','samples2','field']:
    for seed in [20261019,20261020,20261021]:
     if any(r['dataset']==name and r['setting']==setting and r['seed']==seed for r in records):continue
     r=run_setting(data,setting,seed);records.append(r);print('RUN',json.dumps(r),flush=True)
     (OUT/'results.json').write_text(json.dumps(records,indent=2))
