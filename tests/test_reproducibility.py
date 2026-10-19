"""Regression cases for scientific correctness, not implementation mirrors."""
from pathlib import Path
import importlib.util,sys,tempfile,unittest,contextlib,io,math
import numpy as np
from scipy.sparse import coo_matrix
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
READ=load('read_graph',ROOT/'Read-based/get_Snps_Map_reads_block.py')
SNP=load('snp_graph',ROOT/'SNP-based/get_Snps_Map_n_block.py')
PARSER=load('snp_parser',ROOT/'SNP-based/get_init_matrix.py')
METRIC=load('legacy_metric',ROOT/'evaluation/evaluate.py')
NORMALIZER=load('legacy_normalizer',ROOT/'evaluation/normalize_legacy_vcf.py')
TRUTH_FILTER=load('truth_filter',ROOT/'benchmarks/filter_truth.py')
OTHER=load('other_metrics',ROOT/'evaluation/other_metrics.py')
SUBSET=load('subset_alignments',ROOT/'benchmarks/subset_alignments.py')

class ReproducibilityTests(unittest.TestCase):
 def test_read_edges_with_gaps_and_agreement(self):
  # Distant row indices and explicit zero observations must not create false edges.
  a=coo_matrix(([1,-1,1,0],([0,100,5,2],[0,0,0,0])),shape=(101,1))
  READ.snp_data=a.tocsc()
  self.assertEqual(READ.process_column(0),{(0,100):1,(5,100):1})

 def test_component_edges_never_cross_partition(self):
  with self.assertRaisesRegex(ValueError,'crosses'):
   READ.create_component_graphs(None,[{'a','b'},{'c','d'}],[('a','c',1)])

 def test_overlapping_read_blocks_are_not_overwritten(self):
  components={0:{'nodes':['a','b'],'edges':[('a','b',1)]},1:{'nodes':['c','d'],'edges':[('c','d',1)]}}
  # Components agree at column 1 and conflict at separate columns.
  triples=[(0,0,1),(1,0,-1),(0,1,1),(2,1,1),(2,2,1),(3,2,-1)]
  with self.assertRaisesRegex(ValueError,'relative orientation'):
   READ.validate_component_variant_sets(components,triples,{'a':0,'b':1,'c':2,'d':3})
  self.assertTrue(READ.validate_component_variant_sets(components,triples[:3]+triples[4:],{'a':0,'b':1,'c':2,'d':3}))

 def test_fragment_chunk_boundary_keeps_distant_pair_and_normalization(self):
  # One fragment exceeds the previous 1,000-observation chunk threshold.
  # The first/last pair was lost when that fragment was split into two chunks.
  k=1001
  with contextlib.redirect_stdout(io.StringIO()):
   edges,_,_=SNP.compute_edge_weights_optimized([(0,j,.99) for j in range(k)],{10000+j:j for j in range(k)})
  expected=-math.log10((.99*.99+.01*.01)/(.99*.01+.99*.01))/(k-1)
  self.assertAlmostEqual(edges[(10000,11000)],expected,places=12)

 def test_quality_string_advances_across_fragment_runs(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'fragments.txt';p.write_text('2 r1 1 0 3 10 5I+\n')
   x=PARSER.process_phasing_file(p,[('A','C')]*4,[1,2,3,4],{1:0,2:1,3:2,4:3})
   q={(row,col):v for row,col,v in x['sparse_correct_rate']}
   self.assertAlmostEqual(q[(0,0)],.99)
   self.assertAlmostEqual(q[(0,2)],.9999)
   self.assertAlmostEqual(q[(0,3)],.9)

 def test_legacy_metric_keeps_event_definition_and_block_orientation(self):
  reference=[list('12345'),list('AAAAA'),list('CCCCC')]
  inferred=[list('12345'),list('ACCCA'),list('CAAAC')]
  self.assertEqual(min(METRIC.calculate_hamming_error(reference,inferred,o,list('12345'),{})[0] for o in (1,2)),2)
  self.assertEqual(min(sum(METRIC.calculate_switch_error(reference,inferred,o,list('12345'),{})[:2]) for o in (1,2)),1)

 def test_legacy_format_repair_does_not_change_phased_calls(self):
  with tempfile.TemporaryDirectory() as td:
   src=Path(td)/'raw.vcf';dst=Path(td)/'normalized.vcf'
   phased='chr6\t1\t.\tA\tC\t.\tPASS\t.\tGT:PS\t0|1:1'
   unphased='chr6\t2\t.\tA\tC\t.\tPASS\t.\tGT:PS\t0/1'
   src.write_text('#header\n'+phased+'\n'+unphased+'\n:.\n')
   report=NORMALIZER.normalize(src,dst)
   self.assertEqual(report['joined_continuation_lines'],[4])
   self.assertEqual(dst.read_text().splitlines()[1],phased)
   self.assertEqual(dst.read_text().splitlines()[2],unphased+':.')
   src.write_text('#header\n'+phased+'\n:.\n')
   with self.assertRaisesRegex(ValueError,'Unexpected continuation'):
    NORMALIZER.normalize(src,Path(td)/'invalid.vcf')

 def test_truth_coordinates_preserve_the_recorded_legacy_rule(self):
  # BED [10,20) is VCF positions 11..20 under the standard conversion.
  # The supplied legacy script also included POS=10. Both are explicit.
  self.assertTrue(TRUTH_FILTER.keep_position(10,[10],[20],'legacy'))
  self.assertFalse(TRUTH_FILTER.keep_position(10,[10],[20],'standard'))
  for mode in ['legacy','standard']:
   self.assertTrue(TRUTH_FILTER.keep_position(11,[10],[20],mode))
   self.assertTrue(TRUTH_FILTER.keep_position(20,[10],[20],mode))
   self.assertFalse(TRUTH_FILTER.keep_position(21,[10],[20],mode))

 def test_full_output_spans_and_completeness_on_known_vcf(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'calls.vcf';truth=Path(td)/'truth.vcf'
   data=[(10,'0|1','1'),(20,'1|0','1'),(30,'0|1','2'),(70,'1|0','2'),(80,'0/1','.')]
   p.write_text(''.join(f'chr6\t{pos}\t.\tA\tC\t.\tPASS\t.\tGT:PS\t{gt}:{ps}\n' for pos,gt,ps in data))
   truth.write_text(''.join(f'chr6\t{pos}\t.\tA\tC\t.\tPASS\t.\tGT\t0|1\n' for pos,gt,ps in data))
   result=OTHER.summarize(p,truth)
   self.assertEqual(result['phasing_completeness_percent'],80)
   self.assertEqual(result['SNP_completeness_percent'],80)
   self.assertEqual(result['legacy_full_output_N50_bp'],40)

 def test_alignment_subset_preserves_all_overlapping_records(self):
  import pysam
  with tempfile.TemporaryDirectory() as td:
   source=Path(td)/'source.bam';out=Path(td)/'subset.bam'
   header={'HD':{'VN':'1.6','SO':'coordinate'},'SQ':[{'SN':'chr6','LN':100}]}
   with pysam.AlignmentFile(source,'wb',header=header) as f:
    for i,(start,flag) in enumerate([(0,0),(9,0),(15,256),(20,0)]):
     read=pysam.AlignedSegment();read.query_name=f'r{i}';read.query_sequence='AAAAA';read.flag=flag
     read.reference_id=0;read.reference_start=start;read.mapping_quality=5;read.cigar=[(0,5)]
     read.query_qualities=pysam.qualitystring_to_array('IIIII');f.write(read)
   pysam.index(str(source));before=SUBSET.sha(source)
   result=SUBSET.subset(source,out,'chr6',10,20,1)
   with pysam.AlignmentFile(out,'rb') as f:records=list(f.fetch('chr6'))
   self.assertEqual([x.query_name for x in records],['r1','r2'])
   self.assertEqual(records[1].flag,256)
   self.assertEqual(result['alignment_records'],2)
   self.assertEqual(SUBSET.sha(source),before)

if __name__=='__main__':unittest.main()
