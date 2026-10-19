# HG002 benchmark reproduction

The `HG002_CHM13v2/` directory contains three-platform SNP inputs, extracted haplotype-informative fragments, filtered phased truth, and retained outputs for the MHC region and chromosome 22. Pore-C inputs are provided separately under `PoreC/`. `MANIFEST.json` maps each file to its role and gives compressed and uncompressed SHA-256 checksums. The files formerly named `hapcut2_output.txt` are fragment files, **not HapCUT2 phasing results**.

All alignments use CHM13v2 (`chr6` length 172,126,628; `chr22` length 51,324,926). The MHC selection interval in the retained script is chr6:28,477,797–33,448,354. The platform directories describe reads from the same individual; they are not biological replicates. Original instrument studies and the exact historical subsampling and calling commands have not all been accessioned or recovered. Alignment availability is recorded in `ALIGNMENTS.json`; the existing HiFi chr6 release is reused.

## Run a corrected implementation

Install the repository dependencies. Run from the repository root in Linux or WSL:

```bash
python3 benchmarks/run_benchmark.py --platform CycloneSEQ --region chr22 \
  --method snp --backend fast --seed 20261019 --output reproduction_runs/cycloneseq_chr22
```

Use `--platform HiFi` or `--platform ONT`, and `--region MHC` as required. `--method read --backend standard` selects the read-based solver. `--backend mindquantum` retains the existing optional implementation. The fast PyTorch solver cannot be combined with MindQuantum. The wrapper writes into a **new** directory and leaves the packaged inputs unchanged. Post-processing is disabled for these comparison runs.

The corrected implementation preserves whole fragments across graph-building worker batches and fixes quality-string indexing over multiple fragment runs. Thus, newly generated results must be reported with this code revision and do not replace historical manuscript table values. Read-graph edge weights and component membership are preserved; deterministic ordering and component checks prevent invalid partitions and silent phase-set overwrites. If independent conflict components share a variant, the read workflow reports an error and recommends SNP-based phasing, rather than inventing their relative orientation.

## Historical outputs and evaluation

Decompress a retained `*.original.phased.vcf.gz` and the appropriate `truth/*.phased_snps.vcf.gz`, then run:

```bash
python3 evaluation/evaluate.py --phased retained.vcf --truth truth.vcf --json metrics.json
```

The evaluator deliberately retains the manuscript's **event-based SE**: each maximal disagreement run counts once, choosing the orientation with the fewest events per output phase set, divided by pooled unique evaluated sites minus one. HE chooses the orientation with fewer site disagreements per block. Incompatible allele sets remain in the original denominator and are reported separately. This SE is distinct from counting every adjacent phase transition. Run each chromosome separately. JSON contains exact percentages; original table rounding is a separate reporting layer. The evaluator's N50 uses evaluated sites; full output PS spans underlie the manuscript N50.

## Competitors and timing

The historical comparison used HapCUT2 and WhatsHap on a single CPU thread, versus QHap using 16 fragment-extraction workers in retained logs, up to eight graph-construction processes, and one GPU. The original end-to-end QHap times were measured with WSL `time` around **both** `extractHAIRS.py` and the phasing entry point. Internal timing files include only the second stage, and must not be substituted for those wall times. Read mapping and variant calling were outside that timing boundary.

Retained WhatsHap VCF headers identify version 2.8 and the command form below. HapCUT2's historical executable revision and full extraction commands were not recovered. The original tool-specific inputs, fragments, and outputs are under `HG002_CHM13v2/competitors/`. Variant order agrees across tools, but HapCUT2's extractor and QHap's Python extractor produced different fragment observations. Equal variant inputs do not establish identical preprocessing.

```bash
whatshap phase -o phase.vcf --reference=chr22.fa chr22.input.vcf chr22.bam --ignore-read-groups
HAPCUT2 --fragments hapcut2.fragments.txt --VCF chr22.input.vcf --output output.txt
```

The second command is a portable new-run protocol, not a recovered historical invocation. Decompress the relevant tool-specific files first. For extraction from a BAM, record the version and explicit long-read options for the installed HapCUT2 `extractHAIRS`; do not silently substitute QHap fragments. A compatible CHM13v2 reference is described by [NCBI GCA_009914755.4](https://www.ncbi.nlm.nih.gov/datasets/genome/GCA_009914755.4/) and the [CHM13 project](https://github.com/marbl/CHM13); contig names must match the BAM/VCF. Historical reference FASTA hashes were not retained. HapCUT2 block output must be converted using its supplied conversion utility before evaluation. Record wall time, CPU threads, GPU use and peak RAM separately. Historical repeat-level wall times are unavailable, so these instructions do not promise reproduction to the last decimal place.

Some retained HapCUT2 VCFs have a standalone `:.` continuation after an unphased record. `evaluation/normalize_legacy_vcf.py` conservatively joins that exact artifact, rejects unexpected malformed records, and changes no phased genotype or phase-set value. Both raw and normalized files are packaged when needed. Run the evaluator on the normalized version. Re-evaluation of the retained HiFi MHC WhatsHap VCF gives one error/event (about 0.0092%) whereas the historical table says 0.00%; the run association could not be resolved. The historical CycloneSEQ fast retained HE is 24.0473% versus the table's 24.04%. These discrepancies are recorded rather than silently rewriting historical values.

For Pore-C, decompress the four matching `longread.*` and `porec.*` inputs and use the existing `--porec` interface with `a=0.1`. Keep the weighting flag and backend in the run record. Inputs are reused as paired files; do not mix fragment indices with another VCF order.

## Regression verification

```bash
python3 -m unittest discover -s tests -v
```

These tests cover read conflicts at sparse row indices, component endpoints, overlapping phase blocks, whole-fragment batching, multi-run quality indexing, and the preserved SE/HE convention. The optional MindQuantum source files are retained unchanged; a backend accuracy-parity claim would require separate matched experiments.

## Reproduce the 21 parameter controls

```bash
python3 benchmarks/reproduce_controls.py reproduction_runs/controls
```

This uses `historical_snapshot/`, not the corrected graph code, to keep the diagnostic experiments tied to the supplied local implementation. By default it loads four numeric graph archives with explicit vertex and edge order, pairs three technical seeds, and varies iterations, samples, or the deformation term h on CycloneSEQ chr22. There are 21 reported runs: three seeds for four base datasets and three seeds for each of three single-factor CycloneSEQ chr22 parameter controls. It resumes completed records. The tested environment was Python 3.12.3, PyTorch 2.11.0+cu128, NumPy 2.3.2, SciPy 1.18.1, pandas 3.0.6, pysam 0.24.1 and one RTX 5060 Laptop GPU under WSL. Select a CUDA build compatible with the device. These new versions are not claimed to be the historical table environment. Core timing excludes extraction and graph construction; all post-processing is disabled. Full escalation of the standard backend was not tested by these controls.

The final series supersedes the preliminary 21-run series in earlier commits. A baseline replay differed, and its specific cause was not isolated. All reported controls were therefore repeated from the released ordered arrays to control seed-to-vertex assignments across combinations. This changes only the newly added controls, not historical manuscript Tables 1–3. `cache_order_sensitivity.json` records the failed replay and the limits of attributing it to ordering. Use the ordered cache for paired reproduction; rebuilding can change worker aggregation order and floating-point results. Identical seeds alone do not guarantee bitwise replay of GPU computations (PyTorch 2.11 reproducibility notes: https://docs.pytorch.org/docs/2.11/notes/randomness.html).

`score_control_objectives.py control_results --output objectives.json` scores final assignments on the unchanged original signed graph, using the archived VCFs. It reports the sum of original weights crossing the final flip parity, with each undirected edge counted once. Intermediate reparameterized-graph cut values are not directly compared. Higher objective values measure this surrogate, not biological phase accuracy or certified global convergence.

The name `historical_snapshot/` denotes the supplied local source preserved during this audit. It is not a claim that its exact revision generated every historical table row: run-specific source revisions and the original runtime environment were not retained. By default the controls load the released numeric graph caches (with checksum verification and `allow_pickle=False`), preserving retained node/edge order and initialization. This avoids silently changing the diagnostic graphs through the historical worker-batching defect. `--rebuild-graphs` instead reruns the historical builder; report the worker count and check the resulting graph against the archive. Recorded cache build times are original observations, not new rebuild measurements. Use `--dataset data_HIFI_chr22 --seed 20261019` for one functional replay.

`results.json` records externally measured, GPU-synchronized elapsed times for each control. Some module-internal time counters printed in the historical solver logs accumulate across calls in one Python process; do not substitute those cumulative counters for the per-run elapsed times. Functional replay preserves the within-block relative phases of the tested HiFi run, but whole-block haplotype labels can flip; byte-identical VCF output is not promised.

The original repository extractor defaults to eight workers. To request the 16-worker extraction configuration recorded in retained local logs, pass `--threads 16` explicitly. Graph inference separately caps its worker pool at eight. Complete run-to-log association for historical timing entries was not retained.

The original MHC extraction input was the **complete chromosome 6 BAM together with an MHC-region VCF**, not a BAM cropped to MHC. This applies to the original end-to-end timing boundary. Complete chr6 alignment files are distinguished from any convenience `chr6_MHC` subsets in the alignment manifest. Derived subsets contain all alignments overlapping the stated MHC interval, with no quality filtering or downsampling, but their extraction time must not be compared with the historical full-chr6-input timing. Retained fragment files permit inference-only reproduction without substituting an extraction input.

For a new MHC extraction run, decompress the matching `MHC.input.vcf.gz` into a fresh output directory and use the full downloaded chr6 BAM. A current-code template is:

```bash
python3 extractHAIRS.py --bam full_chr6.bam --vcf MHC.input.vcf \
  --out MHC.fragments.txt --threads 16
python3 SNP-based/index_quantum_maxcut_phasing_n_return.py --no-post \
  MHC.input.vcf MHC.fragments.txt
```

These are new-run commands, not recovered historical invocations. Keep the VCF and fragment index order paired. For historical end-to-end comparison, place both commands in the same WSL `time` scope and record all flags; timing only the second command or using a cropped BAM changes the boundary. Use the read entry point with its explicit reported budget, or the SNP `--fast` option for the fast configuration, as described above.

## Truth processing

The author-identified original files are CHM13v2.0_HG2-T2TQ100-V1.1_smvar.vcf and its .benchmark.bed. Unfiltered chromosome subsets with original headers are in truth/source/. The full original VCF checksum is in MANIFEST.json. The supplied historical scripts are preserved under truth_original_scripts/; their hardcoded chr1 defaults are not a portable chr6/chr22 protocol.

Run filter_truth.py with --source, --bed, --chromosome and a fresh --output. It selects one-base REF/ALT records, phased GT 0|1 or 1|0 and BED membership. Default --mode legacy reproduces the historical inclusive test start <= VCF POS <= end. --mode standard uses start <= POS-1 < end. The two modes differ by two chr6 boundary sites (one in MHC); chr22 results are identical. Ordered reconstructed legacy records match all supplied final truth files. The historical mask is retained for the original accuracy evaluation. No PASS filter or normalization step is silently added. For MHC use --start 28477797 --end 33448354 after selecting chr6.

## Read-specific inputs and budgets

ReadBased/{platform}/ contains historical MHC fragments and outputs. HiFi and ONT fragment observations differ from their SNP counterparts; the variant order is identical. CycloneSEQ fragments match in both formulations. The wrapper selects the read-specific inputs and exposes the manuscript's 10,000-iteration/200-sample HiFi/CycloneSEQ and 20,000/1,000 ONT budgets. The audited local and original public read files contain the 10,000/200 default; the exact historical ONT source revision was not retained. The optional MindQuantum files are unchanged and retain their own interface. Direct PyTorch read CLI runs accept --steps and --samples; the selected budget is recorded in run_time.txt.

For full-output N50 and completeness use evaluation/other_metrics.py with --phased and region-specific --truth. Do not use whole-chr6 truth as the MHC completeness denominator. The legacy helper's slash/pipe ratio and the phased/all ratio give identical table values at the original precision for these inputs. Standard versus legacy weighted-median tie behavior is reported explicitly.

Transfer-manuscript table order: Table 1, MHC benchmarks; Table 2, chr22 benchmarks; Table 3, bSB parameter controls; Table 4, the retained HLA application. Original benchmark and HLA numerical bodies are unchanged.
