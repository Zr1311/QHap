# QHap HG002 / CHM13v2 benchmark inputs

This processed-input release supports the original QHap MHC and chromosome 22 comparisons. It permits inspection of retained outputs and rerunning graph inference. It is not a raw instrument archive. All three platforms represent HG002.

## File mapping

- `HiFi/`, `ONT/`, `CycloneSEQ/`: each MHC or chr22 input VCF is paired with its identically prefixed fragment file. Fragment indices refer to that VCF's variant order. `chr6.calls.vcf.gz` and `chr22.calls.vcf.gz` are caller-filtered inputs before historical region-specific selection.
- `*.original.phased.vcf.gz`: retained historical QHap output. CycloneSEQ chr22 used the fast backend. These results are not regenerated with corrected code.
- `*.internal_timing.txt.gz`: timing inside the second phasing script; this excludes extraction and is not the historical WSL pipeline wall time.
- `PoreC/{platform}/{MHC,chr22}/`: paired long-read and Pore-C input VCFs/fragments and retained integrated output. Pore-C is incorporated as pairwise evidence; SNP weighting is `a=0.1`.
- `competitors/`: each baseline's own ordered variant input and retained output. HapCUT2 includes its own extracted fragments, which differ from QHap's Python-extracted observations. A normalized copy is included where raw unphased rows contain the known continuation artifact.
- `truth/`: filtered phased SNP truth files used for historical metrics and the retained benchmark region BED. The source filename identifies CHM13v2.0 HG2-T2TQ100-V1.1 small variants; these files do not replace a fully cited upstream benchmark record.
- `MANIFEST.json` and `SHA256SUMS`: compressed/uncompressed hashes, byte sizes and roles.
- `ALIGNMENTS.json`: separately downloadable BAM/BAI files and chunk restoration hashes. Large alignments are release assets, not Git blobs. Original MHC experiments used the full `chr6.bam` with the MHC-region VCF; any derived `chr6_MHC` subset is a convenience input and is not the input underlying historical end-to-end times.

All compressed text files are deterministic gzip streams. VCF positions are one-based; the MHC selection script used chr6:28,477,797–33,448,354. BAM-header contig lengths are 172,126,628 for chr6 and 51,324,926 for chr22. Fragment rows follow the HapCUT-style sparse observation format. GT and PS fields in output VCFs determine orientation and block membership; lengths use base pairs, and HE/SE are percentages.

## Provenance and reuse

The release preserves supplied processed inputs and renames local labels (`WY` to `CycloneSEQ`; `HIFI` to `HiFi`). It does not infer instrument accessions, exact historical downsampling/calling commands, or a versioned HLA reference. CycloneSEQ and Pore-C source study accessions, upstream benchmark landing/version metadata, and data-specific reuse terms remain author provenance requirements. The repository's Apache-2.0 software licence is not a newly asserted licence for third-party sequencing data. Existing source rights continue to apply.

See the parent benchmark README for methods, environments, commands, metrics and limits. Community sequencing archive deposition and a DOI archive of this snapshot remain to be completed as required by the journal; GitHub availability alone does not establish that all deposition requirements are met.

Complete supplied chr6 and chr22 alignments for all three platforms are listed in ALIGNMENTS.json, together with optional derived MHC subsets. Original MHC extraction/timing used complete chr6 plus the regional VCF. File-level and part-level hashes distinguish original inputs from derived conveniences.
