# HG002 truth on CHM13v2

| File | Contents |
| --- | --- |
| `chr6.phased_snps.vcf.gz` | Filtered phased SNP truth used for chromosome 6 evaluation. |
| `chr22.phased_snps.vcf.gz` | Filtered phased SNP truth used for chromosome 22 evaluation. |
| `MHC.phased_snps.vcf.gz` | Filtered phased SNP truth used for MHC evaluation. |
| `benchmark.bed.gz` | Benchmark regions supplied with the source truth. |

All four files are included in the repository. Uncompress with `gzip -dk data/HG002_CHM13v2/*.gz` on Linux/WSL.

The source files were `CHM13v2.0_HG2-T2TQ100-V1.1_smvar.vcf` and `CHM13v2.0_HG2-T2TQ100-V1.1_smvar.benchmark.bed`. The VCFs here are the processed evaluation inputs, not the complete source callset. Their variant records and phase values are preserved.

The historical processing selected phased heterozygous SNPs and applied the supplied region filter. That script used `start <= VCF_POS <= end`; this convention is retained for reproduction and differs at the left boundary from standard BED conversion. [Filtering scripts and provenance](https://github.com/Zr1311/QHap/tree/a4c5d01b0266949b0ca84a2c41f89a99b6f7de63/benchmarks/truth_original_scripts).
