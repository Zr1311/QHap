# CycloneSEQ (WY) data

HG002 long reads aligned to CHM13v2.

| File | Contents |
| --- | --- |
| `chr6.vcf.gz`, `chr22.vcf.gz` | Filtered SNP calls before selection of the experimental inputs. |
| `MHC.input.vcf.gz`, `chr22.input.vcf.gz` | Exact VCF inputs used in the MHC and chromosome 22 experiments. |
| `MHC.fragments.txt.gz`, `chr22.fragments.txt.gz` | Extracted read observations, paired with the corresponding `*.input.vcf.gz` in the same variant order. |
| `chr6.bam`, `chr6.bam.bai` | Complete chromosome 6 alignment and index; downloaded separately. |
| `chr22.bam`, `chr22.bam.bai` | Complete chromosome 22 alignment and index; downloaded separately. |

From the repository root, download either chromosome:

```bash
python3 data/download.py WY chr6
python3 data/download.py WY chr22
```

The files are restored into `data/WY/`. On Windows, use `python` in place of `python3`.

| BAM | Restored size | Manual download |
| --- | ---: | --- |
| `chr6.bam` | 7.07 GiB | [Release files](https://github.com/Zr1311/QHap/releases/tag/benchmark-hg002-chm13v2-chr6-20261019) |
| `chr22.bam` | 2.17 GiB | [Release files](https://github.com/Zr1311/QHap/releases/tag/benchmark-hg002-chm13v2-20261019) |

Allow about twice the BAM size during restoration. The script verifies SHA-256 checksums, joins any parts in order, and removes the parts after successful restoration.

VCFs and fragment files are already included in the repository. Uncompress them before running QHap, for example `gzip -dk data/WY/*.gz` on Linux/WSL. For the MHC experiment, use **`chr6.bam` with `MHC.input.vcf`** (chr6:28,477,797–33,448,354). The full-chromosome `chr22.vcf` and experimental `chr22.input.vcf` are different inputs.
