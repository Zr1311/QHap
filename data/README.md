# Data

HG002 data used in QHap, aligned to CHM13v2. Start with the folder for the platform you need.

| Folder | Contents |
| --- | --- |
| [WY](WY/) | CycloneSEQ chromosome 6/22 alignments, SNP calls and experimental inputs. |
| [HiFi](HiFi/) | PacBio HiFi chromosome 6/22 alignments, SNP calls and experimental inputs. |
| [ONT](ONT/) | Oxford Nanopore chromosome 6/22 alignments, SNP calls and experimental inputs. |
| [Pore-C](Pore-C/) | Pore-C SNP calls and observations for integration with each platform. |
| [HG002_CHM13v2](HG002_CHM13v2/) | Phased SNP truth and benchmark regions. |

VCFs and fragment files are included as `.gz` files. Clone/download the repository, or open a file on GitHub and use **Download raw file**. Each folder explains its files and commands.

BAMs are larger and are stored in GitHub Releases. From the repository root, for example:

```bash
python3 data/download.py HiFi chr22
```

This restores `data/HiFi/chr22.bam` and `chr22.bam.bai` with checksum verification. Replace `HiFi` with `WY` or `ONT`, and `chr22` with `chr6`, as needed. Python 3.9+ is sufficient; no additional packages are needed for downloading. On Windows use `python`.

Original MHC experiments used **the full `chr6.bam` and the MHC-region VCF**. Use the matching `*.input.vcf` and `*.fragments.txt` to rerun the published experimental inputs. `chr6.vcf.gz` and `chr22.vcf.gz` contain the broader filtered callsets.

[Detailed experimental records and retained outputs](https://github.com/Zr1311/QHap/tree/a4c5d01b0266949b0ca84a2c41f89a99b6f7de63/benchmarks) remain available in the archived revision. File checksums are in [SHA256SUMS](SHA256SUMS). The repository's software licence does not assign new reuse rights to third-party sequencing data.
