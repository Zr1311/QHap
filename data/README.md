# Data

HG002 data used in QHap, aligned to CHM13v2. Start with the folder for the platform you need.

| Folder | Contents |
| --- | --- |
| [WY](WY/) | CycloneSEQ chromosome 6/22 alignments, SNP calls and experimental inputs. |
| [HiFi](HiFi/) | PacBio HiFi chromosome 6/22 alignments, SNP calls and experimental inputs. |
| [ONT](ONT/) | Oxford Nanopore chromosome 6/22 alignments, SNP calls and experimental inputs. |
| [Pore-C](Pore-C/) | Pore-C SNP calls and observations for integration with each platform. |
| [HG002_CHM13v2](HG002_CHM13v2/) | Phased SNP truth and benchmark regions. |

VCFs and fragment files are included as `.gz` files. Clone/download the repository, or open a file on GitHub and use **Download raw file**. Large BAMs and their indexes are downloaded from GitHub Releases using the commands below.

## Run the demos

Install QHap using the [installation guide](../README.md#installation-guide), then activate that environment. Run all commands below from the **QHap repository root in Linux or WSL Bash**. Each example uses a separate folder under `results/`.

The workflow is: **BAM + input VCF → fragment extraction → QHap phasing → phased VCF**. The supplied fragment files also allow you to start directly from the phasing step.

### 1. HiFi chromosome 22 using supplied fragments

This example uses the experimental input VCF and its matching extracted observations. No BAM download is needed.

```bash
mkdir -p results/HiFi_chr22
gzip -dc data/HiFi/chr22.input.vcf.gz > results/HiFi_chr22/input.vcf
gzip -dc data/HiFi/chr22.fragments.txt.gz > results/HiFi_chr22/fragments.txt

python3 SNP-based/index_quantum_maxcut_phasing_n_return.py \
    results/HiFi_chr22/input.vcf results/HiFi_chr22/fragments.txt
```

The final result is `results/HiFi_chr22/phasing_output/phased.vcf`. Runtime information for the phasing step is in `results/HiFi_chr22/run_time.txt`.

### 2. HiFi chromosome 22 starting from BAM

To run extraction as well, use this complete workflow:

```bash
python3 data/download.py HiFi chr22
mkdir -p results/HiFi_chr22_from_bam
gzip -dc data/HiFi/chr22.input.vcf.gz > results/HiFi_chr22_from_bam/input.vcf

python3 extractHAIRS.py \
    --bam data/HiFi/chr22.bam \
    --VCF results/HiFi_chr22_from_bam/input.vcf \
    --out results/HiFi_chr22_from_bam/fragments.txt

python3 SNP-based/index_quantum_maxcut_phasing_n_return.py \
    results/HiFi_chr22_from_bam/input.vcf \
    results/HiFi_chr22_from_bam/fragments.txt
```

The downloader restores `data/HiFi/chr22.bam` and `chr22.bam.bai` and verifies their checksums. The final result is `results/HiFi_chr22_from_bam/phasing_output/phased.vcf`. The phasing script's `run_time.txt` does not include fragment extraction.

### 3. MHC with read-based QHap

```bash
mkdir -p results/HiFi_MHC_read
gzip -dc data/HiFi/MHC.input.vcf.gz > results/HiFi_MHC_read/input.vcf
gzip -dc data/HiFi/MHC.fragments.txt.gz > results/HiFi_MHC_read/fragments.txt

python3 Read-based/index_quantum_maxcut_phasing_n_return.py \
    results/HiFi_MHC_read/input.vcf results/HiFi_MHC_read/fragments.txt
```

The original MHC experiments used **the full `chr6.bam` with `MHC.input.vcf`** (chr6:28,477,797–33,448,354). To start this example from BAM, download with `python3 data/download.py HiFi chr6` and follow the extraction workflow above, using `data/HiFi/chr6.bam` and `results/HiFi_MHC_read/input.vcf`.

### 4. CycloneSEQ chromosome 22 with the fast SNP solver

```bash
mkdir -p results/WY_chr22_fast
gzip -dc data/WY/chr22.input.vcf.gz > results/WY_chr22_fast/input.vcf
gzip -dc data/WY/chr22.fragments.txt.gz > results/WY_chr22_fast/fragments.txt

python3 SNP-based/index_quantum_maxcut_phasing_n_return.py --fast \
    results/WY_chr22_fast/input.vcf results/WY_chr22_fast/fragments.txt
```

`--fast` selects the separate fast bSB implementation; omitting it selects the standard solver. The optional MindQuantum backend is selected with `--mindquantum` after installing its dependencies; it cannot be combined with `--fast`.

### 5. HiFi chromosome 22 with Pore-C

This example uses the supplied long-read and Pore-C observations:

```bash
mkdir -p results/HiFi_chr22_PoreC
gzip -dc data/HiFi/chr22.input.vcf.gz > results/HiFi_chr22_PoreC/input.vcf
gzip -dc data/HiFi/chr22.fragments.txt.gz > results/HiFi_chr22_PoreC/fragments.txt
gzip -dc data/Pore-C/HiFi.chr22.input.vcf.gz > results/HiFi_chr22_PoreC/porec.vcf
gzip -dc data/Pore-C/HiFi.chr22.fragments.txt.gz > results/HiFi_chr22_PoreC/porec.fragments.txt

python3 SNP-based/index_quantum_maxcut_phasing_n_return.py --porec \
    results/HiFi_chr22_PoreC/input.vcf results/HiFi_chr22_PoreC/fragments.txt \
    results/HiFi_chr22_PoreC/porec.vcf results/HiFi_chr22_PoreC/porec.fragments.txt a=0.1
```

The merged inputs and `phasing_output/phased.vcf` are written under `results/HiFi_chr22_PoreC/`. Pore-C BAMs are not included; the provided VCF/fragment pairs support this integration example.

## Choose other inputs and inspect the results

For ONT or WY, replace the platform in the input paths and download command, and choose a new output folder. For Pore-C integration, also use the corresponding `ONT.*` or `WY.*` files in `Pore-C/`.

Use each `*.fragments.txt.gz` with the matching `*.input.vcf.gz`: fragment indices refer to that VCF's variant order. The `chr6.vcf.gz` and `chr22.vcf.gz` files are the broader filtered callsets and are different from the experimental `*.input.vcf.gz` files.

Each example writes `phasing_output/phased.vcf` beside its input VCF. Inspect the `GT` and `PS` fields for phased genotypes and phase-set membership. [HG002_CHM13v2](HG002_CHM13v2/) contains the matching evaluation truth and benchmark regions.

[Detailed experimental records and retained outputs](https://github.com/Zr1311/QHap/tree/a4c5d01b0266949b0ca84a2c41f89a99b6f7de63/benchmarks) remain available in the archived revision. File checksums are in [SHA256SUMS](SHA256SUMS). The repository's software licence does not assign new reuse rights to third-party sequencing data.
