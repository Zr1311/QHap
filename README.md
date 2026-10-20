# QHap: Quantum-Inspired Haplotype Phasing

QHap is a haplotype phasing tool that reformulates haplotype phasing as a weighted Max-Cut problem and solves the resulting combinatorial optimization problem using the ballistic simulated bifurcation (bSB) algorithm.

QHap provides two complementary phasing strategies:

* **Read-based QHap**: sequencing reads are represented as graph vertices, and pairwise allelic relationships between reads are encoded as weighted edges.
* **SNP-based QHap**: heterozygous SNP loci are represented as graph vertices, and edges represent read-supported relationships between SNPs.

Both strategies support optional **Pore-C integration** to improve long-range haplotype block contiguity.

Data used in the QHap manuscript's HG002 phasing experiments, with download instructions, running examples and evaluation code, are available in [data/](data/).

QHap provides a GPU-accelerated PyTorch implementation of simulated bifurcation and optionally supports the BSB solver provided by `mindquantum.algorithm.qaia` through the `--mindquantum` option.

---

## Contents

* [Methods](#methods)
* [System Requirements](#system-requirements)
* [Installation Guide](#installation-guide)
* [Input Data](#input-data)
* [Data Download](data/)
* [Usage](#usage)
* [Optional Fast SNP Solver](#optional-fast-snp-solver)
* [Demo](#demo)
* [Output Files](#output-files)
* [HLA Typing](#hla-typing)
* [Citation](#citation)
* [License](#license)

---

# Methods

QHap provides two complementary haplotype phasing strategies.

## Read-based Method

In the read-based strategy, sequencing reads are treated as graph vertices, while edges encode pairwise allelic relationships between reads.

The resulting weighted graph is formulated as a Max-Cut optimization problem. The graph partition obtained using simulated bifurcation is subsequently used to reconstruct the two haplotypes.

This strategy is suitable for regional haplotype phasing and read-level optimization.

## SNP-based Method

In the SNP-based strategy, heterozygous SNP loci are treated as graph vertices. Weighted relationships between SNPs are constructed using haplotype-informative sequencing fragments.

The resulting weighted graph is converted into a Max-Cut optimization problem and solved using simulated bifurcation. The optimized spin configuration is then mapped back to phased SNP states.

This strategy is designed to efficiently process larger genomic regions.

## Pore-C Integration

Both read-based and SNP-based strategies support optional Pore-C data integration.

Pore-C-derived haplotype information can be combined with conventional sequencing information to improve long-range haplotype block connectivity.

For SNP-based Pore-C phasing, the parameter

```text
a=<value>
```

controls the relative contribution of Pore-C information during graph construction.

## Simulated Bifurcation Backend

QHap includes a PyTorch-based ballistic simulated bifurcation implementation.

The alternative BSB implementation provided by MindQuantum can be enabled using:

```text
--mindquantum
```

---

# System Requirements

## Hardware Requirements

QHap can be executed on a standard x86_64 workstation.

For GPU-accelerated execution, the following hardware is recommended:

* NVIDIA CUDA-capable GPU
* Sufficient system RAM for sequencing data processing
* Sufficient GPU memory for the constructed Max-Cut graph

Memory requirements depend on sequencing depth, the number of heterozygous variants, and graph size.

---

## Software Requirements

QHap has been tested using the following environment:

| Component        | Tested Version                                               |
| ---------------- | ------------------------------------------------------------ |
| Operating system | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| Python           | 3.9.25                                                       |
| CUDA             | 12.8                                                         |
| PyTorch          | 2.8.0+cu128                                                  |
| MindQuantum      | 0.11.0                                                       |

The tested operating system runs under Windows Subsystem for Linux 2 (WSL2).

---

## Python Dependencies

QHap has been tested with the following Python packages:

| Package     |     Version |
| ----------- | ----------: |
| NumPy       |      1.26.4 |
| pandas      |       2.3.3 |
| SciPy       |      1.13.1 |
| pysam       |      0.23.3 |
| pyfaidx     |     0.9.0.4 |
| Numba       |      0.60.0 |
| PyTorch     | 2.8.0+cu128 |
| MindQuantum |      0.11.0 |

MindQuantum is required only when the optional `--mindquantum` backend is used.

---

# Installation Guide

## 1. Clone the Repository

```bash
git clone https://github.com/Zr1311/QHap.git
cd QHap
```

## 2. Create a Python Environment

A dedicated Conda environment is recommended:

```bash
conda create -n qhap python=3.9.25 -y
conda activate qhap
```

## 3. Install Python Dependencies

Install the dependencies used in the tested environment:

```bash
python3 -m pip install \
    numpy==1.26.4 \
    pandas==2.3.3 \
    scipy==1.13.1 \
    pysam==0.23.3 \
    pyfaidx==0.9.0.4 \
    numba==0.60.0
```

Install the CUDA 12.8-compatible PyTorch version:

```bash
python3 -m pip install torch==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

For the optional MindQuantum backend:

```bash
python3 -m pip install mindquantum==0.11.0
```

---

## Verify the Installation

The installed environment can be checked using:

```bash
python3 - <<'PY'
import numpy
import pandas
import scipy
import pysam
import pyfaidx
import numba
import torch

print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scipy:", scipy.__version__)
print("pysam:", pysam.__version__)
print("pyfaidx:", pyfaidx.__version__)
print("numba:", numba.__version__)
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

try:
    import mindquantum
    print("mindquantum:", mindquantum.__version__)
except ImportError:
    print("mindquantum: not installed")
PY
```

The tested environment reports:

```text
numpy: 1.26.4
pandas: 2.3.3
scipy: 1.13.1
pysam: 0.23.3
pyfaidx: 0.9.0.4
numba: 0.60.0
torch: 2.8.0+cu128
CUDA: 12.8
mindquantum: 0.11.0
```

---

## Typical Installation Time

Installation of QHap and its required dependencies typically takes approximately **3–5 minutes** on the tested system.

The exact installation time may vary depending on network speed, package availability, and whether required packages have previously been cached.

---

# Input Data

QHap requires the following input files:

1. A VCF file containing heterozygous variants.
2. A BAM file containing sequencing reads aligned to the corresponding reference genome.
3. The corresponding BAM index (`.bai`) file.

The BAM and VCF files are first processed using `extractHAIRS.py` to generate a haplotype-informative fragment file.

The resulting VCF and fragment files are subsequently used by either the read-based or SNP-based QHap workflow.

For Pore-C integration, an additional Pore-C VCF file and Pore-C fragment file are required.

---

# Usage

The following commands assume that they are executed from the root directory of the QHap repository.

## Step 1 — Fragment Extraction

### Standard Long-read Data

Extract haplotype-informative fragments from a BAM and VCF file:

```bash
python3 extractHAIRS.py \
    --bam <input.bam> \
    --VCF <input.vcf> \
    --out <fragment.txt>
```

### Pore-C Data

For Pore-C sequencing data:

```bash
python3 extractHAIRS.py \
    --bam <input_porec.bam> \
    --VCF <input_porec.vcf> \
    --out <fragment_porec.txt> \
    --porec
```

---

# Read-based Haplotype Phasing

## Standard Mode

```bash
python3 Read-based/index_quantum_maxcut_phasing_n_return.py \
    <input.vcf> \
    <fragment.txt>
```

## Pore-C Integration

```bash
python3 Read-based/index_quantum_maxcut_phasing_n_return.py \
    --porec \
    <input.vcf> \
    <fragment.txt> \
    <input_porec.vcf> \
    <fragment_porec.txt>
```

## MindQuantum Backend

```bash
python3 Read-based/index_quantum_maxcut_phasing_n_return.py \
    --mindquantum \
    <input.vcf> \
    <fragment.txt>
```

## MindQuantum + Pore-C

```bash
python3 Read-based/index_quantum_maxcut_phasing_n_return.py \
    --mindquantum \
    --porec \
    <input.vcf> \
    <fragment.txt> \
    <input_porec.vcf> \
    <fragment_porec.txt>
```

---

# SNP-based Haplotype Phasing

## Standard Mode

```bash
python3 SNP-based/index_quantum_maxcut_phasing_n_return.py \
    <input.vcf> \
    <fragment.txt>
```

## Pore-C Integration

For SNP-based Pore-C integration, specify the relative Pore-C contribution using:

```text
a=<value>
```

For example:

```bash
python3 SNP-based/index_quantum_maxcut_phasing_n_return.py \
    --porec \
    <input.vcf> \
    <fragment.txt> \
    <input_porec.vcf> \
    <fragment_porec.txt> \
    a=0.1
```

The parameter `a` controls the relative contribution of Pore-C-derived information during SNP graph construction.

## MindQuantum Backend

```bash
python3 SNP-based/index_quantum_maxcut_phasing_n_return.py \
    --mindquantum \
    <input.vcf> \
    <fragment.txt>
```

## MindQuantum + Pore-C

```bash
python3 SNP-based/index_quantum_maxcut_phasing_n_return.py \
    --mindquantum \
    --porec \
    <input.vcf> \
    <fragment.txt> \
    <input_porec.vcf> \
    <fragment_porec.txt> \
    a=0.1
```

## Disable Post-processing

Post-processing is enabled by default in the SNP-based workflow.

To disable it:

```bash
python3 SNP-based/index_quantum_maxcut_phasing_n_return.py \
    --no-post \
    <input.vcf> \
    <fragment.txt>
```

The `--no-post` option can also be combined with `--mindquantum`.

---

# Demo

A chromosome-scale demonstration dataset is available for chromosome 6. It pairs the existing `demo/chr6_snp_PASS_filter.vcf` with the full chromosome 6 HiFi alignment file, `chr6.bam`, and its matching index, `chr6.bam.bai`.

The BAM is **3,189,874,315 bytes (approximately 2.97 GiB)**. The BAM and index are distributed through the [chromosome 6 demo release](https://github.com/Zr1311/QHap/releases/tag/demo-chr6-v1); they are downloaded separately from the Git repository. The BAM is split into three binary parts, which must be concatenated in numerical order.

After downloading and restoring the data, the demo directory contains:

```text
demo/
├── chr6.bam
├── chr6.bam.bai
├── chr6_snp_PASS_filter.vcf
├── download_chr6.py
├── SHA256SUMS
└── README.md
```

The BAM contains alignments across chromosome 6, replacing the previous MHC-only example. The VCF is unchanged and contains 155,860 chromosome 6 variant records.

## Download and Restore the Demo Data

From the repository root, run:

```bash
python3 demo/download_chr6.py
```

On Windows, use `python demo/download_chr6.py`. The script uses only the Python standard library, downloads the three BAM parts and the matching index, verifies their SHA-256 checksums, and restores `demo/chr6.bam`. Successfully restored parts are removed by default; use `--keep-parts` to retain them. Re-running the command reuses files whose checksums already match.

Allow at least **7 GB of free disk space** for the download and restoration, plus space for phasing outputs. See [`demo/README.md`](demo/README.md) for manual download, concatenation, and verification commands.

---

## Run the Demo

All commands should be executed from the QHap repository root directory.

### Step 1 — Extract Haplotype-informative Fragments

```bash
python3 extractHAIRS.py \
    --bam demo/chr6.bam \
    --VCF demo/chr6_snp_PASS_filter.vcf \
    --out demo/fragment.txt
```

A successful fragment extraction produces:

```text
demo/fragment.txt
```

---

### Step 2 — Run SNP-based QHap

For this chromosome-scale dataset, start with the SNP-based workflow:

```bash
python3 SNP-based/index_quantum_maxcut_phasing_n_return.py \
    demo/chr6_snp_PASS_filter.vcf \
    demo/fragment.txt
```

Alternatively, the read-based workflow can be tested using:

```bash
python3 Read-based/index_quantum_maxcut_phasing_n_return.py \
    demo/chr6_snp_PASS_filter.vcf \
    demo/fragment.txt
```

---

## Expected Demo Output

A successful QHap run generates a phased VCF file:

```text
demo/phasing_output/phased.vcf
```

The phased VCF contains phased genotypes such as:

```text
0|1
1|0
```

together with phase-set (`PS`) information where applicable.

Runtime information is recorded in:

```text
demo/run_time.txt
```

Depending on the selected workflow, additional intermediate files may also be generated.

---

## Expected Demo Runtime

The expected execution time for the chromosome 6 demo is **under 2 minutes**, excluding environment setup and data download.

Actual execution time depends on hardware, graph size, selected phasing strategy, and selected simulated bifurcation backend. Data download time depends on network speed.

---

# Output Files

The principal QHap output is:

```text
phasing_output/phased.vcf
```

This file contains the final phased variants.

Runtime information is recorded in:

```text
run_time.txt
```

For Pore-C analyses, additional intermediate merged files may be generated, including:

```text
merged.tab
merged_fragment_reads.txt
```

---

# Running QHap on Your Own Data

To apply QHap to a new dataset:

1. Prepare an aligned BAM file.
2. Generate the corresponding BAM index (`.bai`).
3. Prepare a VCF file containing heterozygous variants.
4. Run `extractHAIRS.py` to generate haplotype-informative fragments.
5. Run either the read-based or SNP-based QHap workflow.
6. Use `--porec` when integrating Pore-C data.
7. Use `--mindquantum` when using the MindQuantum BSB backend.
8. Inspect `phasing_output/phased.vcf` for the final phased variants.
9. Inspect `run_time.txt` for runtime information.

---

# HLA Typing

For HLA allele typing based on QHap-phased haplotypes, please refer to:

https://github.com/Roick-Leo/Aster_WGS

---

# Citation

If you use QHap in your research, please cite:

> Zhang, Rui, et al.
> **QHap: Quantum-Inspired Haplotype Phasing.**
> arXiv preprint arXiv:2603.25762 (2026).

Source code:

https://github.com/Zr1311/QHap

---

# License

QHap is distributed under the **Apache License 2.0**.

See the [`LICENSE`](LICENSE) file for the complete license terms.


## Data

[data/](data/) contains inputs for the manuscript's HG002 phasing experiments: WY (CycloneSEQ), HiFi, ONT and Pore-C data, together with truth on CHM13v2. Its README gives download and running examples. [data/evaluate.py](data/evaluate.py) computes SE, HE and the other evaluation metrics using the original scripts' conventions.

## Optional fast SNP solver

Add `--fast` to select `SNP-based/BSB_get_haplotype_fast.py`:

```bash
python3 SNP-based/index_quantum_maxcut_phasing_n_return.py --fast \
    data/WY/chr22.input.vcf data/WY/chr22.fragments.txt
```

Uncompress the inputs first. This backend uses 500–1,000 iterations and 100 samples per base solver call, with at most 10 outer refinement rounds and no adaptive budget escalation. It implements the fast solver used for the CycloneSEQ chromosome 22 experiment. The standard solver remains the default; `--fast` and `--mindquantum` are mutually exclusive.

