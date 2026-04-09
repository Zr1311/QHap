# QHap: Quantum-Inspired Haplotype Phasing

QHap is a haplotype phasing tool that reformulates phasing as a **Max-Cut problem**, solved via a GPU-accelerated **ballistic simulated bifurcation (bSB)** algorithm. It supports Pore-C integration for extended phase block contiguity.

---

## Methods

QHap provides two complementary strategies:

- **Read-based**: Vertices are sequencing reads; edges encode pairwise allelic conflicts. Best suited for regional phasing or moderate-coverage data.
- **SNP-based**: Vertices are SNP loci; edges are weighted by quality-adjusted log-likelihood ratios. Scales efficiently to chromosome-scale tasks.

Both methods support optional **Pore-C** data integration, which improves haplotype block contiguity.

All numerical simulations presented in this work were implemented using the **MindSpore Quantum framework** (`mindquantum`), which provides a powerful environment for quantum computation simulation. QHap can optionally invoke the ballistic simulated bifurcation (BSB) solver from `mindquantum.algorithm.qaia` via the `--mindquantum` flag.

---

## Installation

```bash
git clone https://github.com/Zr1311/QHap.git
cd QHap
```

### Optional: MindSpore Quantum backend

To use the `--mindquantum` flag, install the MindSpore Quantum framework:

```bash
pip install mindquantum
```

---

## Usage

### Step 1 — Fragment Extraction

Extract haplotype-informative fragments from a BAM and VCF file:

```bash
# Standard long-read data
python3 extractHAIRS.py --bam <input.bam> --VCF <input.vcf> --out fragment.txt

# Pore-C data
python3 extractHAIRS.py --bam <input.bam> --VCF <input.vcf> --out fragment.txt --porec
```

---

### Step 2 — Haplotype Phasing

#### Read-based Method

```bash
cd Read-based

# Without Pore-C
python3 Read-based/index_quantum_maxcut_phasing_n_return.py <input.vcf> <fragment.txt>

# With Pore-C integration
python3 Read-based/index_quantum_maxcut_phasing_n_return.py --porec \
    <input.vcf> <fragment.txt> \
    <input_porec.vcf> <fragment_porec.txt>

# Using MindSpore Quantum BSB solver
python3 Read-based/index_quantum_maxcut_phasing_n_return.py --mindquantum \
    <input.vcf> <fragment.txt>

# Combining --mindquantum with --porec
python3 Read-based/index_quantum_maxcut_phasing_n_return.py --mindquantum --porec \
    <input.vcf> <fragment.txt> \
    <input_porec.vcf> <fragment_porec.txt>
```

#### SNP-based Method

```bash
cd SNP-based

# Without Pore-C
python3 SNP-based/index_quantum_maxcut_phasing_n_return.py <input.vcf> <fragment.txt>

# With Pore-C integration (a controls Pore-C weight, default 0.1)
python3 SNP-based/index_quantum_maxcut_phasing_n_return.py --porec \
    <input.vcf> <fragment.txt> \
    <input_porec.vcf> <fragment_porec.txt> \
    a

# Using MindSpore Quantum BSB solver
python3 SNP-based/index_quantum_maxcut_phasing_n_return.py --mindquantum \
    <input.vcf> <fragment.txt>

# Combining --mindquantum with --porec
python3 SNP-based/index_quantum_maxcut_phasing_n_return.py --mindquantum --porec \
    <input.vcf> <fragment.txt> \
    <input_porec.vcf> <fragment_porec.txt> \
    a
```

---

## HLA Typing

For HLA allele typing based on QHap-phased haplotypes, please refer to:
[https://github.com/Roick-Leo/Aster_WGS.git](https://github.com/Roick-Leo/Aster_WGS.git)

---

## Citation

> Zhang, Rui, et al. "QHap: Quantum-Inspired Haplotype Phasing." arXiv preprint arXiv:2603.25762 (2026).

Source code: [https://github.com/Zr1311/QHap](https://github.com/Zr1311/QHap)
