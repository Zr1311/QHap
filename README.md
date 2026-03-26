# QHap: Quantum-Inspired Haplotype Phasing

QHap is a haplotype phasing tool that reformulates phasing as a **Max-Cut problem**, solved via a GPU-accelerated **ballistic simulated bifurcation (bSB)** algorithm. It achieves 4–20× speedup over HapCUT2 and WhatsHap while maintaining comparable accuracy, and supports Pore-C integration for chromosome-scale haplotype reconstruction.

---

## Methods

QHap provides two complementary strategies:

- **Read-based**: Vertices are sequencing reads; edges encode pairwise allelic conflicts. Best suited for regional phasing or moderate-coverage data.
- **SNP-based**: Vertices are SNP loci; edges are weighted by quality-adjusted log-likelihood ratios. Scales efficiently to chromosome-scale tasks.

Both methods support optional **Pore-C** data integration to extend phase block contiguity.

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
# Without Pore-C
python3 Read-based/index_quantum_maxcut_phasing_n_return.py <input.vcf> <fragment.txt>

# With Pore-C integration
python3 Read-based/index_quantum_maxcut_phasing_n_return.py --porec \
    <input.vcf> <fragment.txt> \
    <input.vcf> <fragment_porec.txt>
```

#### SNP-based Method

```bash
# Without Pore-C
python3 SNP-based/index_quantum_maxcut_phasing_n_return.py <input.vcf> <fragment.txt>

# With Pore-C integration (alpha controls Pore-C weight, default 0.1)
python3 SNP-based/index_quantum_maxcut_phasing_n_return.py --porec \
    <input.vcf> <fragment.txt> \
    <input.vcf> <fragment_porec.txt> \
    a=0.1
```

---

## Citation

> Zhang R, Tao X-Z, Chen Y, et al. *QHap: Quantum-Inspired Haplotype Phasing.* 2025.

Source code: [https://github.com/Zr1311/QHap](https://github.com/Zr1311/QHap)
