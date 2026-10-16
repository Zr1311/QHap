# Chromosome 6 HiFi demo

This demo uses the full chromosome 6 HiFi BAM and its matching index. It replaces the earlier MHC-only BAM. The existing `chr6_snp_PASS_filter.vcf` is unchanged and contains 155,860 variant records on chromosome 6.

The BAM and VCF use the contig name `chr6`; the BAM header records a chromosome 6 reference length of 172,126,628 bp. Use the same reference assembly when substituting other input files.

## Download and restore

The alignment data are available from the [chromosome 6 demo release](https://github.com/Zr1311/QHap/releases/tag/demo-chr6-v1). These files are downloaded separately from the Git repository.

| Release asset | Size (bytes) |
| --- | ---: |
| `chr6.bam.part01` | 1,073,741,824 |
| `chr6.bam.part02` | 1,073,741,824 |
| `chr6.bam.part03` | 1,042,390,667 |
| `chr6.bam.bai` | 1,469,760 |
| `SHA256SUMS` | SHA-256 checksums for the parts, restored BAM, and index |

Restoring the three BAM parts in numerical order produces a byte-for-byte copy of `chr6.bam` (3,189,874,315 bytes, approximately 2.97 GiB). Allow at least 7 GB of free disk space during download and restoration, plus space for analysis outputs.

### Automatic download (Linux, macOS, Windows)

From the QHap repository root:

```bash
python3 demo/download_chr6.py
```

On Windows:

```powershell
python demo/download_chr6.py
```

The script requires Python 3.9 or later and uses only the standard library. It verifies each download against the repository's `SHA256SUMS`, restores the BAM, verifies the restored BAM, and removes the downloaded parts after success. Add `--keep-parts` to retain the parts. Re-running the command skips files whose checksums match and retries failed downloads.

### Manual download (Linux or WSL)

Run these commands from the repository root in Bash. They download to `demo/`, concatenate the parts in order, and verify every file:

```bash
(
    set -e
    cd demo
    base_url='https://github.com/Zr1311/QHap/releases/download/demo-chr6-v1'
    for name in chr6.bam.part01 chr6.bam.part02 chr6.bam.part03 chr6.bam.bai; do
        curl --fail --location --retry 3 --output "$name" "$base_url/$name"
    done
    sha256sum --ignore-missing --check SHA256SUMS
    cat chr6.bam.part01 chr6.bam.part02 chr6.bam.part03 > chr6.bam
    sha256sum --check SHA256SUMS
)
```

The `.part01`, `.part02`, and `.part03` files are binary pieces, not individual BAM files or compressed archives. The existing BAI index belongs to the restored BAM; no re-indexing is required.

## Run the demo

All commands below run from the repository root. Install the dependencies described in the [main README](../README.md#installation-guide) first.

```bash
python3 extractHAIRS.py \
    --bam demo/chr6.bam \
    --VCF demo/chr6_snp_PASS_filter.vcf \
    --out demo/fragment.txt

python3 SNP-based/index_quantum_maxcut_phasing_n_return.py \
    demo/chr6_snp_PASS_filter.vcf \
    demo/fragment.txt
```

The SNP-based workflow is the suggested starting point for this chromosome-scale dataset. To use the read-based workflow instead:

```bash
python3 Read-based/index_quantum_maxcut_phasing_n_return.py \
    demo/chr6_snp_PASS_filter.vcf \
    demo/fragment.txt
```

Expected outputs include `demo/fragment.txt`, `demo/phasing_output/phased.vcf`, and `demo/run_time.txt`. The runtime of the previous MHC-only example does not apply. Full chromosome 6 runtime has not been benchmarked here and depends on hardware, sequencing depth, graph size, and solver backend.
