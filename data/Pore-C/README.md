# Pore-C data

HG002 Pore-C variant calls and extracted observations on CHM13v2.

| File | Contents |
| --- | --- |
| `chr6.vcf.gz`, `chr22.vcf.gz` | Filtered chromosome-level SNP calls. |
| `{platform}.{region}.input.vcf.gz` | Pore-C variants matched to the corresponding long-read experiment. |
| `{platform}.{region}.fragments.txt.gz` | Pore-C observations indexed in that input VCF's variant order. |

`platform` is `WY`, `HiFi` or `ONT`; `region` is `MHC` or `chr22`. These files are included in the repository. Pore-C BAMs are not included.

For example, after uncompressing the files, combine HiFi and Pore-C observations for chromosome 22:

```bash
gzip -dk data/HiFi/*.gz data/Pore-C/*.gz
python3 SNP-based/index_quantum_maxcut_phasing_n_return.py --porec \
    data/HiFi/chr22.input.vcf data/HiFi/chr22.fragments.txt \
    data/Pore-C/HiFi.chr22.input.vcf data/Pore-C/HiFi.chr22.fragments.txt a=0.1
```

Use each fragment file with its matching input VCF. The platform-specific Pore-C inputs have different variant selections and are not interchangeable.
