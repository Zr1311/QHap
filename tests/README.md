# Validation of the data and optional fast solver

Checks completed on 20 October 2026. The comparison baseline was commit `194213575c4d42262e5023aa8a9e703560227d30`.

| Check | Result |
| --- | --- |
| Data preservation | All 36 compressed data files reproduce their source contents exactly; fragment indices and quality lengths match their paired VCFs. |
| Download integrity | Six tests passed: ordered restoration, reuse, retained parts, corrupt parts, incorrect complete-file checksum and single-part restoration. A public HiFi chr22 BAI was also downloaded and verified. |
| Default read-based and SNP-based paths | Four-variant fixtures produced the same output records as the original code, with the expected relative phase. |
| Default SNP-based, real input subset | Fixed-seed runs on 128 WY chromosome 22 variants and 804 fragment rows produced identical output records before and after adding the optional flag. |
| Optional fast solver | The known-phase fixture, the real 128-variant subset and the Pore-C fixture ran successfully. The known-phase fixtures had the expected relative phase. |
| Solver selection | Combining `--fast` with `--mindquantum` was rejected. The four MindQuantum source files were preserved. |

The solver checks comprised nine CLI runs in WSL with PyTorch 2.11.0+cu128 and CUDA available. These small tests do not establish that every historical chromosome-scale result is reproduced. MindQuantum execution was not retested.

The supplied local read-based source was also compared with the original public source. The edge calculation, connected components and block assignment have the same logic. The principal computation difference is the worker count: all available CPU processes locally, at most eight publicly. On the same 128-variant subset, both produced the same 33,204 weighted edges and one non-singleton block. No local-only block repair was found, so the public read-based algorithms were retained.

Run the download tests without optional packages or large downloads:

```bash
python3 -m unittest discover -s tests -v
```
