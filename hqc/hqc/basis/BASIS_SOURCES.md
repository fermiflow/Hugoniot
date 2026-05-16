# Basis Set Sources and Citations

This document describes the sources of basis sets used in HQC and the proper citations.

## GTH Basis Sets

The GTH (Goedecker-Teter-Hutter) basis sets in `hqc/basis/gth-raw/` are from CP2K:

**Source:** CP2K Basis Set Library
**License:** GPL v2 or later
**URL:** https://github.com/cp2k/cp2k-data

**Citation:**
```bibtex
@article{VandeVondele2007,
  title={Gaussian basis sets for accurate calculations on molecular systems in gas and condensed phases},
  author={VandeVondele, Joost and Hutter, J{\"u}rg},
  journal={The Journal of Chemical Physics},
  volume={127},
  number={11},
  pages={114105},
  year={2007},
  doi={10.1063/1.2770708}
}
```

## STO-3G and STO-6G Basis Sets

These basis sets are from the EMSL Basis Set Exchange Library, distributed via PySCF.

**Source:** PySCF (Python-based Simulations of Chemistry Framework)
**License:** Apache License 2.0
**URL:** https://github.com/pyscf/pyscf

**PySCF Citation:**
```bibtex
@article{PySCF2020,
  title={Recent developments in the PySCF program package},
  author={Sun, Qiming and Zhang, Xing and Banerjee, Samragni and Bao, Peng and Barbry, Marc and Blunt, Nick S and Bogdanov, Nikolay A and Booth, George H and Chen, Jia and Cui, Zhi-Hao and others},
  journal={The Journal of Chemical Physics},
  volume={153},
  number={2},
  pages={024109},
  year={2020},
  doi={10.1063/5.0006074}
}
```

**Original STO-3G/STO-6G References:**
```bibtex
@article{Hehre1969,
  title={Self-consistent molecular-orbital methods. I. Use of Gaussian expansions of Slater-type atomic orbitals},
  author={Hehre, Warren J and Stewart, Robert F and Pople, John A},
  journal={The Journal of Chemical Physics},
  volume={51},
  number={6},
  pages={2657--2664},
  year={1969},
  doi={10.1063/1.1672392}
}
```

## Usage in Publications

When using HQC with these basis sets, please cite:

1. **HQC itself:**
   ```bibtex
   @software{hqc2026,
     author = {Li, Zihang},
     title = {HQC: Hydrogen Quantum Chemistry with JAX},
     year = {2026},
     url = {https://code.itp.ac.cn/lzh/hydrogen-qc}
   }
   ```

2. **The basis set source** (PySCF or CP2K, depending on which basis you used)

3. **The original basis set paper** (e.g., Hehre et al. 1969 for STO-3G)

## License Compatibility

- **HQC:** MIT License
- **PySCF basis sets:** Apache License 2.0 (compatible with MIT)
- **CP2K basis sets:** GPL v2+ (note: GPL is more restrictive than MIT)

The Apache License 2.0 allows redistribution and modification with proper attribution, which is compatible with HQC's MIT license.

## Adding New Basis Sets

To add basis sets from PySCF:

1. Download from: https://github.com/pyscf/pyscf/tree/master/pyscf/gto/basis
2. Convert format if needed (PySCF uses NWChem format)
3. Add citation information to this file
4. Update README.md with available basis sets
