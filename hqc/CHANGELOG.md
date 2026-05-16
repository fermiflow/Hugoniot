# Changelog

All notable changes to HQC will be documented in this file.

## [0.1.12] - 2026-03-06

Add `hqc.gto` module for isolated molecular systems (non-periodic):
- `hqc.gto.solver`: High-level Hartree-Fock solver with optimized interface for repeated calls
- `hqc.gto.integral`: Fully vectorized Gaussian integral evaluation for GPU acceleration
- `hqc.gto.scf`: Self-consistent field iteration with DIIS acceleration
- `hqc.gto.gto`: Atomic orbital evaluation for wavefunction reconstruction
- `hqc.gto.boys`: Boys function implementation for nuclear attraction integrals

Performance optimizations:
- Precompute basis data and invariants in `make_solver` for efficient repeated calls
- Fully vectorized integral computation: ~93,000x speedup with JIT compilation
- Optimized for geometry optimization and molecular dynamics applications

## [0.1.11] - 2025

Update the total energy in `hqc.pbc.pes` 'dev' mode returns from Eelec to Etot (add Vpp in E).

## [0.1.10]

Add `hqc.pbc.solver` to return more information of HF/DFT solver, including entropy.
Add eval_entropy in `hqc.pbc.solver`.

## [0.1.9]

Add `hqc.pbc.pes` to calculate *potential energy surface (PES)*.
Add `hqc.pbc.potential` to calculate vpp.

## [0.1.8]

**[*Interface change*]**
Add *k-point* support in `hqc.pbc.slater`.

## [0.1.7]

Input and output type check.

## [0.1.6]

Simplify *exchange_correlation_fn* in the scf loop.

## [0.1.5]

Simplify structure, add `hqc.pbc.scf`.

## [0.1.4]

**[*Interface change*]**
Add *k-point* support for DFT method.

## [0.1.3]

**[*Interface change*]**
Add *k-point* support for HF method.

## [0.1.2]

**[*Interface change*]**
Add *E* in the returns of `lcao` function.

## [0.1.1]

Add `hqc.pbc.slater`.

## [0.1.0]

Rename `hqc.pbc.ao` to `hqc.pbc.gto`.
Rename `hqc.pbc.mo` to `hqc.pbc.lcao`.

## [0.0.2]

Update structure, ready to use `pip install -e .`

## [0.0.1]

Initial release.
