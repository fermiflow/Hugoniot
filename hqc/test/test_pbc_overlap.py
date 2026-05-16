"""
Test suite for HQC PBC overlap matrix calculation.

This module tests the overlap matrix calculation for periodic boundary
conditions against PySCF reference implementations.
"""

import jax
import numpy as np
import jax.numpy as jnp
import pytest
from dataclasses import dataclass
from typing import Tuple

jax.config.update("jax_enable_x64", True)

from pyscf.pbc import gto, scf
from hqc.pbc.overlap import make_overlap
from hqc.basis.parse import load_as_str


@dataclass
class OverlapTestConfig:
    """Configuration for overlap matrix tests."""
    n: int = 8
    dim: int = 3
    rs: float = 1.5
    rcut: int = 24
    basis_set: Tuple[str, ...] = ('gth-szv', 'gth-dzv', 'gth-dzvp')
    atol: float = 1e-5
    random_seed: int = 43

    @property
    def L(self) -> float:
        """Box length."""
        return (4/3 * jnp.pi * self.n)**(1/3) * self.rs


def pyscf_overlap(n: int, L: float, rs: float, xp: np.ndarray,
                 basis: str, kpt: np.ndarray) -> np.ndarray:
    """
    Compute overlap matrix using PySCF.

    Args:
        n: Number of protons
        L: Box length (in units of rs)
        rs: Average atomic spacing (Bohr)
        xp: Proton positions, shape (n, 3)
        basis: Basis set name
        kpt: K-point, shape (3,)

    Returns:
        Overlap matrix, shape (n_ao, n_ao)
    """
    xp = xp * rs
    cell = gto.Cell()
    cell.unit = 'B'
    cell.a = np.eye(3) * L * rs
    cell.atom = []
    for ie in range(n):
        cell.atom.append(['H', tuple(xp[ie])])
    cell.spin = 0
    cell.basis = {'H': gto.parse(load_as_str('H', basis), optimize=True)}
    cell.build()

    kpts = [kpt.tolist()]
    kmf = scf.hf.RHF(cell)
    overlap = kmf.get_ovlp(kpt=kpts)

    return overlap


class TestPBCOverlap:
    """Test class for PBC overlap matrix calculation."""

    @pytest.fixture(scope="class")
    def config(self) -> OverlapTestConfig:
        """Test configuration fixture."""
        return OverlapTestConfig()

    @pytest.fixture(scope="class")
    def positions_gamma(self, config: OverlapTestConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate random positions for gamma point test."""
        key = jax.random.PRNGKey(config.random_seed)
        xp = jax.random.uniform(key, (config.n, config.dim), minval=0., maxval=config.L)
        kpt = jnp.array([0., 0., 0.])
        return xp, kpt

    @pytest.fixture(scope="class")
    def positions_kpt(self, config: OverlapTestConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate random positions for k-point test."""
        key = jax.random.PRNGKey(config.random_seed)
        key_p, key_kpt = jax.random.split(key)
        xp = jax.random.uniform(key_p, (config.n, config.dim), minval=0., maxval=config.L)
        kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L, maxval=jnp.pi/config.L)
        return xp, kpt

    def _print_test_header(self, config: OverlapTestConfig, test_name: str,
                          xp: jnp.ndarray, kpt: jnp.ndarray):
        """Print test header information."""
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(f"{'='*60}")
        print(f"System: n={config.n}, rs={config.rs}, L={config.L:.4f}")
        print(f"Cutoff: rcut={config.rcut}")
        print(f"Basis sets: {config.basis_set}")
        print(f"K-point: {kpt}")
        print(f"Positions: xp.shape={xp.shape}")

    def _test_basis(self, config: OverlapTestConfig, basis: str,
                   xp: jnp.ndarray, kpt: jnp.ndarray, gamma: bool):
        """Test overlap matrix for a single basis set."""
        print(f"\n{'-'*60}")
        print(f"Basis: {basis}")
        print(f"{'-'*60}")

        # Compute HQC overlap
        overlap_func = make_overlap(config.n, config.L, config.rs, basis,
                                    config.rcut, gamma=gamma)
        if gamma:
            overlap = overlap_func(xp)
        else:
            overlap = overlap_func(xp, kpt)

        # Compute PySCF reference
        overlap_pyscf = pyscf_overlap(config.n, config.L, config.rs,
                                      np.array(xp), basis, np.array(kpt))

        # Compare
        max_diff = jnp.max(jnp.abs(overlap - overlap_pyscf))
        print(f"  Overlap matrix shape: {overlap.shape}")
        print(f"  Max diff vs PySCF: {max_diff:.2e}")
        assert np.allclose(overlap, overlap_pyscf, atol=config.atol), \
            f"Overlap mismatch for {basis}: max diff = {max_diff:.2e}, atol = {config.atol}"
        print(f"  ✓ Matches PySCF (atol={config.atol})")

    def test_overlap_gamma(self, config, positions_gamma):
        """Test overlap matrix at gamma point."""
        xp, kpt = positions_gamma
        self._print_test_header(config, "Overlap Matrix Gamma Point", xp, kpt)

        for basis in config.basis_set:
            self._test_basis(config, basis, xp, kpt, gamma=True)

    def test_overlap_kpt(self, config, positions_kpt):
        """Test overlap matrix at random k-point."""
        xp, kpt = positions_kpt
        self._print_test_header(config, "Overlap Matrix K-Point", xp, kpt)

        for basis in config.basis_set:
            self._test_basis(config, basis, xp, kpt, gamma=False)


# Legacy test functions for backward compatibility
def test_overlap_gamma():
    """Legacy test function for gamma point."""
    test = TestPBCOverlap()
    config = OverlapTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    xp = jax.random.uniform(key, (config.n, config.dim), minval=0., maxval=config.L)
    kpt = jnp.array([0., 0., 0.])
    test._print_test_header(config, "Overlap Matrix Gamma Point", xp, kpt)
    for basis in config.basis_set:
        test._test_basis(config, basis, xp, kpt, gamma=True)


def test_overlap_kpt():
    """Legacy test function for k-point."""
    test = TestPBCOverlap()
    config = OverlapTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, config.dim), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L, maxval=jnp.pi/config.L)
    test._print_test_header(config, "Overlap Matrix K-Point", xp, kpt)
    for basis in config.basis_set:
        test._test_basis(config, basis, xp, kpt, gamma=False)
