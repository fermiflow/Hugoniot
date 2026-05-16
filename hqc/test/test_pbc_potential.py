"""
Test suite for HQC PBC potential energy calculation.

This module tests the proton-proton potential energy calculation
for periodic boundary conditions against PySCF reference implementations.
"""

import jax
import numpy as np
import jax.numpy as jnp
import pytest
from dataclasses import dataclass

jax.config.update("jax_enable_x64", True)

from pyscf.pbc import gto, scf
from hqc.basis.parse import load_as_str
from hqc.pbc.potential import potential_energy_pp


@dataclass
class PotentialTestConfig:
    """Configuration for potential energy tests."""
    n: int = 14
    rs: float = 1.31
    basis: str = 'gth-szv'
    atol: float = 1e-6
    random_seed: int = 42

    @property
    def L(self) -> float:
        """Box length."""
        return (4/3 * jnp.pi * self.n)**(1/3)


def pyscf_vpp(xp: np.ndarray, L: float, rs: float, basis: str = 'gth-szv') -> float:
    """
    Compute proton-proton potential energy using PySCF.

    Args:
        xp: Proton positions, shape (n, 3)
        L: Box length (in units of rs)
        rs: Average atomic spacing (Bohr)
        basis: Basis set name

    Returns:
        Proton-proton potential energy (Rydberg)
    """
    n = xp.shape[0]
    Ry = 2  # Hartree to Rydberg conversion
    xp = xp * rs

    cell = gto.Cell()
    cell.unit = 'B'
    cell.a = np.eye(3) * L * rs
    cell.atom = []
    for ie in range(n):
        cell.atom.append(['H', tuple(xp[ie])])
    cell.spin = 0
    cell.basis = {'H': gto.parse(load_as_str('H', basis), optimize=True)}
    cell.symmetry = False
    cell.build()

    kmf = scf.hf.RHF(cell)
    return kmf.energy_nuc() * Ry


class TestPBCPotential:
    """Test class for PBC potential energy calculation."""

    @pytest.fixture(scope="class")
    def config(self) -> PotentialTestConfig:
        """Test configuration fixture."""
        return PotentialTestConfig()

    @pytest.fixture(scope="class")
    def positions(self, config: PotentialTestConfig) -> jnp.ndarray:
        """Generate random proton positions."""
        key = jax.random.PRNGKey(config.random_seed)
        xp = jax.random.uniform(key, (config.n, 3), minval=0., maxval=config.L)
        return xp

    def _print_test_info(self, config: PotentialTestConfig, xp: jnp.ndarray):
        """Print test information."""
        print(f"\n{'='*60}")
        print(f"Test: Proton-Proton Potential Energy")
        print(f"{'='*60}")
        print(f"System: n={config.n}, rs={config.rs}, L={config.L:.4f}")
        print(f"Basis: {config.basis}")
        print(f"Positions: xp.shape={xp.shape}")
        print(f"Tolerance: atol={config.atol}")

    def test_pbc_potential(self, config, positions):
        """Test proton-proton potential energy calculation."""
        xp = positions
        self._print_test_info(config, xp)

        # Compute HQC potential
        vpp = potential_energy_pp(xp, config.L, config.rs)
        print(f"\nHQC Vpp: {vpp:.10f} Ry")

        # Compute PySCF reference
        vpp_pyscf = pyscf_vpp(np.array(xp), config.L, config.rs, config.basis)
        print(f"PySCF Vpp: {vpp_pyscf:.10f} Ry")

        # Compare
        diff = abs(vpp - vpp_pyscf)
        print(f"Difference: {diff:.2e} Ry")

        assert jnp.isclose(vpp, vpp_pyscf, atol=config.atol), \
            f"Potential energy mismatch: diff = {diff:.2e}, atol = {config.atol}"
        print(f"✓ Matches PySCF (atol={config.atol})")


# Legacy test function for backward compatibility
def test_pbc_potential():
    """Legacy test function."""
    test = TestPBCPotential()
    config = PotentialTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    xp = jax.random.uniform(key, (config.n, 3), minval=0., maxval=config.L)
    test._print_test_info(config, xp)

    vpp = potential_energy_pp(xp, config.L, config.rs)
    print(f"\nHQC Vpp: {vpp:.10f} Ry")

    vpp_pyscf = pyscf_vpp(np.array(xp), config.L, config.rs, config.basis)
    print(f"PySCF Vpp: {vpp_pyscf:.10f} Ry")

    diff = abs(vpp - vpp_pyscf)
    print(f"Difference: {diff:.2e} Ry")

    assert jnp.isclose(vpp, vpp_pyscf, atol=config.atol), \
        f"Potential energy mismatch: diff = {diff:.2e}, atol = {config.atol}"
    print(f"✓ Matches PySCF (atol={config.atol})")
