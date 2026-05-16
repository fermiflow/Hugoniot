"""
Test suite for HQC PBC GTO evaluation.

This module tests the Gaussian Type Orbital (GTO) evaluation functions
for periodic boundary conditions against PySCF reference implementations.
"""

import jax
import numpy as np
import jax.numpy as jnp
import pytest
from dataclasses import dataclass
from typing import Tuple

jax.config.update("jax_enable_x64", True)

from pyscf.pbc import gto
from hqc.pbc.gto import make_pbc_gto


@dataclass
class GTOTestConfig:
    """Configuration for GTO tests."""
    n: int = 8
    dim: int = 3
    rs: float = 1.5
    rcut: int = 24
    basis_set: Tuple[str, ...] = ('gth-szv', 'gth-dzv', 'gth-dzvp')
    atol: float = 1e-5
    atol_pbc: float = 1e-3
    random_seed: int = 43

    @property
    def L(self) -> float:
        """Box length."""
        return (4/3 * jnp.pi * self.n)**(1/3) * self.rs


def pyscf_eval_gto(L: float, xp: np.ndarray, xe: np.ndarray,
                   basis: str, kpt: np.ndarray) -> np.ndarray:
    """
    Evaluate GTOs using PySCF.

    Args:
        L: Box length
        xp: Proton positions, shape (n, 3)
        xe: Electron positions, shape (n, 3)
        basis: Basis set name
        kpt: K-point, shape (3,)

    Returns:
        GTO values, shape (n_electrons, n_ao)
    """
    cell = gto.Cell()
    cell.unit = 'B'
    for ip in range(xp.shape[0]):
        cell.atom.append(['H', tuple(xp[ip])])
    cell.spin = 0
    cell.basis = basis
    cell.a = np.eye(3) * L
    cell.build()
    kpts = [kpt.tolist()]
    gto_value = cell.pbc_eval_ao("GTOval_sph", xe, kpts=kpts)[0]
    return gto_value


class TestPBCGTO:
    """Test class for PBC GTO evaluation."""

    @pytest.fixture(scope="class")
    def config(self) -> GTOTestConfig:
        """Test configuration fixture."""
        return GTOTestConfig()

    @pytest.fixture(scope="class")
    def positions_gamma(self, config: GTOTestConfig) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generate random positions for gamma point test."""
        key = jax.random.PRNGKey(config.random_seed)
        key_p, key_e = jax.random.split(key)
        xp = jax.random.uniform(key_p, (config.n, config.dim), minval=0., maxval=config.L)
        xe = jax.random.uniform(key_e, (config.n, config.dim), minval=0., maxval=config.L)
        kpt = jnp.array([0., 0., 0.])
        return xp, xe, kpt

    @pytest.fixture(scope="class")
    def positions_kpt(self, config: GTOTestConfig) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generate random positions for k-point test."""
        key = jax.random.PRNGKey(config.random_seed)
        key_p, key_e, key_kpt = jax.random.split(key, 3)
        xp = jax.random.uniform(key_p, (config.n, config.dim), minval=0., maxval=config.L)
        xe = jax.random.uniform(key_e, (config.n, config.dim), minval=0., maxval=config.L)
        kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L, maxval=jnp.pi/config.L)
        return xp, xe, kpt

    def _print_test_header(self, config: GTOTestConfig, test_name: str,
                          xp: jnp.ndarray, xe: jnp.ndarray, kpt: jnp.ndarray):
        """Print test header information."""
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(f"{'='*60}")
        print(f"System: n={config.n}, rs={config.rs}, L={config.L:.4f}")
        print(f"Cutoff: rcut={config.rcut}")
        print(f"Basis sets: {config.basis_set}")
        print(f"K-point: {kpt}")
        print(f"Positions: xp.shape={xp.shape}, xe.shape={xe.shape}")

    def _test_pyscf_agreement(self, gto: jnp.ndarray, gto_pyscf: np.ndarray,
                             basis: str, atol: float):
        """Test agreement with PySCF."""
        max_diff = jnp.max(jnp.abs(gto - gto_pyscf))
        print(f"  Max diff vs PySCF: {max_diff:.2e}")
        assert np.allclose(gto, gto_pyscf, atol=atol), \
            f"GTO mismatch for {basis}: max diff = {max_diff:.2e}, atol = {atol}"
        print(f"  ✓ Matches PySCF (atol={atol})")

    def _test_periodicity(self, eval_gto, xp: jnp.ndarray, xe: jnp.ndarray,
                         config: GTOTestConfig, basis: str):
        """Test periodic boundary conditions."""
        cell = jnp.eye(3)
        image = np.random.randint(-2, 3, size=(config.n, config.dim)).dot(cell.T) * config.L
        gto = eval_gto(xp, xe)
        gto_pbc = eval_gto(xp, xe + image)
        max_diff = jnp.max(jnp.abs(gto - gto_pbc))
        print(f"  Max diff with periodic image: {max_diff:.2e}")
        assert np.allclose(gto, gto_pbc, atol=config.atol_pbc), \
            f"PBC test failed for {basis}: max diff = {max_diff:.2e}"
        print(f"  ✓ Periodicity verified (atol={config.atol_pbc})")

    def _test_jit_compilation(self, eval_gto, xp: jnp.ndarray, xe: jnp.ndarray,
                             kpt: Optional[jnp.ndarray] = None):
        """Test JIT compilation."""
        if kpt is None:
            jax.jit(eval_gto)(xp, xe)
        else:
            jax.jit(eval_gto)(xp, xe, kpt)
        print("  ✓ JIT compilation successful")

    def _test_vmap(self, eval_gto, xp: jnp.ndarray, xe: jnp.ndarray,
                  config: GTOTestConfig, kpt: Optional[jnp.ndarray] = None):
        """Test vmap functionality."""
        # Test xe vmap
        xe2 = jnp.concatenate([xe, xe]).reshape(2, config.n, config.dim)
        if kpt is None:
            jax.vmap(eval_gto, (None, 0), 0)(xp, xe2)
        else:
            jax.vmap(eval_gto, (None, 0, None), 0)(xp, xe2, kpt)
        print("  ✓ xe vmap successful")

        # Test xp vmap
        xp2 = jnp.concatenate([xp, xp]).reshape(2, config.n, config.dim)
        if kpt is None:
            jax.vmap(eval_gto, (0, None), 0)(xp2, xe)
        else:
            jax.vmap(eval_gto, (0, None, None), 0)(xp2, xe, kpt)
        print("  ✓ xp vmap successful")

    def _run_basis_test(self, config: GTOTestConfig, basis: str,
                       xp: jnp.ndarray, xe: jnp.ndarray, kpt: jnp.ndarray,
                       gamma: bool):
        """Run test for a single basis set."""
        print(f"\n{'-'*60}")
        print(f"Basis: {basis}")
        print(f"{'-'*60}")

        # Create GTO evaluator
        eval_gto_novmap = make_pbc_gto(basis, config.L, config.rcut, gamma=gamma)
        if gamma:
            eval_gto = jax.vmap(eval_gto_novmap, (None, 0), 0)
            gto = eval_gto(xp, xe)
        else:
            eval_gto = jax.vmap(eval_gto_novmap, (None, 0, None), 0)
            gto = eval_gto(xp, xe, kpt)

        # Test against PySCF
        gto_pyscf = pyscf_eval_gto(config.L, np.array(xp), np.array(xe), basis, kpt)
        self._test_pyscf_agreement(gto, gto_pyscf, basis, config.atol)

        # Test periodicity (only for gamma point)
        if gamma:
            self._test_periodicity(eval_gto, xp, xe, config, basis)

        # Test JIT compilation
        self._test_jit_compilation(eval_gto, xp, xe, None if gamma else kpt)

        # Test vmap
        self._test_vmap(eval_gto, xp, xe, config, None if gamma else kpt)

    def test_pbc_gto_gamma(self, config, positions_gamma):
        """Test GTO evaluation at gamma point."""
        xp, xe, kpt = positions_gamma
        self._print_test_header(config, "PBC GTO Gamma Point", xp, xe, kpt)

        for basis in config.basis_set:
            self._run_basis_test(config, basis, xp, xe, kpt, gamma=True)

    def test_pbc_gto_kpt(self, config, positions_kpt):
        """Test GTO evaluation at random k-point."""
        xp, xe, kpt = positions_kpt
        self._print_test_header(config, "PBC GTO K-Point", xp, xe, kpt)

        for basis in config.basis_set:
            self._run_basis_test(config, basis, xp, xe, kpt, gamma=False)


# Legacy test functions for backward compatibility
def test_pbc_gto_gamma():
    """Legacy test function for gamma point."""
    test = TestPBCGTO()
    config = GTOTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_e = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, config.dim), minval=0., maxval=config.L)
    xe = jax.random.uniform(key_e, (config.n, config.dim), minval=0., maxval=config.L)
    kpt = jnp.array([0., 0., 0.])
    test._print_test_header(config, "PBC GTO Gamma Point", xp, xe, kpt)
    for basis in config.basis_set:
        test._run_basis_test(config, basis, xp, xe, kpt, gamma=True)


def test_pbc_gto_kpt():
    """Legacy test function for k-point."""
    test = TestPBCGTO()
    config = GTOTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_e, key_kpt = jax.random.split(key, 3)
    xp = jax.random.uniform(key_p, (config.n, config.dim), minval=0., maxval=config.L)
    xe = jax.random.uniform(key_e, (config.n, config.dim), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L, maxval=jnp.pi/config.L)
    test._print_test_header(config, "PBC GTO K-Point", xp, xe, kpt)
    for basis in config.basis_set:
        test._run_basis_test(config, basis, xp, xe, kpt, gamma=False)
