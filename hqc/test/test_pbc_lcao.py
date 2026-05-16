"""
Test suite for HQC PBC LCAO (Linear Combination of Atomic Orbitals).

This module tests the LCAO implementation for periodic boundary conditions
against PySCF reference implementations.
"""

import jax
import numpy as np
import jax.numpy as jnp
import pytest
from dataclasses import dataclass
from typing import Tuple

jax.config.update("jax_enable_x64", True)

from hqc.pbc.lcao import make_lcao
from config import YELLOW, BLUE, GREEN, RESET
from test_pyscf import pyscf_solver


@dataclass
class LCAOTestConfig:
    """Configuration for LCAO tests."""
    n: int = 4
    dim: int = 3
    rs: float = 1.5
    basis_set: Tuple[str, ...] = ('gth-dzv', 'gth-dzvp')
    xc: str = 'lda,vwn'
    rcut: int = 24
    grid_length: float = 0.12
    smearing_sigma: float = 0.1
    atol_mo: float = 1e-2
    atol_energy: float = 1e-3
    random_seed: int = 42

    @property
    def L(self) -> float:
        """Box length."""
        return (4/3 * jnp.pi * self.n)**(1/3)


@dataclass
class LCAOParams:
    """Parameters for a specific LCAO test."""
    dft: bool
    gamma: bool
    diis: bool
    smearing: bool

    @property
    def name(self) -> str:
        """Generate descriptive test name."""
        method = "dft" if self.dft else "hf"
        kpoint = "gamma" if self.gamma else "kpt"
        accel = "diis" if self.diis else "fp"
        smear = "_smearing" if self.smearing else ""
        return f"{method}_{kpoint}_{accel}{smear}"


class TestPBCLCAO:
    """Test class for PBC LCAO."""

    @pytest.fixture(scope="class")
    def config(self) -> LCAOTestConfig:
        """Test configuration fixture."""
        return LCAOTestConfig()

    @pytest.fixture(scope="class")
    def positions(self, config: LCAOTestConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate random positions and k-point."""
        key = jax.random.PRNGKey(config.random_seed)
        key_p, key_kpt = jax.random.split(key)
        xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
        kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                                maxval=jnp.pi/config.L/config.rs)
        return xp, kpt

    def _print_test_info(self, config: LCAOTestConfig, params: LCAOParams,
                        xp: jnp.ndarray, kpoint: jnp.ndarray):
        """Print test configuration information."""
        print(f"\n{YELLOW}============= Test: {params.name} ============={RESET}")
        print(f"{BLUE}Method:{RESET} {'DFT' if params.dft else 'HF'}")
        print(f"{BLUE}K-point:{RESET} {'Gamma' if params.gamma else 'Random'}")
        print(f"{BLUE}Acceleration:{RESET} {'DIIS' if params.diis else 'Fixed-point'}")
        print(f"{BLUE}Smearing:{RESET} {params.smearing}")
        print(f"{BLUE}System:{RESET} n={config.n}, rs={config.rs}, L={config.L:.4f}")
        print(f"{BLUE}K-point value:{RESET} {kpoint}")
        print(f"{BLUE}Basis set:{RESET} {config.basis_set}")
        print(f"{BLUE}Grid length:{RESET} {config.grid_length}")
        print(f"{BLUE}Positions shape:{RESET} {xp.shape}")

    def _normalize_mo_coeff(self, mo_coeff: jnp.ndarray) -> jnp.ndarray:
        """Normalize MO coefficients by phase."""
        return mo_coeff @ jnp.diag(jnp.sign(mo_coeff[0]).conjugate())

    def _assert_close(self, name: str, hqc_val, pyscf_val, atol: float):
        """Assert values are close and print comparison."""
        diff = jnp.max(jnp.abs(hqc_val - pyscf_val))
        print(f"{BLUE}{name} max diff:{RESET} {diff:.2e}")
        assert np.allclose(hqc_val, pyscf_val, atol=atol), \
            f"{name} mismatch: max diff = {diff:.2e}, atol = {atol}"
        print(f"{GREEN}{name} matches PySCF ✓{RESET}")

    def _run_lcao_test(self, config: LCAOTestConfig, params: LCAOParams,
                      xp: jnp.ndarray, kpt: jnp.ndarray, basis: str):
        """Run a single LCAO test for given basis."""
        print(f"\n{YELLOW}===== Basis: {basis} ====={RESET}")

        # Determine k-point
        kpoint = jnp.array([0., 0., 0.]) if params.gamma else kpt

        # Run PySCF reference
        pyscf_data = pyscf_solver(
            config.n, config.L, config.rs, xp, kpoint, basis,
            ifdft=params.dft, xc=config.xc, smearing=params.smearing,
            smearing_method='fermi', smearing_sigma=config.smearing_sigma
        )

        # Run HQC LCAO
        lcao = make_lcao(
            config.n, config.L, config.rs, basis,
            grid_length=config.grid_length, diis=params.diis, dft=params.dft,
            smearing=params.smearing, smearing_sigma=config.smearing_sigma,
            gamma=params.gamma
        )

        if params.gamma:
            mo_coeff, bands, E = lcao(xp)
        else:
            mo_coeff, bands, E = lcao(xp, kpoint)

        # Normalize and compare MO coefficients
        mo_coeff_norm = self._normalize_mo_coeff(mo_coeff)
        mo_coeff_pyscf_norm = self._normalize_mo_coeff(pyscf_data["mo_coeff"])
        self._assert_close("MO coefficients", mo_coeff_norm, mo_coeff_pyscf_norm, config.atol_mo)

        # Compare band energies
        print(f"{BLUE}Bands (HQC):{RESET}\n{bands}")
        print(f"{BLUE}Bands (PySCF):{RESET}\n{pyscf_data['bands']}")
        self._assert_close("Band energies", bands, pyscf_data["bands"], config.atol_energy)

        # Compare total energy
        print(f"{BLUE}Total energy (HQC):{RESET} {E:.10f}")
        print(f"{BLUE}Total energy (PySCF):{RESET} {pyscf_data['Eelec']:.10f}")
        self._assert_close("Total energy", E, pyscf_data["Eelec"], config.atol_energy)

    def _run_test(self, config: LCAOTestConfig, params: LCAOParams,
                 xp: jnp.ndarray, kpt: jnp.ndarray):
        """Run test for all basis sets."""
        kpoint = jnp.array([0., 0., 0.]) if params.gamma else kpt
        self._print_test_info(config, params, xp, kpoint)

        for basis in config.basis_set:
            self._run_lcao_test(config, params, xp, kpt, basis)

    # Test methods
    def test_hf_gamma_diis(self, config, positions):
        """Test HF with gamma point and DIIS."""
        xp, kpt = positions
        params = LCAOParams(dft=False, gamma=True, diis=True, smearing=False)
        self._run_test(config, params, xp, kpt)

    def test_hf_gamma_fp(self, config, positions):
        """Test HF with gamma point and fixed-point iteration."""
        xp, kpt = positions
        params = LCAOParams(dft=False, gamma=True, diis=False, smearing=False)
        self._run_test(config, params, xp, kpt)

    def test_hf_gamma_diis_smearing(self, config, positions):
        """Test HF with gamma point, DIIS, and smearing."""
        xp, kpt = positions
        params = LCAOParams(dft=False, gamma=True, diis=True, smearing=True)
        self._run_test(config, params, xp, kpt)

    @pytest.mark.skip(reason="PySCF k-point tests cause segmentation fault")
    def test_hf_kpt_diis(self, config, positions):
        """Test HF with k-point and DIIS."""
        xp, kpt = positions
        params = LCAOParams(dft=False, gamma=False, diis=True, smearing=False)
        self._run_test(config, params, xp, kpt)

    @pytest.mark.skip(reason="PySCF DFT tests cause segmentation fault")
    def test_dft_gamma_diis(self, config, positions):
        """Test DFT with gamma point and DIIS."""
        xp, kpt = positions
        params = LCAOParams(dft=True, gamma=True, diis=True, smearing=False)
        self._run_test(config, params, xp, kpt)


# Legacy test functions for backward compatibility
def test_hf_gamma_diis():
    """Legacy test function."""
    test = TestPBCLCAO()
    config = LCAOTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = LCAOParams(dft=False, gamma=True, diis=True, smearing=False)
    test._run_test(config, params, xp, kpt)


def test_hf_kpt_diis():
    """Legacy test function."""
    test = TestPBCLCAO()
    config = LCAOTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = LCAOParams(dft=False, gamma=False, diis=True, smearing=False)
    test._run_test(config, params, xp, kpt)


def test_dft_gamma_diis():
    """Legacy test function."""
    test = TestPBCLCAO()
    config = LCAOTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = LCAOParams(dft=True, gamma=True, diis=True, smearing=False)
    test._run_test(config, params, xp, kpt)


def test_dft_kpt_diis():
    """Legacy test function."""
    test = TestPBCLCAO()
    config = LCAOTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = LCAOParams(dft=True, gamma=False, diis=True, smearing=False)
    test._run_test(config, params, xp, kpt)


def test_hf_gamma_fp():
    """Legacy test function."""
    test = TestPBCLCAO()
    config = LCAOTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = LCAOParams(dft=False, gamma=True, diis=False, smearing=False)
    test._run_test(config, params, xp, kpt)


def test_hf_kpt_fp():
    """Legacy test function."""
    test = TestPBCLCAO()
    config = LCAOTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = LCAOParams(dft=False, gamma=False, diis=False, smearing=False)
    test._run_test(config, params, xp, kpt)


def test_dft_gamma_fp():
    """Legacy test function."""
    test = TestPBCLCAO()
    config = LCAOTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = LCAOParams(dft=True, gamma=True, diis=False, smearing=False)
    test._run_test(config, params, xp, kpt)


def test_dft_kpt_fp():
    """Legacy test function."""
    test = TestPBCLCAO()
    config = LCAOTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = LCAOParams(dft=True, gamma=False, diis=False, smearing=False)
    test._run_test(config, params, xp, kpt)


def test_hf_gamma_diis_smearing():
    """Legacy test function."""
    test = TestPBCLCAO()
    config = LCAOTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = LCAOParams(dft=False, gamma=True, diis=True, smearing=True)
    test._run_test(config, params, xp, kpt)


def test_hf_kpt_diis_smearing():
    """Legacy test function."""
    test = TestPBCLCAO()
    config = LCAOTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = LCAOParams(dft=False, gamma=False, diis=True, smearing=True)
    test._run_test(config, params, xp, kpt)


def test_dft_gamma_diis_smearing():
    """Legacy test function."""
    test = TestPBCLCAO()
    config = LCAOTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = LCAOParams(dft=True, gamma=True, diis=True, smearing=True)
    test._run_test(config, params, xp, kpt)


def test_dft_kpt_diis_smearing():
    """Legacy test function."""
    test = TestPBCLCAO()
    config = LCAOTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = LCAOParams(dft=True, gamma=False, diis=True, smearing=True)
    test._run_test(config, params, xp, kpt)
