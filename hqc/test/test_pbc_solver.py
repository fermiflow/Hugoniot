"""
Test suite for HQC PBC solver.

This module tests the HQC periodic boundary condition solver against PySCF
reference implementations for various configurations:
- HF vs DFT
- Gamma point vs k-point
- With/without DIIS acceleration
- With/without smearing
"""

import jax
import numpy as np
import jax.numpy as jnp
import pytest
from dataclasses import dataclass
from typing import Tuple, Optional

jax.config.update("jax_enable_x64", True)

from hqc.pbc.solver import make_solver
from config import YELLOW, BLUE, GREEN, RESET
from test_pyscf import pyscf_solver


@dataclass
class TestConfig:
    """Configuration for PBC solver tests."""
    n: int = 4
    dim: int = 3
    rs: float = 1.5
    basis_set: Tuple[str, ...] = ('gth-dzv', 'gth-dzvp')
    xc: str = 'lda,vwn'
    rcut: int = 24
    grid_length: float = 0.12
    smearing_sigma: float = 0.1
    atol_mo: float = 1e-2
    atol_dm: float = 1e-2
    atol_energy: float = 1e-3

    @property
    def L(self) -> float:
        """Box length."""
        return (4/3 * jnp.pi * self.n)**(1/3)


@dataclass
class SolverParams:
    """Parameters for a specific solver test."""
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


class TestPBCSolver:
    """Test class for PBC solver."""

    @pytest.fixture(scope="class")
    def config(self) -> TestConfig:
        """Test configuration fixture."""
        return TestConfig()

    @pytest.fixture(scope="class")
    def positions(self, config: TestConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate random positions and k-point."""
        key = jax.random.PRNGKey(42)
        key_p, key_kpt = jax.random.split(key)
        xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
        kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                                maxval=jnp.pi/config.L/config.rs)
        return xp, kpt

    def _print_test_info(self, config: TestConfig, params: SolverParams,
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

    def _run_solver_test(self, config: TestConfig, params: SolverParams,
                        xp: jnp.ndarray, kpt: jnp.ndarray, basis: str):
        """Run a single solver test for given basis."""
        print(f"\n{YELLOW}===== Basis: {basis} ====={RESET}")

        # Determine k-point
        kpoint = jnp.array([0., 0., 0.]) if params.gamma else kpt

        # Run PySCF reference
        pyscf_data = pyscf_solver(
            config.n, config.L, config.rs, xp, kpoint, basis,
            ifdft=params.dft, xc=config.xc, smearing=params.smearing,
            smearing_method='fermi', smearing_sigma=config.smearing_sigma
        )

        # Run HQC solver
        solver = make_solver(
            config.n, config.L, config.rs, basis,
            grid_length=config.grid_length, diis=params.diis, dft=params.dft,
            smearing=params.smearing, smearing_sigma=config.smearing_sigma,
            gamma=params.gamma
        )

        if params.gamma:
            mo_coeff, dm, bands, E, Ki, Vep, Vee, Se, converged = solver(xp)
        else:
            mo_coeff, dm, bands, E, Ki, Vep, Vee, Se, converged = solver(xp, kpoint)

        # Check convergence
        assert converged, "Solver did not converge"
        print(f"{GREEN}Solver converged ✓{RESET}")

        # Normalize and compare MO coefficients
        mo_coeff_norm = self._normalize_mo_coeff(mo_coeff)
        mo_coeff_pyscf_norm = self._normalize_mo_coeff(pyscf_data["mo_coeff"])
        self._assert_close("MO coefficients", mo_coeff_norm, mo_coeff_pyscf_norm, config.atol_mo)

        # Compare density matrix
        self._assert_close("Density matrix", dm, pyscf_data["dm"], config.atol_dm)

        # Compare band energies
        print(f"{BLUE}Bands (HQC):{RESET}\n{bands}")
        print(f"{BLUE}Bands (PySCF):{RESET}\n{pyscf_data['bands']}")
        self._assert_close("Band energies", bands, pyscf_data["bands"], config.atol_energy)

        # Compare total energy
        print(f"{BLUE}Total energy (HQC):{RESET} {E:.10f}")
        print(f"{BLUE}Total energy (PySCF):{RESET} {pyscf_data['Eelec']:.10f}")
        self._assert_close("Total energy", E, pyscf_data["Eelec"], config.atol_energy)

        # Compare core energy
        Ecore = Ki + Vep
        print(f"{BLUE}Core energy (HQC):{RESET} {Ecore:.10f}")
        print(f"{BLUE}Core energy (PySCF):{RESET} {pyscf_data['Ecore']:.10f}")
        self._assert_close("Core energy", Ecore, pyscf_data["Ecore"], config.atol_energy)

        # Compare electron-electron energy
        print(f"{BLUE}Vee (HQC):{RESET} {Vee:.10f}")
        print(f"{BLUE}Vee (PySCF):{RESET} {pyscf_data['Vee']:.10f}")
        self._assert_close("Electron-electron energy", Vee, pyscf_data["Vee"], config.atol_energy)

        # Compare entropy
        print(f"{BLUE}Entropy (HQC):{RESET} {Se:.10f}")
        print(f"{BLUE}Entropy (PySCF):{RESET} {pyscf_data['Se']:.10f}")
        self._assert_close("Entropy", Se, pyscf_data["Se"], config.atol_energy)

    def _run_test(self, config: TestConfig, params: SolverParams,
                 xp: jnp.ndarray, kpt: jnp.ndarray):
        """Run test for all basis sets."""
        kpoint = jnp.array([0., 0., 0.]) if params.gamma else kpt
        self._print_test_info(config, params, xp, kpoint)

        for basis in config.basis_set:
            self._run_solver_test(config, params, xp, kpt, basis)

    # Test methods using parametrize for cleaner organization
    @pytest.mark.parametrize("params", [
        SolverParams(dft=False, gamma=True, diis=True, smearing=False),
    ])
    def test_hf_gamma_diis(self, config, positions, params):
        """Test HF with gamma point and DIIS."""
        xp, kpt = positions
        self._run_test(config, params, xp, kpt)

    @pytest.mark.parametrize("params", [
        SolverParams(dft=False, gamma=True, diis=False, smearing=False),
    ])
    def test_hf_gamma_fp(self, config, positions, params):
        """Test HF with gamma point and fixed-point iteration."""
        xp, kpt = positions
        self._run_test(config, params, xp, kpt)

    @pytest.mark.parametrize("params", [
        SolverParams(dft=False, gamma=True, diis=True, smearing=True),
    ])
    def test_hf_gamma_diis_smearing(self, config, positions, params):
        """Test HF with gamma point, DIIS, and smearing."""
        xp, kpt = positions
        self._run_test(config, params, xp, kpt)

    @pytest.mark.skip(reason="PySCF k-point tests cause segmentation fault")
    @pytest.mark.parametrize("params", [
        SolverParams(dft=False, gamma=False, diis=True, smearing=False),
    ])
    def test_hf_kpt_diis(self, config, positions, params):
        """Test HF with k-point and DIIS."""
        xp, kpt = positions
        self._run_test(config, params, xp, kpt)

    @pytest.mark.skip(reason="PySCF DFT tests cause segmentation fault")
    @pytest.mark.parametrize("params", [
        SolverParams(dft=True, gamma=True, diis=True, smearing=False),
    ])
    def test_dft_gamma_diis(self, config, positions, params):
        """Test DFT with gamma point and DIIS."""
        xp, kpt = positions
        self._run_test(config, params, xp, kpt)


# Legacy test functions for backward compatibility
def test_hf_gamma_diis():
    """Legacy test function."""
    test = TestPBCSolver()
    config = TestConfig()
    key = jax.random.PRNGKey(42)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = SolverParams(dft=False, gamma=True, diis=True, smearing=False)
    test._run_test(config, params, xp, kpt)


def test_hf_gamma_fp():
    """Legacy test function."""
    test = TestPBCSolver()
    config = TestConfig()
    key = jax.random.PRNGKey(42)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = SolverParams(dft=False, gamma=True, diis=False, smearing=False)
    test._run_test(config, params, xp, kpt)


def test_hf_gamma_diis_smearing():
    """Legacy test function."""
    test = TestPBCSolver()
    config = TestConfig()
    key = jax.random.PRNGKey(42)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = SolverParams(dft=False, gamma=True, diis=True, smearing=True)
    test._run_test(config, params, xp, kpt)
