"""
Test suite for HQC PBC PES (Potential Energy Surface).

This module tests the potential energy surface calculation for periodic
boundary conditions against PySCF reference implementations.
"""

import jax
import numpy as np
import jax.numpy as jnp
import pytest
from dataclasses import dataclass
from typing import Tuple

jax.config.update("jax_enable_x64", True)

from hqc.pbc.pes import make_pes
from config import YELLOW, BLUE, GREEN, RESET
from test_pyscf import pyscf_solver


@dataclass
class PESTestConfig:
    """Configuration for PES tests."""
    n: int = 4
    dim: int = 3
    rs: float = 1.5
    basis_set: Tuple[str, ...] = ('gth-dzv',)  # Single basis for faster testing
    xc: str = 'lda,vwn'
    rcut: int = 24
    grid_length: float = 0.12
    smearing_sigma: float = 0.1
    atol_energy: float = 1e-3
    random_seed: int = 42

    @property
    def L(self) -> float:
        """Box length."""
        return (4/3 * jnp.pi * self.n)**(1/3)


@dataclass
class PESParams:
    """Parameters for a specific PES test."""
    dft: bool
    gamma: bool
    smearing: bool

    @property
    def name(self) -> str:
        """Generate descriptive test name."""
        method = "dft" if self.dft else "hf"
        kpoint = "gamma" if self.gamma else "kpt"
        smear = "_smearing" if self.smearing else ""
        return f"pes_{method}_{kpoint}{smear}"


class TestPBCPES:
    """Test class for PBC PES."""

    @pytest.fixture(scope="class")
    def config(self) -> PESTestConfig:
        """Test configuration fixture."""
        return PESTestConfig()

    @pytest.fixture(scope="class")
    def positions(self, config: PESTestConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate random positions and k-point."""
        key = jax.random.PRNGKey(config.random_seed)
        key_p, key_kpt = jax.random.split(key)
        xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
        kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                                maxval=jnp.pi/config.L/config.rs)
        return xp, kpt

    def _print_test_info(self, config: PESTestConfig, params: PESParams,
                        xp: jnp.ndarray, kpoint: jnp.ndarray):
        """Print test configuration information."""
        print(f"\n{YELLOW}============= Test: {params.name} ============={RESET}")
        print(f"{BLUE}Method:{RESET} {'DFT' if params.dft else 'HF'}")
        print(f"{BLUE}K-point:{RESET} {'Gamma' if params.gamma else 'Random'}")
        print(f"{BLUE}Smearing:{RESET} {params.smearing}")
        if params.smearing:
            print(f"{BLUE}Smearing sigma:{RESET} {config.smearing_sigma}")
        print(f"{BLUE}System:{RESET} n={config.n}, rs={config.rs}, L={config.L:.4f}")
        print(f"{BLUE}K-point value:{RESET} {kpoint}")
        print(f"{BLUE}Basis set:{RESET} {config.basis_set}")
        print(f"{BLUE}Grid length:{RESET} {config.grid_length}")
        print(f"{BLUE}Positions shape:{RESET} {xp.shape}")

    def _assert_close(self, name: str, hqc_val, pyscf_val, atol: float):
        """Assert values are close and print comparison."""
        diff = abs(hqc_val - pyscf_val)
        print(f"{BLUE}{name}:{RESET}")
        print(f"  HQC:   {hqc_val:.10f}")
        print(f"  PySCF: {pyscf_val:.10f}")
        print(f"  Diff:  {diff:.2e}")
        assert np.allclose(hqc_val, pyscf_val, atol=atol), \
            f"{name} mismatch: diff = {diff:.2e}, atol = {atol}"
        print(f"{GREEN}  ✓ Matches PySCF{RESET}")

    def _run_pes_test(self, config: PESTestConfig, params: PESParams,
                     xp: jnp.ndarray, kpt: jnp.ndarray, basis: str):
        """Run a single PES test for given basis."""
        print(f"\n{YELLOW}===== Basis: {basis} ====={RESET}")

        # Determine k-point
        kpoint = jnp.array([0., 0., 0.]) if params.gamma else kpt

        # Run PySCF reference
        pyscf_data = pyscf_solver(
            config.n, config.L, config.rs, xp, kpoint, basis,
            ifdft=params.dft, xc=config.xc, smearing=params.smearing,
            smearing_method='fermi', smearing_sigma=config.smearing_sigma
        )

        # Run HQC PES
        pes = make_pes(
            config.n, config.L, config.rs, basis,
            grid_length=config.grid_length, dft=params.dft,
            smearing=params.smearing, smearing_sigma=config.smearing_sigma,
            gamma=params.gamma, mode='dev'
        )

        if params.gamma:
            E, Ki, Vep, Vee, Vpp, Se, converged = pes(xp)
        else:
            E, Ki, Vep, Vee, Vpp, Se, converged = pes(xp, kpoint)

        # Check convergence
        print(f"{BLUE}Converged:{RESET} {converged}")
        assert converged, "PES calculation did not converge"
        print(f"{GREEN}✓ PES converged{RESET}")

        # Compare energies
        print(f"\n{BLUE}Energy Components:{RESET}")
        self._assert_close("Total energy (Etot)", E, pyscf_data['Etot'], config.atol_energy)

        Ecore = Ki + Vep
        self._assert_close("Core energy (Ki+Vep)", Ecore, pyscf_data["Ecore"], config.atol_energy)

        self._assert_close("Electron-electron energy (Vee)", Vee, pyscf_data["Vee"], config.atol_energy)

        self._assert_close("Proton-proton energy (Vpp)", Vpp, pyscf_data["Vpp"], config.atol_energy)

        self._assert_close("Entropy (Se)", Se, pyscf_data["Se"], config.atol_energy)

    def _run_test(self, config: PESTestConfig, params: PESParams,
                 xp: jnp.ndarray, kpt: jnp.ndarray):
        """Run test for all basis sets."""
        kpoint = jnp.array([0., 0., 0.]) if params.gamma else kpt
        self._print_test_info(config, params, xp, kpoint)

        for basis in config.basis_set:
            self._run_pes_test(config, params, xp, kpt, basis)

    # Test methods
    def test_pes_hf_gamma(self, config, positions):
        """Test PES for HF with gamma point."""
        xp, kpt = positions
        params = PESParams(dft=False, gamma=True, smearing=False)
        self._run_test(config, params, xp, kpt)

    def test_pes_hf_gamma_smearing(self, config, positions):
        """Test PES for HF with gamma point and smearing."""
        xp, kpt = positions
        params = PESParams(dft=False, gamma=True, smearing=True)
        self._run_test(config, params, xp, kpt)

    @pytest.mark.skip(reason="PySCF k-point tests cause segmentation fault")
    def test_pes_hf_kpt(self, config, positions):
        """Test PES for HF with k-point."""
        xp, kpt = positions
        params = PESParams(dft=False, gamma=False, smearing=False)
        self._run_test(config, params, xp, kpt)

    @pytest.mark.skip(reason="PySCF DFT tests cause segmentation fault")
    def test_pes_dft_gamma(self, config, positions):
        """Test PES for DFT with gamma point."""
        xp, kpt = positions
        params = PESParams(dft=True, gamma=True, smearing=True)
        self._run_test(config, params, xp, kpt)


# Legacy test functions for backward compatibility
def test_pes_hf_gamma():
    """Legacy test function."""
    test = TestPBCPES()
    config = PESTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = PESParams(dft=False, gamma=True, smearing=False)
    test._run_test(config, params, xp, kpt)


def test_pes_hf_kpt():
    """Legacy test function."""
    test = TestPBCPES()
    config = PESTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = PESParams(dft=False, gamma=False, smearing=False)
    test._run_test(config, params, xp, kpt)


def test_pes_dft_gamma():
    """Legacy test function."""
    test = TestPBCPES()
    config = PESTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = PESParams(dft=True, gamma=True, smearing=True)
    test._run_test(config, params, xp, kpt)


def test_pes_dft_kpt():
    """Legacy test function."""
    test = TestPBCPES()
    config = PESTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = PESParams(dft=True, gamma=False, smearing=True)
    test._run_test(config, params, xp, kpt)


def test_pes_hf_gamma_smearing():
    """Legacy test function."""
    test = TestPBCPES()
    config = PESTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = PESParams(dft=False, gamma=True, smearing=True)
    test._run_test(config, params, xp, kpt)


def test_pes_hf_kpt_smearing():
    """Legacy test function."""
    test = TestPBCPES()
    config = PESTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = PESParams(dft=False, gamma=False, smearing=True)
    test._run_test(config, params, xp, kpt)


def test_pes_dft_gamma_smearing():
    """Legacy test function."""
    test = TestPBCPES()
    config = PESTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = PESParams(dft=True, gamma=True, smearing=True)
    test._run_test(config, params, xp, kpt)


def test_pes_dft_kpt_smearing():
    """Legacy test function."""
    test = TestPBCPES()
    config = PESTestConfig()
    key = jax.random.PRNGKey(config.random_seed)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (config.n, 3), minval=0., maxval=config.L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/config.L/config.rs,
                            maxval=jnp.pi/config.L/config.rs)
    params = PESParams(dft=True, gamma=False, smearing=True)
    test._run_test(config, params, xp, kpt)
