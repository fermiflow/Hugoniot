"""
Test STO-3G and STO-6G basis sets.
"""

import pytest
import jax
import jax.numpy as jnp
from hqc.gto.solver import make_solver

jax.config.update("jax_enable_x64", True)


class TestSTOBasis:
    """Test STO-nG basis sets."""

    def test_sto3g_h2(self):
        """Test H2 molecule with STO-3G basis."""
        atom_positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        atom_charges = jnp.array([1.0, 1.0])
        n_electrons = 2

        hf = make_solver(atom_charges, n_electrons, basis='sto-3g', use_jit=False)
        result = hf(atom_positions)

        print(f"\nH2 with STO-3G:")
        print(f"  Converged: {result['converged']}")
        print(f"  Energy: {result['energy']:.6f} Ha")

        assert result['converged']
        # STO-3G gives slightly different energy than GTH-SZV
        assert -1.2 < result['energy'] < -1.0

    def test_sto6g_h2(self):
        """Test H2 molecule with STO-6G basis."""
        atom_positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        atom_charges = jnp.array([1.0, 1.0])
        n_electrons = 2

        hf = make_solver(atom_charges, n_electrons, basis='sto-6g', use_jit=False)
        result = hf(atom_positions)

        print(f"\nH2 with STO-6G:")
        print(f"  Converged: {result['converged']}")
        print(f"  Energy: {result['energy']:.6f} Ha")

        assert result['converged']
        # STO-6G should give better energy than STO-3G
        assert -1.2 < result['energy'] < -1.0

    def test_sto3g_vs_sto6g(self):
        """Compare STO-3G and STO-6G energies."""
        atom_positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        atom_charges = jnp.array([1.0, 1.0])
        n_electrons = 2

        hf_3g = make_solver(atom_charges, n_electrons, basis='sto-3g', use_jit=False)
        result_3g = hf_3g(atom_positions)

        hf_6g = make_solver(atom_charges, n_electrons, basis='sto-6g', use_jit=False)
        result_6g = hf_6g(atom_positions)

        print(f"\nBasis set comparison:")
        print(f"  STO-3G: {result_3g['energy']:.6f} Ha")
        print(f"  STO-6G: {result_6g['energy']:.6f} Ha")
        print(f"  Difference: {abs(result_3g['energy'] - result_6g['energy']):.6f} Ha")

        # STO-6G should give lower (more negative) energy
        assert result_6g['energy'] < result_3g['energy']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
