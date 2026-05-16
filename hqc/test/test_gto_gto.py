"""
Test GTO integral matrix construction against PySCF.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf
from hqc.gto.integral import prepare_basis_data, build_integral_matrices_vec

jax.config.update("jax_enable_x64", True)


class TestGTOIntegrals:
    """Test GTO integral matrix construction."""

    def test_h2_gth_szv_overlap(self):
        """Test overlap matrix for H2 molecule with GTH-SZV basis."""
        # H2 molecule at 1.4 Bohr separation
        atom_positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        atom_charges = jnp.array([1.0, 1.0])

        basis_data = prepare_basis_data(atom_charges, basis='gth-szv')
        S, T, V, eri = build_integral_matrices_vec(atom_positions, basis_data)

        # Compare with PySCF
        mol = gto.M(atom='H 0 0 0; H 1.4 0 0', basis='gth-szv', unit='B')
        S_pyscf = mol.intor('int1e_ovlp')

        print("\nHQC overlap matrix:")
        print(S)
        print("\nPySCF overlap matrix:")
        print(S_pyscf)

        assert S.shape == S_pyscf.shape
        assert jnp.allclose(S, S_pyscf, atol=1e-6)

    def test_single_h_atom(self):
        """Test integrals for single H atom."""
        atom_positions = jnp.array([[0.0, 0.0, 0.0]])
        atom_charges = jnp.array([1.0])

        basis_data = prepare_basis_data(atom_charges, basis='gth-szv')
        S, T, V, eri = build_integral_matrices_vec(atom_positions, basis_data)

        # For single atom, overlap diagonal should be normalized to 1
        print(f"\nS shape: {S.shape}")
        print(f"S matrix:\n{S}")
        assert S.shape == (1, 1)
        assert jnp.abs(S[0, 0] - 1.0) < 1e-10, f"Overlap not normalized: S[0,0] = {S[0,0]}"

        # Kinetic energy should be positive
        assert T[0, 0] > 0

        # Nuclear attraction should be negative
        assert V[0, 0] < 0

        # ERI should be positive
        assert eri[0, 0, 0, 0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
