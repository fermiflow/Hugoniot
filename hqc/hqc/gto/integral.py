"""
Gaussian integral evaluation for non-periodic systems.

This module handles:
- Primitive Gaussian integrals (overlap, kinetic, nuclear attraction, ERI)
- Vectorized contracted GTO integral matrix construction
- Loading basis sets and building complete integral matrices

Implements analytical formulas for:
- Overlap integrals: S_μν = ∫ φ_μ(r) φ_ν(r) dr
- Kinetic energy integrals: T_μν = ∫ φ_μ(r) (-½∇²) φ_ν(r) dr
- Nuclear attraction integrals: V_μν = ∫ φ_μ(r) (-Z/|r-R|) φ_ν(r) dr
- Electron repulsion integrals: (μν|λσ) = ∫∫ φ_μ(r₁)φ_ν(r₁) (1/|r₁-r₂|) φ_λ(r₂)φ_σ(r₂) dr₁dr₂

For Gaussian primitives: φ = N·exp(-α|r-R|²)·x^l·y^m·z^n
"""

import jax
import jax.numpy as jnp
import numpy as np
from hqc.basis.parse import parse_gto, parse_quant_num, normalize_gto_coeff
from hqc.gto.boys import boys_function_array


# ============================================================================
# Primitive Gaussian Integrals (Vectorized)
# ============================================================================

def overlap_primitive_s_vec(alpha_a, Ra, alpha_b, Rb):
    """Vectorized overlap integral for s-type primitives.

    Args:
        alpha_a: shape (...,)
        Ra: shape (..., 3)
        alpha_b: shape (...,)
        Rb: shape (..., 3)
    Returns:
        S: shape (...)
    """
    p = alpha_a + alpha_b
    P = (alpha_a[..., None] * Ra + alpha_b[..., None] * Rb) / p[..., None]
    AB = Ra - Rb
    K_AB = jnp.exp(-alpha_a * alpha_b * jnp.sum(AB * AB, axis=-1) / p)
    return jnp.power(jnp.pi / p, 1.5) * K_AB


def kinetic_primitive_s_vec(alpha_a, Ra, alpha_b, Rb):
    """Vectorized kinetic energy integral for s-type primitives."""
    p = alpha_a + alpha_b
    AB = Ra - Rb
    AB2 = jnp.sum(AB * AB, axis=-1)
    K_AB = jnp.exp(-alpha_a * alpha_b * AB2 / p)
    S = jnp.power(jnp.pi / p, 1.5) * K_AB
    T = alpha_a * alpha_b / p * (3.0 - 2.0 * alpha_a * alpha_b * AB2 / p) * S
    return T


def nuclear_primitive_s_vec(alpha_a, Ra, alpha_b, Rb, Rc, Zc):
    """Vectorized nuclear attraction integral for s-type primitives."""
    p = alpha_a + alpha_b
    P = (alpha_a[..., None] * Ra + alpha_b[..., None] * Rb) / p[..., None]
    AB = Ra - Rb
    K_AB = jnp.exp(-alpha_a * alpha_b * jnp.sum(AB * AB, axis=-1) / p)
    PC = P - Rc
    T = p * jnp.sum(PC * PC, axis=-1)
    F0 = boys_function_array(0, T)[0]
    V = -2.0 * jnp.pi / p * K_AB * Zc * F0
    return V


def eri_primitive_s_vec(alpha_a, Ra, alpha_b, Rb, alpha_c, Rc, alpha_d, Rd):
    """Vectorized electron repulsion integral for s-type primitives."""
    p_ab = alpha_a + alpha_b
    P = (alpha_a[..., None] * Ra + alpha_b[..., None] * Rb) / p_ab[..., None]
    AB = Ra - Rb
    K_AB = jnp.exp(-alpha_a * alpha_b * jnp.sum(AB * AB, axis=-1) / p_ab)

    p_cd = alpha_c + alpha_d
    Q = (alpha_c[..., None] * Rc + alpha_d[..., None] * Rd) / p_cd[..., None]
    CD = Rc - Rd
    K_CD = jnp.exp(-alpha_c * alpha_d * jnp.sum(CD * CD, axis=-1) / p_cd)

    rho = p_ab * p_cd / (p_ab + p_cd)
    PQ = P - Q
    T = rho * jnp.sum(PQ * PQ, axis=-1)
    F0 = boys_function_array(0, T)[0]

    eri = 2.0 * jnp.power(jnp.pi, 2.5) / (p_ab * p_cd * jnp.sqrt(p_ab + p_cd)) * K_AB * K_CD * F0
    return eri


# ============================================================================
# Basis Set Preprocessing (for make_solver)
# ============================================================================

def prepare_basis_data(atom_charges, basis='sto-3g'):
    """Prepare basis set data that doesn't depend on atom positions.

    This function is called once in make_solver to extract all basis information.
    Uses numpy and for loops since it's only called once during preprocessing.

    Args:
        atom_charges: array of shape (n_atoms,), nuclear charges
        basis: str, basis set name

    Returns:
        basis_data: dict containing:
            - n_ao_per_atom: int, number of AOs per atom
            - n_primitives: int, number of primitive Gaussians per contracted GTO
            - alphas: array of shape (n_sets, n_primitives), exponents
            - coeffs: array of shape (n_sets, n_primitives), contraction coefficients
            - n_sets: int, number of contracted GTO sets per atom
    """
    # Assume all atoms are hydrogen for now
    quant_num = parse_quant_num('H', basis)
    gto_coeffs_raw = parse_gto('H', basis)
    gto_coeffs = normalize_gto_coeff(quant_num, gto_coeffs_raw)

    # Apply cartesian to spherical conversion
    for set_i in range(len(quant_num)):
        l_min = quant_num[set_i][1]
        l_max = quant_num[set_i][2]
        for l in range(l_min, l_max + 1):
            i_min = np.sum(quant_num[set_i][4:l-l_min+4]) + 1
            i_max = np.sum(quant_num[set_i][4:l-l_min+5]) + 1
            for i in range(i_min, i_max):
                if l == 0:
                    gto_coeffs[set_i][:, i:i+1] = gto_coeffs[set_i][:, i:i+1] / np.sqrt(4*np.pi)
                elif l == 1:
                    gto_coeffs[set_i][:, i:i+1] = gto_coeffs[set_i][:, i:i+1] * np.sqrt(3/(4*np.pi))
                else:
                    raise NotImplementedError("Only s and p orbitals are currently supported")

    # Count AOs per atom (only s-type for now)
    n_ao_per_atom = 0
    for i, qn in enumerate(quant_num):
        l_min, l_max = qn[1], qn[2]
        if l_min == 0 and l_max == 0:
            n_ao_per_atom += 1
        else:
            raise NotImplementedError("Only s-type orbitals are currently supported")

    # Extract alphas and coeffs for each set
    n_sets = len(quant_num)
    n_primitives = gto_coeffs[0].shape[0]

    alphas = np.zeros((n_sets, n_primitives))
    coeffs = np.zeros((n_sets, n_primitives))

    for set_idx in range(n_sets):
        alphas[set_idx] = gto_coeffs[set_idx][:, 0]
        coeffs[set_idx] = gto_coeffs[set_idx][:, 1]

    return {
        'n_ao_per_atom': n_ao_per_atom,
        'n_primitives': n_primitives,
        'alphas': jnp.array(alphas),
        'coeffs': jnp.array(coeffs),
        'n_sets': n_sets,
        'n_atoms': len(atom_charges),
        'atom_charges': jnp.array(atom_charges)
    }


# ============================================================================
# Vectorized Integral Matrix Construction
# ============================================================================

def build_integral_matrices_vec(atom_positions, basis_data):
    """Build integral matrices using vectorized operations.

    This function is designed to be called repeatedly with different atom_positions.
    All basis information is pre-computed in basis_data.

    Args:
        atom_positions: array of shape (n_atoms, 3)
        basis_data: dict from prepare_basis_data()

    Returns:
        S, T, V, eri: integral matrices
    """
    atom_positions = jnp.asarray(atom_positions)
    n_atoms = basis_data['n_atoms']
    n_sets = basis_data['n_sets']
    n_primitives = basis_data['n_primitives']
    alphas = basis_data['alphas']  # (n_sets, n_primitives)
    coeffs = basis_data['coeffs']  # (n_sets, n_primitives)
    atom_charges = basis_data['atom_charges']

    n_ao = n_atoms * n_sets

    # Build arrays of centers, alphas, coeffs for all basis functions
    # Shape: (n_ao, n_primitives, ...)
    centers = jnp.repeat(atom_positions, n_sets, axis=0)  # (n_ao, 3)
    alphas_all = jnp.tile(alphas, (n_atoms, 1))  # (n_ao, n_primitives)
    coeffs_all = jnp.tile(coeffs, (n_atoms, 1))  # (n_ao, n_primitives)

    # Build overlap matrix S
    # For each pair (i,j), sum over primitives (k,l)
    # S[i,j] = sum_k sum_l coeff_i[k] * coeff_j[l] * overlap(alpha_i[k], R_i, alpha_j[l], R_j)

    # Expand to (n_ao, n_ao, n_primitives, n_primitives)
    alpha_i = alphas_all[:, None, :, None]  # (n_ao, 1, n_primitives, 1)
    alpha_j = alphas_all[None, :, None, :]  # (1, n_ao, 1, n_primitives)
    coeff_i = coeffs_all[:, None, :, None]
    coeff_j = coeffs_all[None, :, None, :]
    R_i = centers[:, None, None, None, :]  # (n_ao, 1, 1, 1, 3)
    R_j = centers[None, :, None, None, :]  # (1, n_ao, 1, 1, 3)

    # Compute all primitive overlaps
    S_prim = overlap_primitive_s_vec(alpha_i, R_i, alpha_j, R_j)  # (n_ao, n_ao, n_primitives, n_primitives)
    S = jnp.sum(coeff_i * coeff_j * S_prim, axis=(2, 3))  # (n_ao, n_ao)

    # Build kinetic matrix T
    T_prim = kinetic_primitive_s_vec(alpha_i, R_i, alpha_j, R_j)
    T = jnp.sum(coeff_i * coeff_j * T_prim, axis=(2, 3))

    # Build nuclear attraction matrix V
    # Need to sum over all nuclei
    V = jnp.zeros((n_ao, n_ao))
    for atom_idx in range(n_atoms):
        Rc = atom_positions[atom_idx]
        Zc = atom_charges[atom_idx]
        V_prim = nuclear_primitive_s_vec(alpha_i, R_i, alpha_j, R_j, Rc, Zc)
        V = V + jnp.sum(coeff_i * coeff_j * V_prim, axis=(2, 3))

    # Build ERI tensor
    # Shape: (n_ao, n_ao, n_ao, n_ao)
    # eri[i,j,k,l] = sum over all 4 primitive indices

    alpha_i = alphas_all[:, None, None, None, :, None, None, None]  # (n_ao, 1, 1, 1, n_prim, 1, 1, 1)
    alpha_j = alphas_all[None, :, None, None, None, :, None, None]  # (1, n_ao, 1, 1, 1, n_prim, 1, 1)
    alpha_k = alphas_all[None, None, :, None, None, None, :, None]  # (1, 1, n_ao, 1, 1, 1, n_prim, 1)
    alpha_l = alphas_all[None, None, None, :, None, None, None, :]  # (1, 1, 1, n_ao, 1, 1, 1, n_prim)

    coeff_i = coeffs_all[:, None, None, None, :, None, None, None]
    coeff_j = coeffs_all[None, :, None, None, None, :, None, None]
    coeff_k = coeffs_all[None, None, :, None, None, None, :, None]
    coeff_l = coeffs_all[None, None, None, :, None, None, None, :]

    R_i = centers[:, None, None, None, None, None, None, None, :]  # (n_ao, 1, 1, 1, 1, 1, 1, 1, 3)
    R_j = centers[None, :, None, None, None, None, None, None, :]
    R_k = centers[None, None, :, None, None, None, None, None, :]
    R_l = centers[None, None, None, :, None, None, None, None, :]

    eri_prim = eri_primitive_s_vec(alpha_i, R_i, alpha_j, R_j, alpha_k, R_k, alpha_l, R_l)
    eri = jnp.sum(coeff_i * coeff_j * coeff_k * coeff_l * eri_prim, axis=(4, 5, 6, 7))

    return S, T, V, eri
