import jax
import numpy as np
import jax.numpy as jnp
from typing import Callable
from functools import partial
from jax import vmap, grad, jit, jacfwd

from hqc.basis.parse import load_as_str, parse_quant_num, parse_gto, normalize_gto_coeff


def make_overlap(n: int, L: float, rs: float, 
                 basis: str, 
                 rcut: float = 24, 
                 gamma: bool = True, 
                 use_jit: bool = True, 
                 remat: bool = False) -> Callable:
    """
        Make the overlap function for a periodic system.
        Inputs:
            n: int, number of hydrogen atoms in unit cell.
            L: int, side length of unit cell, unit: rs.
            rs: float, Wigner-Seitz radius, unit: Bohr.
            basis: str, basis set name.
            rcut: float, cutoff radius, recommended value: rcut >= 18.
                         the larger rcut is, the more accurate the result is.
            gamma: bool, if True, return overlap(xp) for gamma point only, the output of overlap(xp) is real. 
                         if False, return overlap(xp, kpt) for a single k-point, the output of overlap(xp, kpt) is complex.
            use_jit: bool, if True, use jit to compile the overlap function.
            remat: bool, if True, use remat to compile the overlap function.
    """
    element = 'H'

    # load basis parameters
    def load_gto(element, basis):
        """
            Load basis parameters from basis file.
        """
        gto_coeffs = parse_gto(element, basis)
        quant_num = parse_quant_num(element, basis)
        norm_gto_coeffs = normalize_gto_coeff(quant_num, gto_coeffs)
        num_sets = len(quant_num)

        l_min = 10
        l_max = 0
        for set_i in range(num_sets):
            if quant_num[set_i][1] < l_min:
                l_min = quant_num[set_i][1]
            if quant_num[set_i][2] > l_max:
                l_max = quant_num[set_i][2]
        return quant_num, norm_gto_coeffs, num_sets, l_min, l_max
    
    quant_num, norm_gto_coeffs, num_sets, l_min, l_max = load_gto(element, basis)
    if l_min != 0:
        raise ValueError("l_min != 0")
    if l_max > 1:
        raise ValueError("l_max > 1 is not supported yet.")

    def _make_alpha():
        """
            List all alpha of GTO orbitals in a 1d array.
            Returns:
                alpha: array of shape (n_all_alpha,)
        """
        all_alpha = jnp.array([])
        for set_i in range(num_sets):
            all_alpha = jnp.concatenate((all_alpha, norm_gto_coeffs[set_i][:, 0]), axis=0)
        return all_alpha

    all_alpha = _make_alpha() # (n_all_alpha,)

    def _eval_num_cart(l):
        """
            Return the number of cartesian functions for angular momentum l.
        Returns:
            num_cart: int
        """
        return (l+2)*(l+1)//2

    def _make_alpha_coeff_cart():
        """
            Returns:
                alpha_coeff_cart: array of shape (n_all_alpha, n_gto_cart+1)
                alpha and coefficients all GTO orbitals in a 2d array.
        """
        n_gto = 0
        n_gto_s = 0
        n_gto_p = 0
        num_alpha = np.zeros(num_sets, dtype=int)

        for set_i in range(num_sets):
            num_alpha[set_i] = norm_gto_coeffs[set_i].shape[0]
            l_1 = quant_num[set_i][1] # min l in set_i
            l_2 = quant_num[set_i][2] # max l in set_i
            for l in range(l_1, l_2+1):
                i_1 = np.sum(quant_num[set_i][4:l-l_1+4])+1
                i_2 = np.sum(quant_num[set_i][4:l-l_1+5])+1
                for i in range(i_1, i_2):
                    if l == 0:
                        norm_gto_coeffs[set_i][:, i:i+1] = \
                            norm_gto_coeffs[set_i][:, i:i+1]/jnp.sqrt(4*jnp.pi) # cartesian to spherical
                        n_gto_s += 1
                    elif l == 1:
                        norm_gto_coeffs[set_i][:, i:i+1] = \
                            norm_gto_coeffs[set_i][:, i:i+1]*jnp.sqrt(3/(4*jnp.pi)) # cartesian to spherical
                        n_gto_p += 1
                    else:
                        raise ValueError("l_max > 1 is not supported yet.")
                    n_gto += 1

        alpha_coeff_cart = all_alpha.reshape(np.sum(num_alpha), 1)
        for set_i in range(0, num_sets):
            alpha_coeff_cart = np.hstack((alpha_coeff_cart, np.insert(np.zeros((np.sum(num_alpha)-num_alpha[set_i], 
                norm_gto_coeffs[set_i].shape[1]-1)), np.sum(num_alpha[:set_i]),
                norm_gto_coeffs[set_i][:, 1:], axis=0)))

        if n_gto+1 != alpha_coeff_cart.shape[1]:
            raise ValueError('n_gto != alpha_coeff_cart.shape[0]')

        return alpha_coeff_cart, n_gto_s, n_gto_p

    alpha_coeff_cart, n_gto_s, n_gto_p = _make_alpha_coeff_cart() # (n_all_alpha, n_gto_cart+1)

    L *= rs
    cell = jnp.eye(3)

    def gen_lattice():
        """
            Return lattice T within the cutoff radius in real space.
            OUTPUT:
                lattice: (n_lattice, 3), unit: Bohr.
        """
        tmax = rcut//(min(jnp.linalg.norm(cell, axis=-1))*L)
        nt = np.arange(-tmax, tmax+1)
        nis = np.meshgrid(*( [nt]*3 ))
        lattice = np.array([ni.flatten() for ni in nis]).T.dot(cell.T)*L
        lattice2 = (lattice**2).sum(axis=-1)
        lattice = lattice[lattice2<=rcut**2] # (n_lattice, 3)
        return lattice

    lattice = gen_lattice()

    # load coefficients of the basis
    coeffs = alpha_coeff_cart[:, 1:]
    n_ao = n*(n_gto_s+3*n_gto_p) # number of atomic orbitals
    # intermediate variables
    sum_alpha = all_alpha[:, None] + all_alpha[None, :]  # (n_all_alpha, n_all_alpha)
    prod_alpha = all_alpha[:, None] * all_alpha[None, :]  # (n_all_alpha, n_all_alpha)
    alpha2 = prod_alpha / sum_alpha  # (n_all_alpha, n_all_alpha)

    def _make_overlap():
        """
            make overlap function.
            if l_max == 0, return eval_overlap_s.
            if l_max == 1, return eval_overlap_sp.
            d and higher orbital is not support yet.
        """
        def _eval_intermediate_integral(xp1, xp2):
            """
                Evaluate intermediate overlap integrals.
                The original function is used for s orbital integral, use jax.grad for p orbitals.
                Args:
                    xp1: array of shape (3,), position of protons.
                    xp2: array of shape (3,), position of protons.
                Returns:
                    overlap: array of shape (n_all_alpha, n_all_alpha), overlap integrals.
            """
            Rmnc = jnp.sum(jnp.square(xp1[None, :] - xp2[None, :] + lattice), axis=1)
            _ovlp = jnp.pi**1.5*jnp.einsum('ij,ijl->ijl', 1/jnp.power(sum_alpha, 1.5), jnp.exp(-jnp.einsum('ij,l->ijl', alpha2, Rmnc)))
            overlap = jnp.sum(_ovlp, axis=2)
            return overlap
        
        def _eval_intermediate_integral_kpt(xp1, xp2, kpt):
            """
                Evaluate intermediate overlap integrals at k point.
                The original function is used for s orbital integral, use jax.grad for p orbitals.
                Args:
                    xp1: array of shape (3,), position of protons.
                    xp2: array of shape (3,), position of protons.
                    kpt: array of shape (3,), k point. 1BZ: (-pi/L, pi/L)
                Returns:
                    overlap: array of shape (n_all_alpha, n_all_alpha), overlap integrals.
            """
            Rmnc = jnp.sum(jnp.square(xp1[None, :] - xp2[None, :] + lattice), axis=1)
            _ovlp = jnp.pi**1.5*jnp.einsum('ij,ijl,l->ijl', 1/jnp.power(sum_alpha, 1.5), 
                    jnp.exp(-jnp.einsum('ij,l->ijl', alpha2, Rmnc)), jnp.exp(-1j*lattice@kpt))
            overlap = jnp.sum(_ovlp, axis=2)
            return overlap
    
        def eval_overlap_s(xp1, xp2):
            """
                Evaluate overlap matrix for s orbitals.
                Args:
                    xp1: array of shape (3,), position of protons.
                    xp2: array of shape (3,), position of protons.
                Returns:
                    ovlp: array of shape (n_ao_one_atom, n_ao_one_atom), overlap matrix.
            """
            overlap_s = _eval_intermediate_integral(xp1, xp2)
            ovlp_s = jnp.einsum('ip,jq,ij->pq', coeffs, coeffs, overlap_s)
            return ovlp_s
        
        def eval_overlap_s_kpt(xp1, xp2, kpt):
            """
                Evaluate overlap matrix for s orbitals.
                Args:
                    xp1: array of shape (3,), position of protons.
                    xp2: array of shape (3,), position of protons.
                    kpt: array of shape (3,), k point. 1BZ: (-pi/L, pi/L)
                Returns:
                    ovlp: array of shape (n_ao_one_atom, n_ao_one_atom), overlap matrix.
            """
            overlap_s = _eval_intermediate_integral_kpt(xp1, xp2, kpt)
            ovlp_s = jnp.einsum('ip,jq,ij->pq', coeffs, coeffs, overlap_s)
            return ovlp_s

        def _matrix_s_p(integral_s, integral_sp, integral_ps, integral_p):
            """
                internal function for s and p orbital matrix elements calculation.
            """
            matrix_s = jnp.einsum('ip,jq,ij->pq', coeffs, coeffs, integral_s)
            matrix_sp = jnp.einsum('ip,jq,j,ijb->pqb', coeffs, coeffs, 1/all_alpha, integral_sp)/2
            matrix_ps = jnp.einsum('ip,jq,i,ija->pqa', coeffs, coeffs, 1/all_alpha, integral_ps)/2
            matrix_p = jnp.einsum('ip,jq,i,j,ijab->pqab', coeffs, coeffs, 1/all_alpha, 1/all_alpha, integral_p)/4

            matrix_s = matrix_s[:n_gto_s, :n_gto_s]
            matrix_sp = matrix_sp[:n_gto_s, -n_gto_p:].reshape(n_gto_s, 3*n_gto_p)
            matrix_ps = matrix_ps[-n_gto_p:, :n_gto_s].transpose(0,2,1).reshape(3*n_gto_p, n_gto_s)
            matrix_p = matrix_p[-n_gto_p:, -n_gto_p:].transpose(0,2,1,3).reshape(3*n_gto_p, 3*n_gto_p)

            matrix = jnp.hstack((jnp.vstack((matrix_s, matrix_ps)), jnp.vstack((matrix_sp, matrix_p))))
            return matrix
        
        def eval_overlap_sp(xp1, xp2):
            """
                Evaluate overlap matrix for s and p orbitals.
                Args:
                    xp1: array of shape (3,), position of protons.
                    xp2: array of shape (3,), position of protons.
                Returns:
                    ovlp: array of shape (n_ao, n_ao), overlap matrix.
            """
            overlap_s = _eval_intermediate_integral(xp1, xp2)
            overlap_sp  = jacfwd(_eval_intermediate_integral, argnums=1)(xp1, xp2)
            overlap_ps = jacfwd(_eval_intermediate_integral, argnums=0)(xp1, xp2)
            overlap_p = jacfwd(jacfwd(_eval_intermediate_integral), argnums=1)(xp1, xp2)

            ovlp = _matrix_s_p(overlap_s, overlap_sp, overlap_ps, overlap_p)
            return ovlp
        
        def eval_overlap_sp_kpt(xp1, xp2, kpt):
            """
                Evaluate overlap matrix for s and p orbitals.
                Args:
                    xp1: array of shape (3,), position of protons.
                    xp2: array of shape (3,), position of protons.
                    kpt: array of shape (3,), k point. 1BZ: (-pi/L, pi/L)
                Returns:
                    ovlp: array of shape (n_ao, n_ao), overlap matrix.
            """
            overlap_s = _eval_intermediate_integral_kpt(xp1, xp2, kpt)
            overlap_sp = jacfwd(_eval_intermediate_integral_kpt, argnums=1)(xp1, xp2, kpt)
            overlap_ps = jacfwd(_eval_intermediate_integral_kpt, argnums=0)(xp1, xp2, kpt)
            overlap_p = jacfwd(jacfwd(_eval_intermediate_integral_kpt), argnums=1)(xp1, xp2, kpt)

            ovlp = _matrix_s_p(overlap_s, overlap_sp, overlap_ps, overlap_p)
            return ovlp
        
        if gamma:
            if l_max == 0:
                return eval_overlap_s
            elif l_max == 1:
                return eval_overlap_sp
            else:
                raise ValueError("l_max > 1 is not supported yet.")
        else:
            if l_max == 0:
                return eval_overlap_s_kpt
            elif l_max == 1:
                return eval_overlap_sp_kpt
            else:
                raise ValueError("l_max > 1 is not supported yet.")

    if gamma:
        eval_overlap_noreshape = vmap(vmap(_make_overlap(), (0, None), 0), (None, 0), 2)
    else:
        eval_overlap_kpt_noreshape = vmap(vmap(_make_overlap(), (0, None, None), 0), (None, 0, None), 2)

    def overlap(xp: jnp.ndarray) -> jnp.ndarray:
        """
            Evaluate overlap matrix for a single configuration at gamma point.
            Args:
                xp: array of shape (n, 3), positions of protons.
            Returns:
                ovlp: array of shape (n_ao, n_ao), overlap matrix, real.
        """
        xp *= rs
        return eval_overlap_noreshape(xp, xp).reshape(n_ao, n_ao)

    def overlap_kpt(xp: jnp.ndarray, kpt: jnp.ndarray) -> jnp.ndarray:
        """
            Evaluate overlap matrix for a single configuration at k point.
            Args:
                xp: array of shape (n, 3), positions of protons.
                kpt: array of shape (3,), k point. 1BZ: (-pi/L, pi/L)
            Returns:
                ovlp: array of shape (n_ao, n_ao), overlap matrix, complex.
        """
        xp *= rs
        return eval_overlap_kpt_noreshape(xp, xp, kpt).reshape(n_ao, n_ao)

    if gamma:
        overlap_func = overlap
    else:
        overlap_func = overlap_kpt
    
    if use_jit:
        return jit(overlap_func)
    else:
        return overlap_func
