"""
SCF (Self-Consistent Field) solver for non-periodic systems.

Implements both fixed-point iteration and DIIS acceleration.
Based on hqc/pbc/scf.py but simplified for non-periodic systems.
"""

import jax
import jax.numpy as jnp


def make_scf(diis=True, diis_space=8, diis_start_cycle=1, diis_damp=0.0,
             tol=1e-7, max_cycle=100):
    """
    Make SCF function for non-periodic GTO calculations.

    Args:
        diis: bool, whether to use DIIS acceleration
        diis_space: int, number of vectors in DIIS space
        diis_start_cycle: int, start DIIS after this cycle
        diis_damp: float, DIIS damping factor (0 = no damping)
        tol: float, convergence tolerance for energy
        max_cycle: int, maximum number of SCF iterations

    Returns:
        scf: function to perform SCF iteration
            Interface: scf(v_ovlp, Hcore, dm_init, hartree_fn, exchange_correlation_fn,
                          density_matrix_fn, errvec_sdf_fn)
            Returns: (mo_coeff, w1, E, converged)
    """

    def fixed_point_scf(v_ovlp, Hcore, dm_init, hartree_fn, exchange_correlation_fn,
                       density_matrix_fn, errvec_sdf_fn):
        """
        Fixed point iteration for SCF.

        Args:
            v_ovlp: array of shape (n_ao, n_ao), orthonormal matrix of overlap matrix S
                V^T @ S @ V = I
            Hcore: array of shape (n_ao, n_ao), core Hamiltonian matrix (T + V)
            dm_init: array of shape (n_ao, n_ao), initial guess of density matrix
            hartree_fn: function to compute Hartree (Coulomb) matrix
                J = hartree_fn(dm)
            exchange_correlation_fn: function to compute exchange-correlation
                Exc, Vxc = exchange_correlation_fn(dm)
                For HF: Vxc = -0.5*K, Exc = -0.5*Tr(K@dm)
                For DFT: Vxc and Exc from functional
            density_matrix_fn: function to compute density matrix
                dm = density_matrix_fn(mo_coeff, w1)
            errvec_sdf_fn: function to compute DIIS error vector
                errvec = errvec_sdf_fn(dm, F) = S@dm@F - F@dm@S

        Returns:
            mo_coeff: array of shape (n_ao, n_mo), molecular orbital coefficients
            w1: array of shape (n_mo,), orbital energies
            E: float, total electronic energy
            converged: bool, whether SCF converged
        """
        mo_coeff_init = jnp.empty_like(dm_init)
        w_init = jnp.empty_like(dm_init[0], dtype=jnp.float64)

        def body_fun(carry):
            _, E, _, dm, _, loop = carry

            # Hartree and exchange-correlation terms
            J = hartree_fn(dm)
            Exc, Vxc = exchange_correlation_fn(dm)

            # Total energy
            E_new = 0.5 * jnp.einsum('pq,qp', 2*Hcore + J, dm).real + Exc

            # Fock matrix
            F = Hcore + J + Vxc

            # Diagonalization in orthonormal basis
            F1 = jnp.einsum('pq,qr,rs->ps', v_ovlp.T, F, v_ovlp)
            w1, c1 = jnp.linalg.eigh(F1)

            # Transform back to AO basis
            mo_coeff = jnp.dot(v_ovlp, c1)
            dm = density_matrix_fn(mo_coeff, w1)

            return E, E_new, mo_coeff, dm, w1, loop + 1

        def cond_fun(carry):
            E_prev, E_new, _, _, _, loop = carry
            return (jnp.abs(E_new - E_prev) > tol) & (loop < max_cycle)

        _, E, mo_coeff, dm, w1, loop = jax.lax.while_loop(
            cond_fun, body_fun,
            (0.0, 1.0, mo_coeff_init, dm_init, w_init, 0)
        )
        converged = jnp.logical_not(loop == max_cycle)

        return mo_coeff, w1, E, converged

    def diis_scf(v_ovlp, Hcore, dm_init, hartree_fn, exchange_correlation_fn,
                density_matrix_fn, errvec_sdf_fn):
        """
        DIIS-accelerated SCF iteration.

        Same interface as fixed_point_scf but uses DIIS to accelerate convergence.
        """
        mo_coeff_init = jnp.empty_like(dm_init)
        w_init = jnp.empty_like(dm_init[0], dtype=jnp.float64)

        # Initialize F and error vector series for DIIS
        J = hartree_fn(dm_init)
        Vxc = exchange_correlation_fn(dm_init)[1]
        F_init = Hcore + J + Vxc
        errvec_init = errvec_sdf_fn(dm_init, F_init)
        F_k = jnp.repeat(F_init[None, ...], diis_space, axis=0)
        errvec_k = jnp.repeat(errvec_init[None, ...], diis_space, axis=0)

        def fp_body_fun(carry):
            """Fixed-point iterations before DIIS starts."""
            _, E, _, _, loop, F_k, errvec_k = carry

            # Use last Fock matrix
            F = F_k[-1]

            # Diagonalization
            F1 = jnp.einsum('pq,qr,rs->ps', v_ovlp.T, F, v_ovlp)
            w1, c1 = jnp.linalg.eigh(F1)

            # Update MO coefficients and density matrix
            mo_coeff = jnp.dot(v_ovlp, c1)
            dm = density_matrix_fn(mo_coeff, w1)

            # Compute new Fock matrix
            J = hartree_fn(dm)
            Exc, Vxc = exchange_correlation_fn(dm)
            E_new = 0.5 * jnp.einsum('pq,qp', 2*Hcore + J, dm).real + Exc
            F = Hcore + J + Vxc

            # Update error vector
            errvec = errvec_sdf_fn(dm, F)

            # Update DIIS history
            F_k = jnp.concatenate((F_k[1:], jnp.array([F])), axis=0)
            errvec_k = jnp.concatenate((errvec_k[1:], jnp.array([errvec])), axis=0)

            return E, E_new, mo_coeff, w1, loop + 1, F_k, errvec_k

        def fp_cond_fun(carry):
            return carry[4] < diis_start_cycle + diis_space

        _, E, mo_coeff, w1, loop, F_k, errvec_k = jax.lax.while_loop(
            fp_cond_fun, fp_body_fun,
            (0.0, 1.0, mo_coeff_init, w_init, 0, F_k, errvec_k)
        )

        def diis_body_fun(carry):
            """DIIS-accelerated iterations."""
            _, E, _, _, loop, F_k, errvec_k = carry

            # Compute DIIS coefficients
            B = jnp.einsum('imn,jmn->ij', errvec_k, errvec_k)
            # Add regularization for numerical stability
            eta = 1e-12 * (jnp.trace(B) / diis_space + 1.0)
            B = B + eta * jnp.eye(diis_space)

            # Solve DIIS equations: B @ c = -1, sum(c) = 1
            temp1 = -jnp.ones((diis_space, 1))
            temp2 = jnp.array([jnp.append(-jnp.ones(diis_space), 0.0)])
            h = jnp.concatenate((jnp.concatenate((B, temp1), axis=1), temp2), axis=0)
            g = jnp.append(jnp.zeros(diis_space), -1.0)
            c_k = jnp.linalg.solve(h, g)[:diis_space]

            # Extrapolate Fock matrix
            _F = jnp.einsum('k,kab->ab', c_k, F_k)

            # Apply damping
            _F = (1 - diis_damp) * _F + diis_damp * F_k[-1]

            # Diagonalization
            F1 = jnp.einsum('pq,qr,rs->ps', v_ovlp.T, _F, v_ovlp)
            w1, c1 = jnp.linalg.eigh(F1)

            # Update MO coefficients and density matrix
            mo_coeff = jnp.dot(v_ovlp, c1)
            dm = density_matrix_fn(mo_coeff, w1)

            # Compute new Fock matrix and energy
            J = hartree_fn(dm)
            Exc, Vxc = exchange_correlation_fn(dm)
            E_new = 0.5 * jnp.einsum('pq,qp', 2*Hcore + J, dm).real + Exc
            F = Hcore + J + Vxc

            # Update error vector
            errvec = errvec_sdf_fn(dm, F)

            # Update DIIS history
            F_k = jnp.concatenate((F_k[1:], jnp.array([F])), axis=0)
            errvec_k = jnp.concatenate((errvec_k[1:], jnp.array([errvec])), axis=0)

            return E, E_new, mo_coeff, w1, loop + 1, F_k, errvec_k

        def diis_cond_fun(carry):
            E_prev, E_new, _, _, loop, _, errvec_k = carry
            err_norm = jnp.linalg.norm(errvec_k[-1])
            err_tol = tol * 10.0  # Slightly relaxed error tolerance
            return ((jnp.abs(E_new - E_prev) > tol) | (err_norm > err_tol)) & (loop < max_cycle)

        _, E, mo_coeff, w1, loop, F_k, errvec_k = jax.lax.while_loop(
            diis_cond_fun, diis_body_fun,
            (E - 1.0, E, mo_coeff, w1, loop, F_k, errvec_k)
        )
        converged = jnp.logical_not(loop == max_cycle)

        return mo_coeff, w1, E, converged

    if diis:
        return diis_scf
    else:
        return fixed_point_scf
