import jax
import numpy as np
import jax.numpy as jnp

def make_scf(diis=True, diis_space=8, diis_start_cycle=1, diis_damp=0, tol=1e-7, max_cycle=100):
    """
        Make SCF function for 'hqc.pbc.lcao'.
        Input:
            diis: bool, whether to use DIIS
            diis_space: int, number of vectors in DIIS space
            diis_start_cycle: int, start DIIS after this cycle
            diis_damp: float, DIIS damping factor
            tol: float, tolerance
            max_iter: int, maximum number of iterations
        Output:
            scf: function to perform SCF
                 if diis is True, scf returns DIIS SCF function
                 if diis is False, scf returns fixed point SCF function
                 These two functions have the same Interface.
    """
    def fixed_point_scf(v_ovlp, Hcore, dm_init, hartree_fn, exchange_correlation_fn, 
                        density_matrix_fn, errvec_sdf_fn):
        """
            Fixed point iteration for SCF.
            Input:
                v_ovlp: array of shape (n_ao, n_ao), orthonormal matrix of overlap matrix S
                    V^{dagger}SV = I
                Hcore: array of shape (n_ao, n_ao), core Hamiltonian matrix
                    Hcore = T + V
                dm_init: array of shape (n_ao, n_ao), initial guess of density matrix
                hartree_fn: function to compute coulomb repulsion between electrons (Hartree term)
                    J = hartree_fn(dm), where dm and J has shape (n_ao, n_ao)
                exchange_correlation_fn: function to compute exchange-correlation energy
                    Exc, Vxc = exchange_correlation_fn(dm), where Exc is a real float,
                    dm and Vxc has shape (n_ao, n_ao)
                    for Hartree-Fock, Vxc = -0.5*K, for DFT, Vxc = Vxc
                    F = Hcore + J + Vxc
                density_matrix_fn: function to compute density matrix
                    dm = density_matrix_fn(mo_coeff, w1), where mo_coeff has shape (n_ao, n_mo)
                    and w1 has shape (n_mo,), dm has shape (n_ao, n_ao)
                errvec_sdf_fn: function to compute error vector for DIIS
                    errvec = errvec_sdf_fn(dm, F), where dm and F has shape (n_ao, n_ao)
                    errvec has shape (n_ao, n_ao), errvec = SDF-FDS
                    Note: this function is not used in fixed point iteration.
            Output:
                mo_coeff: array of shape (n_ao, n_mo), molecular orbitals
                w1: array of shape (n_mo,), orbital energies
                E: float, total energy
                converged: bool, whether the SCF converges
        """
        mo_coeff_init = jnp.empty_like(dm_init)
        w_init = jnp.empty_like(dm_init[0], dtype=jnp.float64)

        def body_fun(carry):
            _, E, _, dm, _, loop = carry

            # hartree and exchange-correlation term
            J = hartree_fn(dm)
            Exc, Vxc = exchange_correlation_fn(dm)

            # energy
            E_new = 0.5*jnp.einsum('pq,qp', 2*Hcore+J, dm).real+Exc

            # Fock matrix
            F = Hcore + J + Vxc

            # diagonalization
            F1 = jnp.einsum('pq,qr,rs->ps', v_ovlp.T.conjugate(), F, v_ovlp)
            w1, c1 = jnp.linalg.eigh(F1)

            # molecular orbitals and density matrix
            mo_coeff = jnp.dot(v_ovlp, c1) # (n_ao, n_mo)
            dm = density_matrix_fn(mo_coeff, w1) # (n_ao, n_ao)

            # ======================= debug =======================
            # jax.debug.print("fp loop: {x}", x=loop)
            # jax.debug.print("E:{x}, E_new:{y}", x=E, y=E_new)
            # =====================================================

            return E, E_new, mo_coeff, dm, w1, loop+1
        
        def cond_fun(carry):
            return (abs(carry[1] - carry[0]) > tol) * (carry[5] < max_cycle)
            
        _, E, mo_coeff, dm, w1, loop = jax.lax.while_loop(cond_fun, body_fun, (0., 1., mo_coeff_init, dm_init, w_init, 0))
        converged = jnp.logical_not(loop==max_cycle)

        return mo_coeff, w1, E, converged

    def diis_scf(v_ovlp, Hcore, dm_init, hartree_fn, exchange_correlation_fn, 
                 density_matrix_fn, errvec_sdf_fn):
        """
            DIIS for SCF.
            Input:
                v_ovlp: array of shape (n_ao, n_ao), orthonormal matrix of overlap matrix S
                    V^{dagger}SV = I
                Hcore: array of shape (n_ao, n_ao), core Hamiltonian matrix
                    Hcore = T + V
                dm_init: array of shape (n_ao, n_ao), initial guess of density matrix
                hartree_fn: function to compute coulomb repulsion between electrons (Hartree term)
                    J = hartree_fn(dm), where dm and J has shape (n_ao, n_ao)
                exchange_correlation_fn: function to compute exchange-correlation energy
                    Exc, Vxc = exchange_correlation_fn(dm), where Exc is a real float,
                    dm and Vxc has shape (n_ao, n_ao)
                    for Hartree-Fock, Vxc = -0.5*K, for DFT, Vxc = Vxc
                    F = Hcore + J + Vxc
                density_matrix_fn: function to compute density matrix
                    dm = density_matrix_fn(mo_coeff, w1), where mo_coeff has shape (n_ao, n_mo)
                    and w1 has shape (n_mo,), dm has shape (n_ao, n_ao)
                errvec_sdf_fn: function to compute error vector for DIIS
                    errvec = errvec_sdf_fn(dm, F), where dm and F has shape (n_ao, n_ao)
                    errvec has shape (n_ao, n_ao), errvec = SDF-FDS
            Output:
                mo_coeff: array of shape (n_ao, n_mo), molecular orbitals
                w1: array of shape (n_mo,), orbital energies
                E: float, total energy
                converged: bool, whether the SCF converges
        """
        mo_coeff_init = jnp.empty_like(dm_init)
        w_init = jnp.empty_like(dm_init[0], dtype=jnp.float64)

        # initial F and error vector series for DIIS
        J = hartree_fn(dm_init)
        Vxc = exchange_correlation_fn(dm_init)[1]
        F_init = Hcore + J + Vxc
        errvec_init = errvec_sdf_fn(dm_init, F_init)
        F_k = jnp.repeat(F_init[None, ...], diis_space, axis=0)
        errvec_k = jnp.repeat(errvec_init[None, ...], diis_space, axis=0)

        def fp_body_fun(carry):
            _, E, _, _, loop, F_k, errvec_k = carry

            # last Fock matrix
            F = F_k[-1]

            # diagonalization
            F1 = jnp.einsum('pq,qr,rs->ps', v_ovlp.T.conjugate(), F, v_ovlp)
            w1, c1 = jnp.linalg.eigh(F1)

            # next molecular orbitals and density matrix
            mo_coeff = jnp.dot(v_ovlp, c1) # (n_ao, n_mo)
            dm = density_matrix_fn(mo_coeff, w1) # (n_ao, n_ao)

            # hartree and exchange-correlation term
            J = hartree_fn(dm)
            Exc, Vxc = exchange_correlation_fn(dm)

            # next energy
            E_new = 0.5*jnp.einsum('pq,qp', 2*Hcore+J, dm).real+Exc

            # next Fock matrix
            F = Hcore + J + Vxc

            # next error vector
            errvec = errvec_sdf_fn(dm, F)
            
            # update F and error vector series for DIIS
            F_k = jnp.concatenate((F_k[1:], jnp.array([F])), axis=0)
            errvec_k = jnp.concatenate((errvec_k[1:], jnp.array([errvec])), axis=0)

            # ======================= debug =======================
            # jax.debug.print("fp loop: {x}", x=loop)
            # jax.debug.print("E:{x}, E_new:{y}", x=E, y=E_new)
            # =====================================================

            return E, E_new, mo_coeff, w1, loop+1, F_k, errvec_k
        
        def fp_cond_fun(carry):
            return carry[4] < diis_start_cycle + diis_space
            
        _, E, mo_coeff, w1, loop, F_k, errvec_k = jax.lax.while_loop(fp_cond_fun, fp_body_fun, 
                                                (0., 1., mo_coeff_init, w_init, 0, F_k, errvec_k))
                                        
        def diis_body_fun(carry):
            _, E, _, _, loop, F_k, errvec_k = carry

            # get DIIS c_k
            B = jnp.einsum('imn,jmn->ij', errvec_k, errvec_k)
            temp1 = -jnp.ones((diis_space, 1))
            temp2 = jnp.array([jnp.append(-jnp.ones(diis_space), 0)])
            h = jnp.concatenate((jnp.concatenate((B, temp1), axis=1), temp2), axis=0)
            g = jnp.append(jnp.zeros(diis_space), -1)
            c_k = jnp.linalg.solve(h, g)[:diis_space]

            # guess Fock matrix
            _F = jnp.einsum('k,kab->ab', c_k, F_k)

            # damp
            _F = (1 - diis_damp) * _F + diis_damp * F_k[-1]

            # diagonalization
            F1 = jnp.einsum('pq,qr,rs->ps', v_ovlp.T.conjugate(), _F, v_ovlp)
            w1, c1 = jnp.linalg.eigh(F1)

            # molecular orbitals and density matrix
            mo_coeff = jnp.dot(v_ovlp, c1) # (n_ao, n_mo)
            dm = density_matrix_fn(mo_coeff, w1) # (n_ao, n_ao)

            # hartree and exchange-correlation term
            J = hartree_fn(dm)
            Exc, Vxc = exchange_correlation_fn(dm)

            # next energy
            E_new = 0.5*jnp.einsum('pq,qp', 2*Hcore+J, dm).real+Exc

            # next Fock matrix
            F = Hcore + J + Vxc

            # next error vector
            errvec = errvec_sdf_fn(dm, F)

            # update F and error vector series for DIIS
            F_k = jnp.concatenate((F_k[1:], jnp.array([F])), axis=0)
            errvec_k = jnp.concatenate((errvec_k[1:], jnp.array([errvec])), axis=0)

            # ======================= debug =======================
            # jax.debug.print("diis loop: {x}", x=loop)
            # jax.debug.print("E:{x}, E_new:{y}", x=E, y=E_new)
            # =====================================================

            return (E, E_new, mo_coeff, w1, loop+1, F_k, errvec_k)

        def diis_cond_fun(carry):
            return (jnp.abs(carry[1] - carry[0]) > tol) * (carry[4] < max_cycle)

        _, E, mo_coeff, w1, loop, F_k, errvec_k = jax.lax.while_loop(diis_cond_fun, diis_body_fun, 
                                                (E-1., E, mo_coeff, w1, loop, F_k, errvec_k))
        converged = jnp.logical_not(loop==max_cycle)

        return mo_coeff, w1, E, converged

    if diis:
        return diis_scf
    else:
        return fixed_point_scf