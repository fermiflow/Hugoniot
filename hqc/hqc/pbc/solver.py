import jax
import numpy as np
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple
from jax import vmap, grad, jit, jacfwd

from hqc.pbc.scf import make_scf
from hqc.pbc.gto import make_pbc_gto
from hqc.tools.smearing import make_occupation_func
from hqc.tools.excor import make_exchange_func, make_correlation_func
from hqc.basis.parse import load_as_str, parse_quant_num, parse_gto, normalize_gto_coeff

def make_solver(n: int, L: float, rs: float, basis: str,
                rcut: float = 24, 
                tol: float = 1e-7, 
                max_cycle: int = 100, 
                grid_length: float = 0.12,
                diis: bool = True, 
                diis_space: int = 8, 
                diis_start_cycle: int = 1, 
                diis_damp: float = 0.,
                use_jit: bool = True,
                dft: bool = False, 
                xc: str = 'lda,vwn',
                smearing: bool = False,
                smearing_method: str = 'fermi',
                smearing_sigma: float = 0.,
                search_method: str = 'newton', 
                search_cycle: int = 100, 
                search_tol: float= 1e-7,
                gamma: bool = True) -> Callable:
    """
        Make PBC solver function.
        INPUT:
            n: int, number of hydrogen atoms in unit cell.
            L: float, side length of unit cell, unit: rs.
            rs: float, unit: Bohr
            basis: gto basis name, eg:'gth-szv'.
            tol: the tolerance for convergence.
            max_cycle: the maximum number of iterations.
            grid_length: the grid length for real space grid, unit: Bohr.
            diis: if True, use DIIS.
            diis_space: the number of vectors in DIIS space.
            diis_start_cycle: the cycle to start DIIS.
            diis_damp: the damping factor for DIIS.
            use_jit: if True, use jit.
            dft: if True, use DFT, if False, use HF.
            xc: exchange-correlation functional.
            smearing: if True, use smearing.
            smearing_method: 'fermi' or 'gauss'.
            smearing_sigma: smearing width, unit: Hartree.
            search_method: 'bisect' or 'newton'.
            search_cycle: the maximum number of iterations for search.
            search_tol: the tolerance for searching mu.
            gamma: bool, if True, return solver(xp) for gamma point only, 
                         else, return solver(xp, kpt) for a single k-point.
        OUTPUT:
            solver: solver function.
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
    
    Ry = 2
    L *= rs
    cell = jnp.eye(3)
    Omega = jnp.linalg.det(cell)*L**3

    # pyscf grid=0.12
    n_grid = round(L/grid_length/2)*2+1 # odd number
    n_grid3 = n_grid**3

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
    n_mo = n_ao # number of molecular orbitals
    # intermediate variables
    sum_alpha = all_alpha[:, None] + all_alpha[None, :]  # (n_all_alpha, n_all_alpha)
    prod_alpha = all_alpha[:, None] * all_alpha[None, :]  # (n_all_alpha, n_all_alpha)
    alpha2 = prod_alpha / sum_alpha  # (n_all_alpha, n_all_alpha)

    def make_mesh():
        # 1D Rmesh
        grid_eris = jnp.arange(n_grid)-n_grid//2 # (n_grid, ), nx = ny = nz = n_grid
        Rmesh_1d = (grid_eris+n_grid//2)*L/n_grid # (nx,)
        Rmesh = (jnp.array(jnp.meshgrid(*( [Rmesh_1d]*3 ))).transpose(1,2,3,0)).reshape(-1, 3) # (nx, ny, nz, 3)

        # Gmesh
        mesh_eris = jnp.array(jnp.meshgrid(*( [grid_eris]*3 ))).transpose(1,2,3,0) # (nx, ny, nz, 3)
        Gmesh = mesh_eris.dot(jnp.linalg.inv(cell))*2*jnp.pi/L # (nx, ny, nz, 3), range (-n_grid*pi/L, n_grid*pi/L)

        # Coulomb potential on reciprocal space
        Gnorm2_eris = jnp.sum(jnp.square(Gmesh), axis=3)
        VG = 4 * jnp.pi/Omega/Gnorm2_eris # (nx, ny, nz)
        VG = VG.at[n_grid//2,n_grid//2,n_grid//2].set(0)

        # omega = 1 range separation
        # VG_lr = VG * jnp.exp(-Gnorm2/4/omega)

        return Rmesh_1d, Rmesh, Gmesh, VG
    
    Rmesh_1d, Rmesh, Gmesh, VG = make_mesh()

    if gamma:
        # eval pbc gaussian power x on Rmesh_1d
        eval_pbc_gaussian_power_x, power2cart, alpha_coeff_gto_cart2sph = make_pbc_gto(basis, L, gamma=gamma, lcao_xyz=True)
        eval_pbc_gaussian_power_x = vmap(eval_pbc_gaussian_power_x)
        eval_pbc_gaussian_power_x = vmap(eval_pbc_gaussian_power_x)
        eval_pbc_gaussian_power_x_Rmesh1D = lambda xp: eval_pbc_gaussian_power_x(Rmesh_1d[None, None, :]-xp[:, :, None])
        
        # evaluate atomic orbitals on Rmesh
        eval_pbc_ao = make_pbc_gto(basis, L, gamma=gamma, lcao_xyz=False)
        eval_pbc_ao = vmap(eval_pbc_ao, (None, 0), 1)
        eval_pbc_ao_Rmesh = lambda xp: eval_pbc_ao(xp, Rmesh)
        
    else:
        # eval pbc gaussian power x on Rmesh_1d at kpt
        eval_pbc_gaussian_power_x_kpt, power2cart, alpha_coeff_gto_cart2sph = make_pbc_gto(basis, L, gamma=gamma, lcao_xyz=True)
        eval_pbc_gaussian_power_x_kpt = vmap(eval_pbc_gaussian_power_x_kpt, (0, 0), 0)
        eval_pbc_gaussian_power_x_kpt = vmap(eval_pbc_gaussian_power_x_kpt, (0, None), 0)
        eval_pbc_gaussian_power_x_kpt_Rmesh1D = lambda xp, kpt: eval_pbc_gaussian_power_x_kpt(Rmesh_1d[None, None, :]-xp[:, :, None], kpt)
        
        # evaluate atomic orbitals on Rmesh
        eval_pbc_ao_kpt = make_pbc_gto(basis, L, gamma=gamma, lcao_xyz=False)
        eval_pbc_ao_kpt = vmap(eval_pbc_ao_kpt, (None, 0, None), 1)
        eval_pbc_ao_kpt_Rmesh = lambda xp, kpt: eval_pbc_ao_kpt(xp, Rmesh, kpt)

    def make_overlap_kinetic():
        """
            make overlap and kinetic function.
            if l_max == 0, return eval_overlap_kinetic_s.
            if l_max == 1, return eval_overlap_kinetic_sp.
            d and higher orbital is not support yet.
        """
        def _eval_intermediate_integral(xp1, xp2):
            """
                Evaluate intermediate overlap and kinetic integrals.
                The original function is used for s orbital integral, use jax.grad for p orbitals.
                Args:
                    xp1: array of shape (3,), position of protons.
                    xp2: array of shape (3,), position of protons.
                Returns:
                    overlap: array of shape (n_all_alpha, n_all_alpha), overlap integrals.
                    kinetic: array of shape (n_all_alpha, n_all_alpha), kinetic energy integrals.
            """
            Rmnc = jnp.sum(jnp.square(xp1[None, :] - xp2[None, :] + lattice), axis=1)
            _ovlp = jnp.pi**1.5*jnp.einsum('ij,ijl->ijl', 1/jnp.power(sum_alpha, 1.5), jnp.exp(-jnp.einsum('ij,l->ijl', alpha2, Rmnc)))
            overlap = jnp.sum(_ovlp, axis=2)
            kinetic = jnp.einsum('ijl,ij,ijl->ij', _ovlp, alpha2, 3-2*jnp.einsum('ij,l->ijl', alpha2, Rmnc))
            return overlap, kinetic
        
        def _eval_intermediate_integral_kpt(xp1, xp2, kpt):
            """
                Evaluate intermediate overlap and kinetic integrals at k point.
                The original function is used for s orbital integral, use jax.grad for p orbitals.
                Args:
                    xp1: array of shape (3,), position of protons.
                    xp2: array of shape (3,), position of protons.
                    kpt: array of shape (3,), k point. 1BZ: (-pi/L/rs, pi/L/rs)
                Returns:
                    overlap: array of shape (n_all_alpha, n_all_alpha), overlap integrals.
                    kinetic: array of shape (n_all_alpha, n_all_alpha), kinetic energy integrals.
            """
            Rmnc = jnp.sum(jnp.square(xp1[None, :] - xp2[None, :] + lattice), axis=1)
            _ovlp = jnp.pi**1.5*jnp.einsum('ij,ijl,l->ijl', 1/jnp.power(sum_alpha, 1.5), 
                    jnp.exp(-jnp.einsum('ij,l->ijl', alpha2, Rmnc)), jnp.exp(-1j*lattice@kpt))
            overlap = jnp.sum(_ovlp, axis=2)
            kinetic = jnp.einsum('ijl,ij,ijl->ij', _ovlp, alpha2, 3-2*jnp.einsum('ij,l->ijl', alpha2, Rmnc))
            return overlap, kinetic
    
        def eval_overlap_kinetic_s(xp1, xp2):
            """
                Evaluate overlap and kinetic matrix for s orbitals.
                Args:
                    xp1: array of shape (3,), position of protons.
                    xp2: array of shape (3,), position of protons.
                Returns:
                    ovlp: array of shape (n_ao_one_atom, n_ao_one_atom), overlap matrix.
                    T: array of shape (n_ao_one_atom, n_ao_one_atom), kinetic matrix.
            """
            overlap_s, kinetic_s = _eval_intermediate_integral(xp1, xp2)
            ovlp_s = jnp.einsum('ip,jq,ij->pq', coeffs, coeffs, overlap_s)
            T_s = jnp.einsum('ip,jq,ij->pq', coeffs, coeffs, kinetic_s)
            return ovlp_s, T_s
        
        def eval_overlap_kinetic_s_kpt(xp1, xp2, kpt):
            """
                Evaluate overlap and kinetic matrix for s orbitals.
                Args:
                    xp1: array of shape (3,), position of protons.
                    xp2: array of shape (3,), position of protons.
                    kpt: array of shape (3,), k point. 1BZ: (-pi/L/rs, pi/L/rs)
                Returns:
                    ovlp: array of shape (n_ao_one_atom, n_ao_one_atom), overlap matrix.
                    T: array of shape (n_ao_one_atom, n_ao_one_atom), kinetic matrix.
            """
            overlap_s, kinetic_s = _eval_intermediate_integral_kpt(xp1, xp2, kpt)
            ovlp_s = jnp.einsum('ip,jq,ij->pq', coeffs, coeffs, overlap_s)
            T_s = jnp.einsum('ip,jq,ij->pq', coeffs, coeffs, kinetic_s)
            return ovlp_s, T_s

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
        
        def eval_overlap_kinetic_sp(xp1, xp2):
            """
                Evaluate overlap and kinetic matrix for s and p orbitals.
                Args:
                    xp1: array of shape (3,), position of protons.
                    xp2: array of shape (3,), position of protons.
                Returns:
                    ovlp: array of shape (n_ao, n_ao), overlap matrix.
                    T: array of shape (n_ao, n_ao), kinetic matrix.
            """
            overlap_s, kinetic_s = _eval_intermediate_integral(xp1, xp2)
            overlap_sp, kinetic_sp = jacfwd(_eval_intermediate_integral, argnums=1)(xp1, xp2)
            overlap_ps, kinetic_ps = jacfwd(_eval_intermediate_integral, argnums=0)(xp1, xp2)
            overlap_p, kinetic_p = jacfwd(jacfwd(_eval_intermediate_integral), argnums=1)(xp1, xp2)

            ovlp = _matrix_s_p(overlap_s, overlap_sp, overlap_ps, overlap_p)
            T = _matrix_s_p(kinetic_s, kinetic_sp, kinetic_ps, kinetic_p)
            return ovlp, T
        
        def eval_overlap_kinetic_sp_kpt(xp1, xp2, kpt):
            """
                Evaluate overlap and kinetic matrix for s and p orbitals.
                Args:
                    xp1: array of shape (3,), position of protons.
                    xp2: array of shape (3,), position of protons.
                    kpt: array of shape (3,), k point. 1BZ: (-pi/L/rs, pi/L/rs)
                Returns:
                    ovlp: array of shape (n_ao, n_ao), overlap matrix.
                    T: array of shape (n_ao, n_ao), kinetic matrix.
            """
            overlap_s, kinetic_s = _eval_intermediate_integral_kpt(xp1, xp2, kpt)
            overlap_sp, kinetic_sp = jacfwd(_eval_intermediate_integral_kpt, argnums=1)(xp1, xp2, kpt)
            overlap_ps, kinetic_ps = jacfwd(_eval_intermediate_integral_kpt, argnums=0)(xp1, xp2, kpt)
            overlap_p, kinetic_p = jacfwd(jacfwd(_eval_intermediate_integral_kpt), argnums=1)(xp1, xp2, kpt)

            ovlp = _matrix_s_p(overlap_s, overlap_sp, overlap_ps, overlap_p)
            T = _matrix_s_p(kinetic_s, kinetic_sp, kinetic_ps, kinetic_p)
            return ovlp, T
        
        if gamma:
            if l_max == 0:
                return eval_overlap_kinetic_s
            elif l_max == 1:
                return eval_overlap_kinetic_sp
            else:
                raise ValueError("l_max > 1 is not supported yet.")
        else:
            if l_max == 0:
                return eval_overlap_kinetic_s_kpt
            elif l_max == 1:
                return eval_overlap_kinetic_sp_kpt
            else:
                raise ValueError("l_max > 1 is not supported yet.")

    if gamma:
        eval_overlap_kinetic_noreshape = vmap(vmap(make_overlap_kinetic(), (0, None), (0, 0)), (None, 0), (2, 2))
    else:
        eval_overlap_kinetic_kpt_noreshape = vmap(vmap(make_overlap_kinetic(), (0, None, None), (0, 0)), (None, 0, None), (2, 2))

    def eval_overlap_kinetic(xp1, xp2):
        ovlp, T = eval_overlap_kinetic_noreshape(xp1, xp2)
        return ovlp.reshape(n_ao, n_ao), T.reshape(n_ao, n_ao)
    
    def eval_overlap_kinetic_kpt(xp1, xp2, kpt):
        ovlp, T = eval_overlap_kinetic_kpt_noreshape(xp1, xp2, kpt)
        return ovlp.reshape(n_ao, n_ao), T.reshape(n_ao, n_ao)
    
    def eval_vep_eris(xp, pbc_gaussian_power_xyz):
        """
            Use jax.lax.scan to calculate vep matrix and electron repulsion integrals (eris).
            INPUT:
                xp: array of shape (n, 3), position of protons.
                pbc_gaussian_power_xyz: array of shape (n, 3, n_grid_eris, n_all_alpha, n_l)'
                one dimensional gaussian power for each proton.
            OUTPUT:
                vep: array of shape (n_ao, n_ao), Vep matrix.
                eris0: array of shape (n_ao, n_ao, n_ao, n_ao), two-electron repulsion integrals at gamma point.
                eris: array of shape (n_ao, n_ao, n_ao, n_ao), two-electron repulsion integrals.

        """
        unknown1 = 0.22578495

        SI = jnp.sum(jnp.exp(-1j*Gmesh.dot(xp.T)), axis=3) # (nx, ny, nz)
        vlocG = -SI * VG # (nx, ny, nz)
        vlocG_real_image = jnp.real(vlocG) + jnp.imag(vlocG)

        def body_fun(carry, pbc_gaussian_power_xyz_one):
            pbc_gaussian_power2R_xyz = jnp.einsum('ndgal,dgbk->dgnablk', pbc_gaussian_power_xyz, pbc_gaussian_power_xyz_one)
            pbc_gaussian_power2G_xyz = jnp.fft.fft(pbc_gaussian_power2R_xyz, axis=1)*jnp.linalg.det(cell)**(1/3)*(L/n_grid)
            pbc_gaussian_power2G_xyz = jnp.concatenate([pbc_gaussian_power2G_xyz[:,-n_grid//2+1:], pbc_gaussian_power2G_xyz[:,:-n_grid//2+1]], axis=1)
            pbc_gaussian_cart2G_xyz = jnp.einsum('dgnablk,dlc,dke->dgnabce', pbc_gaussian_power2G_xyz, power2cart, power2cart)
        
            pbc_gaussian_cart2G = jnp.einsum('xnabce,ynabce,znabce->yxznabce', pbc_gaussian_cart2G_xyz[0], pbc_gaussian_cart2G_xyz[1],
                                            pbc_gaussian_cart2G_xyz[2])
            pbc_gaussian_cart2G_real_image = jnp.real(pbc_gaussian_cart2G) + jnp.imag(pbc_gaussian_cart2G)
            pbc_gto_cart2G = jnp.einsum('yxznabce,aco,bep->yxznop', pbc_gaussian_cart2G_real_image, alpha_coeff_gto_cart2sph, alpha_coeff_gto_cart2sph)
            return carry, pbc_gto_cart2G
        
        _, rhoG = jax.lax.scan(body_fun, None, pbc_gaussian_power_xyz)
        rhoG = (rhoG.transpose(1,2,3,4,5,0,6)).reshape(n_grid, n_grid, n_grid, n_ao, n_ao) # (n_grid3_eris, n_ao, n_ao)
        rhoG0 = rhoG[n_grid//2,n_grid//2,n_grid//2]
        
        # vep matrix
        vep = jnp.einsum('xyz,xyzpq->pq', vlocG_real_image, rhoG)

        # # # eris
        # # einsum (vmap)
        # # max batchsize 56 (no jit)
        # # hf time: 7.2888023853302 (no jit)
        # # max batchsize 40 (jit)
        # # hf time: 2.2167017459869385 (jit)
        eris = jnp.einsum('xyz,xyzrs,xyzqp->prsq', VG, rhoG, rhoG)

        # # for x
        # # max batchsize 68 (no jit)
        # # hf time: 7.186677932739258 (no jit)
        # # max batchsize 78 (jit)
        # # hf time: 4.62881875038147 (jit)
        # eris = jnp.zeros((n_ao, n_ao, n_ao, n_ao))
        # for ix in range(n_grid):
        #     eris += jnp.einsum('yz,yzrs,yzqp->prsq', VG[ix], rhoG[ix], rhoG[ix])

        # # for xy
        # # max batchsize 68 (no jit)
        # # hf time: 8.341909408569336 (no jit)
        # # max batchsize 118 (jit)
        # # hf time: 13.894800424575806 (jit)
        # eris = jnp.zeros((n_ao, n_ao, n_ao, n_ao))
        # for ix in range(n_grid):
        #     for iy in range(n_grid):
        #         eris += jnp.einsum('z,zrs,zqp->prsq', VG[ix, iy], rhoG[ix, iy], rhoG[ix, iy])
        
        # # # jax.lax.scan
        # # max batchsize 108 (jit)
        # # hf time: 11.287713766098022 (jit)
        # VG1 = VG.reshape(n_grid3)
        # rhoG = rhoG.reshape(n_grid3, n_ao, n_ao)
        # eris = jnp.zeros((n_ao, n_ao, n_ao, n_ao))
        # def body_fun(carry, x):
        #     carry += jnp.einsum('rs,qp->prsq', x[1], x[1])*x[0]
        #     return carry, 0
        # eris, _ = jax.lax.scan(body_fun, eris, xs=[VG1, rhoG], length=n_grid3)

        # jax.lax.scan
        # max batchsize 108 (jit)
        # hf time: 11.287713766098022 (jit)
        # VG1 = VG.reshape(n_grid3)
        # rhoG = rhoG.reshape(n_grid3, n_ao, n_ao)
        # eris = jnp.zeros((n_ao, n_ao, n_ao, n_ao))
        # def body_fun(carry, x):
        #     carry += jnp.einsum('rs,qp->prsq', x[1], x[1])*x[0]
        #     return carry, 0
        # eris, _ = jax.lax.scan(body_fun, eris, xs=[VG1, rhoG], length=n_grid3, unroll=n_grid**2)

        # del rhoG
        eris0 = jnp.einsum('rs,qp->prsq', rhoG0, rhoG0)*4*jnp.pi/L/jnp.linalg.det(cell)**(1/3)*unknown1

        return vep, eris, eris0

    def eval_vep_eris_new(xp, pbc_gaussian_power_xyz):
        """
            Use jax.lax.scan to calculate vep matrix and electron repulsion integrals (eris).
            INPUT:
                xp: array of shape (n, 3), position of protons.
                pbc_gaussian_power_xyz: array of shape (n, 3, n_grid_eris, n_all_alpha, n_l)'
                one dimensional gaussian power for each proton.
            OUTPUT:
                vep: array of shape (n_ao, n_ao), Vep matrix.
                eris0: array of shape (n_ao, n_ao, n_ao, n_ao), two-electron repulsion integrals at gamma point.
                eris: array of shape (n_ao, n_ao, n_ao, n_ao), two-electron repulsion integrals.

        """
        unknown1 = 0.22578495

        SI = jnp.sum(jnp.exp(-1j*Gmesh.dot(xp.T)), axis=3) # (nx, ny, nz)
        vlocG = -SI * VG # (nx, ny, nz)
        vlocG_real_image = jnp.real(vlocG) + jnp.imag(vlocG)

        def body_fun(carry, pbc_gaussian_power_xyz_one):
            pbc_gaussian_power2R_xyz = jnp.einsum('ndgal,dgbk->dgnablk', pbc_gaussian_power_xyz, pbc_gaussian_power_xyz_one)
            pbc_gaussian_power2G_xyz = jnp.fft.fft(pbc_gaussian_power2R_xyz, axis=1)*jnp.linalg.det(cell)**(1/3)*(L/n_grid)
            pbc_gaussian_power2G_xyz = jnp.concatenate([pbc_gaussian_power2G_xyz[:,-n_grid//2+1:], pbc_gaussian_power2G_xyz[:,:-n_grid//2+1]], axis=1)
            pbc_gaussian_cart2G_xyz = jnp.einsum('dgnablk,dlc,dke->dgnabce', pbc_gaussian_power2G_xyz, power2cart, power2cart)
        
            pbc_gaussian_cart2G = jnp.einsum('xnabce,ynabce,znabce->yxznabce', pbc_gaussian_cart2G_xyz[0], pbc_gaussian_cart2G_xyz[1],
                                            pbc_gaussian_cart2G_xyz[2])
            pbc_gaussian_cart2G_real_image = jnp.real(pbc_gaussian_cart2G) + jnp.imag(pbc_gaussian_cart2G)
            pbc_gto_cart2G = jnp.einsum('yxznabce,aco,bep->yxznop', pbc_gaussian_cart2G_real_image, alpha_coeff_gto_cart2sph, alpha_coeff_gto_cart2sph)
            return carry, pbc_gto_cart2G
        
        _, rhoG = jax.lax.scan(body_fun, None, pbc_gaussian_power_xyz)
        rhoG = (rhoG.transpose(1,2,3,4,5,0,6)).reshape(n_grid, n_grid, n_grid, n_ao, n_ao) # (n_grid, n_grid, n_grid, n_ao, n_ao)
    
        # vep matrix
        vep = jnp.einsum('xyz,xyzpq->pq', vlocG_real_image, rhoG) # (n_ao, n_ao)

        # eris
        # eris0 = jnp.einsum('rs,qp->prsq', rhoG[n_grid//2,n_grid//2,n_grid//2], 
        #                    rhoG[n_grid//2,n_grid//2,n_grid//2])*4*jnp.pi/L/jnp.linalg.det(cell)**(1/3)*unknown1
        # eris = jnp.einsum('xyz,xyzrs,xyzqp->prsq', VG, rhoG, rhoG)


        # return vep, eris, eris0
        return vep, rhoG

    def eval_vep_eris_kpt(xp, pbc_gaussian_power_xyz):
        """
            k-point version.
            Use jax.lax.scan to calculate vep matrix and electron repulsion integrals (eris).
            INPUT:
                xp: array of shape (n, 3), position of protons.
                pbc_gaussian_power_xyz: array of shape (n, 3, n_grid_eris, n_all_alpha, n_l)'
                one dimensional gaussian power for each proton.
            OUTPUT:
                vep: array of shape (n_ao, n_ao), Vep matrix.
                eris0: array of shape (n_ao, n_ao, n_ao, n_ao), two-electron repulsion integrals at gamma point.
                eris: array of shape (n_ao, n_ao, n_ao, n_ao), two-electron repulsion integrals.

        """
        unknown1 = 0.22578495

        SI = jnp.sum(jnp.exp(-1j*Gmesh.dot(xp.T)), axis=3) # (nx, ny, nz)
        vlocG = -SI * VG # (nx, ny, nz)

        def body_fun(carry, pbc_gaussian_power_xyz_one):
            pbc_gaussian_power2R_xyz = jnp.einsum('ndgal,dgbk->dgnablk', pbc_gaussian_power_xyz, pbc_gaussian_power_xyz_one.conjugate())
            pbc_gaussian_power2G_xyz = jnp.fft.fft(pbc_gaussian_power2R_xyz, axis=1)*jnp.linalg.det(cell)**(1/3)*(L/n_grid)
            pbc_gaussian_power2G_xyz = jnp.concatenate([pbc_gaussian_power2G_xyz[:,-n_grid//2+1:], pbc_gaussian_power2G_xyz[:,:-n_grid//2+1]], axis=1)
            pbc_gaussian_cart2G_xyz = jnp.einsum('dgnablk,dlc,dke->dgnabce', pbc_gaussian_power2G_xyz, power2cart, power2cart)
        
            pbc_gaussian_cart2G = jnp.einsum('xnabce,ynabce,znabce->yxznabce', pbc_gaussian_cart2G_xyz[0], pbc_gaussian_cart2G_xyz[1],
                                            pbc_gaussian_cart2G_xyz[2])
            pbc_gto_cart2G = jnp.einsum('yxznabce,aco,bep->yxznop', pbc_gaussian_cart2G, alpha_coeff_gto_cart2sph, alpha_coeff_gto_cart2sph)
            return carry, pbc_gto_cart2G
        
        _, rhoG = jax.lax.scan(body_fun, None, pbc_gaussian_power_xyz)
        rhoG = (rhoG.transpose(1,2,3,4,5,0,6)).reshape(n_grid, n_grid, n_grid, n_ao, n_ao) # (n_grid3_eris, n_ao, n_ao)
        rhoG0 = rhoG[n_grid//2,n_grid//2,n_grid//2]
        
        # vep matrix
        vep = jnp.einsum('xyz,xyzpq->pq', vlocG, rhoG.conjugate())

        # # # eris
        # # einsum (vmap)
        # # max batchsize 56 (no jit)
        # # hf time: 7.2888023853302 (no jit)
        # # max batchsize 40 (jit)
        # # hf time: 2.2167017459869385 (jit)
        eris = jnp.einsum('xyz,xyzrs,xyzpq->prsq', VG, rhoG, rhoG.conjugate())

        # # for x
        # # max batchsize 68 (no jit)
        # # hf time: 7.186677932739258 (no jit)
        # # max batchsize 78 (jit)
        # # hf time: 4.62881875038147 (jit)
        # eris = jnp.zeros((n_ao, n_ao, n_ao, n_ao))
        # for ix in range(n_grid):
        #     eris += jnp.einsum('yz,yzrs,yzqp->prsq', VG[ix], rhoG[ix], rhoG[ix])

        # # for xy
        # # max batchsize 68 (no jit)
        # # hf time: 8.341909408569336 (no jit)
        # # max batchsize 118 (jit)
        # # hf time: 13.894800424575806 (jit)
        # eris = jnp.zeros((n_ao, n_ao, n_ao, n_ao))
        # for ix in range(n_grid):
        #     for iy in range(n_grid):
        #         eris += jnp.einsum('z,zrs,zqp->prsq', VG[ix, iy], rhoG[ix, iy], rhoG[ix, iy])
        
        # # # jax.lax.scan
        # # max batchsize 108 (jit)
        # # hf time: 11.287713766098022 (jit)
        # VG1 = VG.reshape(n_grid3)
        # rhoG = rhoG.reshape(n_grid3, n_ao, n_ao)
        # eris = jnp.zeros((n_ao, n_ao, n_ao, n_ao))
        # def body_fun(carry, x):
        #     carry += jnp.einsum('rs,qp->prsq', x[1], x[1])*x[0]
        #     return carry, 0
        # eris, _ = jax.lax.scan(body_fun, eris, xs=[VG1, rhoG], length=n_grid3)

        # jax.lax.scan
        # max batchsize 108 (jit)
        # hf time: 11.287713766098022 (jit)
        # VG1 = VG.reshape(n_grid3)
        # rhoG = rhoG.reshape(n_grid3, n_ao, n_ao)
        # eris = jnp.zeros((n_ao, n_ao, n_ao, n_ao))
        # def body_fun(carry, x):
        #     carry += jnp.einsum('rs,qp->prsq', x[1], x[1])*x[0]
        #     return carry, 0
        # eris, _ = jax.lax.scan(body_fun, eris, xs=[VG1, rhoG], length=n_grid3, unroll=n_grid**2)

        # del rhoG
        eris0 = jnp.einsum('rs,qp->prsq', rhoG0, rhoG0)*4*jnp.pi/L/jnp.linalg.det(cell)**(1/3)*unknown1

        return vep, eris, eris0

    def eval_vep_eris_new_kpt(xp, pbc_gaussian_power_xyz):
        """
            k-point version.
            Use jax.lax.scan to calculate vep matrix and electron repulsion integrals (eris).
            INPUT:
                xp: array of shape (n, 3), position of protons.
                pbc_gaussian_power_xyz: array of shape (n, 3, n_grid_eris, n_all_alpha, n_l)'
                one dimensional gaussian power for each proton.
            OUTPUT:
                vep: array of shape (n_ao, n_ao), Vep matrix.
                eris0: array of shape (n_ao, n_ao, n_ao, n_ao), two-electron repulsion integrals at gamma point.
                eris: array of shape (n_ao, n_ao, n_ao, n_ao), two-electron repulsion integrals.

        """
        SI = jnp.sum(jnp.exp(-1j*Gmesh.dot(xp.T)), axis=3) # (nx, ny, nz)
        vlocG = -SI * VG # (nx, ny, nz)

        def body_fun(carry, pbc_gaussian_power_xyz_one):
            pbc_gaussian_power2R_xyz = jnp.einsum('ndgal,dgbk->dgnablk', pbc_gaussian_power_xyz, pbc_gaussian_power_xyz_one.conjugate())
            pbc_gaussian_power2G_xyz = jnp.fft.fft(pbc_gaussian_power2R_xyz, axis=1)*jnp.linalg.det(cell)**(1/3)*(L/n_grid)
            pbc_gaussian_power2G_xyz = jnp.concatenate([pbc_gaussian_power2G_xyz[:,-n_grid//2+1:], pbc_gaussian_power2G_xyz[:,:-n_grid//2+1]], axis=1)
            pbc_gaussian_cart2G_xyz = jnp.einsum('dgnablk,dlc,dke->dgnabce', pbc_gaussian_power2G_xyz, power2cart, power2cart)
        
            pbc_gaussian_cart2G = jnp.einsum('xnabce,ynabce,znabce->yxznabce', pbc_gaussian_cart2G_xyz[0], pbc_gaussian_cart2G_xyz[1],
                                            pbc_gaussian_cart2G_xyz[2])
            pbc_gto_cart2G = jnp.einsum('yxznabce,aco,bep->yxznop', pbc_gaussian_cart2G, alpha_coeff_gto_cart2sph, alpha_coeff_gto_cart2sph)
            return carry, pbc_gto_cart2G
        
        _, rhoG = jax.lax.scan(body_fun, None, pbc_gaussian_power_xyz)
        rhoG = (rhoG.transpose(1,2,3,4,5,0,6)).reshape(n_grid, n_grid, n_grid, n_ao, n_ao) # (n_grid3_eris, n_ao, n_ao)

        # vep matrix
        vep = jnp.einsum('xyz,xyzpq->pq', vlocG, rhoG.conjugate())

        return vep, rhoG

    occupation = make_occupation_func(n, n_mo, smearing=smearing, smearing_method=smearing_method, 
                                      smearing_sigma=smearing_sigma, search_method=search_method, 
                                      search_cycle=search_cycle, search_tol=search_tol)

    def density_matrix(mo_coeff, w1):
        """ 
            density matrix for closed-shell system. (Hermitian)
            Args:
                mo_coeff: array of shape (n_ao, n_mo), molecular orbital coefficients.
                w1: array of shape (n_mo,), orbital energies.
            Returns:
                dm: array of shape (n_ao, n_ao), density matrix.
        """
        dm_mo = jnp.diag(occupation(w1))
        dm = jnp.einsum('ab,bc,dc->ad', mo_coeff, dm_mo, mo_coeff.conjugate())
        return dm

    def eval_entropy(occ, epsilon=1e-15):
        """
            Evaluate entropy (closed-shell situation).
            Args:
                occ: array of shape (n_mo,), orbital occupation numbers.
                epsilon: float, small number to avoid log(0).
            Returns:
                entropy: float, entropy.
        """
        entropy = -2*(jnp.sum(jnp.where(occ/2 > epsilon, occ/2*jnp.log(occ/2), 0)) + \
                      jnp.sum(jnp.where(1-occ/2 > epsilon, (1-occ/2)*jnp.log(1-occ/2), 0)))
        return entropy

    def hartree(eris, dm):
        """
            Hartree matrix.
            Args:
                eris: array of shape (n_ao, n_ao, n_ao, n_ao), two-electron repulsion integrals.
                dm: array of shape (n_ao, n_ao), density matrix.
            Returns:
                hartree matrix: array of shape (n_ao, n_ao)
        """
        J = jnp.einsum('rs,prsq->pq', dm, eris)
        return J

    def exchange(eris, dm):
        """
            Exchange matrix.
            Args:
                eris: array of shape (n_ao, n_ao, n_ao, n_ao), two-electron repulsion integrals.
                dm: array of shape (n_ao, n_ao), density matrix.
            Returns:
                exchange matrix: array of shape (n_ao, n_ao)
        """
        K = jnp.einsum('rs,pqsr->pq', dm, eris)
        return K

    def hartree_rhoG(rhoG, dm):
        """
            Hartree matrix.
            Args:
                eris: array of shape (n_ao, n_ao, n_ao, n_ao), two-electron repulsion integrals.
                dm: array of shape (n_ao, n_ao), density matrix.
            Returns:
                hartree matrix: array of shape (n_ao, n_ao)
        """
        # eris = jnp.einsum('xyz,xyzrs,xyzpq->prsq', VG, rhoG, rhoG.conjugate())
        # J = jnp.einsum('rs,prsq->pq', dm, eris)
        rho = jnp.einsum('rs,xyzrs->xyz', dm, rhoG) # (n_grid, n_grid, n_grid)
        J = jnp.einsum('xyz,xyz,xyzpq->pq', rho, VG, rhoG.conjugate()) # (n_ao, n_ao)
        return J

    def exchange_rhoG(rhoG, mo_coeff):
        """
            Exchange matrix.
            Args:
                eris: array of shape (n_ao, n_ao, n_ao, n_ao), two-electron repulsion integrals.
                dm: array of shape (n_ao, n_ao), density matrix.
            Returns:
                exchange matrix: array of shape (n_ao, n_ao)
        """
        unknown1 = 0.22578495
        # eris0 = jnp.einsum('rs,qp->prsq', rhoG[n_grid//2,n_grid//2,n_grid//2], 
        #                    rhoG[n_grid//2,n_grid//2,n_grid//2])*4*jnp.pi/L/jnp.linalg.det(cell)**(1/3)*unknown1
        # eris = jnp.einsum('xyz,xyzrs,xyzqp->prsq', VG, rhoG, rhoG)
        # K = jnp.einsum('rs,pqsr->pq', dm, eris)
        rhoG1 = jnp.einsum('xyzrs,sn,nm->xyzrm', rhoG, mo_coeff, jnp.sqrt(dm0))
        K = jnp.einsum('xyzrm,xyzsm,xyz->rs', rhoG1, rhoG1, VG)
        K0 = jnp.einsum('rm,sm->rs', rhoG1[n_grid//2,n_grid//2,n_grid//2], rhoG1[n_grid//2,n_grid//2,n_grid//2])*4*jnp.pi/L/jnp.linalg.det(cell)**(1/3)*unknown1
        return K + K0

    # DFT functionals
    if dft:
        parts = xc.split(',')
        exchange, correlation = parts[0], parts[1]
        exc_functional = lambda rho: make_exchange_func(exchange)(rho) + make_correlation_func(correlation)(rho)
        vxc_functional = lambda rho: exc_functional(rho) + rho * grad(exc_functional)(rho) # probe speed so slow
        exc_rho_functional = lambda rho: rho * exc_functional(rho)

        # vmap
        vxc_functional = vmap(vxc_functional)
        exc_rho_functional = vmap(exc_rho_functional)

    def Exc_Vxc_integral(ao_Rmesh, dm):
        """
            Evaluate Exc integral and Vxc integral matrix.
            INPUT:
                ao_Rmesh: array of shape (n_ao, n_grid3), ao value on real space mesh.
                dm: array of shape (n_ao, n_ao), density matrix.
            OUTPUT:
                Exc: float, Exc integral.
                Vxc: array of shape (n_ao, n_ao), Vxc integral matrix.
        """
        rho_Rmesh = jnp.einsum('pr,qr,pq->r', ao_Rmesh, ao_Rmesh.conjugate(), dm).real # test probe
        Exc = jnp.sum(exc_rho_functional(rho_Rmesh))*(L/n_grid)**3*jnp.linalg.det(cell)
        Vxc_Rmesh = vxc_functional(rho_Rmesh) # (n_grid3,)
        Vxc = jnp.einsum('pr,qr,r->pq', ao_Rmesh.conjugate(), ao_Rmesh, Vxc_Rmesh)*(L/n_grid)**3*jnp.linalg.det(cell)
        return Exc, Vxc

    def get_diis_errvec_sdf(s, d, f):
        """
            Get error vector for DIIS.
        """
        return s @ d @ f - f @ d @ s

    scf = make_scf(diis=diis, diis_space=diis_space, diis_start_cycle=diis_start_cycle, 
                   diis_damp=diis_damp, tol=tol, max_cycle=max_cycle)

    def hf_gamma(xp: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float, float, float, float, bool]:
        """
            PBC Hartree Fock. kpt = (0,0,0)
            INPUT:
                xp: array of shape (n, dim), position of protons in rs unit.
                    Warining: xp * rs is in Bohr unit, xp is in rs unit.
            OUTPUT:
                mo_coeff: array of shape (n_ao, n_mo), molecular orbital coefficients.
                dm: array of shape (n_ao, n_ao), density matrix.
                bands: array of shape (n_mo,), orbital energies, Unit: Rydberg.
                E: float, total energy of the electrons, Note that vpp is not include in E, Unit: Rydberg.
                Ki: float, kinetic energy, Unit: Rydberg.
                Vep: float, electron-proton potential energy, Unit: Rydberg.
                Vee: float, electron-electron potential energy, Unit: Rydberg.
                Se: float, entropy.
                converged: bool, whether the calculation is converged.
        """
        assert xp.shape[0] == n
        xp *= rs

        # overlap and kinetic initialization
        ovlp, T = eval_overlap_kinetic(xp, xp)

        # diagonalization of overlap
        w, u = jnp.linalg.eigh(ovlp)
        v = jnp.dot(u, jnp.diag(w**(-0.5)))

        # potential (Vep), Hartree & Exchange integral initialization
        pbc_gaussian_power_xyz = eval_pbc_gaussian_power_x_Rmesh1D(xp) # (n, 3, n_grid_eris, n_all_alpha, n_l)
        V, eris, eris0 = eval_vep_eris(xp, pbc_gaussian_power_xyz)

        # core Hamiltonian
        Hcore = T + V

        # intialize molecular orbital
        f1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), Hcore, v)
        w1, c1 = jnp.linalg.eigh(f1)

        # Hcore initial guess (1e initial guess)
        mo_coeff = jnp.dot(v, c1)
        dm_init = density_matrix(mo_coeff, w1)

        hartree_fn = lambda dm: hartree(eris, dm)
        def exchange_fn(dm):
            Vx = -0.5*exchange(eris+eris0, dm)
            Ex = 0.5*jnp.einsum('pq,qp', Vx, dm).real
            return Ex, Vx
        errvec_sdf_fn = lambda dm, F: get_diis_errvec_sdf(ovlp, dm, F)

        # fixed point scf iteration
        mo_coeff, w1, E, converged = scf(v, Hcore, dm_init, hartree_fn, exchange_fn, 
                                         density_matrix, errvec_sdf_fn)

        # other quantities
        dm = density_matrix(mo_coeff, w1)
        J = hartree_fn(dm)
        Ex = exchange_fn(dm)[0]
        Ki = jnp.einsum('pq,qp', T, dm).real
        Vep = jnp.einsum('pq,qp', V, dm).real
        Vee = 0.5*jnp.einsum('pq,qp', J, dm).real + Ex
        Se = eval_entropy(occupation(w1))

        return mo_coeff, dm, w1 * Ry, E * Ry, Ki * Ry, Vep * Ry, Vee * Ry, Se, converged
    
    def hf_kpt(xp: jnp.ndarray, kpt: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float, float, float, float, bool]:
        """
            PBC Hartree Fock at kpt.
            INPUT:
                xp: array of shape (n, dim), position of protons in rs unit.
                    Warining: xp * rs is in Bohr unit, xp is in rs unit.
                kpt: array of shape (3,), k-point. (Unit: 1/Bohr)
                    1BZ: (-pi/L/rs, pi/L/rs)
            OUTPUT:
                mo_coeff: array of shape (n_ao, n_mo), molecular orbital coefficients.
                dm: array of shape (n_ao, n_ao), density matrix.
                bands: array of shape (n_mo,), orbital energies, Unit: Rydberg.
                E: float, total energy of the electrons, Note that vpp is not include in E, Unit: Rydberg.
                Ki: float, kinetic energy, Unit: Rydberg.
                Vep: float, electron-proton potential energy, Unit: Rydberg.
                Vee: float, electron-electron potential energy, Unit: Rydberg.
                Se: float, entropy.
                converged: bool, whether the calculation is converged
        """
        assert xp.shape[0] == n
        xp *= rs

        # overlap and kinetic initialization
        ovlp, T = eval_overlap_kinetic_kpt(xp, xp, kpt)

        # diagonalization of overlap
        w, u = jnp.linalg.eigh(ovlp)
        v = jnp.dot(u, jnp.diag(w**(-0.5)))

        # potential (Vep), Hartree & Exchange integral initialization
        pbc_gaussian_power_xyz = eval_pbc_gaussian_power_x_kpt_Rmesh1D(xp, kpt) # (n, 3, n_grid_eris, n_all_alpha, n_l)
        V, eris, eris0 = eval_vep_eris_kpt(xp, pbc_gaussian_power_xyz)

        # core Hamiltonian
        Hcore = T + V

        # intialize molecular orbital
        f1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), Hcore, v)
        # add random noise to f1 for a different initialization
        #key = jax.random.PRNGKey(42)
        #f1 = f1 + jax.random.normal(key, f1.shape) * 0.001
        #f1 = (f1+ f1.T)/2.0
        w1, c1 = jnp.linalg.eigh(f1)

        # Hcore initial guess (1e initial guess)
        mo_coeff = jnp.dot(v, c1) # (n_ao, n_mo)
        dm_init = density_matrix(mo_coeff, w1) # (n_ao, n_ao)

        hartree_fn = lambda dm: hartree(eris, dm)
        def exchange_fn(dm):
            Vx = -0.5*exchange(eris+eris0, dm)
            Ex = 0.5*jnp.einsum('pq,qp', Vx, dm).real
            return Ex, Vx
        errvec_sdf_fn = lambda dm, F: get_diis_errvec_sdf(ovlp, dm, F)

        # fixed point scf iteration
        mo_coeff, w1, E, converged = scf(v, Hcore, dm_init, hartree_fn, exchange_fn, 
                                         density_matrix, errvec_sdf_fn)

        # other quantities
        dm = density_matrix(mo_coeff, w1)
        J = hartree_fn(dm)
        Ex = exchange_fn(dm)[0]
        Ki = jnp.einsum('pq,qp', T, dm).real
        Vep = jnp.einsum('pq,qp', V, dm).real
        Vee = 0.5*jnp.einsum('pq,qp', J, dm).real + Ex
        Se = eval_entropy(occupation(w1))

        return mo_coeff, dm, w1 * Ry, E * Ry, Ki * Ry, Vep * Ry, Vee * Ry, Se, converged

    def dft_gamma(xp: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float, float, float, float, bool]:
        """
            PBC DFT. kpt = (0,0,0)
            INPUT:
                xp: array of shape (n, dim), position of protons in rs unit.
                    Warining: xp * rs is in Bohr unit, xp is in rs unit.
            OUTPUT:
                mo_coeff: array of shape (n_ao, n_mo), molecular orbital coefficients.
                dm: array of shape (n_ao, n_ao), density matrix.
                bands: array of shape (n_mo,), orbital energies, Unit: Rydberg.
                E: float, total energy of the electrons, Note that vpp is not include in E, Unit: Rydberg.
                Ki: float, kinetic energy, Unit: Rydberg.
                Vep: float, electron-proton potential energy, Unit: Rydberg.
                Vee: float, electron-electron potential energy, Unit: Rydberg.
                Se: float, entropy.
                converged: bool, whether the calculation is converged.
        """
        assert xp.shape[0] == n
        xp *= rs

        # overlap and kinetic initialization
        ovlp, T = eval_overlap_kinetic(xp, xp)

        # diagonalization of overlap
        w, u = jnp.linalg.eigh(ovlp)
        v = jnp.dot(u, jnp.diag(w**(-0.5)))

        # potential (Vep), Hartree & Exchange & correlation integral initialization
        pbc_gaussian_power_xyz = eval_pbc_gaussian_power_x_Rmesh1D(xp) # (n, 3, n_grid_eris, n_all_alpha, n_l)
        V, rhoG = eval_vep_eris_new(xp, pbc_gaussian_power_xyz) # V (n_ao, n_ao), rhoG (n_grid, n_grid, n_grid, n_ao, n_ao)
        ao_Rmesh = eval_pbc_ao_Rmesh(xp) # (n_ao, n_grid3) ao value on real space mesh
        eval_Exc_Vxc = lambda dm: Exc_Vxc_integral(ao_Rmesh, dm)
                                                                                                                                                                                                                                                                                                             
        # core Hamiltonian
        Hcore = T + V

        # intialize molecular orbital
        f1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), Hcore, v)
        w1, c1 = jnp.linalg.eigh(f1)

        # Hcore initial guess (1e initial guess)
        mo_coeff = jnp.dot(v, c1) # (n_ao, n_mo)
        dm_init = density_matrix(mo_coeff, w1) # (n_ao, n_ao)

        hartree_fn = lambda dm: hartree_rhoG(rhoG, dm)
        errvec_sdf_fn = lambda dm, F: get_diis_errvec_sdf(ovlp, dm, F)

        # fixed point scf iteration
        mo_coeff, w1, E, converged = scf(v, Hcore, dm_init, hartree_fn, eval_Exc_Vxc, 
                                         density_matrix, errvec_sdf_fn)

        # other quantities
        dm = density_matrix(mo_coeff, w1)
        J = hartree_fn(dm)
        Ex = eval_Exc_Vxc(dm)[0]
        Ki = jnp.einsum('pq,qp', T, dm).real
        Vep = jnp.einsum('pq,qp', V, dm).real
        Vee = 0.5*jnp.einsum('pq,qp', J, dm).real + Ex
        Se = eval_entropy(occupation(w1))

        return mo_coeff, dm, w1 * Ry, E * Ry, Ki * Ry, Vep * Ry, Vee * Ry, Se, converged

    def dft_kpt(xp: jnp.ndarray, kpt: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float, float, float, float, bool]:
        """
            PBC DFT at kpt.
            INPUT:
                xp: array of shape (n, dim), position of protons in rs unit.
                    Warining: xp * rs is in Bohr unit, xp is in rs unit.
                kpt: array of shape (3,), k-point. (Unit: 1/Bohr)
                    1BZ: (-pi/L/rs, pi/L/rs)
            OUTPUT:
                mo_coeff: array of shape (n_ao, n_mo), molecular orbital coefficients.
                dm: array of shape (n_ao, n_ao), density matrix.
                bands: array of shape (n_mo,), orbital energies, Unit: Rydberg.
                E: float, total energy of the electrons, Note that vpp is not include in E, Unit: Rydberg.
                Ki: float, kinetic energy, Unit: Rydberg.
                Vep: float, electron-proton potential energy, Unit: Rydberg.
                Vee: float, electron-electron potential energy, Unit: Rydberg.
                Se: float, entropy.
                converged: bool, whether the calculation is converged
        """
        assert xp.shape[0] == n
        xp *= rs

        # overlap and kinetic initialization
        ovlp, T = eval_overlap_kinetic_kpt(xp, xp, kpt)

        # diagonalization of overlap
        w, u = jnp.linalg.eigh(ovlp)
        v = jnp.dot(u, jnp.diag(w**(-0.5)))

        # potential (Vep), Hartree & Exchange & correlation integral initialization
        pbc_gaussian_power_xyz = eval_pbc_gaussian_power_x_kpt_Rmesh1D(xp, kpt) # (n, 3, n_grid_eris, n_all_alpha, n_l)
        V, rhoG = eval_vep_eris_new_kpt(xp, pbc_gaussian_power_xyz) # V (n_ao, n_ao), rhoG (n_grid, n_grid, n_grid, n_ao, n_ao)
        ao_Rmesh = eval_pbc_ao_kpt_Rmesh(xp, kpt) # (n_ao, n_grid3) ao value on real space mesh
        eval_Exc_Vxc = lambda dm: Exc_Vxc_integral(ao_Rmesh, dm)
                                                                                                                                                                                                                                                                                                             
        # core Hamiltonian
        Hcore = T + V

        # intialize molecular orbital
        f1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), Hcore, v)
        w1, c1 = jnp.linalg.eigh(f1)

        # Hcore initial guess (1e initial guess)
        mo_coeff = jnp.dot(v, c1) # (n_ao, n_mo)
        dm_init = density_matrix(mo_coeff, w1) # (n_ao, n_ao)

        hartree_fn = lambda dm: hartree_rhoG(rhoG, dm)
        errvec_sdf_fn = lambda dm, F: get_diis_errvec_sdf(ovlp, dm, F)

        # fixed point scf iteration
        mo_coeff, w1, E, converged = scf(v, Hcore, dm_init, hartree_fn, eval_Exc_Vxc, 
                                         density_matrix, errvec_sdf_fn)
        # other quantities
        dm = density_matrix(mo_coeff, w1)
        J = hartree_fn(dm)
        Ex = eval_Exc_Vxc(dm)[0]
        Ki = jnp.einsum('pq,qp', T, dm).real
        Vep = jnp.einsum('pq,qp', V, dm).real
        Vee = 0.5*jnp.einsum('pq,qp', J, dm).real + Ex
        Se = eval_entropy(occupation(w1))

        return mo_coeff, dm, w1 * Ry, E * Ry, Ki * Ry, Vep * Ry, Vee * Ry, Se, converged

    if gamma:
        if dft:
            solver = dft_gamma
        else:
            solver = hf_gamma
    else:
        if dft:
            solver = dft_kpt
        else:
            solver = hf_kpt

    if use_jit:
        return jit(solver)
    else:
        return solver
            