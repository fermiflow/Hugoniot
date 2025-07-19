import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, jacfwd
from pyscf.pbc import dft, gto, scf
from functools import partial

from src.orbitals import make_pbc_gto
from src.excor import make_exchange_func, make_correlation_func
from src.basis.parse import load_as_str, parse_quant_num, parse_gto, normalize_gto_coeff

def gen_lattice(cell, L, rcut=24):
    """
        Return lattice T within the cutoff radius in real space.

        INPUT:
            cell: (dim, dim)
            L: float
            cell * L is the basic vector of unit cell.

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

def make_lcao(n, L, rs, basis='gth-dzv', 
            rcut=24, tol=1e-7, max_cycle=50, grid_length=0.5, 
            diis=True, diis_space=8, diis_start_cycle=1, diis_damp=0,
            use_jit=True, dft=True, xc='lda,vwn',
            smearing=True, smearing_method='fermi', smearing_sigma=0.):
    """
        Make PBC Hartree Fock function.
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
        OUTPUT:
            lcao: lcao function.
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
    print("n_grid:", n_grid)
    n_grid3 = n_grid**3

    lattice = gen_lattice(cell, L, rcut=rcut)

    # load coefficients of the basis
    coeffs = alpha_coeff_cart[:, 1:]
    n_ao = n*(n_gto_s+3*n_gto_p)
    print("n_ao:", n_ao)
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

    # eval pbc gaussian power x on Rmesh_1d
    eval_pbc_gaussian_power_x, power2cart, alpha_coeff_gto_cart2sph = make_pbc_gto(basis, L, lcao_xyz=True)
    eval_pbc_gaussian_power_x = vmap(eval_pbc_gaussian_power_x)
    eval_pbc_gaussian_power_x = vmap(eval_pbc_gaussian_power_x)
    eval_pbc_gaussian_power_x_Rmesh1D = lambda xp: eval_pbc_gaussian_power_x(Rmesh_1d[None, None, :]-xp[:, :, None])
    
    # evaluate atomic orbitals on Rmesh
    eval_pbc_ao = make_pbc_gto(basis, L, lcao_xyz=False)
    eval_pbc_ao = vmap(eval_pbc_ao, (None, 0), 1)
    eval_pbc_ao_Rmesh = lambda xp: eval_pbc_ao(xp, Rmesh)
    
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
        
        if l_max == 0:
            return eval_overlap_kinetic_s
        elif l_max == 1:
            return eval_overlap_kinetic_sp
        else:
            raise ValueError("l_max > 1 is not supported yet.")
    
    eval_overlap_kinetic_noreshape = vmap(vmap(make_overlap_kinetic(), (0, None), (0, 0)), (None, 0), (2, 2))

    def eval_overlap_kinetic(xp1, xp2):
        ovlp, T = eval_overlap_kinetic_noreshape(xp1, xp2)
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
        # eris = jnp.einsum('xyz,xyzrs,xyzqp->prsq', VG, rhoG, rhoG)

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
        #         eris += jnp.einsum('zn \,zrs,zqp->prsq', VG[ix, iy], rhoG[ix, iy], rhoG[ix, iy])
        
        # # jax.lax.scan
        # max batchsize 108 (jit)
        # hf time: 11.287713766098022 (jit)
        VG1 = VG.reshape(n_grid3)
        rhoG = rhoG.reshape(n_grid3, n_ao, n_ao)
        eris = jnp.zeros((n_ao, n_ao, n_ao, n_ao))
        def body_fun(carry, x):
            carry += jnp.einsum('rs,qp->prsq', x[1], x[1])*x[0]
            return carry, 0
        eris, _ = jax.lax.scan(body_fun, eris, xs=[VG1, rhoG], length=n_grid3)

        # # jax.lax.scan
        # # max batchsize 108 (jit)
        # # hf time: 11.287713766098022 (jit)
        # VG1 = VG.reshape(n_grid3)
        # rhoG = rhoG.reshape(n_grid3, n_ao, n_ao)
        # eris = jnp.zeros((n_ao, n_ao, n_ao, n_ao))
        # def body_fun(carry, x):
        #     carry += jnp.einsum('rs,qp->prsq', x[1], x[1])*x[0]
        #     return carry, 0
        # eris, _ = jax.lax.scan(body_fun, eris, xs=[VG1, rhoG], length=n_grid3, unroll=n_grid**2)

        del rhoG
        eris0 = jnp.einsum('rs,qp->prsq', rhoG0, rhoG0)*4*jnp.pi/L/jnp.linalg.det(cell)**(1/3)*unknown1

        # ======================= debug =======================
        # jax.debug.print("rhoG.shape:{x}", x=rhoG.shape)
        # jax.debug.print("rhoG.size:{x}", x=rhoG.nbytes)
        # jax.debug.print("VG.shape:{x}", x=VG.shape)
        # jax.debug.print("VG.size:{x}", x=VG.nbytes)
        # jax.debug.print("eris0.shape:{x}", x=eris0.shape)
        # jax.debug.print("eris0.size:{x}", x=eris0.nbytes)
        # jax.debug.print("eris.shape:{x}", x=eris.shape)
        # jax.debug.print("eris.size:{x}", x=eris.nbytes)
        # =====================================================

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

        # ======================= debug =======================
        # jax.debug.print("rhoG.shape:{x}", x=rhoG.shape)
        # jax.debug.print("rhoG.size:{x}", x=rhoG.nbytes)
        # jax.debug.print("VG.shape:{x}", x=VG.shape)
        # jax.debug.print("VG.size:{x}", x=VG.nbytes)
        # jax.debug.print("eris0.shape:{x}", x=eris0.shape)
        # jax.debug.print("eris0.size:{x}", x=eris0.nbytes)
        # jax.debug.print("eris.shape:{x}", x=eris.shape)
        # jax.debug.print("eris.size:{x}", x=eris.nbytes)
        # =====================================================

        # return vep, eris, eris0
        return vep, rhoG

    def gaussian_f_func(mu, w1):
        """
            Gaussing smearing f function. (closed-shell situation)
        """
        pass

    def fermi_dirac_f_func(mu, w1):
        """
            Fermi-Dirac distribution f function. (closed-shell situation)
                2/(exp((w1_i-mu)/sigma)+1)
            Input:
                mu: float, chemical potential, Unit: Hartree.
                w1: array of shape (n_ao,), orbital energies, Unit: Hartree.
            Output:
                N_fd: array of shape (n_ao,), Fermi-Dirac distribution f function.
        """
        return 2/(jnp.exp((w1-mu)/smearing_sigma)+1)
    
    fermi_dirac_N_func = lambda mu, w1: jnp.sum(fermi_dirac_f_func(mu, w1))-n
    fermi_dirac_N2_func1 = grad(lambda mu, w1: (fermi_dirac_N_func(mu, w1))**2) # first derivative
    fermi_dirac_N2_func2 = grad(fermi_dirac_N2_func1) # second derivative
    fermi_dirac_Newton_iter_func = lambda mu, w1: mu - fermi_dirac_N2_func1(mu, w1)/jnp.abs(fermi_dirac_N2_func2(mu, w1)) # Newton iteration

    def search_mu_newton(w1, Newton_iter_func, max_cycle=10, mu_tol=1e-7):
        """
            Search chemical potential by Newton minimization method.
            Input:
                w1: array of shape (n_ao,), orbital energies, Unit: Hartree.
                Newton_iter_func: Newton iteration function.
                max_cycle: int, maximum number of cycles.
                mu_tol: float, tolerance.
            Output:
                mu: float, chemical potential, Unit: Hartree.
        """
        mu_init = w1[n//2]

        def body_fun(carry):
            _, mu, loop = carry
            mu_new = Newton_iter_func(mu)

            # ======================= debug =======================
            # jax.debug.print("======= smearing =======")
            # jax.debug.print("loop: {x}", x=loop)
            # jax.debug.print("mu:{x}, mu_new:{y}", x=mu, y=mu_new)
            # =====================================================

            return mu, mu_new, loop+1
        
        def cond_fun(carry):
            return jnp.abs(carry[1] - carry[0]) > mu_tol
            
        _, mu, _ = jax.lax.while_loop(cond_fun, body_fun, (1., mu_init, 0))

        return mu

    def search_mu_bisect(w1, N_func, max_cycle=100, mu_tol=1e-7):
        """
            Search chemical potential by bisection algorithm
            Input:
                w1: array of shape (n_ao,), orbital energies, Unit: Hartree.
                N_func: N(mu)-n, find mu make N_func(mu) == 0.
                max_cycle: int, maximum number of cycles.
                mu_tol: float, tolerance.
            Output:
                mu: float, chemical potential, Unit: Hartree.
        """
        mu_lo = jnp.min(w1) - smearing_sigma * jnp.log(2*n_ao/n-1)
        mu_hi = jnp.max(w1) - smearing_sigma * jnp.log(2*n_ao/n-1)

        def body_fun(carry):
            mu_lo, mu_hi, _, _, loop = carry
            mu_mid = (mu_lo + mu_hi) / 2
            y_mid = N_func(mu_mid)
            mu_lo = (y_mid < 0) * mu_mid + (y_mid >= 0) * mu_lo
            mu_hi = (y_mid >= 0) * mu_mid + (y_mid < 0) * mu_hi

            # ======================= debug =======================
            # jax.debug.print("======= smearing =======")
            # jax.debug.print("loop: {x}", x=loop)
            # jax.debug.print("mu_lo:{x}, mu_hi:{y}", x=mu_lo, y=mu_hi)
            # =====================================================

            return mu_lo, mu_hi, N_func(mu_lo), N_func(mu_hi), loop+1
        
        def cond_fun(carry):
            return (carry[2] < -mu_tol) * (carry[3] > mu_tol) * (carry[4] < max_cycle)
            
        mu_lo, mu_hi, y_lo, y_hi, loop = jax.lax.while_loop(cond_fun, body_fun, (mu_lo, mu_hi, -1., 1., 0))

        return (mu_lo > -mu_tol) * mu_lo + (mu_hi < mu_tol) * mu_hi + (mu_lo + mu_hi)/2 * (loop == max_cycle)

    def density_martix_fermi_mo(w1):
        """
            Fermi-Dirac distribution density matrix in molecular orbital representation. (Hermitian)
            Determine the chemical potential by Newton minimization method.
            Input:
                w1: array of shape (n_ao,), orbital energies.
            Output:
                dm: array of shape (n_ao, n_ao), Fermi-Dirac density matrix.
        """
        # # Newton method
        # Newton_iter_func = lambda mu: fermi_dirac_Newton_iter_func(mu, w1)
        # mu = search_mu_newton(w1, Newton_iter_func)

        # bisection method
        N_func = lambda mu: fermi_dirac_N_func(mu, w1)
        mu = search_mu_bisect(w1, N_func)
        return jnp.diag(fermi_dirac_f_func(mu, w1))

    def density_matrix_mo(w1):
        """
            density matrix in molecular orbital representation. (Hermitian)
        """
        return 2*jnp.diag(jnp.concatenate((jnp.ones(n//2), jnp.zeros(n_ao-n//2))))
    
    if smearing:
        if smearing_method == 'fermi':
            density_matrix_mo_func = density_martix_fermi_mo
        else:
            raise ValueError("smearing_method not supported yet.")
    else:
        density_matrix_mo_func = density_matrix_mo

    def density_matrix(mo_coeff, dm_mo):
        """
            density matrix for closed shell system. (Hermitian)
        """
        dm = jnp.einsum('ab,bc,dc->ad', mo_coeff, dm_mo, mo_coeff)
        return dm

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
        # eris = jnp.einsum('xyz,xyzrs,xyzqp->prsq', VG, rhoG, rhoG)
        # J = jnp.einsum('rs,prsq->pq', dm, eris)
        rho = jnp.einsum('rs,xyzrs->xyz', dm, rhoG) # (n_grid, n_grid, n_grid)
        J = jnp.einsum('xyz,xyz,xyzqp->pq', rho, VG, rhoG) # (n_ao, n_ao)
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
        rho_Rmesh = jnp.einsum('pr,qr,pq->r', ao_Rmesh, ao_Rmesh, dm) # test probe
        Exc = jnp.sum(exc_rho_functional(rho_Rmesh))*(L/n_grid)**3*jnp.linalg.det(cell)
        Vxc_Rmesh = vxc_functional(rho_Rmesh) # (n_grid3,)
        Vxc = jnp.einsum('pr,qr,r->pq', ao_Rmesh, ao_Rmesh, Vxc_Rmesh)*(L/n_grid)**3*jnp.linalg.det(cell)
        return Exc, Vxc

    def get_diis_errvec_sdf(s, d, f):
        """
            Get error vector for DIIS.
        """
        return s @ d @ f - f @ d @ s

    def hf_fp(xp):
        """
            PBC Hartree Fock with Vee. kpt = (0,0,0)
            INPUT:
                xp: array of shape (n, dim), position of protons.
            OUTPUT:
                energy without vpp, unit: Rydberg.
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

        mo_coeff = jnp.dot(v, c1)
        dm_mo = density_matrix_mo_func(w1)
        dm = density_matrix(mo_coeff, dm_mo)

        # ======================= debug =======================
        # jax.debug.print("w of ovlp:\n{x}", x=w)
        # jax.debug.print("w**(-0.5) of ovlp:\n{x}", x=w**(-0.5))
        # jax.debug.print("u of ovlp:\n{x}", x=u)
        # jax.debug.print("v of ovlp:\n{x}", x=v)
        # jax.debug.print("Hcore:\n{x}", x=Hcore)
        # jax.debug.print("f1:\n{x}", x=f1)
        # jax.debug.print("initial mo_coeff:\n{x}", x=mo_coeff)
        # jax.debug.print("initial w1:\n{x}", x=w1)
        # jax.debug.print("begin scf loop")
        # =====================================================

        # scf loop
        def body_fun(carry):
            _, E, mo_coeff, dm, w1, loop = carry

            # Hartree & Exchange
            J = hartree(eris, dm)
            K = exchange(eris+eris0, dm)

            # Fock matrix
            F = Hcore + J - 0.5 * K

            # diagonalization
            f1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), F, v)
            w1, c1 = jnp.linalg.eigh(f1)

            # molecular orbitals and density matrix
            mo_coeff = jnp.dot(v, c1) # (n_ao, n_mo)
            dm_mo = density_matrix_mo_func(w1)  # (n_mo, n_mo)
            dm = density_matrix(mo_coeff, dm_mo) # (n_ao, n_ao)

            # energy
            E_new = 0.5*jnp.einsum('pq,qp', F+Hcore, dm)

            # ======================= debug =======================
            # jax.debug.print("======= fp =======")
            # jax.debug.print("loop: {x}", x=loop)
            # jax.debug.print("F:\n{x}", x=F)
            # jax.debug.print("w1:\n{x}", x=w1)
            # jax.debug.print("E:{x}, E_new:{y}", x=E, y=E_new)
            # jax.debug.print(jax.Device.addressable_memories())
            # jax.debug.print(jax.Device.default_memory)
            # jax.debug.print(jax.Device.memory)
            # jax.debug.print(jax.Device.memory_stats)
            # =====================================================

            return (E, E_new, mo_coeff, dm, w1, loop+1)
        
        def cond_fun(carry):
            return (abs(carry[1] - carry[0]) > tol) * (carry[5] < max_cycle)
            
        _, E, mo_coeff, dm, w1, loop = jax.lax.while_loop(cond_fun, body_fun, (0., 1., mo_coeff, dm, w, 0))

        # ======================= debug =======================
        # jax.debug.print("end scf loop {x}", x=loop)
        # =====================================================

        return mo_coeff[:, ::-1]+0j, w1[::-1] * Ry
    
    def hf_diis(xp):
        """
            PBC Hartree Fock with Vee. kpt = (0,0,0)
            INPUT:
                xp: array of shape (n, dim), position of protons.
            OUTPUT:
                energy without vpp, unit: Rydberg.
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

        # V, rhoG = eval_vep_eris_new(xp, pbc_gaussian_power_xyz)

        # core Hamiltonian
        Hcore = T + V

        # intialize molecular orbital
        f1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), Hcore, v)
        w1, c1 = jnp.linalg.eigh(f1)

        # Hcore initial guess (1e initial guess)
        mo_coeff = jnp.dot(v, c1) # (n_ao, n_mo)
        dm_mo = density_matrix_mo_func(w1) # (n_mo, n_mo)
        dm = density_matrix(mo_coeff, dm_mo) # (n_ao, n_ao)

        # initial J and K
        J = hartree(eris, dm)
        K = exchange(eris+eris0, dm)

        # J = hartree_rhoG(rhoG, dm)
        # K = exchange_rhoG(rhoG, mo_coeff)

        # initial F
        F_init = Hcore + J - 0.5 * K

        # initial error vector
        errvec_init = get_diis_errvec_sdf(ovlp, dm, F_init)

        # initial F and error vector series for DIIS
        F_k = jnp.repeat(F_init[None, ...], diis_space, axis=0)
        errvec_k = jnp.repeat(errvec_init[None, ...], diis_space, axis=0)
        
        # ======================= debug =======================
        # jax.debug.print("J-J_new:\n{x}", x=J-J_new)
        # jax.debug.print("K-K_new:\n{x}", x=K-K_new)
        # jax.debug.print("w of ovlp:\n{x}", x=w)
        # jax.debug.print("dm-dm_new:\n{x}", x=dm-dm_new)
        # jax.debug.print("w**(-0.5) of ovlp:\n{x}", x=w**(-0.5))
        # jax.debug.print("u of ovlp:\n{x}", x=u)
        # jax.debug.print("v of ovlp:\n{x}", x=v)
        # jax.debug.print("Hcore:\n{x}", x=Hcore)
        # jax.debug.print("f1:\n{x}", x=f1)
        # jax.debug.print("initial mo_coeff:\n{x}", x=mo_coeff)
        # jax.debug.print("initial dm:\n{x}", x=dm)
        # jax.debug.print("w1 of F_init:\n{x}", x=w1)
        # jax.debug.print("initial dm_mo:\n{x}", x=dm_mo)
        # jax.debug.print("number of particles:{x}", x=jnp.trace(ovlp @ dm))
        # jax.debug.print("number of particles:{x}", x=jnp.trace(dm1))
        # jax.debug.print("max element of errvev:{x}", x=jnp.max(errvec_init))
        # jax.debug.print("e:\n{x}", x=jnp.diag(w1))
        # jax.debug.print("FC-SCe:\n{x}", x=F_init@mo_coeff-ovlp@mo_coeff@jnp.diag(w1))
        # jax.debug.print("initial w1:\n{x}", x=w1)
        # jax.debug.print("begin scf loop")
        # jax.debug.print("initial F_k.shape:\n{x}", x=F_k.shape)
        # jax.debug.print("initial F_k:\n{x}", x=F_k)
        # jax.debug.print("initial errvec_k.shape:\n{x}", x=errvec_k.shape)
        # =====================================================

        # fixed point iteration
        def fp_body_fun(carry):
            _, E, _, _, loop, F_k, errvec_k = carry

            # last Fock matrix
            F = F_k[-1]

            # diagonalization
            F1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), F, v)
            w1, c1 = jnp.linalg.eigh(F1)

            # next molecular orbitals and density matrix
            mo_coeff = jnp.dot(v, c1) # (n_ao, n_mo)
            dm_mo = density_matrix_mo_func(w1) # (n_mo, n_mo)
            dm = density_matrix(mo_coeff, dm_mo) # (n_ao, n_ao)

            # next energy
            E_new = 0.5*jnp.einsum('pq,qp', F+Hcore, dm)

            # next Fock matrix
            J = hartree(eris, dm)
            K = exchange(eris+eris0, dm)
            # J = hartree_rhoG(rhoG, dm)
            # K = exchange_rhoG(rhoG, mo_coeff)
            F = Hcore + J - 0.5 * K
            
            # next error vector
            errvec = get_diis_errvec_sdf(ovlp, dm, F)
            
            # update F and error vector series for DIIS
            F_k = jnp.concatenate((F_k[1:], jnp.array([F])), axis=0)
            errvec_k = jnp.concatenate((errvec_k[1:], jnp.array([errvec])), axis=0)

            # ======================= debug =======================
            # jax.debug.print("======= fp =======")
            # jax.debug.print("loop: {x}", x=loop)
            # jax.debug.print("F:\n{x}", x=F)
            # jax.debug.print("F-F_dagger:\n{x}", x=F-F.T.conjugate())
            # jax.debug.print("w1:\n{x}", x=w1)
            # jax.debug.print("max element of errvev:{x}", x=jnp.max(errvec))
            # jax.debug.print("E:{x}, E_new:{y}", x=E, y=E_new)
            # jax.debug.print("number of particles:{x}", x=jnp.trace(ovlp @ dm))
            # jax.debug.print("number of particles:{x}", x=jnp.trace(dm1))
            # jax.debug.print("latest error vector:{x}", x=errvec)
            # jax.debug.print(jax.Device.addressable_memories())
            # jax.debug.print(jax.Device.default_memory)
            # jax.debug.print(jax.Device.memory)
            # jax.debug.print(jax.Device.memory_stats)
            # =====================================================

            return (E, E_new, mo_coeff, w1, loop+1, F_k, errvec_k)
        
        def fp_cond_fun(carry):
            return carry[4] < diis_start_cycle + diis_space
            
        _, E, mo_coeff, w1, loop, F_k, errvec_k = jax.lax.while_loop(fp_cond_fun, fp_body_fun, (0., 0., mo_coeff, w1, 0, F_k, errvec_k))

        # ======================= debug =======================
        # jax.debug.print("end scf loop {x}", x=loop-1)
        # jax.debug.print("F_k:\n{x}", x=F_k)
        # jax.debug.print("errvec_k:\n{x}", x=errvec_k)
        # =====================================================

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
            F1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), _F, v)
            w1, c1 = jnp.linalg.eigh(F1)

            # molecular orbitals and density matrix
            mo_coeff = jnp.dot(v, c1) # (n_ao, n_mo)
            dm_mo = density_matrix_mo_func(w1) # (n_mo, n_mo)
            dm = density_matrix(mo_coeff, dm_mo) # (n_ao, n_ao)

            # next Fock matrix
            J = hartree(eris, dm)
            K = exchange(eris+eris0, dm)
            # J = hartree_rhoG(rhoG, dm)
            # K = exchange_rhoG(rhoG, mo_coeff)
            F = Hcore + J - 0.5 * K

            # next energy
            E_new = 0.5*jnp.einsum('pq,qp', F+Hcore, dm)

            # next error vector
            errvec = get_diis_errvec_sdf(ovlp, dm, F)

            # update F and error vector series for DIIS
            F_k = jnp.concatenate((F_k[1:], jnp.array([F])), axis=0)
            errvec_k = jnp.concatenate((errvec_k[1:], jnp.array([errvec])), axis=0)

            # ======================= debug =======================
            # jax.debug.print("======= diis =======")
            # w_diis, v_diis = jnp.linalg.eigh(h)
            # jax.debug.print("latest errvec:{x}", x=errvec)
            # jax.debug.print("loop: {x}", x=loop)
            # jax.debug.print("max element of errvev:{x}", x=jnp.max(errvec))
            # jax.debug.print("w of diis h:\n{x}", x=w_diis)
            # jax.debug.print("c_k: {x}", x=c_k)
            # jax.debug.print("F:\n{x}", x=F)
            # jax.debug.print("w1:\n{x}", x=w1)
            # jax.debug.print("E:{x}, E_new:{y}", x=E, y=E_new)
            # jax.debug.print("number of particles:{x}", x=jnp.trace(ovlp @ dm))
            # jax.debug.print("number of particles:{x}", x=jnp.trace(dm1))
            # jax.debug.print(jax.Device.addressable_memories())
            # jax.debug.print(jax.Device.default_memory)
            # jax.debug.print(jax.Device.memory)
            # jax.debug.print(jax.Device.memory_stats)
            # =====================================================

            return (E, E_new, mo_coeff, w1, loop+1, F_k, errvec_k)

        def diis_cond_fun(carry):
            return (jnp.abs(carry[1] - carry[0]) > tol) * (carry[4] < max_cycle)

        _, E, mo_coeff, w1, loop, F_k, errvec_k = jax.lax.while_loop(diis_cond_fun, diis_body_fun, (E-1., E, mo_coeff, w1, loop, F_k, errvec_k))

        return mo_coeff[:, ::-1]+0j, w1[::-1] * Ry

    def dft_fp(xp):
        """
            PBC DFT with Vee. kpt = (0,0,0)
            INPUT:
                xp: array of shape (n, dim), position of protons.
            OUTPUT:
                energy without vpp, unit: Rydberg.
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

        mo_coeff = jnp.dot(v, c1) # (n_ao, n_mo)
        dm_mo = density_matrix_mo_func(w1) # (n_mo, n_mo)
        dm = density_matrix(mo_coeff, dm_mo) # (n_ao, n_ao)

        # ======================= debug =======================
        # jax.debug.print("w of ovlp:\n{x}", x=w)
        # jax.debug.print("w**(-0.5) of ovlp:\n{x}", x=w**(-0.5))
        # jax.debug.print("u of ovlp:\n{x}", x=u)
        # jax.debug.print("v of ovlp:\n{x}", x=v)
        # jax.debug.print("Hcore:\n{x}", x=Hcore)
        # jax.debug.print("f1:\n{x}", x=f1)
        # jax.debug.print("initial mo_coeff:\n{x}", x=mo_coeff)
        # jax.debug.print("initial w1:\n{x}", x=w1)
        # jax.debug.print("begin scf loop")
        # =====================================================

        # scf loop
        def body_fun(carry):
            _, E, _, dm, _, loop = carry

            # Hartree & Exchange
            J = hartree_rhoG(rhoG, dm)
            Exc, Vxc = eval_Exc_Vxc(dm)

            # energy
            E_new = jnp.einsum('pq,qp', Hcore + 0.5*J, dm) + Exc

            # Fock matrix
            F = Hcore + J + Vxc

            # diagonalization
            f1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), F, v)
            w1, c1 = jnp.linalg.eigh(f1)

            # molecular orbitals and density matrix
            mo_coeff = jnp.dot(v, c1) # (n_ao, n_mo)
            dm_mo = density_matrix_mo_func(w1) # (n_mo, n_mo)
            dm = density_matrix(mo_coeff, dm_mo) # (n_ao, n_ao)

            # ======================= debug =======================
            # jax.debug.print("======= fp =======")
            # jax.debug.print("loop: {x}", x=loop)
            # jax.debug.print("F:\n{x}", x=F)
            # jax.debug.print("w1:\n{x}", x=w1)
            # jax.debug.print("E:{x}, E_new:{y}", x=E, y=E_new)
            # jax.debug.print(jax.Device.addressable_memories())
            # jax.debug.print(jax.Device.default_memory)
            # jax.debug.print(jax.Device.memory)
            # jax.debug.print(jax.Device.memory_stats)
            # =====================================================

            return (E, E_new, mo_coeff, dm, w1, loop+1)
        
        def cond_fun(carry):
            return (abs(carry[1] - carry[0]) > tol) * (carry[5] < max_cycle)
            
        _, E, mo_coeff, dm, w1, loop = jax.lax.while_loop(cond_fun, body_fun, (0., 1., mo_coeff, dm, w, 0))

        # ======================= debug =======================
        # jax.debug.print("end scf loop {x}", x=loop)
        # =====================================================

        return mo_coeff[:, ::-1]+0j, w1[::-1] * Ry

    def dft_diis(xp):
        """
            PBC DFT with Vee. kpt = (0,0,0)
            INPUT:
                xp: array of shape (n, dim), position of protons.
            OUTPUT:
                energy without vpp, unit: Rydberg.
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
        dm_mo = density_matrix_mo_func(w1) # (n_mo, n_mo)
        dm = density_matrix(mo_coeff, dm_mo) # (n_ao, n_ao)

        # initial J & XC
        J = hartree_rhoG(rhoG, dm)
        Exc, Vxc = eval_Exc_Vxc(dm)

        # initial F
        F_init = Hcore + J + Vxc

        # initial error vector
        errvec_init = get_diis_errvec_sdf(ovlp, dm, F_init)

        # initial F and error vector series for DIIS
        F_k = jnp.repeat(F_init[None, ...], diis_space, axis=0)
        errvec_k = jnp.repeat(errvec_init[None, ...], diis_space, axis=0)
        
        # ======================= debug =======================
        # jax.debug.print("J-J_new:\n{x}", x=J-J_new)
        # jax.debug.print("K-K_new:\n{x}", x=K-K_new)
        # jax.debug.print("w of ovlp:\n{x}", x=w)
        # jax.debug.print("dm-dm_new:\n{x}", x=dm-dm_new)
        # jax.debug.print("w**(-0.5) of ovlp:\n{x}", x=w**(-0.5))
        # jax.debug.print("u of ovlp:\n{x}", x=u)
        # jax.debug.print("v of ovlp:\n{x}", x=v)
        # jax.debug.print("Hcore:\n{x}", x=Hcore)
        # jax.debug.print("f1:\n{x}", x=f1)
        # jax.debug.print("initial mo_coeff:\n{x}", x=mo_coeff)
        # jax.debug.print("initial dm:\n{x}", x=dm)
        # jax.debug.print("w1 of F_init:\n{x}", x=w1)
        # jax.debug.print("initial dm_mo:\n{x}", x=dm_mo)
        # jax.debug.print("number of particles:{x}", x=jnp.trace(ovlp @ dm))
        # jax.debug.print("number of particles:{x}", x=jnp.trace(dm1))
        # jax.debug.print("max element of errvev:{x}", x=jnp.max(errvec_init))
        # jax.debug.print("e:\n{x}", x=jnp.diag(w1))
        # jax.debug.print("FC-SCe:\n{x}", x=F_init@mo_coeff-ovlp@mo_coeff@jnp.diag(w1))
        # jax.debug.print("initial w1:\n{x}", x=w1)
        # jax.debug.print("begin scf loop")
        # jax.debug.print("initial F_k.shape:\n{x}", x=F_k.shape)
        # jax.debug.print("initial F_k:\n{x}", x=F_k)
        # jax.debug.print("initial errvec_k.shape:\n{x}", x=errvec_k.shape)
        # =====================================================

        # fixed point iteration
        def fp_body_fun(carry):
            _, E, _, _, loop, F_k, errvec_k = carry

            # last Fock matrix
            F = F_k[-1]

            # diagonalization
            F1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), F, v)
            w1, c1 = jnp.linalg.eigh(F1)

            # next molecular orbitals and density matrix
            mo_coeff = jnp.dot(v, c1) # (n_ao, n_mo)
            dm_mo = density_matrix_mo_func(w1) # (n_mo, n_mo)
            dm = density_matrix(mo_coeff, dm_mo) # (n_ao, n_ao)

            # next J and xc
            J = hartree_rhoG(rhoG, dm)
            Exc, Vxc = eval_Exc_Vxc(dm)

            # next energy
            E_new = jnp.einsum('pq,qp', Hcore + 0.5*J, dm) + Exc
 
            # next Fock matrix
            F = Hcore + J + Vxc
            
            # next error vector
            errvec = get_diis_errvec_sdf(ovlp, dm, F)
            
            # update F and error vector series for DIIS
            F_k = jnp.concatenate((F_k[1:], jnp.array([F])), axis=0)
            errvec_k = jnp.concatenate((errvec_k[1:], jnp.array([errvec])), axis=0)

            # ======================= debug =======================
            # jax.debug.print("======= fp =======")
            # jax.debug.print("loop: {x}", x=loop)
            # jax.debug.print("F:\n{x}", x=F)
            # jax.debug.print("F-F_dagger:\n{x}", x=F-F.T.conjugate())
            # jax.debug.print("w1:\n{x}", x=w1)
            # jax.debug.print("max element of errvev:{x}", x=jnp.max(errvec))
            # jax.debug.print("E:{x}, E_new:{y}", x=E, y=E_new)
            # jax.debug.print("number of particles:{x}", x=jnp.trace(ovlp @ dm))
            # jax.debug.print("number of particles:{x}", x=jnp.trace(dm1))
            # jax.debug.print("latest error vector:{x}", x=errvec)
            # jax.debug.print(jax.Device.addressable_memories())
            # jax.debug.print(jax.Device.default_memory)
            # jax.debug.print(jax.Device.memory)
            # jax.debug.print(jax.Device.memory_stats)
            # =====================================================

            return (E, E_new, mo_coeff, w1, loop+1, F_k, errvec_k)
        
        def fp_cond_fun(carry):
            return carry[4] < diis_start_cycle + diis_space
            
        _, E, mo_coeff, w1, loop, F_k, errvec_k = jax.lax.while_loop(fp_cond_fun, fp_body_fun, (0., 0., mo_coeff, w1, 0, F_k, errvec_k))

        # ======================= debug =======================
        # jax.debug.print("end scf loop {x}", x=loop-1)
        # jax.debug.print("F_k:\n{x}", x=F_k)
        # jax.debug.print("errvec_k:\n{x}", x=errvec_k)
        # =====================================================

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
            F1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), _F, v)
            w1, c1 = jnp.linalg.eigh(F1)

            # molecular orbitals and density matrix
            mo_coeff = jnp.dot(v, c1) # (n_ao, n_mo)
            dm_mo = density_matrix_mo_func(w1) # (n_mo, n_mo)
            dm = density_matrix(mo_coeff, dm_mo) # (n_ao, n_ao)

            # next J anx xc
            J = hartree_rhoG(rhoG, dm)
            Exc, Vxc = eval_Exc_Vxc(dm)

            # next energy
            E_new = jnp.einsum('pq,qp', Hcore + 0.5*J, dm) + Exc

            # next Fock matrix
            F = Hcore + J + Vxc

            # next error vector
            errvec = get_diis_errvec_sdf(ovlp, dm, F)

            # update F and error vector series for DIIS
            F_k = jnp.concatenate((F_k[1:], jnp.array([F])), axis=0)
            errvec_k = jnp.concatenate((errvec_k[1:], jnp.array([errvec])), axis=0)

            # ======================= debug =======================
            # jax.debug.print("======= diis =======")
            # w_diis, v_diis = jnp.linalg.eigh(h)
            # jax.debug.print("latest errvec:{x}", x=errvec)
            # jax.debug.print("loop: {x}", x=loop)
            # jax.debug.print("max element of errvev:{x}", x=jnp.max(errvec))
            # jax.debug.print("w of diis h:\n{x}", x=w_diis)
            # jax.debug.print("c_k: {x}", x=c_k)
            # jax.debug.print("F:\n{x}", x=F)
            # jax.debug.print("w1:\n{x}", x=w1)
            # jax.debug.print("E:{x}, E_new:{y}", x=E, y=E_new)
            # jax.debug.print("Kinetic energy:{x}", x=jnp.einsum('pq,qp', T, dm))
            # jax.debug.print("Potential energy:{x}", x=jnp.einsum('pq,qp', V, dm))
            # jax.debug.print("Hartree energy:{x}", x=0.5*jnp.einsum('pq,qp', J, dm))
            # jax.debug.print("Exchange & correlation energy:{x}", x=Exc)
            # jax.debug.print("number of particles:{x}", x=jnp.trace(ovlp @ dm))
            # jax.debug.print("number of particles:{x}", x=jnp.trace(dm1))
            # jax.debug.print(jax.Device.addressable_memories())
            # jax.debug.print(jax.Device.default_memory)
            # jax.debug.print(jax.Device.memory)
            # jax.debug.print(jax.Device.memory_stats)
            # =====================================================

            return (E, E_new, mo_coeff, w1, loop+1, F_k, errvec_k)

        def diis_cond_fun(carry):
            return (jnp.abs(carry[1] - carry[0]) > tol) * (carry[4] < max_cycle)

        _, E, mo_coeff, w1, loop, F_k, errvec_k = jax.lax.while_loop(diis_cond_fun, diis_body_fun, (E-1., E, mo_coeff, w1, loop, F_k, errvec_k))

        return mo_coeff[:, ::-1]+0j, w1[::-1] * Ry

    if dft:
        if diis:
            lcao = dft_diis
        else:
            lcao = dft_fp
    else:
        if diis:
            lcao = hf_diis
        else:
            lcao = hf_fp

    if use_jit:
        return jit(lcao)
    else:
        return lcao

def pyscf_hf(n, L, rs, sigma, xp, basis='sto-3g', hf0=False, smearing=False, smearing_method='fermi'):
    """
        Pyscf Hartree Fock solver for hydrogen.
    INPUT:
        n: number of protons.
        L: side length of unit cell, unit: rs.
        rs: average atomic spacing, unit: Bohr
        sigma: smearing width, unit: Hartree.
        xp: array of shape (n, dim), position of protons.
        basis: gto basis name, eg:'gth-szv', 'gth-tzv2p', 'gth-qzv3p'.
        hf0: if True, do Hartree Fock scf without Vpp.
        smearing: if True, use Fermi-Dirac smearing
            (finite temperature Hartree Fock or thermal Hartree Fock).
    OUTPUT:
        mo_coeff: molecular orbitals coefficients, complex array of shape (n_ao, n_mo).
        bands: energy bands of corresponding molecular orbitals, 
            ranking of energy from low to high. array of shape (n_mo,).
    """
    Ry = 2
    xp *= rs
    cell = gto.Cell()
    cell.unit = 'B'
    cell.a = np.eye(3) * L * rs
    cell.atom = []
    for ie in range(n):
        cell.atom.append(['H', tuple(xp[ie])])
    cell.spin = 0
    cell.basis = {'H':gto.parse(load_as_str('H', basis), optimize=True)}
    cell.build()

    kmf = scf.hf.RHF(cell)
    # kmf.diis = False
    kmf.init_guess = '1e'
    if hf0:
        kmf.max_cycle = 1
        kmf.get_veff = lambda *args, **kwargs: np.zeros(kmf.get_hcore().shape)
    if smearing:
        kmf = scf.addons.smearing_(kmf, sigma=sigma, method=smearing_method)
    kmf.verbose = 4
    kmf.kernel()
    mo_coeff = kmf.mo_coeff  # (n_ao, n_mo)
    bands = kmf.get_bands(kpts_band=cell.make_kpts([1,1,1]))[0][0]

    # print("pyscf overlap:\n", kmf.get_ovlp())
    # print("pyscf kinetic:\n", kmf.get_ovlp)
    # print("pyscf potential:\n", kmf.get_vnuc())
    # print("pyscf Hcore:\n", kmf.get_hcore())
    print("pyscf e_tot:", (kmf.e_tot - kmf.energy_nuc())*Ry)

    return mo_coeff[:,::-1]+0j, bands[::-1] * Ry 

def pyscf_dft(n, L, rs, sigma, xp, basis='sto-3g', xc='lda,', smearing=False, smearing_method='fermi'):
    """
        Pyscf DFT solver for hydrogen.
    INPUT:
        n: number of protons.
        L: side length of unit cell, unit: rs.
        rs: average atomic spacing, unit: Bohr
        sigma: smearing width, unit: Hartree.
        xp: array of shape (n, dim), position of protons.
        basis: gto basis name, eg:'gth-szv', 'gth-tzv2p', 'gth-qzv3p'.
        xc: exchange correlation functional, eg:'lda', 'pbe', 'b3lyp'.
        smearing: if True, use Fermi-Dirac smearing
            (finite temperature Hartree Fock or thermal Hartree Fock).
    OUTPUT:
        mo_coeff: molecular orbitals coefficients, complex array of shape (n_ao, n_mo).
        bands: energy bands of corresponding molecular orbitals, 
            ranking of energy from low to high. array of shape (n_mo,).
    """
    Ry = 2
    xp *= rs
    cell = gto.Cell()
    cell.unit = 'B'
    cell.a = np.eye(3) * L * rs
    cell.atom = []
    for ie in range(n):
        cell.atom.append(['H', tuple(xp[ie])])
    cell.spin = 0
    cell.basis = {'H':gto.parse(load_as_str('H', basis), optimize=True)}
    cell.build()

    kmf = dft.RKS(cell)
    if smearing:
        kmf = scf.addons.smearing_(kmf, sigma=sigma, method=smearing_method)
    kmf.xc = xc
    # kmf.diis = False
    kmf.verbose = 4
    kmf.kernel()
    mo_coeff = kmf.mo_coeff  # (n_ao, n_mo)
    bands = kmf.get_bands(kpts_band=cell.make_kpts([1,1,1]))[0][0]

    print("pyscf e_elec (Ha):", kmf.e_tot-kmf.energy_nuc())
    # print("pyscf e_elec (Ha):", kmf.energy_elec())
    # print("pyscf e_nuc (Ha):", kmf.energy_nuc())
    
    return mo_coeff+0j, bands * Ry 

if __name__ == "__main__":
    from jax import config
    config.update("jax_enable_x64", True)

    n = 14
    rs = 1.25
    L = (4/3*jnp.pi*n)**(1/3)
    T = 145381.48927717
    basis = 'gth-dzv'
    xc = 'lda'
    smearing = False
    kpt = jnp.array([0, 0, 0])

    beta = 157888.088922572/T # inverse temperature in unit of 1/Ry
    Ry = 2 # Ha/Ry
    sigma = 1/beta/Ry # temperature in Hartree unit
    if n == 4:
        s = jnp.array([[0.82, 0.826, 0.842],
                    [0.24, 0.5, 0.842],
                    [0.1, 0.25, 0.1],
                    [0.5, 0.0, 0.0]])
    elif n == 14:
        s = jnp.array( [[0.222171,  0.53566,  0.579785],
                        [0.779669,  0.464566,  0.37398],
                        [0.213454,  0.97441,  0.753668],
                        [0.390099,  0.731473,  0.403714],
                        [0.756045,  0.902379,  0.214993],
                        [0.330075,  0.00246738,  0.433778],
                        [0.91655,  0.112157,  0.493196],
                        [0.0235088,  0.117353,  0.628366],
                        [0.519162,  0.693898,  0.761833],
                        [0.902309,  0.377603,  0.763541],
                        [0.00753097,  0.690769,  0.97936],
                        [0.534081,  0.856997,  0.996808],
                        [0.907683,  0.0194549,  0.91836],
                        [0.262901,  0.287673,  0.882681]], dtype=jnp.float64)
    else:
        s = jnp.random.uniform(0, 1, (n, 3))
    s *= L
    
    key = jax.random.PRNGKey(42)
    xe = jax.random.uniform(key, minval=0, maxval=L, shape=(n, 3), dtype=jnp.float64)

    state_idx = jnp.array([2,3,2,3])

    print("======= parameters =======")
    print("basis:", basis)


    print("======= hf =======")
    hf = make_hf(n, L, rs, basis)
    mo_coeff, energy = hf(s)
    # print("mo_coeff:\n", mo_coeff)
    print("energy:\n", energy)

    # from orbitals import make_lcao_orbitals
    # lcao_orbitals = make_lcao_orbitals(n, L, rs, basis)
    # slater_up, slater_dn = lcao_orbitals(s, xe, mo_coeff, state_idx)
    # print("slater_up:\n", slater_up)
    # print("slater_dn:\n", slater_dn)

    # print("======= pyscf =======")
    # mo_coeff_hf, energy_hf = pyscf_hf(n, L, rs, sigma, s, basis, smearing=smearing)
    # # print("pyscf mo_coeff:\n", mo_coeff_hf)
    # print("pyscf energy:\n", energy_hf)
