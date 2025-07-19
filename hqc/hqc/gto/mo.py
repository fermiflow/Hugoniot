import jax
import jax.numpy as jnp
from hqc.gto.ao import make_ao

Ry = 2
delta = 1e-8
coeff_sto3g = jnp.array([[3.42525091, 0.15432897],
                        [0.62391373, 0.53532814],
                        [0.16885540, 0.44463454]])
coeff_sto6g = jnp.array([[35.52322122, 0.00916359628],
                        [6.513143725, 0.04936149294],
                        [1.822142904, 0.16853830490],
                        [0.625955266, 0.37056279970],
                        [0.243076747, 0.41649152980],
                        [0.100112428, 0.13033408410]])

def make_hf(n, basis='sto3g'):
    """
        Make Hartree Fock function.
        INPUT:
            n: int, number of hydrogen atoms in unit cell.
            basis: gto basis name, eg:'sto3g'.

        OUTPUT:
            hf: hartree fock function.
            
    """
    
    # coefficients of the basis
    if basis == 'sto3g':
        alpha = coeff_sto3g[:, 0]  # (3,)
        coeff = coeff_sto3g[:, 1:2].T  # (1, 3)
    elif basis == 'sto6g':
        alpha = coeff_sto6g[:, 0]  # (6,)
        coeff = coeff_sto6g[:, 1:2].T  # (1, 6)
    dim_mat = n * coeff.shape[0]
    
    # intermediate varaibles
    sum_alpha = alpha[:, None] + alpha[None, :]
    pro_alpha = jnp.einsum('i,j->ij', alpha, alpha)
    alpha2 = pro_alpha / sum_alpha

    ao = jax.vmap(make_ao(basis), (None, 0), 0)
     
    # erf function
    def f0(x):
        x += delta # should be better
        return jax.lax.erf(x)/x

    def hf(xp):
        """
            Hartree Fock without vee.
            INPUT:
                xp: array of shape (n, dim), position of protons.

            OUTPUT:
                (energy, mo_coeff)
                energy unit: Rydberg.
                mo_coeff: coefficient of molecular orbitals on atomic orbitals, shape:(n_ao, n_mo).
        """
        assert n == xp.shape[0]

        # overlap
        Rmn = jnp.sum(jnp.square(xp[:, None, :] - xp[None, :, :]), axis=2) # (n, n)
        _ovlp = 2**1.5*jnp.einsum('pi,qj,ij,ijmn->mpinqj', coeff, coeff, jnp.power(pro_alpha, 0.75)/jnp.power(sum_alpha, 1.5), 
                jnp.exp(-jnp.einsum('ij,mn->ijmn', alpha2, Rmn)))
        ovlp = jnp.reshape(jnp.einsum('mpinqj->mpnq', _ovlp), (dim_mat, dim_mat))

        # kinetic
        K = jnp.reshape(jnp.einsum('mpinqj,ij,ijmn->mpnq', _ovlp, alpha2, 3-2*jnp.einsum('ij,mn->ijmn', alpha2, Rmn)), (dim_mat, dim_mat))
        
        # potential
        rminj = (xp[:,None,None,None,:]*alpha[None,:,None,None,None]+xp[None,None,:,None,:]*alpha[None,None,None,:,None])/sum_alpha[None, :, None, :, None]
        x = jnp.einsum('ij,vminj->vminj', jnp.sqrt(sum_alpha), jnp.linalg.norm(xp[:,None,None,None,None,:]-rminj[None,...], axis=5))
        V = -jnp.reshape(jnp.einsum('mpinqj,ij,vminj->mpnq', _ovlp, jnp.sqrt(sum_alpha), f0(x)), (dim_mat, dim_mat))

        # core Hamiltonian
        hcore = K + V

        # diagonalization
        w, u = jnp.linalg.eigh(ovlp)
        v = jnp.dot(u, jnp.diag(w**(-0.5)))
        f1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), hcore, v)
        w1, c1 = jnp.linalg.eigh(f1)
        mo_coeff = jnp.dot(v, c1) # (n_ao, n_mo)
        e = 2 * jnp.sum(w1[0:n//2])

        return Ry * e, mo_coeff # this E is without vpp

    def logpsi(mo_coeff, xp, xe):
        """
            log of hartree fock wavefunction.
            INPUT:
                mo_coeff: coefficient of molecular orbitals on atomic orbitals, shape:(n_ao, n_mo).
                xp: array of shape (n, dim), position of protons.
                xe: array of shape (n, dim), position of electrons.

            OUTPUT:
                logpsi

        """
        assert xe.shape[0] == n

        mo_up = mo_dn = mo_coeff[..., 0:n//2] # (n_ao, n_up/n_dn)
        ao_all = ao(xp, xe) # (n, n_ao)
        ao_up = ao_all[:n//2] # (n_up, n_ao)
        ao_dn = ao_all[n//2:] # (n_dn, n_ao)
        slater_up = jnp.dot(ao_up, mo_up) # (n_up, n_up)
        slater_dn = jnp.dot(ao_dn, mo_dn) # (n_dn, n_dn)
        sign_up, logabsdet_up = jnp.linalg.slogdet(slater_up)
        sign_dn, logabsdet_dn = jnp.linalg.slogdet(slater_dn)
        sign = sign_up * sign_dn
        logabsdet = logabsdet_up + logabsdet_dn
        return jnp.log(sign) + logabsdet

    return hf, logpsi


if __name__ == "__main__":
    import numpy as np
    from pyscf import gto, scf
    from jax.config import config   
    config.update("jax_enable_x64", True)
    
    Ry = 2
    n, dim = 4, 3
    rs = 1.25
    L = (4/3*jnp.pi*n)**(1/3)*rs
    key = jax.random.PRNGKey(42)
    xp = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
    key = jax.random.PRNGKey(43)
    xe = jax.random.uniform(key, (n, dim), minval=0., maxval=L)

    hf, logpsi = make_hf()
    E, mo_coeff = hf(xp)
    print("energy:", E)
    print("mo_coeff:\n", mo_coeff)
    # print(logpsi(mo_coeff, xe))

    mol = gto.Mole()
    mol.unit = 'B'
    for i in range(n):
        mol.atom.append(['H', tuple(xp[i])])
    mol.basis = 'sto3g'
    mol.build()
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.max_cycle = 1
    mf.get_veff = lambda *args: np.zeros(mf.get_hcore().shape)
    mf.kernel()
    print("energy:", Ry*(mf.e_tot-mf.energy_nuc()))
    print("mo_coeff:\n", mf.mo_coeff)
