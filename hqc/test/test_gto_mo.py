from config import *
from pyscf import gto, scf
from hqc.gto.mo import make_hf

def zerovee(xp, basis):
    Ry = 2
    n = xp.shape[0]
    mol = gto.Mole()
    mol.unit = 'B'
    for i in range(n):
        mol.atom.append(['H', tuple(xp[i])])
    mol.basis = basis
    mol.build()
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.max_cycle = 1
    mf.get_veff = lambda *args: np.zeros(mf.get_hcore().shape)
    mf.kernel()

    mo_up, mo_dn = mf.mo_coeff[..., 0:n//2], mf.mo_coeff[..., 0:n//2]

    def logpsi(xe):
        ao_all = mol.eval_ao("GTOval_sph", xe) # (n, n_ao)
        ao_up = ao_all[:n//2] # (n_up, n_ao)
        ao_dn = ao_all[n//2:] # (n_dn, n_ao)
        slater_up = jnp.dot(ao_up, mo_up) # (n_up, n_up)
        slater_dn = jnp.dot(ao_dn, mo_dn) # (n_dn, n_dn)
        sign_up, logabsdet_up = jnp.linalg.slogdet(slater_up)
        sign_dn, logabsdet_dn = jnp.linalg.slogdet(slater_dn)
        sign = sign_up * sign_dn
        logabsdet = logabsdet_up + logabsdet_dn
        return jnp.log(sign) + logabsdet

    return Ry*(mf.e_tot-mf.energy_nuc()), mf.mo_coeff, logpsi

def test_gto_mo():
    Ry = 2
    n, dim = 14, 3
    rs = 1.25
    L = (4/3*jnp.pi*n)**(1/3)*rs
    basis_set = ['sto3g', 'sto6g']
    key = jax.random.PRNGKey(42)
    xp = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
    key = jax.random.PRNGKey(43)
    xe = jax.random.uniform(key, (n, dim), minval=0., maxval=L)

    basis_set = ['sto3g', 'sto6g']
    for basis in basis_set:

        hf, logpsi = make_hf(n, basis)
        E, mo_coeff = hf(xp)
        pyscf_E, pyscf_mo_coeff, pyscf_logpsi = zerovee(xp, basis)

        # test energy
        assert np.allclose(E, pyscf_E)

        # test mo_coeff
        # assert np.allclose(mo_coeff, pyscf_mo_coeff)

        # test psi
        assert np.allclose(logpsi(mo_coeff, xp, xe), pyscf_logpsi(xe))

        # hf jit, vmap, test
        jax.jit(hf)(xp)
        xp2 = jnp.concatenate([xp, xp]).reshape(2, n, dim)
        jax.vmap(hf)(xp2)

        # logpsi jit, vmap, grad test
        jax.jit(jax.grad(logpsi))(mo_coeff, xp, xe)
        xe2 = jnp.concatenate([xe, xe]).reshape(2, n, dim)
        jax.vmap(logpsi, (None, None, 0), 0)(mo_coeff, xp, xe2)

