import jax
import jax.numpy as jnp
import numpy as np
import itertools
from hqc.pbc.lcao import make_lcao
from test_slater import vmc_slater_hf

from test_pbc_lcao import pyscf_dft, pyscf_hf
jax.config.update("jax_enable_x64", True)

def make_atoms(
    ncopy = [2, 2, 2],
):
  """make atom pyscf style coords
  """
  lattice = (4/3*np.pi*2)**(1/3) # lattice constant for bcc lattice, length unit rs*a0
  atom_strs = []
  for ii,jj,kk in itertools.product(range(ncopy[0]), range(ncopy[1]), range(ncopy[2])):
    xx = ii * lattice
    yy = jj * lattice
    zz = kk * lattice
    atom_strs.append([xx, yy, zz])
    atom_strs.append([xx+0.5*lattice, yy+0.5*lattice, zz+0.5*lattice])
  return np.array(atom_strs)

def test_bcc_solid_hf():
    dim = 3
    rs = 1.31
    basis_set = ['gth-szv'] # , 'gth-dzv', 'gth-dzvp']
    rcut = 24
    grid_length = 0.12
    dft = False
    xc = "lda,vwn"
    smearing = False
    sigma = 0.0 # smearing parameter 
    perturbation = 0.0 # perturbation strength for atom position
    tol = 1e-10
    max_cycle = 1000
    gamma = True
    diis = False
    diis_space = 8
    diis_start_cycle = 1
    diis_damp = 0.0

    # bcc crystal
    xp = make_atoms([2, 2, 2]) 
    n = xp.shape[0]
    L = (4/3*jnp.pi*n)**(1/3)

    key = jax.random.PRNGKey(42)
    xp += jax.random.normal(key, (n, dim)) * perturbation
    xp = xp - L * jnp.floor(xp/L)

    key = jax.random.PRNGKey(43)
    if gamma:
        kpt = jnp.zeros(3)
    else:
        kpt = jax.random.uniform(key, (3,), minval=-jnp.pi/L/rs, maxval=jnp.pi/L/rs)

    # uniform
    # xp = jax.random.uniform(key, (n, dim), minval=0., maxval=L)

    print("\n============= begin test =============")
    print("n:", n)
    print("rs:", rs)
    print("L:", L)
    print("basis_set:", basis_set)
    print("rcut:", rcut)
    print("grid_length:", grid_length)
    print("tol:", tol)
    print("max_cycle:", max_cycle)
    print("diis:", diis)
    print("diis_space:", diis_space)
    print("diis_start_cycle:", diis_start_cycle)
    print("diis_damp:", diis_damp)
    print("hf:", not dft)
    if dft:
        print("xc:", xc)
    print("smearing:", smearing)
    if smearing:
        print("smearing sigma:", sigma)
    print("gamma:", gamma)
    if not gamma:
        print("kpt:", kpt)
    print("perturbation:", perturbation)
    print("xp:\n", xp)

    for basis in basis_set:
        print("\n==========", basis, "==========")
        if dft: 
            mo_coeff_pyscf, bands_pyscf, E_pyscf = pyscf_dft(n, L, rs, sigma, xp, basis, kpt, xc=xc, smearing=smearing)
        else:
            mo_coeff_pyscf, bands_pyscf, E_pyscf = pyscf_hf(n, L, rs, sigma, xp, basis, kpt, smearing=smearing)

        lcao = make_lcao(n, L, rs, basis, grid_length=grid_length, dft=dft, smearing=smearing, smearing_sigma=sigma, 
                         diis=diis, diis_space=diis_space, diis_start_cycle=diis_start_cycle, diis_damp=diis_damp, 
                         tol=tol, max_cycle=max_cycle, gamma=gamma)
        if gamma:
            mo_coeff, bands, E = lcao(xp)
        else:
            mo_coeff, bands, E = lcao(xp, kpt)

        print ("E", E)
        print ("E_pyscf", E_pyscf)

        mo_coeff = mo_coeff @ jnp.diag(jnp.sign(mo_coeff[0]).conjugate())
        mo_coeff_pyscf = mo_coeff_pyscf @ jnp.diag(jnp.sign(mo_coeff_pyscf[0]).conjugate())

        print ("coef diff", np.abs(mo_coeff - mo_coeff_pyscf).max())
        print ("band diff", np.abs(bands - bands_pyscf).max())

        print("bands:\n", bands)
        print("bands_pyscf:\n", bands_pyscf)

        dm = mo_coeff[:, :n//2] @ (mo_coeff[:, :n//2].T)
        dm_pyscf = mo_coeff_pyscf[:, :n//2] @ (mo_coeff_pyscf[:, :n//2].T)

        print ("dm diff", np.abs(dm - dm_pyscf).max())
        
        #np.set_printoptions(suppress=True)
        #print("dm:\n", dm)
        #print("dm_pyscf:\n", dm_pyscf)
        #print("diff:\n", dm - dm_pyscf)

def test_bcc_solid_hf_kpt():
    dim = 3
    rs = 1.31
    basis_set = ['gth-szv'] # , 'gth-dzv', 'gth-dzvp']
    rcut = 24
    grid_length = 0.12
    dft = False
    xc = "lda,vwn"
    smearing = False
    sigma = 0.0 # smearing parameter 
    perturbation = 0.0 # perturbation strength for atom position
    tol = 1e-7
    max_cycle = 100
    gamma = False
    diis = True
    diis_space = 8
    diis_start_cycle = 1
    diis_damp = 0.0

    # bcc crystal
    xp = make_atoms([2, 2, 2]) 
    n = xp.shape[0]
    L = (4/3*jnp.pi*n)**(1/3)

    key = jax.random.PRNGKey(42)
    xp += jax.random.normal(key, (n, dim)) * perturbation
    xp = xp - L * jnp.floor(xp/L)

    key = jax.random.PRNGKey(43)
    if gamma:
        kpt = jnp.zeros(3)
    else:
        kpt = jax.random.uniform(key, (3,), minval=-jnp.pi/L/rs, maxval=jnp.pi/L/rs)

    # Baldereschi
    kpt = jnp.ones(3)*jnp.pi/L/rs/2

    # uniform
    # xp = jax.random.uniform(key, (n, dim), minval=0., maxval=L)

    print("\n============= begin test =============")
    print("n:", n)
    print("rs:", rs)
    print("L:", L)
    print("basis_set:", basis_set)
    print("rcut:", rcut)
    print("grid_length:", grid_length)
    print("tol:", tol)
    print("max_cycle:", max_cycle)
    print("diis:", diis)
    print("diis_space:", diis_space)
    print("diis_start_cycle:", diis_start_cycle)
    print("diis_damp:", diis_damp)
    print("hf:", not dft)
    if dft:
        print("xc:", xc)
    print("smearing:", smearing)
    if smearing:
        print("smearing sigma:", sigma)
    print("gamma:", gamma)
    if not gamma:
        print("kpt:", kpt)
    print("perturbation:", perturbation)
    print("xp:\n", xp)

    for basis in basis_set:
        print("\n==========", basis, "==========")
        if dft: 
            mo_coeff_pyscf, bands_pyscf, E_pyscf = pyscf_dft(n, L, rs, sigma, xp, basis, kpt, xc=xc, smearing=smearing)
        else:
            mo_coeff_pyscf, bands_pyscf, E_pyscf = pyscf_hf(n, L, rs, sigma, xp, basis, kpt, smearing=smearing)

        lcao = make_lcao(n, L, rs, basis, grid_length=grid_length, dft=dft, smearing=smearing, smearing_sigma=sigma, 
                         diis=diis, diis_space=diis_space, diis_start_cycle=diis_start_cycle, diis_damp=diis_damp, 
                         tol=tol, max_cycle=max_cycle, gamma=gamma)
        if gamma:
            mo_coeff, bands, E = lcao(xp)
        else:
            mo_coeff, bands, E = lcao(xp, kpt)

        print ("E", E)
        print ("E_pyscf", E_pyscf)

        mo_coeff = mo_coeff @ jnp.diag(jnp.sign(mo_coeff[0]).conjugate())
        mo_coeff_pyscf = mo_coeff_pyscf @ jnp.diag(jnp.sign(mo_coeff_pyscf[0]).conjugate())

        print ("coef diff", np.abs(mo_coeff - mo_coeff_pyscf).max())
        print ("band diff", np.abs(bands - bands_pyscf).max())

        print("bands:\n", bands)
        print("bands_pyscf:\n", bands_pyscf)

        dm = mo_coeff[:, :n//2] @ (mo_coeff[:, :n//2].T)
        dm_pyscf = mo_coeff_pyscf[:, :n//2] @ (mo_coeff_pyscf[:, :n//2].T)

        print ("dm diff", np.abs(dm - dm_pyscf).max())
        
        #np.set_printoptions(suppress=True)
        #print("dm:\n", dm)
        #print("dm_pyscf:\n", dm_pyscf)
        #print("diff:\n", dm - dm_pyscf)

def test_bcc_solid_hf_mcmc():
    dim = 3
    rs = 1.31
    basis = 'gth-dzv'
    rcut = 24
    grid_length = 0.12
    smearing = False
    sigma = 0.0 # smearing parameter 
    perturbation = 0.1 # perturbation strength for atom position
    gamma = True
    batchsize = 256
    mc_steps = 400
    mc_width = 0.06
    therm_steps = 5
    sample_steps = 10

    # bcc crystal
    xp = make_atoms([2, 2, 2])
    n = xp.shape[0]
    L = (4/3*jnp.pi*n)**(1/3)

    key = jax.random.PRNGKey(42)
    xp += jax.random.normal(key, (n, dim)) * perturbation
    xp = xp - L * jnp.floor(xp/L)

    key = jax.random.PRNGKey(43)
    kpt = jax.random.uniform(key, (3,), minval=-jnp.pi/L/rs, maxval=jnp.pi/L/rs)

    # uniform
    #xp = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
    '''
    rs = 2.0
    n = 14
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

    L = (4/3*jnp.pi*n)**(1/3)
    xp = s * L
    '''

    print("\n============= begin test =============")
    print("n:", n)
    print("rs:", rs)
    print("L:", L)
    print("basis:", basis)
    print("rcut:", rcut)
    print("grid_length:", grid_length)
    print("smearing:", smearing)
    print("perturbation:", perturbation)
    print("xp:\n", xp)
    print("gamma:", gamma)
    if not gamma:
        print("kpt:", kpt)
    print("batchsize:", batchsize)
    print("mc_steps:", mc_steps)
    print("mc_width:", mc_width)
    print("therm_steps:", therm_steps)
    print("sample_steps:", sample_steps)
    
    vmc_slater_hf(xp, rs, basis, kpt, rcut, grid_length, smearing, sigma, gamma,
                  batchsize, mc_steps, mc_width, therm_steps, sample_steps)

def test_bcc_solid_hf_mcmc_kpt():
    dim = 3
    rs = 1.31
    basis = 'gth-dzv'
    rcut = 24
    grid_length = 0.12
    smearing = False
    sigma = 0.0 # smearing parameter 
    perturbation = 0.1 # perturbation strength for atom position
    gamma = False
    batchsize = 256
    mc_steps = 400
    mc_width = 0.06
    therm_steps = 5
    sample_steps = 10

    # bcc crystal
    xp = make_atoms([2, 2, 2])
    n = xp.shape[0]
    L = (4/3*jnp.pi*n)**(1/3)

    key = jax.random.PRNGKey(42)
    xp += jax.random.normal(key, (n, dim)) * perturbation
    xp = xp - L * jnp.floor(xp/L)

    key = jax.random.PRNGKey(43)
    kpt = jax.random.uniform(key, (3,), minval=-jnp.pi/L/rs, maxval=jnp.pi/L/rs)

    print("\n============= begin test =============")
    print("n:", n)
    print("rs:", rs)
    print("L:", L)
    print("basis:", basis)
    print("rcut:", rcut)
    print("grid_length:", grid_length)
    print("smearing:", smearing)
    print("perturbation:", perturbation)
    print("xp:\n", xp)
    print("gamma:", gamma)
    if not gamma:
        print("kpt:", kpt)
    print("batchsize:", batchsize)
    print("mc_steps:", mc_steps)
    print("mc_width:", mc_width)
    print("therm_steps:", therm_steps)
    print("sample_steps:", sample_steps)
    
    vmc_slater_hf(xp, rs, basis, kpt, rcut, grid_length, smearing, sigma, gamma,
                  batchsize, mc_steps, mc_width, therm_steps, sample_steps)

if __name__=='__main__':
    # test_bcc_solid_hf()
    test_bcc_solid_hf_kpt()
    # test_bcc_solid_hf_mcmc()
    # test_bcc_solid_hf_mcmc_kpt()
