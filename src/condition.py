import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
from functools import partial

@partial(jax.jit, static_argnums=-1)
def Z(Es, beta, N):
    """ For N non-interacting fermions in M single-particle states with energies
    (E_1, E_2, ..., E_M), compute the partition function using the recursive relation
            Z_m(n) = sum_{j=n-1}^{m-1} e^{- beta E_{M-j}} Z_j(n-1)
            (n = 1,2,...,N, m = n,n+1,...,n+M-N),
    where Z_m(n) = sum_{M-m+1 <= i_1 < i_2 < ... < i_n <= M} e^{- beta (E_{i_1} + ... + E_{i_n})}
    denotes the partition function of n non-interacting fermions occupying the "last" m states.

    Args:
        Es (Array, shape: (M,)): single-particle energy spectrum.
        beta (float): inverse temperature.
        N (int): system size.
    
    Returns:
        the log partition function lnZ_{m}(n) (n = 0,1,2,...,N, m = n,n+1,...,n+M-N),
        represented by a 2D array of shape (N+1, M-N+1).
        NOTE: if we align the #states index m, this array will look like a tilted parallelogram:
        Z_M(N)  Z_{M-1}(N)   Z_{M-2}(N)   ... ... ... Z_N(N)
                Z_{M-1}(N-1) Z_{M-2}(N-1) ... ... ... Z_N(N-1) Z_{N-1}(N-1)
                             Z_{M-2}(N-2) ... ... ... Z_N(N-2) Z_{N-1}(N-2) Z_{N-2}(N-2)
                                          ... ... ... ... ... ... ... ... ... ... ... ... ... ...
                                                Z_{M-N+1}(1) Z_{M-N}(1) Z_{M-N-1}(1) ... ... ... Z_2(1) Z_1(1)
                                                             Z_{M-N}(0) Z_{M-N-1}(0) ... ... ... Z_2(0) Z_1(0) Z_0(0)
        This function computes the values in this parallelogram ROW BY ROW.
        'n' and 'm' are in range:
            0 <= n <= N (N is the number of spinless particles)
            n <= m <= M (M is the number of states, Es.shape[0])
        lnZ (Array, shape: (N+1, M-N+1)):
            lnZ[n,m] = lnZ_{m+n+1}(n+1): log partition functions of 'n+1' spinless fermions on 'm+n+1' states.
            'n' and 'm' are the row and column indices of z.
        This function is built by Hao Xie.
    """
    M, = Es.shape
    assert M >= N
    def body_fun(n, lnZ):
        # jnp.roll(Es, n-1)[N-1:M] is another way to write Es[N-n:M-n+1], so as
        # to make the indexing operation jittable.
        es = jnp.roll(Es, n-1)[N-1:M][::-1]
        logit = -beta * es + lnZ[n-1]
        lnZ_n = jax.lax.cumlogsumexp(logit)
        lnZ = lnZ.at[n].set(lnZ_n)
        return lnZ
    # Z_{n, k} = Z_{n+k}(n), where k = 0, 1, ..., M-N.
    lnZ_init = jnp.empty((N+1, M-N+1)).at[0].set(0.)
    lnZ = jax.lax.fori_loop(1, N+1, body_fun, lnZ_init)
    return lnZ

def make_cond_prob(beta, N):
    """ For N non-interacting fermions in M single-particle states, compute
    the conditional log-probability of one given the others, using the relations
            p(i_1)                      = e^{- beta E_{i_1}} Z_{M-i_1}(N-1) / Z_M(N),
            p(i_2|i_1)                  = e^{- beta E_{i_2}} Z_{M-i_2}(N-2) / Z_{M-i_1}(N-1),
            p(i_3|i_1,i_2)              = e^{- beta E_{i_3}} Z_{M-i_3}(N-3) / Z_{M-i_2}(N-2),
            ... ... ... ... ... ... ...
            p(i_{N-1}|i_1,...,i_{N-2})  = e^{- beta E_{i_{N-1}}} Z_{M-i_{N-1}}(1) / Z_{M-i_{N-2}}(2),
            p(i_N|i_1,...,i_{N-1})      = e^{- beta E_{i_N}} Z_{M-i_N}(0) / Z_{M-i_{N-1}}(1).

    Args:
        Es (Array, shape: (M,)): single-particle energy spectrum.
        beta (float): inverse temperature.
        N (int): system size.

    Returns:
        cond_logp_fn (callable): function that take a set of state index
            1<= i_1 < i_2 < ... < i_N <= M, computes the corresponding
            conditional log-probability distributions logp(i_1), logp(i_2|i_1),
            logp(i_3|i_1,i_2), ..., logp(i_N|i_1,i_2,...,i_{N-1}).
            The output is aranged in a (N, M) array of (normalized) logits.
        logp_fn (callable): similar as cond_logp_fn, but proceed to compute the
            joint log-probability of the specifically given set of index.
            The output is a scalar.
        This function is built by Hao Xie.
    """
    rangeN = jnp.arange(N+1)[::-1]

    def _mask(indices, M):
        mask = jnp.tril(jnp.ones((N, M)), k=M-N)
        idx_lb = jnp.concatenate( (jnp.array([-1]), indices[:-1]) )
        mask = jnp.where(jnp.arange(M) > idx_lb[:, None], mask, 0.)
        return mask

    def cond_logp_fn(indices, Es):
        M, = Es.shape
        lnZ = Z(Es, beta, N) # (N+1, M-N+1)
        logits = jnp.full((N, M), -jnp.inf)
        logits = logits.at[jnp.arange(N)[:,None], jnp.arange(N)[:,None] + jnp.arange(M-N+1)].set(lnZ[rangeN[1:], ::-1])
        # print(logits)
        indices_aug = jnp.concatenate([jnp.array([0]), indices[:-1] + 1])
        logits = logits - beta*Es - lnZ[rangeN[:-1], M-indices_aug-rangeN[:-1]][:, None]
        mask = _mask(indices, M)
        logits = jnp.where(mask, logits, -jnp.inf)
        return logits
    
    def single_cond_logp_fn(previous_index, n, Es):
        """ Conditional log-probability of the next fermion when the previous_index 
            is given and n fermions left .
        Args:
            previous_index: int in range [0, M-n)
            n: int
        Return:
            logit: logp(index|previous_index)
        NOTE: this function is not masked, but it's ok.
        """
        M, = Es.shape
        lnZ = Z(Es, beta, N) # (N+1, M-N+1)
        logit = jnp.full(M, -jnp.inf)
        if n == N:
            logit = logit.at[jnp.arange(M-N+1)].set(lnZ[N-1,::-1])
            logit = logit - beta*Es - lnZ[N, M-N]
        else:
            # assert N-n-1 < previous_index < M-n
            logit = logit.at[jnp.arange(N-n,M-n+1)].set(lnZ[n-1,::-1])
            logit = logit - beta*Es - lnZ[n, M-n-1-previous_index]
            # logit = logit.at[jnp.arange(previous_index+1)].set(-jnp.inf)
        return logit

    return cond_logp_fn, single_cond_logp_fn

if __name__ == '__main__':
    from lcao import make_hf
    n, dim = 14, 3
    rs = 1.86
    T = 31250.0
    beta = 157888.088922572/T # inverse temperature in unit of 1/Ry
    basis = 'sto-3g'
    L = (4/3*jnp.pi*n)**(1/3)
    # s = jnp.array( [[0.222171,  0.53566,  0.579785],
    #                 [0.779669,  0.464566,  0.37398],
    #                 [0.213454,  0.97441,  0.753668],
    #                 [0.390099,  0.731473,  0.403714],
    #                 [0.756045,  0.902379,  0.214993],
    #                 [0.330075,  0.00246738,  0.433778],
    #                 [0.91655,  0.112157,  0.493196],
    #                 [0.0235088,  0.117353,  0.628366],
    #                 [0.519162,  0.693898,  0.761833],
    #                 [0.902309,  0.377603,  0.763541],
    #                 [0.00753097,  0.690769,  0.97936],
    #                 [0.534081,  0.856997,  0.996808],
    #                 [0.907683,  0.0194549,  0.91836],
    #                 [0.262901,  0.287673,  0.882681]], dtype=jnp.float64)
    # s *= L
    kpt = jnp.array([0,0,0])

    # hf, _ = make_hf(n, L, rs, basis=basis)
    # _, Es = hf(s, kpt) # Es is in unit of Ry
    Es = jnp.array([1.9173509, 1.5602368, 1.4923979, 1.3089842, 1.1989565, 1.0040728, 0.96070105, 0.9023717, 0.62099594, 0.35862455, 0.22602847, 0.1905089, -0.10513657, -0.8374768])
    num_states = Es.shape[0]
    print("====================== Es*beta ======================\n", Es*beta)
    print("====================== Z ======================\n", Z(Es, beta, n//2))

    from autoregressive import Transformer
    import haiku as hk
    from jax.flatten_util import ravel_pytree

    key = jax.random.PRNGKey(42)
    nlayers = 2
    modelsize = 16
    nheads = 4
    nhidden = 32
    remat = False
    def forward_fn(state):
        model = Transformer(num_states, nlayers, modelsize, nheads, nhidden, remat=remat)
        return model(state)
    van = hk.transform(forward_fn)
    state_idx_dummy = jnp.array([jnp.tile(jnp.arange(n//2, dtype=jnp.float64), 2)]).T
    # state_idx_dummy = sp_indices[-n:].astype(jnp.float64)
    params_van = van.init(key, state_idx_dummy)

    raveled_params_van, _ = ravel_pytree(params_van)
    print("#parameters in the autoregressive model: %d" % raveled_params_van.size)

    from sampler import make_autoregressive_sampler
    sampler, logprob_van_novmap = make_autoregressive_sampler(van, n, num_states, beta)
    logprob_e = jax.vmap(logprob_van_novmap, (None, 0, 0), 0)

    batch = 1
    Es_batch = jnp.tile(Es, (batch, 1))
    state_idx = sampler(params_van, key, batch, Es_batch)
    logp_e = logprob_e(params_van, state_idx, Es_batch) # (B, )
    print(logp_e)

    print("====================== mine ======================")
    cond_logp_fn, single_cond_logp_fn = make_cond_prob(beta, n//2)
    for i in range(n//2):
        temp = jax.vmap(single_cond_logp_fn, (0, None, 0), 0)(state_idx[:, i-1], n//2-i%(n//2), Es_batch)
        print(temp)
    
    cond_logits_up = cond_logp_fn(state_idx[0,:n//2], Es) # (n//2, num_states)
    print("==================== benchmark ====================\n", cond_logits_up)
    