import jax
import jax.numpy as jnp
from functools import partial

from src.utils import shard, replicate

def make_loss(log_prob, Es, beta):
    
    def loss_fn(params, state_idx):

        logp = log_prob(params, state_idx)
        E = Es[state_idx].sum(axis=-1)
        F = jax.lax.stop_gradient(logp / beta + E)

        E_mean = jax.lax.pmean(jnp.mean(E), axis_name='p')
        F_mean = jax.lax.pmean(jnp.mean(F), axis_name='p')
        S_mean = jax.lax.pmean(jnp.mean(-logp), axis_name='p')

        F_var = jax.lax.pmean(jnp.mean((F_mean - F)**2), axis_name='p')
        E_var = jax.lax.pmean(jnp.mean((E_mean - E)**2), axis_name='p')
        S_var = jax.lax.pmean(jnp.mean((S_mean + logp)**2), axis_name='p')

        gradF = jnp.mean(logp * (F - F_mean))

        auxiliary_data = {"E_mean": E_mean, "E_var": E_var,
                          "F_mean": F_mean, "F_var": F_var,
                          "S_mean": S_mean, "S_var": S_var,
                         }

        return gradF, auxiliary_data

    return loss_fn

def pretrain(van, params_van,
             log_prob_novmap, sampler, 
             n, dim, beta, L, Emax,
             path, key,
             lr, sr, damping, max_norm,
             batch, epoch=10000):

    log_prob = jax.vmap(log_prob_novmap, (None, 0), 0)

    from orbitals import sp_orbitals
    sp_indices, Es = sp_orbitals(dim, Emax)
    sp_indices = jnp.array(sp_indices[::-1])
    Es = (2*jnp.pi/L)**2 * jnp.array(Es[::-1])

    from mpmath import mpf, mp
    from analytic import Z_E
    F, E, S = Z_E(n//2, dim, mpf(str(L)), mpf(str(beta)), Emax)
    print("Analytic results for the thermodynamic quantities: "
            "F: %s, E: %s, S: %s" % (mp.nstr(2*F), mp.nstr(2*E), mp.nstr(2*S)))
    
    Fexact = float(2*F)

    loss_fn = make_loss(log_prob, Es, beta)

    num_devices = jax.device_count()
    print ('Number of devices in pretraining:', num_devices)
    print ('Batchsize in pretraining:', batch)

    if batch %num_devices !=0:
        raise ValueError('Batch size must be divisible by number of devices, '
                         'got batch size {} for {} devices.'.format(
                          batch, num_devices))
    import optax
    if sr:
        from sampler import make_classical_score
        score_fn = make_classical_score(log_prob_novmap)
        from sr import fisher_sr
        optimizer = fisher_sr(score_fn, damping, max_norm)
        print("Optimizer fisher_sr: damping = %.5f, max_norm = %.5f." % (damping, max_norm))
    else:
        optimizer = optax.adam(lr)
        print("Optimizer adam: lr = %.3f." % lr)

    @partial(jax.pmap, axis_name='p')
    def update(params_van, opt_state, key):
        key, subkey = jax.random.split(key)
        state_idx = sampler(params_van, subkey, batch//num_devices)

        grads, aux = jax.grad(loss_fn, argnums=0, has_aux=True)(params_van, state_idx)
        grads = jax.lax.pmean(grads, axis_name="p")
        updates, opt_state = optimizer.update(grads, opt_state,
                                params=(params_van, state_idx) if sr else None)
        params_van = optax.apply_updates(params_van, updates)

        return params_van, opt_state, key, aux

    import os
    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w", buffering=1, newline="\n")

    params_van = replicate(params_van, num_devices)
    opt_state = jax.pmap(optimizer.init)(params_van)
    key, *subkeys = jax.random.split(key, num_devices+1)
    sharded_key = shard(jnp.stack(subkeys))

    for i in range(1, epoch+1):
        params_van, opt_state, sharded_key, aux = update(params_van, opt_state, sharded_key)
        aux = jax.tree_map(lambda x: x[0], aux)

        E, E_var, F, F_var, S, S_var = aux["E_mean"], aux["E_var"], \
                                       aux["F_mean"], aux["F_var"], \
                                       aux["S_mean"], aux["S_var"]

        E_err = jnp.sqrt(E_var / batch)
        F_err = jnp.sqrt(F_var / batch)
        S_err = jnp.sqrt(S_var / batch)

        print("iter: %04d" % i,
                "F:", F, "F_err:", E_err,
                "E:", E, "E_err:", F_err,
                "S:", S, "S_err:", S_err)

        f.write( ("%6d" + "  %.6f"*6 + "\n") % (i, F, F_err,
                                                   E, E_err,
                                                   S, S_err) )
    
        if jnp.abs((F-Fexact)) < max(F_err, 1E-6): break 
    return jax.tree_map(lambda x: x[0], params_van)

