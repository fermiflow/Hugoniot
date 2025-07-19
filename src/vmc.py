import jax
import jax.numpy as jnp

from functools import partial

from src.mcmc import mcmc, mala
from src.potential import potential_energy

@partial(jax.pmap, axis_name="p",
                   in_axes=(0,
                            None, 0, 
                            None, None, 0, 0, 
                            None, None, 0, 0, 
                            None, None, 
                            None, None, None, None, None),
                   static_broadcasted_argnums=(1, 3, 4, 7, 8, 16))
def sample_s_and_x(key,
                   sampler, params_van, 
                   logprob, force_fn_p, s, params_flow,
                   logpsi2, force_fn_e, x, params_wfn,
                   mc_proton_steps, mc_electron_steps,
                   mc_proton_width, mc_electron_width, L, lcao, kpt):
    """
        Generate new proton samples of shape (batchsize, n, dim), as well as coordinate sample
    of shape (batchsize, n, dim), from the sample of last optimization step.
    """
    key, key_momenta, key_proton, key_electron = jax.random.split(key, 4)

    batchsize, dim = s.shape[0], s.shape[2]
    
    # proton move
    # s, _, _, _, proton_acc_rate = mala(lambda s: logprob(params_flow, s), 
    #                                    lambda s: force_fn_p(params_flow, s), 
    #                                    s, key_proton, mc_proton_steps, mc_proton_width)
    s, proton_acc_rate = mcmc(lambda s: logprob(params_flow, s),
                              s, key_proton, mc_proton_steps, mc_proton_width)
    s -= L * jnp.floor(s/L)

    # solve HF on new s
    mo_coeff, bands = lcao(s)

    # sample hf state
    state_idx = sampler(params_van, key_momenta, batchsize, bands)
    
    # electron move
    x, electron_acc_rate = mcmc(lambda x: logpsi2(x, params_wfn, s, state_idx, mo_coeff), 
                                x, key_electron, mc_electron_steps, mc_electron_width)
    x -= L * jnp.floor(x/L)

    return key, state_idx, mo_coeff, bands, s, x, proton_acc_rate, electron_acc_rate

####################################################################################

def make_loss(logprob_p, logprob_e, logpsi, logpsi_grad_laplacian, kappa, G, L, rs, Vconst, beta, clip_factor_flow, clip_factor_van, clip_factor_wfn):

    def observable_and_lossfn(params_flow, params_van, params_wfn, state_idx, mo_coeff, bands, s, x, key):
        '''
            s: (B, n, dim)
            x: (B, n, dim)
        '''

        logp_e = logprob_e(params_van, state_idx, bands) # (B, )
        logp_p = logprob_p(params_flow, s) # (B,)

        print("logp_e.shape", logp_e.shape)
        print("logp_p.shape", logp_p.shape)
        grad, laplacian = logpsi_grad_laplacian(x, params_wfn, s, state_idx, mo_coeff, key)
        print("grad.shape:", grad.shape)
        print("laplacian.shape:", laplacian.shape)

        kinetic = -laplacian - (grad**2).sum(axis=(-2, -1))

        v_pp, v_ep, v_ee = potential_energy(jnp.concatenate([s, x], axis=1), kappa, G, L, rs) 
        v_pp += Vconst
        v_ee += Vconst
        
        Eloc = kinetic + (v_ep + v_ee) # (B, ) 
        print("Eloc.shape", Eloc.shape)
        Floc = (logp_p + logp_e)*rs**2/ beta + Eloc.real + v_pp # (B,)

        #pressure in Gpa using viral theorem 
        # 1 Ry/Bohr^3 = 14710.513242194795 GPa 
        #http://greif.geo.berkeley.edu/~driver/conversions.html
        P = (2*kinetic.real + (v_pp + v_ep + v_ee) )/(3*(L*rs)**3)* 14710.513242194795

        EPloc = Eloc * P

        K, K2, Vpp, Vpp2, Vep, Vep2, Vee, Vee2, \
        P, P2, E, E2, EP, F, F2, Sp, Sp2, Se, Se2 = \
        jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="p"), 
                     (kinetic.real.mean(), (kinetic.real**2).mean(),
                      v_pp.mean(), (v_pp**2).mean(), 
                      v_ep.mean(), (v_ep**2).mean(), 
                      v_ee.mean(), (v_ee**2).mean(), 
                      P.mean(), (P**2).mean(), 
                      Eloc.real.mean(), (Eloc.real**2).mean(),
                      EPloc.real.mean(),
                      Floc.mean(), (Floc**2).mean(),
                      -logp_p.mean(), (logp_p**2).mean(), 
                      -logp_e.mean(), (logp_e**2).mean()
                     )
                    )

        observable = {"K": K, "K2": K2,
                      "Vpp": Vpp, "Vpp2": Vpp2,
                      "Vep": Vep, "Vep2": Vep2,
                      "Vee": Vee, "Vee2": Vee2,
                      "P": P, "P2": P2,
                      "E": E, "E2": E2,
                      "EP": EP,
                      "F": F, "F2": F2,
                      "Sp": Sp, "Sp2": Sp2,
                      "Se": Se, "Se2": Se2
                      }

        def flow_lossfn(params_flow):
            logp_states = logprob_p(params_flow, s) # (B,)

            tv = jax.lax.pmean(jnp.abs(Floc - F).mean(), axis_name="p")
            Floc_clipped = jnp.clip(Floc, F - clip_factor_flow*tv, F + clip_factor_flow*tv)
            gradF = (logp_states * (Floc_clipped - F)).mean()
            return gradF

        def van_lossfn(params_van):
            logp_states = logprob_e(params_van, state_idx, bands) # (B, )

            tv = jax.lax.pmean(jnp.abs(Floc - F).mean(), axis_name="p")
            Floc_clipped = jnp.clip(Floc, F - clip_factor_van*tv, F + clip_factor_van*tv)
            gradF = (logp_states * (Floc_clipped - F)).mean()
            return gradF

        def wfn_lossfn(params_wfn):
            logpsix = logpsi(x, params_wfn, s, state_idx, mo_coeff) # (B,)

            tv = jax.lax.pmean(jnp.abs(Eloc - E).mean(), axis_name="p")
            Eloc_clipped = jnp.clip(Eloc, E - clip_factor_wfn*tv, E + clip_factor_wfn*tv)
            gradF = 2 * (logpsix * (Eloc_clipped - E).conj()).real.mean()
            return gradF

        return observable, flow_lossfn, van_lossfn, wfn_lossfn

    return observable_and_lossfn

def make_loss_sr(logprob_p, logprob_e, logpsi, logpsi_grad_laplacian, kappa, G, L, rs, Vconst, beta, clip_factor):

    def observable_and_lossfn(params_flow, params_van, params_wfn, state_idx, mo_coeff, bands, s, x, key):
        '''
            s: (B, n, dim)
            x: (B, n, dim)
        '''

        logp_e = logprob_e(params_van, state_idx, bands) # (B, )
        logp_p = logprob_p(params_flow, s) # (B,)

        print("logp_e.shape", logp_e.shape)
        print("logp_p.shape", logp_p.shape)
        grad, laplacian = logpsi_grad_laplacian(x, params_wfn, s, state_idx, mo_coeff, key)
        print("grad.shape:", grad.shape)
        print("laplacian.shape:", laplacian.shape)

        kinetic = -laplacian - (grad**2).sum(axis=(-2, -1))

        v_pp, v_ep, v_ee = potential_energy(jnp.concatenate([s, x], axis=1), kappa, G, L, rs) 
        v_pp += Vconst
        v_ee += Vconst
        
        Eloc = kinetic + (v_ep + v_ee) # (B, ) 
        print("Eloc.shape", Eloc.shape)
        Floc = (logp_p + logp_e)*rs**2/ beta + Eloc.real + v_pp # (B,)
        
        #pressure in Gpa using viral theorem 
        # 1 Ry/Bohr^3 = 14710.513242194795 GPa 
        #http://greif.geo.berkeley.edu/~driver/conversions.html
        P = (2*kinetic.real + (v_pp + v_ep + v_ee) )/(3*(L*rs)**3)* 14710.513242194795

        EPloc = Eloc * P

        K, K2, Vpp, Vpp2, Vep, Vep2, Vee, Vee2, \
        P, P2, E, E2, EP, F, F2, Sp, Sp2, Se, Se2 = \
        jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="p"), 
                     (kinetic.real.mean(), (kinetic.real**2).mean(),
                      v_pp.mean(), (v_pp**2).mean(), 
                      v_ep.mean(), (v_ep**2).mean(), 
                      v_ee.mean(), (v_ee**2).mean(), 
                      P.mean(), (P**2).mean(), 
                      Eloc.real.mean(), (Eloc.real**2).mean(),
                      EPloc.real.mean(),
                      Floc.mean(), (Floc**2).mean(),
                      -logp_p.mean(), (logp_p**2).mean(), 
                      -logp_e.mean(), (logp_e**2).mean()
                     )
                    )

        observable = {"K": K, "K2": K2,
                      "Vpp": Vpp, "Vpp2": Vpp2,
                      "Vep": Vep, "Vep2": Vep2,
                      "Vee": Vee, "Vee2": Vee2,
                      "P": P, "P2": P2,
                      "E": E, "E2": E2,
                      "EP": EP,
                      "F": F, "F2": F2,
                      "Sp": Sp, "Sp2": Sp2,
                      "Se": Se, "Se2": Se2
                      }

        def flow_lossfn(params_flow):
            logp_states = logprob_p(params_flow, s) # (B,)

            tv = jax.lax.pmean(jnp.abs(Floc - F).mean(), axis_name="p")
            Floc_clipped = jnp.clip(Floc, F - clip_factor*tv, F + clip_factor*tv)
            gradF = (logp_states * Floc_clipped).mean()
            flow_score = logp_states.mean()
            return gradF, flow_score

        def van_lossfn(params_van):
            logp_states = logprob_e(params_van, state_idx, bands) # (B, )

            tv = jax.lax.pmean(jnp.abs(Floc - F).mean(), axis_name="p")
            Floc_clipped = jnp.clip(Floc, F - clip_factor*tv, F + clip_factor*tv)
            gradF = (logp_states * Floc_clipped).mean()
            van_score = logp_states.mean()
            return gradF, van_score

        def wfn_lossfn(params_wfn):
            logpsix = logpsi(x, params_wfn, s, state_idx, mo_coeff) # (B,)

            tv = jax.lax.pmean(jnp.abs(Eloc - E).mean(), axis_name="p")
            Eloc_clipped = jnp.clip(Eloc, E - clip_factor*tv, E + clip_factor*tv)
            gradF = 2 * (logpsix * Eloc_clipped.conj()).real.mean()
            wfn_score = 2 * logpsix.real.mean()
            return gradF, wfn_score

        return observable, flow_lossfn, van_lossfn, wfn_lossfn

    return observable_and_lossfn


@partial(jax.pmap, axis_name="p",
                   in_axes=(0, None, None, 0, 0, None, None, None),
                   static_broadcasted_argnums=(1, 2))
def sample_s(key, logprob, force_fn_p, s, params_flow,
             mc_proton_steps, mc_proton_width, L):
    """
        Generate new proton samples of shape (batchsize, n, dim).
        Input:
            key: PRNG key
            logprob: Callable, log probability function
            force_fn_p: Callable, force function
            s: jnp.ndarray, proton samples of shape (batchsize, n, dim)
            params_flow: Dict, flow parameters
            mc_proton_steps: int, number of MCMC steps
            mc_proton_width: float, width of MCMC step
            L: float, box size in rs unit.
        Output:
            key_new: new PRNG key
            s: jnp.ndarray, new proton samples of shape (batchsize, n, dim)
            proton_acc_rate: float, acceptance rate of proton move
        Note:
            coordinates in 's' are in range [0, L)
    """
    batchsize, dim = s.shape[0], s.shape[2]
    key_new, key_proton = jax.random.split(key)
    # proton move
    # s, _, _, _, proton_acc_rate = mala(lambda s: logprob(params_flow, s), 
    #                                    lambda s: force_fn_p(params_flow, s), 
    #                                    s, key_proton, mc_proton_steps, mc_proton_width)
    s, proton_acc_rate = mcmc(lambda s: logprob(params_flow, s),
                              s, key_proton, mc_proton_steps, mc_proton_width)
    s -= L * jnp.floor(s/L)
    return key_new, s, proton_acc_rate

def make_loss_pretrain_flow(logprob_p, pes, L, rs, reciprocal_beta, clip_factor):
    """
        Make loss function for pretraining flow.
        Input:
            logprob_p: Callable, log probability function
            pes: Callable, potential energy surface function
            rs: float, Wigner-Seitz radius
            beta: float, inverse temperature
            clip_factor: float, clipping factor
        Output:
            observable_and_lossfn: Callable, observable and loss function
    """

    def observable_and_lossfn(params_flow, s):
        '''
            s: (B, n, dim)
        '''
        observable = pes(s)
        Eloc = (observable[:, 0]-observable[:, 4])*rs**2 # (B,)
        kinetic = observable[:, 1]*rs**2 # (B,)
        v_ep = observable[:, 2]*rs**2 # (B,)
        v_ee = observable[:, 3]*rs**2 # (B,)
        v_pp = observable[:, 4]*rs**2 # (B,)
        se = observable[:, 5] # (B,)
        converged = observable[:, 6] # (B,)

        logp_p = logprob_p(params_flow, s) # (B,)
        Floc = (logp_p-se)*rs**2*reciprocal_beta + Eloc + v_pp # (B,)

        P = (2*kinetic + (v_pp + v_ep + v_ee) )/(3*(L*rs)**3)* 14710.513242194795 # (B,) GPa
        EPloc = (Eloc + v_pp) * P

        F, F2, E, E2, K, K2, Vep, Vep2, Vee, Vee2, Vpp, Vpp2, \
        P, P2, ep, Sp, Sp2, Se, Se2, convergence = \
            jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="p"), 
                            (Floc.mean(), (Floc**2).mean(), 
                             Eloc.mean(), (Eloc**2).mean(),
                             kinetic.mean(), (kinetic**2).mean(), 
                             v_ep.mean(), (v_ep**2).mean(),
                             v_ee.mean(), (v_ee**2).mean(),
                             v_pp.mean(), (v_pp**2).mean(),
                             P.mean(), (P**2).mean(),
                             EPloc.mean(),
                             -logp_p.mean(), (logp_p**2).mean(),
                             se.mean(), (se**2).mean(),
                             converged.mean()
                            )
                        )

        observable = {"F": F, "F2": F2, 
                      "E": E, "E2": E2, 
                      "K": K, "K2": K2,
                      "Vep": Vep, "Vep2": Vep2,
                      "Vee": Vee, "Vee2": Vee2,
                      "Vpp": Vpp, "Vpp2": Vpp2,
                      "P": P, "P2": P2,
                      "ep": ep,
                      "Sp": Sp, "Sp2": Sp2,
                      "Se": Se, "Se2": Se2,
                      "convergence": convergence}

        def flow_lossfn(params_flow):
            logp_states = logprob_p(params_flow, s) # (B,)
            tv = jax.lax.pmean(jnp.abs(Floc - F).mean(), axis_name="p")
            Floc_clipped = jnp.clip(Floc, F - clip_factor*tv, F + clip_factor*tv)
            gradF = (logp_states * (Floc_clipped-F)).mean()
            return gradF

        return observable, flow_lossfn

    return observable_and_lossfn

def make_loss_pretrain_flow_sr(logprob_p, pes, L, rs, reciprocal_beta, clip_factor):
    """
        Make loss function for pretraining flow.
        Input:
            logprob_p: Callable, log probability function
            pes: Callable, potential energy surface function
            L: float, box size in rs unit
            rs: float, Wigner-Seitz radius
            reciprocal_beta: float, temperature in Ry, 1/beta
            clip_factor: float, clipping factor
        Output:
            observable_and_lossfn: Callable, observable and loss function
    """

    def observable_and_lossfn(params_flow, s, key):
        '''
            s: (B, n, dim)
        '''
        observable = pes(s)
        Eloc = (observable[:, 0]-observable[:, 4])*rs**2 # (B,)
        kinetic = observable[:, 1]*rs**2 # (B,)
        v_ep = observable[:, 2]*rs**2 # (B,)
        v_ee = observable[:, 3]*rs**2 # (B,)
        v_pp = observable[:, 4]*rs**2 # (B,)
        se = observable[:, 5] # (B,)
        converged = observable[:, 6] # (B,)

        logp_p = logprob_p(params_flow, s) # (B,)
        Floc = (logp_p-se)*rs**2*reciprocal_beta + Eloc + v_pp # (B,)

        P = (2*kinetic + (v_pp + v_ep + v_ee) )/(3*(L*rs)**3)* 14710.513242194795 # (B,) GPa
        EPloc = (Eloc + v_pp) * P

        F, F2, E, E2, K, K2, Vep, Vep2, Vee, Vee2, Vpp, Vpp2, \
        P, P2, ep, Sp, Sp2, Se, Se2, convergence = \
            jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="p"), 
                            (Floc.mean(), (Floc**2).mean(), 
                             Eloc.mean(), (Eloc**2).mean(),
                             kinetic.mean(), (kinetic**2).mean(), 
                             v_ep.mean(), (v_ep**2).mean(),
                             v_ee.mean(), (v_ee**2).mean(),
                             v_pp.mean(), (v_pp**2).mean(),
                             P.mean(), (P**2).mean(),
                             EPloc.mean(),
                             -logp_p.mean(), (logp_p**2).mean(),
                             se.mean(), (se**2).mean(),
                             converged.mean()
                            )
                        )

        observable = {"F": F, "F2": F2, 
                      "E": E, "E2": E2, 
                      "K": K, "K2": K2,
                      "Vep": Vep, "Vep2": Vep2,
                      "Vee": Vee, "Vee2": Vee2,
                      "Vpp": Vpp, "Vpp2": Vpp2,
                      "P": P, "P2": P2,
                      "ep": ep,
                      "Sp": Sp, "Sp2": Sp2,
                      "Se": Se, "Se2": Se2,
                      "convergence": convergence}

        def flow_lossfn(params_flow):
            logp_states = logprob_p(params_flow, s) # (B,)
            tv = jax.lax.pmean(jnp.abs(Floc - F).mean(), axis_name="p")
            Floc_clipped = jnp.clip(Floc, F - clip_factor*tv, F + clip_factor*tv)
            gradF = (logp_states * Floc_clipped).mean()
            flow_score = logp_states.mean()
            return gradF, flow_score

        return observable, flow_lossfn

    return observable_and_lossfn