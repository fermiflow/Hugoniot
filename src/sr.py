"""
    Second-order optimization algorithm using stochastic reconfiguration.
    The design of API signatures is in parallel with the package `optax`.
"""
import jax
import jax.numpy as jnp
import numpy as np
import math
from jax.flatten_util import ravel_pytree
from optax._src import base
from functools import partial

from src.utils import compose
from src.ad import make_vjp_real, make_vjp_complex

def pytree_as_whole_block(pytree):
    structure = jax.tree_util.tree_structure(pytree)
    return lambda block: jax.tree_util.tree_structure(block) == structure

_make_state_ckpt = lambda ckpt_keys: lambda state: {
    key: val for key, val in state.items() if key in ckpt_keys
}

_update_treedef = jax.tree_util.tree_structure(tuple(range(2)))

# Make arbitrary function `f` callable with pytree as arguments.
tree_fn = lambda f: lambda *args: jax.tree_util.tree_map(f, *args)

block_ravel_pytree = lambda block_fn: lambda pytree: jax.tree_util.tree_map(
    lambda x: ravel_pytree(x)[0], pytree, is_leaf=block_fn
)
block_unravel_pytree = lambda block_fn: lambda pytree: jax.tree_util.tree_map(
    lambda x: ravel_pytree(x)[1], pytree, is_leaf=block_fn
)

def _update(fisher, grad, damping, *, mode="dense", 
            init_vec=None, precondition=False, tol=1e-10, maxiter=None, style="cg"):
    """ Solve the linear system (fisher + damping I) x = grad using various methods.
        To obtain the final parameter update, one should add a minus sign and an
    appropriate scaling factor; see function _scale.
    """
    if mode == "dense":
        assert fisher.shape == grad.shape*2
        fisher += damping * jnp.eye(fisher.shape[0])
        update = jax.scipy.linalg.solve(fisher, grad)
    elif mode == "score":
        fisher_vec_prod, fisher_diag_inv_vec_prod = fisher
        fisher = lambda x: fisher_vec_prod(x) + damping*x
        if precondition:
            M = lambda x: fisher_diag_inv_vec_prod(x, damping)
        else:
            M = None
        if style == "cg":
            update, _ = jax.scipy.sparse.linalg.cg(fisher, grad, x0=init_vec, tol=tol, maxiter=maxiter, M=M)
        elif style == "gmres":
            update, _ = jax.scipy.sparse.linalg.gmres(fisher, grad, x0=init_vec, tol=tol, maxiter=maxiter, M=M)
        else:
            raise ValueError(f"unknown solver style {style}")
    elif mode in ["svd", "qr"]:
        fisher_inv_vec_prod = fisher
        update = fisher_inv_vec_prod(grad, damping)
    else:
        raise ValueError(f"unknown mode {mode}")

    return update

def _scale(update, grad, lr, maxnorm):
    """ Scale the update according to gradnorm. """
    gnorm = jnp.sum(grad * update) 
    scale = jnp.minimum(jnp.sqrt(maxnorm/gnorm), lr)
    update *= -scale
    return update, gnorm

def get_updates(fisher, grad, lr, damping, maxnorm):
    fisher += damping * jnp.eye(fisher.shape[0])
    update = jax.scipy.linalg.solve(fisher, grad)
    gnorm = jnp.sum(grad * update) 
    scale = jnp.minimum(jnp.sqrt(maxnorm/gnorm), lr)
    update *= -scale
    return update

####################################################################################

def _make_fisher_sr(score_fn, block_fn, acc_steps, alpha, lr, decay, damping, maxnorm, type, save_fisher=False):
    """
        SR for a probabilistic model, which is also known as the
        natural gradient descent in machine-learning literatures.

    INPUT:
        block_fn: a function that specifies the subtrees in the parameter pytree
            that should be treated as whole blocks.
    """

    def init_fn(params):
        block_raveled_params = block_ravel_pytree(block_fn)(params)
        return {
            "fisher_ewm": jax.tree_util.tree_map(lambda x: jnp.zeros((x.size, x.size)), block_raveled_params),
            "acc_step": 0, 
            "mix_fisher": False, 
            "step": 0, 
            "gnorm": None,
        }

    ckpt_keys = ("step", "gnorm")
    if save_fisher: ckpt_keys += ("fisher_ewm", "mix_fisher",)
    state_ckpt_fn = _make_state_ckpt(ckpt_keys)

    def fishers_fn_classical(params, x, state, *args):

        score = score_fn(params, x, *args)
        batchshape = x.shape[:-2]
        batchdim = len(batchshape)
        score = compose(*([jax.vmap] * batchdim))(block_ravel_pytree(block_fn))(score)
        print("score.shape:", tree_fn(lambda score: score.shape)(score))  # (*batchshape, nparams)

        fisher = jax.lax.pmean(
            tree_fn(lambda score: jnp.einsum("...i,...j->ij", score, score) / np.prod(score.shape[:-1]))(score),
            axis_name="p"
        )

        state["fisher_ewm"] = jax.lax.cond(
            state["mix_fisher"],
            tree_fn(lambda x_ewm, x: (1-alpha)**(1/acc_steps) * x_ewm + alpha/(1-alpha)**(1-(1+state["acc_step"])/acc_steps) * x/acc_steps),
            tree_fn(lambda x_ewm, x: x_ewm + x/acc_steps),
            state["fisher_ewm"], fisher
        )
        state["acc_step"] = (state["acc_step"] + 1) % acc_steps
        return state

    def fishers_fn_occupation(params, state_idx, state, *args):

        score = score_fn(params, state_idx, *args)
        batchshape = state_idx.shape[:-1]
        batchdim = len(batchshape)
        score = compose(*([jax.vmap] * batchdim))(block_ravel_pytree(block_fn))(score)
        print("score.shape:", tree_fn(lambda score: score.shape)(score))  # (*batchshape, nparams)

        fisher = jax.lax.pmean(
            tree_fn(lambda score: jnp.einsum("...i,...j->ij", score, score) / np.prod(score.shape[:-1]))(score),
            axis_name="p"
        )

        state["fisher_ewm"] = jax.lax.cond(
            state["mix_fisher"],
            tree_fn(lambda x_ewm, x: (1-alpha)**(1/acc_steps) * x_ewm + alpha/(1-alpha)**(1-(1+state["acc_step"])/acc_steps) * x/acc_steps),
            tree_fn(lambda x_ewm, x: x_ewm + x/acc_steps),
            state["fisher_ewm"], fisher
        )
        state["acc_step"] = (state["acc_step"] + 1) % acc_steps
        return state

    def fishers_fn_quantum(params, x, state, *args):

        score = score_fn(x, params, *args)
        batchshape = x.shape[:-2]
        batchdim = len(batchshape)
        score = compose(*( [jax.vmap]*batchdim ))(block_ravel_pytree(block_fn))(score)
        print("score.shape:", tree_fn(lambda score: score.shape)(score))  # (*batchshape, nparams)
        
        fisher_term1 = jax.lax.pmean(
            tree_fn(lambda score: jnp.einsum("...i,...j->ij", score.conj(), score).real / np.prod(score.shape[:-1]))(score),
            axis_name="p"
        )
        score_mean = jax.tree_util.tree_map(lambda score: score.mean(axis=-2), score) # (*batchshape[:-1], nparams) 
        fisher_term2 = jax.lax.pmean(
            tree_fn(lambda score: jnp.einsum("...i,...j->ij", score.conj(), score).real / np.prod(score.shape[:-1]))(score_mean), 
            axis_name="p"
        )
        factor = 1. - 1. / (1 + decay*state["step"])
        fisher = tree_fn(lambda x, y: x - factor * y)(fisher_term1, fisher_term2)

        state["fisher_ewm"] = jax.lax.cond(
            state["mix_fisher"], 
            tree_fn(lambda x_ewm, x: (1-alpha)**(1/acc_steps) * x_ewm + alpha/(1-alpha)**(1-(1+state["acc_step"])/acc_steps) * x/acc_steps), 
            tree_fn(lambda x_ewm, x: x_ewm + x/acc_steps),
            state["fisher_ewm"], fisher
        )
        state["acc_step"] = (state["acc_step"] + 1) % acc_steps
        return state

    def update_fn(grad, state, params):
        fisher = state["fisher_ewm"]
        grad_raveled, unravel_fn = block_ravel_pytree(block_fn)(grad), block_unravel_pytree(block_fn)(grad)
        print("grad.shape:", tree_fn(lambda x: x.shape)(grad_raveled))

        raw_update = _update(fisher, grad_raveled, damping, mode="dense")
        step_lr = lr / (1 + decay*state["step"])
        update_raveled, gnorm = _scale(raw_update, grad_raveled, step_lr, maxnorm)
        update = unravel_fn(update_raveled)

        state["gnorm"] = gnorm
        state["step"] += 1
        # If we have finished a step, we can mix fisher information matrix from now on.
        state["mix_fisher"] = True

        return update, state

    if type == "quantum":
        fishers_fn = fishers_fn_quantum
    elif type == "classical":
        fishers_fn = fishers_fn_classical
    elif type == "occupation":
        fishers_fn = fishers_fn_occupation
    else:
        raise NotImplementedError

    return fishers_fn, state_ckpt_fn, base.GradientTransformation(init_fn, update_fn)


quantum_fisher_sr = partial(_make_fisher_sr, type="quantum")
classical_fisher_sr = partial(_make_fisher_sr, type="classical")
occupation_fisher_sr = partial(_make_fisher_sr, type="occupation")

####################################################################################
# -- naive sr implementation for testing purpose -- #

FisherSRState = base.EmptyState

def fisher_sr(score_fn, damping, max_norm):
    """
    SR for a classical probabilistic model, which is also known as the
    natural gradient descent in machine-learning literatures.
    """

    def init_fn(params):
        return FisherSRState()

    def update_fn(grads, state, params):
        """
            NOTE: as the computation of Fisher information metric calls for the
        Monte-Carlo sample `state_indices`, we manually place them within the
        `params` argument.
        """
        params, state_indices = params

        grads_raveled, grads_unravel_fn = ravel_pytree(grads)
        print("grads.shape:", grads_raveled.shape)

        score = score_fn(params, state_indices)
        score_raveled = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])(score)
        print("score.shape:", score_raveled.shape)

        batch_per_device = score_raveled.shape[0]

        fisher = jax.lax.pmean(score_raveled.T.dot(score_raveled) / batch_per_device, axis_name='p')
        fisher += damping * jnp.eye(fisher.shape[0])
        updates_raveled = jax.scipy.linalg.solve(fisher, grads_raveled)
        #scale gradient according to gradnorm
        gnorm = jnp.sum(grads_raveled * updates_raveled)
        scale = jnp.minimum(jnp.sqrt(max_norm/gnorm), 1)
        updates_raveled *= -scale
        updates = grads_unravel_fn(updates_raveled)

        return updates, state

    return base.GradientTransformation(init_fn, update_fn)


def hybrid_fisher_sr(flow_score_fn, van_score_fn, wfn_score_fn, lr_flow, lr_van, lr_wfn, decay, damping_flow, damping_van, damping_wfn, maxnorm_flow, maxnorm_van, maxnorm_wfn):
    """
        Hybrid SR for both a classical probabilistic model and a set of
    quantum basis wavefunction ansatz.
    """

    def init_fn(params):
        return {'step': 0}

    def fishers_fn(params_flow, params_van, params_wfn, state_idx, mo_coeff, bands, s, x, state):

        flow_score = flow_score_fn(params_flow, s)
        flow_score = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])(flow_score)

        van_score = van_score_fn(params_van, state_idx, bands)
        van_score = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])(van_score)

        wfn_score = wfn_score_fn(x, params_wfn, s, state_idx, mo_coeff)
        wfn_score = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])(wfn_score)

        batchsize = flow_score.shape[0]
        print("flow_score.shape:", flow_score.shape)
        print("van_score.shape:", van_score.shape)
        print("wfn_score.shape:", wfn_score.shape)
        
        flow_fisher = jax.lax.pmean(
                    flow_score.T.dot(flow_score) / batchsize,
                    axis_name="p")

        van_fisher = jax.lax.pmean(
                    van_score.T.dot(van_score) / batchsize,
                    axis_name="p")

        wfn_score_mean = jax.lax.pmean(wfn_score.mean(axis=0), axis_name="p")

        wfn_fisher = jax.lax.pmean(
                    wfn_score.conj().T.dot(wfn_score).real / batchsize,  
                    axis_name="p")

        return flow_fisher, van_fisher, wfn_fisher, wfn_score_mean


    def update_fn(grads, state, params):
        """
            NOTE: as the computation of (classical and quantum) Fisher information
        metrics calls for the Monte-Carlo sample `s` and `x`, we manually
        place them within the `params` argument.
        """
        grad_flow, grad_van, grad_wfn = grads

        fisher_flow, fisher_van, fisher_wfn, wfn_score_mean = params
        fisher_wfn -= (wfn_score_mean.conj()[:, None] * wfn_score_mean).real

        grad_flow, params_flow_unravel_fn = ravel_pytree(grad_flow)
        grad_van, params_van_unravel_fn = ravel_pytree(grad_van)
        grad_wfn, params_wfn_unravel_fn = ravel_pytree(grad_wfn)

        print("grad_flow.shape:", grad_flow.shape)
        print("grad_van.shape:", grad_van.shape)
        print("grad_wfn.shape:", grad_wfn.shape)

        update_params_flow = params_flow_unravel_fn(get_updates(fisher_flow, grad_flow, lr_flow/(1+decay*state['step']), damping_flow, maxnorm_flow))

        update_params_van = params_van_unravel_fn(get_updates(fisher_van, grad_van, lr_van/(1+decay*state['step']), damping_van, maxnorm_van))

        update_params_wfn = params_wfn_unravel_fn(get_updates(fisher_wfn, grad_wfn, lr_wfn/(1+decay*state['step']), damping_wfn, maxnorm_wfn))

        state["step"] += 1

        return (update_params_flow, update_params_van, update_params_wfn), state

    return fishers_fn, base.GradientTransformation(init_fn, update_fn)


def hybrid_fisher_sr_flow(flow_score_fn, lr_flow, decay, damping_flow, maxnorm_flow):
    """
        Hybrid SR for both a classical probabilistic model and a set of
    quantum basis wavefunction ansatz.
    """

    def init_fn(params):
        return {'step': 0}

    def fishers_fn(params_flow, s, state):

        flow_score = flow_score_fn(params_flow, s)
        flow_score = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])(flow_score)

        batchsize = flow_score.shape[0]
        print("flow_score.shape:", flow_score.shape)
        
        flow_fisher = jax.lax.pmean(
                    flow_score.T.dot(flow_score) / batchsize,
                    axis_name="p")

        return flow_fisher

    def update_fn(grad_flow, state, fisher_flow):
        """
            NOTE: as the computation of (classical and quantum) Fisher information
        metrics calls for the Monte-Carlo sample `s` and `x`, we manually
        place them within the `params` argument.
        """

        grad_flow, params_flow_unravel_fn = ravel_pytree(grad_flow)

        print("grad_flow.shape:", grad_flow.shape)

        update_params_flow = params_flow_unravel_fn(get_updates(fisher_flow, grad_flow, lr_flow/(1+decay*state['step']), damping_flow, maxnorm_flow))

        state["step"] += 1

        return update_params_flow, state

    return fishers_fn, base.GradientTransformation(init_fn, update_fn)

