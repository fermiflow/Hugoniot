# copy from https://code.itp.ac.cn/xinyangd/quantumhydrogen/-/blob/dev_exci/qhydrogen/optimizer/cg.py

import jax
import jax.numpy as jnp
import numpy as np
import math
from jax.flatten_util import ravel_pytree
from optax._src import base
from functools import partial

from src.utils import compose
from src.sr import _update, _scale


_make_state_ckpt = lambda ckpt_keys: lambda state: {
    key: val for key, val in state.items() if key in ckpt_keys
}

def make_para_mode(mode):
    ret = None
    if mode == "svd":
        ret = "all_gather"
    elif mode == "qr":
        ret = "all_to_all"
    else:
        raise ValueError(f"unknown mode {mode}")
    return ret

def pad_score(ss, divide,):
    np = ss.shape[-1]
    upnp = math.ceil(np / divide) * divide
    np_pad = upnp - np
    if np_pad > 0:
        pad = jnp.zeros( [*ss.shape[:-1], np_pad], dtype=ss.dtype )
        ret = jnp.concatenate([ss, pad], axis=-1)
        return ret
    else:
        return ss

def svd(A):
    """
        A manual implementation of SVD using the eigen-decomposition of A A^T or A^T A,
    which appears to be much more efficient than jax.scipy.linalg.svd.
    """
    M, N = A.shape
    if M < N:
        s2, U = jax.scipy.linalg.eigh(A.dot(A.T))
        s2, U = s2[::-1], U[:, ::-1]
        s = jnp.sqrt(jnp.abs(s2))
        Vh = (U/s).T.dot(A)
    else:
        s2, V = jax.scipy.linalg.eigh(A.T.dot(A))
        s2, V = s2[::-1], V[:, ::-1]
        s = jnp.sqrt(jnp.abs(s2))
        U = A.dot(V/s)
        Vh = V.T
    return U, s, Vh

def chol_qr(A, shift=None, para_mode="all_gather"):
    """
        Compute the Cholesky decomposition of the real symmetric positive-definite matrix:
    A^T A + shift I = R^T R, where A is a MxN matrix (M>=N), and R is an NxN (invertible)
    upper triangular matrix. A MxN matrix Q is then computed such that A = QR.

    Note when shift=0, this function reduces to the usual QR decomposition.
    """
    M, N = A.shape      # np(np_loc) x nb
    if para_mode == "all_gather":
      a = A.T @ A
    elif para_mode == "all_to_all":
      # nb x nb
      a = jax.lax.psum(A.T @ A, axis_name="p")
    else:
      raise ValueError(f"unknown para_mode {para_mode}")
    if shift is None:
        shift = 1.2e-15 * (M*N + N*(N+1)) * a.trace(0,-1,-2).max()
    R = jax.scipy.linalg.cholesky(a + shift * jnp.eye(N, dtype=A.dtype), lower=False)
    Q = jax.lax.linalg.triangular_solve(R, A, left_side=False, lower=False)
    return Q, R

def qr_implicit_naive(A, vec, damping):
    # nb x np(np_loc)
    N, M = A.shape
    nd = jax.device_count()
    real_np = vec.shape[0]
    vec = pad_score(vec, nd)
    assert M == vec.shape[0]//nd, f"{M} != {vec.shape[0]//nd}"
    # nb x nb
    a = jax.lax.psum(A @ A.T, axis_name="p")
    a = a + damping * jnp.eye(N, dtype=A.dtype)
    # np(np_loc)  fetch the local part of vector
    vec = vec.reshape(nd,-1)[jax.lax.axis_index("p")]
    rhs = jax.lax.psum(A @ vec, axis_name="p")
    y = jax.scipy.linalg.solve(a, rhs, assume_a="pos")
    # np(np_loc)
    ret = 1./damping * (vec - A.T.dot(y))
    ret = jax.lax.all_gather(ret, "p", axis=0, tiled=True)
    ret = jnp.split(ret, [real_np], axis=0)[0]
    return ret

def qr_implicit_gatherd_psum(A, vec, damping):
    # nb x np(np_loc)
    N, M = A.shape
    nd = jax.device_count()
    real_np = vec.shape[0]
    vec = pad_score(vec, nd)
    assert M == vec.shape[0]//nd, f"{M} != {vec.shape[0]//nd}"
    # np(np_loc)  fetch the local part of vector
    vec = vec.reshape(nd,-1)[jax.lax.axis_index("p")]
    Atv = jnp.concatenate([A.T, vec.reshape(-1,1)], axis=-1)
    # nb x nb, nb x 1
    a, rhs = jnp.split(
      jax.lax.psum(A @ Atv, axis_name="p"), [N], axis=-1)
    a = a + damping * jnp.eye(N, dtype=A.dtype)
    y = jax.scipy.linalg.solve(a, rhs.reshape(-1), assume_a="pos")
    # np(np_loc)
    ret = 1./damping * (vec - A.T.dot(y))
    ret = jax.lax.all_gather(ret, "p", axis=0, tiled=True)
    ret = jnp.split(ret, [real_np], axis=0)[0]
    return ret

qr_implicit = qr_implicit_gatherd_psum

def _convert_score_quantum(score, factor, para_mode="all_gather"):
    """
        Convert the quantum fisher information matrix of electrons as follows:
                            1/B Re(S^† S) = A^T A,
    where S is the (complex) score function of shape (*batchshape, nparams), and
    B = prod(batchshape) is the total batch size. The output A is a REAL matrix
    of shape (2B, nparams).

        Note this function should be placed in some pmapped subroutines.
    """
    *batchshape, nparams = score.shape
    c = 1. - jnp.sqrt(1. - factor)
    score_center = score - c * score.mean(axis=-2, keepdims=True)
    # nb_loc x np
    score_center = score_center.reshape(-1, nparams)
    if para_mode == "all_gather":
        # nb x np
        score_center = jax.lax.all_gather(score_center, "p", axis=0, tiled=True)
        print("gathered score shape:", score_center.shape)
    elif para_mode == "all_to_all":
        ss = score_center
        nb_loc,_ = ss.shape
        nd = jax.device_count()
        # nb_loc x np, pad so the np can be divided by nd
        ss = pad_score(ss, nd)
        np_loc = ss.shape[-1] // nd
        ss = ss.reshape(nb_loc, nd, np_loc)
        # due to the bug https://github.com/google/jax/issues/18122
        # we have to do all_to_all for real and imag parts one by one.
        ss_real = jax.lax.all_to_all(ss.real, "p", 1, 0, tiled=True)
        ss_imag = jax.lax.all_to_all(ss.imag, "p", 1, 0, tiled=True)
        ss = ss_real + 1.J * ss_imag
        ss = ss.reshape(nb_loc*nd, np_loc)
        # nb x np_loc
        score_center = ss
        print("transposed score shape:", score_center.shape)
    else:
        raise ValueError(f"unknown para_mode {para_mode}")
    assert score_center.shape[0] == jax.device_count() * np.prod(batchshape)
    score_center /= jnp.sqrt(score_center.shape[0])
    score_center = jnp.concatenate([score_center.real, score_center.imag], axis=0)
    # nb x np(np_loc)
    return score_center

def _convert_score_classical(score, factor, para_mode="all_gather"):
    """
        Convert the classical fisher information matrix.
        This is basically a copy of the quantum case with real matrices
    """
    *batchshape, nparams = score.shape
    score_center = score.reshape(-1, nparams)
    if para_mode == "all_gather":
        # nb x np
        score_center = jax.lax.all_gather(score_center, "p", axis=0, tiled=True)
        print("gathered score shape:", score_center.shape)
    elif para_mode == "all_to_all":
        ss = score_center
        nb_loc, _ = ss.shape
        nd = jax.device_count()
        # nb_loc x np, pad so the np can be divided by nd
        ss = pad_score(ss, nd)
        np_loc = ss.shape[-1] // nd
        ss = ss.reshape(nb_loc, nd, np_loc)
        ss = jax.lax.all_to_all(ss, "p", 1, 0, tiled=True)
        ss = ss.reshape(nb_loc * nd, np_loc)
        # nb x np_loc
        score_center = ss
        print("transposed score shape:", score_center.shape)
    else:
        raise ValueError(f"unknown para_mode {para_mode}")
    assert score_center.shape[0] == jax.device_count() * np.prod(batchshape)
    score_center /= jnp.sqrt(score_center.shape[0])
    # nb x np(np_loc)
    return score_center

####################################################################################

def _make_fisher_cg(logpsi, score_fn, acc_steps, gamma, lr, decay, damping, maxnorm, *,
                    mode="score",
                    expose_fisher_vec_prod=False,
                    init_vec_last_step=False,
                    solver_precondition=False,
                    solver_maxiter=None,
                    solver_tol=1e-10,
                    solver_style="cg",
                    type="quantum",
                    van=False
                    ):
    """
    Args:
        gamma (float): the decaying parameter for the ProxSR method.
            The recommended value is 0.01. gamma = 1 means ordinary sr without damping,
            one recovers the original version with no history info.

        The args init_vec_last_step, solver_precondition, solver_maxiter, solver_tol
            and solver_style are used only for the mode "score".
    """

    assert (type in ["quantum", "classical"])

    # This is for van and flow model, both are classical
    # for flow model, x has shape (..., n, dim) so batchidx = -2
    # for van, state_idx has shape (..., n) so batchidx = -1
    batchidx = -1 if van else -2

    def init_fn(params, x, *args):
        """
            NOTE: in this implementation, the optimizer state should be pmapped
        along the first axis, since it contains scores of the sample data that
        are used to compute the Fisher information matrix.
        """
        batchshape = x.shape[:batchidx]
        print("init optimizer state with batchshape:", batchshape)
        raveled_params, _ = ravel_pytree(params)
        score_dummy = logpsi(x, params, *args)
        return {
            "score": jnp.empty((acc_steps, *batchshape, raveled_params.size), dtype=score_dummy.dtype),
            "acc_step": 0,
            "mix_fisher": False,
            "step": 0,
            "gnorm": None,
            "last_update": None,
            "grad_sum": None,
            "update_sum": None,
        }

    ckpt_keys = ("step", "gnorm")

    def fisher_fn(params, x, state, *args):
        score = score_fn(x, params, *args)
        batchshape = x.shape[:batchidx]
        batchdim = len(batchshape)
        score = compose(*([jax.vmap] * batchdim))(lambda pytree: ravel_pytree(pytree)[0])(score)
        print("score.shape:", score.shape)  # (*batchshape, nparams)

        state["score"] = state["score"].at[state["acc_step"]].set(score)
        state["acc_step"] = (state["acc_step"] + 1) % acc_steps
        return state

    if mode == "score":
        if type == "classical": raise RuntimeError("score mode in fisher cg only works for quantum case")
        def make_fisher_vec_prod(params, state):
            score = state["score"]  # (acc_steps, *batchshape, nparams)
            batchshape = score.shape[:-1]
            factor = 1. - 1. / (1 + decay * state["step"])
            score_center = score - factor * score.mean(axis=-2, keepdims=True)

            fisher_diag = jax.lax.pmean(
                jnp.einsum("...i,...i->i", score.conj(), score_center).real / np.prod(batchshape),
                axis_name="p"
            )

            def fisher_diag_inv_vec_prod(vec, damping):
                return vec / (fisher_diag + damping)

            def fisher_vec_prod(vec):
                jvp = jnp.einsum("...i,i->...", score_center, vec)
                return jax.lax.pmean(
                    jnp.einsum("...i,...->i", score.conj(), jvp).real / np.prod(batchshape),
                    axis_name="p"
                )

            return fisher_vec_prod, fisher_diag_inv_vec_prod
    elif mode in ["svd", "qr"]:
        _convert_score = _convert_score_quantum if type == "quantum" else _convert_score_classical
        def make_fisher_vec_prod(params, state, para_mode=None):
            """
                Note for mode "svd" and "qr", this function actually makes INVERSE
            of the fisher information matrix - vector product.
            """
            para_mode = make_para_mode(mode) if para_mode is None else para_mode
            score = state["score"]  # (acc_steps, *batchshape, nparams)
            factor = 1. - 1. / (1 + decay * state["step"])
            real_np = score.shape[-1]
            A = _convert_score(score, factor, para_mode=para_mode)

            def _svd(vec, damping):
                # s: (B,), Vh: (B, nparams)
                _, s, Vh = svd(A)
                s2inv = 1. / (s ** 2 + damping)
                return jnp.einsum("ia,a,aj,j->i", Vh.T, s2inv, Vh, vec) \
                    + 1. / damping * (vec - Vh.T.dot(Vh.dot(vec)))

            def _qr(vec, damping):
                # Q: (nparams, B)
                Q, _ = chol_qr(A.T, shift=damping, para_mode=para_mode)
                if para_mode == "all_gather":
                    QQtv = Q.dot(Q.T.dot(vec))
                elif para_mode == "all_to_all":
                    np_loc, nb = Q.shape
                    nd = jax.device_count()
                    Q = Q.reshape(np_loc, nd, nb // nd)
                    Q = jax.lax.all_to_all(Q, "p", 1, 0, tiled=True)
                    # np x nb_loc
                    Q = Q.reshape(np_loc * nd, nb // nd)
                    Q, pad = jnp.split(Q, [real_np], axis=0)
                    QQtv = Q.dot(Q.T.dot(vec))
                    QQtv = jax.lax.psum(QQtv, axis_name="p")
                else:
                    raise ValueError(f"unknown para_mode {para_mode}")
                ret = 1. / damping * (vec - QQtv)
                return ret

            def _qr_implicit(vec, damping):
                return qr_implicit(A, vec, damping)

            fisher_inv_vec_prod = {
                "svd": _svd,
                "qr": _qr_implicit,
            }
            return fisher_inv_vec_prod[mode]

    state_ckpt_fn = _make_state_ckpt(ckpt_keys)

    def update_fn(grad, state, params):
        fisher = make_fisher_vec_prod(params, state)
        grad_raveled, unravel_fn = ravel_pytree(grad)
        print("grad.shape:", grad_raveled.shape)
        if state["last_update"] is not None:
            grad_raveled += damping * (1. - gamma) * state["last_update"]

        raw_update = _update(fisher, grad_raveled, damping,
                             mode=mode,
                             init_vec=state["last_update"] if init_vec_last_step else None,
                             precondition=solver_precondition,
                             tol=solver_tol,
                             maxiter=solver_maxiter,
                             style=solver_style,
                             )
        state["last_update"] = raw_update
        print("raw_update.shape:", raw_update.shape)

        step_lr = lr / (1 + decay * state["step"])
        update_raveled, gnorm = _scale(raw_update, grad_raveled, step_lr, maxnorm)
        update = unravel_fn(update_raveled)

        state["gnorm"] = gnorm
        state["step"] += 1
        # If we have finished a step, we can mix fisher information matrix from now on.
        state["mix_fisher"] = True

        # for debugging purpose
        state["grad_norm"] = jnp.linalg.norm(grad_raveled)
        state["update_norm"] = jnp.linalg.norm(raw_update)
        state["supdate_norm"] = jnp.linalg.norm(update_raveled)

        return update, state

    fishers = [fisher_fn]
    if expose_fisher_vec_prod: fishers.append(make_fisher_vec_prod)
    return *fishers, state_ckpt_fn, base.GradientTransformation(init_fn, update_fn)


quantum_fisher_cg = partial(_make_fisher_cg, type="quantum")
classical_fisher_cg = partial(_make_fisher_cg, type="classical")
