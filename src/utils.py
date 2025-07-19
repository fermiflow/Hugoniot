import jax
import jax.numpy as jnp
import functools
from typing import Sequence, Optional 
import math
import itertools

p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))

shard = jax.pmap(lambda x: x)

def compose(*decs):
    """ Compose a sequence of function decorators. """
    return functools.reduce(lambda dec1, dec2: lambda f: dec1(dec2(f)), decs)

def replicate(pytree, num_devices):
    dummy_input = jnp.empty(num_devices)
    return jax.pmap(lambda _: pytree)(dummy_input)

def all_gather(x, axis_name, *, axis=0, tiled=False):
    return jax.pmap(
        lambda x: jax.lax.all_gather(x, axis_name, axis=axis, tiled=tiled), 
        axis_name=axis_name
    )(x)

def logdet_matmul(xs: Sequence[jnp.ndarray],
                  logw: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Combines determinants and takes dot product with weights in log-domain.
  We use the log-sum-exp trick to reduce numerical instabilities.
  Args:
    xs: FermiNet orbitals in each determinant. Either of length 1 with shape
      (ndet, nelectron, nelectron) (full_det=True) or length 2 with shapes
      (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta) (full_det=False,
      determinants are factorised into block-diagonals for each spin channel).
    w: weight of each determinant. If none, a uniform weight is assumed.
  Returns:
    sum_i exp(logw_i) D_i in the log domain, where logw_i is the log-weight of D_i, the i-th
    determinant (or product of the i-th determinant in each spin channel, if
    full_det is not used).
  """
  # Special case to avoid taking log(0) if any matrix is of size 1x1.
  # We can avoid this by not going into the log domain and skipping the
  # log-sum-exp trick.
  det1 = functools.reduce(
      lambda a, b: a * b,
      [x.reshape(-1) for x in xs if x.shape[-1] == 1],
      1
  )

  # Compute the logdet for all matrices larger than 1x1
  sign_in, logdet = functools.reduce(
      lambda a, b: (a[0] * b[0], a[1] + b[1]),
      [jnp.linalg.slogdet(x) for x in xs if x.shape[-1] > 1],
      (1, 0)
  )

  if logw is not None:
    logdet = logw + logdet

  # log-sum-exp trick
  maxlogdet = jnp.max(logdet)
  det = sign_in * det1 * jnp.exp(logdet - maxlogdet)
  result = jnp.sum(det)

  sign_out = result/jnp.abs(result)
  log_out = jnp.log(jnp.abs(result)) + maxlogdet
  return sign_out, log_out

def cubic_init(n,spacing):
    K = math.ceil(n**(1 / 3))
    x = jnp.linspace(0, K*spacing, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    return jnp.array(position[:n])

def clipped_force_fn(force_fn, max_norm, x):
    f = force_fn(x)
    g_norm = jnp.linalg.norm(f, axis=-1, keepdims=True) 
    scale = jnp.minimum(1.0, max_norm/g_norm)
    return f*scale
