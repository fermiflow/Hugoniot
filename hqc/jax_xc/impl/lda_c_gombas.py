"""Generated from lda_c_gombas.mpl."""

import jax
import jax.numpy as jnp
from jax.numpy import array as array
from jax.numpy import int32 as int32
from jax.numpy import nan as nan
from typing import Callable, Optional
from .utils import *


def pol(p, r, s=(None, None, None), l=(None, None), tau=(None, None)):
  params = p.params
  (r0, r1), (s0, s1, s2), (l0, l1), (tau0, tau1) = r, s, l, tau
  t2 = jnp.cbrt(r0 + r1)
  t3 = 0.1e1 / t2
  t10 = jnp.log((t3 + 0.239e1) * t2)
  res = -0.357e-1 / (0.1e1 + 0.562e-1 * t3) - 0.311e-1 * t10
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(r0)
  t2 = 0.1e1 / t1
  t9 = jnp.log((t2 + 0.239e1) * t1)
  res = -0.357e-1 / (0.1e1 + 0.562e-1 * t2) - 0.311e-1 * t9
  return res


def invoke(
  p: NamedTuple, rho: Callable, r: jnp.ndarray, mo: Optional[Callable] = None,
  deorbitalize: Optional[float] = None,
):
  args = rho_to_arguments(p, rho, r, mo, deorbitalize)
  ret = pol(p, *args) if p.nspin == 2 else unpol(p, *args)
  dens = args[0] if p.nspin == 1 else sum(args[0])
  ret = jnp.float64(dens >= p.dens_threshold) * ret
  return ret