"""Generated from lda_c_wigner.mpl."""

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
  t2 = (r0 - r1) ** 2
  t3 = r0 + r1
  t4 = t3 ** 2
  t9 = jnp.cbrt(3)
  t11 = jnp.cbrt(0.1e1 / jnp.pi)
  t13 = jnp.cbrt(4)
  t14 = t13 ** 2
  t15 = jnp.cbrt(t3)
  res = (0.1e1 - t2 / t4) * params.a / (params.b + t9 * t11 * t14 / t15 / 0.4e1)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(3)
  t3 = jnp.cbrt(0.1e1 / jnp.pi)
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = jnp.cbrt(r0)
  res = params.a / (params.b + t1 * t3 * t6 / t7 / 0.4e1)
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