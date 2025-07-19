"""Generated from lda_c_rc04.mpl."""

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
  t2 = r0 + r1
  t4 = (r0 - r1) / t2
  t5 = 0.1e1 + t4
  t7 = jnp.cbrt(p.zeta_threshold)
  t8 = t7 ** 2
  t9 = jnp.cbrt(t5)
  t10 = t9 ** 2
  t11 = jnp.where(t5 <= p.zeta_threshold, t8, t10)
  t12 = 0.1e1 - t4
  t14 = jnp.cbrt(t12)
  t15 = t14 ** 2
  t16 = jnp.where(t12 <= p.zeta_threshold, t8, t15)
  t18 = t11 / 0.2e1 + t16 / 0.2e1
  t19 = t18 ** 2
  t21 = jnp.cbrt(3)
  t23 = jnp.cbrt(0.1e1 / jnp.pi)
  t25 = jnp.cbrt(4)
  t26 = t25 ** 2
  t27 = jnp.cbrt(t2)
  t33 = jnp.arctan(0.488827e1 + 0.79425925 * t21 * t23 * t26 / t27)
  t37 = t21 ** 2
  res = t19 * t18 * (-0.655868 * t33 + 0.897889) * t37 / t23 * t25 * t27 / 0.3e1
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t2 = jnp.cbrt(p.zeta_threshold)
  t3 = t2 ** 2
  t4 = jnp.where(0.1e1 <= p.zeta_threshold, t3, 1)
  t5 = t4 ** 2
  t7 = jnp.cbrt(3)
  t9 = jnp.cbrt(0.1e1 / jnp.pi)
  t11 = jnp.cbrt(4)
  t12 = t11 ** 2
  t13 = jnp.cbrt(r0)
  t19 = jnp.arctan(0.488827e1 + 0.79425925 * t7 * t9 * t12 / t13)
  t23 = t7 ** 2
  res = t5 * t4 * (-0.655868 * t19 + 0.897889) * t23 / t9 * t11 * t13 / 0.3e1
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