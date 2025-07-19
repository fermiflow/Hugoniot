"""Generated from gga_c_wi.mpl."""

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
  t2 = s0 + 0.2e1 * s1 + s2
  t4 = r0 + r1
  t5 = t4 ** 2
  t6 = jnp.cbrt(t4)
  t7 = t6 ** 2
  t9 = 0.1e1 / t7 / t5
  t12 = jnp.exp(-params.k * t2 * t9)
  t16 = jnp.cbrt(3)
  t18 = jnp.cbrt(0.1e1 / jnp.pi)
  t20 = jnp.cbrt(4)
  t21 = t20 ** 2
  t25 = t16 ** 2
  t26 = jnp.cbrt(jnp.pi)
  t28 = jnp.sqrt(t2)
  t30 = t5 ** 2
  t36 = jnp.sqrt(t28 / t6 / t4)
  res = (params.b * t2 * t9 * t12 + params.a) / (params.c + t16 * t18 * t21 / t6 * (0.1e1 + params.d * t20 * t25 * t26 * t36 * t28 * t2 / t30 / 0.3e1) / 0.4e1)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t2 = r0 ** 2
  t3 = jnp.cbrt(r0)
  t4 = t3 ** 2
  t6 = 0.1e1 / t4 / t2
  t9 = jnp.exp(-params.k * s0 * t6)
  t13 = jnp.cbrt(3)
  t15 = jnp.cbrt(0.1e1 / jnp.pi)
  t17 = jnp.cbrt(4)
  t18 = t17 ** 2
  t22 = t13 ** 2
  t23 = jnp.cbrt(jnp.pi)
  t25 = jnp.sqrt(s0)
  t27 = t2 ** 2
  t33 = jnp.sqrt(t25 / t3 / r0)
  res = (params.b * s0 * t6 * t9 + params.a) / (params.c + t13 * t15 * t18 / t3 * (0.1e1 + params.d * t17 * t22 * t23 * t33 * t25 * s0 / t27 / 0.3e1) / 0.4e1)
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