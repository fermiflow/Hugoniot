"""Generated from gga_x_2d_b86.mpl."""

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
  t2 = jnp.sqrt(jnp.pi)
  t3 = 0.1e1 / t2
  t4 = r0 + r1
  t5 = 0.1e1 / t4
  t8 = 0.2e1 * r0 * t5 <= p.zeta_threshold
  t9 = p.zeta_threshold - 0.1e1
  t12 = 0.2e1 * r1 * t5 <= p.zeta_threshold
  t13 = -t9
  t15 = (r0 - r1) * t5
  t16 = jnp.where(t12, t13, t15)
  t17 = jnp.where(t8, t9, t16)
  t18 = 0.1e1 + t17
  t20 = jnp.sqrt(p.zeta_threshold)
  t21 = t20 * p.zeta_threshold
  t22 = jnp.sqrt(t18)
  t24 = jnp.where(t18 <= p.zeta_threshold, t21, t22 * t18)
  t26 = jnp.sqrt(0.2e1)
  t28 = jnp.sqrt(t4)
  t29 = r0 ** 2
  t32 = s0 / t29 / r0
  t42 = jnp.where(r0 <= p.dens_threshold, 0, -0.2e1 / 0.3e1 * t3 * t24 * t26 * t28 * (0.1e1 + 0.2105e-2 * t32) / (0.1e1 + 0.119e-3 * t32))
  t44 = jnp.where(t8, t13, -t15)
  t45 = jnp.where(t12, t9, t44)
  t46 = 0.1e1 + t45
  t48 = jnp.sqrt(t46)
  t50 = jnp.where(t46 <= p.zeta_threshold, t21, t48 * t46)
  t53 = r1 ** 2
  t56 = s2 / t53 / r1
  t66 = jnp.where(r1 <= p.dens_threshold, 0, -0.2e1 / 0.3e1 * t3 * t50 * t26 * t28 * (0.1e1 + 0.2105e-2 * t56) / (0.1e1 + 0.119e-3 * t56))
  res = t42 + t66
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = jnp.sqrt(jnp.pi)
  t5 = 0.1e1 <= p.zeta_threshold
  t6 = p.zeta_threshold - 0.1e1
  t8 = jnp.where(t5, -t6, 0)
  t9 = jnp.where(t5, t6, t8)
  t10 = 0.1e1 + t9
  t12 = jnp.sqrt(p.zeta_threshold)
  t14 = jnp.sqrt(t10)
  t16 = jnp.where(t10 <= p.zeta_threshold, t12 * p.zeta_threshold, t14 * t10)
  t18 = jnp.sqrt(0.2e1)
  t20 = jnp.sqrt(r0)
  t21 = r0 ** 2
  t24 = s0 / t21 / r0
  t34 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.2e1 / 0.3e1 / t3 * t16 * t18 * t20 * (0.1e1 + 0.421e-2 * t24) / (0.1e1 + 0.238e-3 * t24))
  res = 0.2e1 * t34
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