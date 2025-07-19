"""Generated from mgga_x_2d_prhg07.mpl."""

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
  t3 = 0.1e1 / t2
  t6 = 0.2e1 * r0 * t3 <= p.zeta_threshold
  t7 = p.zeta_threshold - 0.1e1
  t10 = 0.2e1 * r1 * t3 <= p.zeta_threshold
  t11 = -t7
  t13 = (r0 - r1) * t3
  t14 = jnp.where(t10, t11, t13)
  t15 = jnp.where(t6, t7, t14)
  t16 = 0.1e1 + t15
  t18 = jnp.sqrt(p.zeta_threshold)
  t19 = t18 * p.zeta_threshold
  t20 = jnp.sqrt(t16)
  t22 = jnp.where(t16 <= p.zeta_threshold, t19, t20 * t16)
  t24 = jnp.sqrt(0.2e1)
  t25 = jnp.sqrt(t2)
  t26 = t24 * t25
  t27 = r0 ** 2
  t28 = 0.1e1 / t27
  t37 = 0.1e1 / jnp.pi
  t38 = (l0 * t28 / 0.4e1 - tau0 * t28 + s0 / t27 / r0 / 0.8e1) * t37
  t40 = jnp.where(-0.9999999999 < t38, t38, -0.9999999999)
  t41 = jnp.exp(-1)
  t43 = lambertw(t40 * t41)
  t46 = jax.scipy.special.i0(t43 / 0.2e1 + 0.1e1 / 0.2e1)
  t50 = jnp.where(r0 <= p.dens_threshold, 0, -jnp.pi * t22 * t26 * t46 / 0.8e1)
  t52 = jnp.where(t6, t11, -t13)
  t53 = jnp.where(t10, t7, t52)
  t54 = 0.1e1 + t53
  t56 = jnp.sqrt(t54)
  t58 = jnp.where(t54 <= p.zeta_threshold, t19, t56 * t54)
  t60 = r1 ** 2
  t61 = 0.1e1 / t60
  t70 = (l1 * t61 / 0.4e1 - tau1 * t61 + s2 / t60 / r1 / 0.8e1) * t37
  t72 = jnp.where(-0.9999999999 < t70, t70, -0.9999999999)
  t74 = lambertw(t72 * t41)
  t77 = jax.scipy.special.i0(t74 / 0.2e1 + 0.1e1 / 0.2e1)
  t81 = jnp.where(r1 <= p.dens_threshold, 0, -jnp.pi * t58 * t26 * t77 / 0.8e1)
  res = t50 + t81
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = 0.1e1 <= p.zeta_threshold
  t4 = p.zeta_threshold - 0.1e1
  t6 = jnp.where(t3, -t4, 0)
  t7 = jnp.where(t3, t4, t6)
  t8 = 0.1e1 + t7
  t10 = jnp.sqrt(p.zeta_threshold)
  t12 = jnp.sqrt(t8)
  t14 = jnp.where(t8 <= p.zeta_threshold, t10 * p.zeta_threshold, t12 * t8)
  t16 = jnp.sqrt(0.2e1)
  t17 = jnp.sqrt(r0)
  t19 = r0 ** 2
  t20 = 0.1e1 / t19
  t31 = (l0 * t20 / 0.2e1 - 0.2e1 * tau0 * t20 + s0 / t19 / r0 / 0.4e1) / jnp.pi
  t33 = jnp.where(-0.9999999999 < t31, t31, -0.9999999999)
  t34 = jnp.exp(-1)
  t36 = lambertw(t33 * t34)
  t39 = jax.scipy.special.i0(t36 / 0.2e1 + 0.1e1 / 0.2e1)
  t43 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -jnp.pi * t14 * t16 * t17 * t39 / 0.8e1)
  res = 0.2e1 * t43
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