"""Generated from mgga_x_lta.mpl."""

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
  t2 = jnp.cbrt(3)
  t3 = jnp.cbrt(jnp.pi)
  t5 = t2 / t3
  t6 = r0 + r1
  t7 = 0.1e1 / t6
  t10 = 0.2e1 * r0 * t7 <= p.zeta_threshold
  t11 = p.zeta_threshold - 0.1e1
  t14 = 0.2e1 * r1 * t7 <= p.zeta_threshold
  t15 = -t11
  t17 = (r0 - r1) * t7
  t18 = jnp.where(t14, t15, t17)
  t19 = jnp.where(t10, t11, t18)
  t20 = 0.1e1 + t19
  t22 = jnp.cbrt(p.zeta_threshold)
  t23 = t22 * p.zeta_threshold
  t24 = jnp.cbrt(t20)
  t26 = jnp.where(t20 <= p.zeta_threshold, t23, t24 * t20)
  t27 = jnp.cbrt(t6)
  t29 = jnp.cbrt(r0)
  t30 = t29 ** 2
  t34 = jnp.cbrt(6)
  t35 = jnp.pi ** 2
  t36 = jnp.cbrt(t35)
  t37 = t36 ** 2
  t39 = t34 / t37
  t42 = 0.4e1 / 0.5e1 * params.ltafrac
  t43 = (0.5e1 / 0.9e1 * tau0 / t30 / r0 * t39) ** t42
  t47 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * t43)
  t49 = jnp.where(t10, t15, -t17)
  t50 = jnp.where(t14, t11, t49)
  t51 = 0.1e1 + t50
  t53 = jnp.cbrt(t51)
  t55 = jnp.where(t51 <= p.zeta_threshold, t23, t53 * t51)
  t57 = jnp.cbrt(r1)
  t58 = t57 ** 2
  t64 = (0.5e1 / 0.9e1 * tau1 / t58 / r1 * t39) ** t42
  t68 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t55 * t27 * t64)
  res = t47 + t68
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = jnp.cbrt(3)
  t4 = jnp.cbrt(jnp.pi)
  t7 = 0.1e1 <= p.zeta_threshold
  t8 = p.zeta_threshold - 0.1e1
  t10 = jnp.where(t7, -t8, 0)
  t11 = jnp.where(t7, t8, t10)
  t12 = 0.1e1 + t11
  t14 = jnp.cbrt(p.zeta_threshold)
  t16 = jnp.cbrt(t12)
  t18 = jnp.where(t12 <= p.zeta_threshold, t14 * p.zeta_threshold, t16 * t12)
  t19 = jnp.cbrt(r0)
  t21 = jnp.cbrt(2)
  t22 = t21 ** 2
  t24 = t19 ** 2
  t27 = jnp.cbrt(6)
  t29 = jnp.pi ** 2
  t30 = jnp.cbrt(t29)
  t31 = t30 ** 2
  t37 = (0.5e1 / 0.9e1 * tau0 * t22 / t24 / r0 * t27 / t31) ** (0.4e1 / 0.5e1 * params.ltafrac)
  t41 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * t37)
  res = 0.2e1 * t41
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