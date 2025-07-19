"""Generated from gga_x_pbetrans.mpl."""

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
  t29 = jnp.pi ** 2
  t30 = jnp.cbrt(t29)
  t31 = t2 * t30
  t32 = jnp.cbrt(6)
  t33 = t32 ** 2
  t35 = t33 / t30
  t36 = jnp.sqrt(s0)
  t37 = jnp.cbrt(r0)
  t46 = jnp.exp(-0.2e1 * t31 * (t35 * t36 / t37 / r0 / 0.12e2 - 0.3e1))
  t49 = 0.413 / (0.1e1 + t46)
  t50 = 0.1227e1 - t49
  t51 = t30 ** 2
  t53 = t32 / t51
  t54 = r0 ** 2
  t55 = t37 ** 2
  t70 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + t50 * (0.1e1 - t50 / (0.1227e1 - t49 + 0.91249999999999999998e-2 * t53 * s0 / t55 / t54))))
  t72 = jnp.where(t10, t15, -t17)
  t73 = jnp.where(t14, t11, t72)
  t74 = 0.1e1 + t73
  t76 = jnp.cbrt(t74)
  t78 = jnp.where(t74 <= p.zeta_threshold, t23, t76 * t74)
  t80 = jnp.sqrt(s2)
  t81 = jnp.cbrt(r1)
  t90 = jnp.exp(-0.2e1 * t31 * (t35 * t80 / t81 / r1 / 0.12e2 - 0.3e1))
  t93 = 0.413 / (0.1e1 + t90)
  t94 = 0.1227e1 - t93
  t95 = r1 ** 2
  t96 = t81 ** 2
  t111 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t78 * t27 * (0.1e1 + t94 * (0.1e1 - t94 / (0.1227e1 - t93 + 0.91249999999999999998e-2 * t53 * s2 / t96 / t95))))
  res = t70 + t111
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
  t21 = jnp.pi ** 2
  t22 = jnp.cbrt(t21)
  t24 = jnp.cbrt(6)
  t25 = t24 ** 2
  t28 = jnp.sqrt(s0)
  t29 = jnp.cbrt(2)
  t39 = jnp.exp(-0.2e1 * t3 * t22 * (t25 / t22 * t28 * t29 / t19 / r0 / 0.12e2 - 0.3e1))
  t42 = 0.413 / (0.1e1 + t39)
  t43 = 0.1227e1 - t42
  t44 = t22 ** 2
  t47 = t29 ** 2
  t49 = r0 ** 2
  t50 = t19 ** 2
  t65 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + t43 * (0.1e1 - t43 / (0.1227e1 - t42 + 0.91249999999999999998e-2 * t24 / t44 * s0 * t47 / t50 / t49))))
  res = 0.2e1 * t65
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