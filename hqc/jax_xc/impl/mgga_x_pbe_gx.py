"""Generated from mgga_x_pbe_gx.mpl."""

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
  t28 = jnp.cbrt(t6)
  t29 = jnp.cbrt(2)
  t30 = t2 ** 2
  t32 = jnp.cbrt(4)
  t34 = 0.8e1 / 0.27e2 * t29 * t30 * t32
  t35 = jnp.cbrt(r0)
  t36 = t35 ** 2
  t40 = r0 ** 2
  t43 = s0 / t36 / t40
  t46 = jnp.cbrt(6)
  t48 = jnp.pi ** 2
  t49 = jnp.cbrt(t48)
  t50 = t49 ** 2
  t51 = 0.1e1 / t50
  t52 = (tau0 / t36 / r0 - t43 / 0.8e1) * t46 * t51
  t59 = 0.1e1 - t34
  t64 = 0.5e1 / 0.9e1 * t52
  t65 = 0.1e1 - t64
  t66 = Heaviside(t65)
  t74 = Heaviside(-t65)
  t84 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * ((t34 + 0.5e1 / 0.9e1 * t52 * (0.827411 - 0.35753333333333333333 * t52) / (0.1e1 - 0.45341611111111111111 * t52) * t59) * t66 + (0.1e1 + 0.148 * t65 / (0.1e1 + t64)) * t74) / (0.1e1 + 0.1015549e-2 * t43))
  t86 = jnp.where(t10, t15, -t17)
  t87 = jnp.where(t14, t11, t86)
  t88 = 0.1e1 + t87
  t90 = jnp.cbrt(t88)
  t92 = jnp.where(t88 <= p.zeta_threshold, t23, t90 * t88)
  t94 = jnp.cbrt(r1)
  t95 = t94 ** 2
  t99 = r1 ** 2
  t102 = s2 / t95 / t99
  t106 = (tau1 / t95 / r1 - t102 / 0.8e1) * t46 * t51
  t117 = 0.5e1 / 0.9e1 * t106
  t118 = 0.1e1 - t117
  t119 = Heaviside(t118)
  t127 = Heaviside(-t118)
  t137 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t92 * t28 * ((t34 + 0.5e1 / 0.9e1 * t106 * (0.827411 - 0.35753333333333333333 * t106) / (0.1e1 - 0.45341611111111111111 * t106) * t59) * t119 + (0.1e1 + 0.148 * t118 / (0.1e1 + t117)) * t127) / (0.1e1 + 0.1015549e-2 * t102))
  res = t84 + t137
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
  t20 = jnp.cbrt(r0)
  t21 = jnp.cbrt(2)
  t22 = t3 ** 2
  t24 = jnp.cbrt(4)
  t26 = 0.8e1 / 0.27e2 * t21 * t22 * t24
  t27 = t21 ** 2
  t29 = t20 ** 2
  t34 = r0 ** 2
  t37 = s0 * t27 / t29 / t34
  t40 = jnp.cbrt(6)
  t42 = jnp.pi ** 2
  t43 = jnp.cbrt(t42)
  t44 = t43 ** 2
  t46 = (tau0 * t27 / t29 / r0 - t37 / 0.8e1) * t40 / t44
  t58 = 0.5e1 / 0.9e1 * t46
  t59 = 0.1e1 - t58
  t60 = Heaviside(t59)
  t68 = Heaviside(-t59)
  t78 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * ((t26 + 0.5e1 / 0.9e1 * t46 * (0.827411 - 0.35753333333333333333 * t46) / (0.1e1 - 0.45341611111111111111 * t46) * (0.1e1 - t26)) * t60 + (0.1e1 + 0.148 * t59 / (0.1e1 + t58)) * t68) / (0.1e1 + 0.1015549e-2 * t37))
  res = 0.2e1 * t78
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