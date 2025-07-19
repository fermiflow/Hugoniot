"""Generated from mgga_x_br89.mpl."""

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
  t18 = jnp.cbrt(p.zeta_threshold)
  t19 = t18 * p.zeta_threshold
  t20 = jnp.cbrt(t16)
  t22 = jnp.where(t16 <= p.zeta_threshold, t19, t20 * t16)
  t23 = jnp.cbrt(t2)
  t26 = jnp.cbrt(0.1e1 / jnp.pi)
  t28 = jnp.cbrt(4)
  t29 = 0.1e1 / t26 * t28
  t31 = jnp.cbrt(r0)
  t32 = t31 ** 2
  t34 = 0.1e1 / t32 / r0
  t41 = r0 ** 2
  t46 = l0 * t34 / 0.6e1 - 0.2e1 / 0.3e1 * params.gamma * tau0 * t34 + params.gamma * s0 / t32 / t41 / 0.12e2
  t47 = jnp.abs(t46)
  t50 = jnp.where(0. < t46, 0.5e-12, -0.5e-12)
  t51 = jnp.where(t47 < 0.5e-12, t50, t46)
  t52 = br89_x(t51)
  t54 = jnp.exp(t52 / 0.3e1)
  t55 = jnp.exp(-t52)
  t62 = jnp.cbrt(6)
  t63 = t62 ** 2
  t64 = jnp.pi ** 2
  t65 = jnp.cbrt(t64)
  t66 = t65 ** 2
  t68 = 0.3e1 / 0.1e2 * t63 * t66
  t69 = tau0 * t34
  t70 = t68 - t69
  t71 = t68 + t69
  t74 = t70 ** 2
  t76 = t71 ** 2
  t81 = t74 ** 2
  t83 = t76 ** 2
  t94 = jnp.where(r0 <= p.dens_threshold, 0, -t22 * t23 * t29 * t54 * (0.1e1 - t55 * (0.1e1 + t52 / 0.2e1)) / t52 * (0.1e1 + params.at * (t70 / t71 - 0.2e1 * t74 * t70 / t76 / t71 + t81 * t70 / t83 / t71)) / 0.4e1)
  t96 = jnp.where(t6, t11, -t13)
  t97 = jnp.where(t10, t7, t96)
  t98 = 0.1e1 + t97
  t100 = jnp.cbrt(t98)
  t102 = jnp.where(t98 <= p.zeta_threshold, t19, t100 * t98)
  t105 = jnp.cbrt(r1)
  t106 = t105 ** 2
  t108 = 0.1e1 / t106 / r1
  t115 = r1 ** 2
  t120 = l1 * t108 / 0.6e1 - 0.2e1 / 0.3e1 * params.gamma * tau1 * t108 + params.gamma * s2 / t106 / t115 / 0.12e2
  t121 = jnp.abs(t120)
  t124 = jnp.where(0. < t120, 0.5e-12, -0.5e-12)
  t125 = jnp.where(t121 < 0.5e-12, t124, t120)
  t126 = br89_x(t125)
  t128 = jnp.exp(t126 / 0.3e1)
  t129 = jnp.exp(-t126)
  t136 = tau1 * t108
  t137 = t68 - t136
  t138 = t68 + t136
  t141 = t137 ** 2
  t143 = t138 ** 2
  t148 = t141 ** 2
  t150 = t143 ** 2
  t161 = jnp.where(r1 <= p.dens_threshold, 0, -t102 * t23 * t29 * t128 * (0.1e1 - t129 * (0.1e1 + t126 / 0.2e1)) / t126 * (0.1e1 + params.at * (t137 / t138 - 0.2e1 * t141 * t137 / t143 / t138 + t148 * t137 / t150 / t138)) / 0.4e1)
  res = t94 + t161
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = 0.1e1 <= p.zeta_threshold
  t4 = p.zeta_threshold - 0.1e1
  t6 = jnp.where(t3, -t4, 0)
  t7 = jnp.where(t3, t4, t6)
  t8 = 0.1e1 + t7
  t10 = jnp.cbrt(p.zeta_threshold)
  t12 = jnp.cbrt(t8)
  t14 = jnp.where(t8 <= p.zeta_threshold, t10 * p.zeta_threshold, t12 * t8)
  t15 = jnp.cbrt(r0)
  t18 = jnp.cbrt(0.1e1 / jnp.pi)
  t20 = jnp.cbrt(4)
  t23 = jnp.cbrt(2)
  t24 = t23 ** 2
  t26 = t15 ** 2
  t28 = 0.1e1 / t26 / r0
  t36 = r0 ** 2
  t42 = l0 * t24 * t28 / 0.6e1 - 0.2e1 / 0.3e1 * params.gamma * tau0 * t24 * t28 + params.gamma * s0 * t24 / t26 / t36 / 0.12e2
  t43 = jnp.abs(t42)
  t46 = jnp.where(0. < t42, 0.5e-12, -0.5e-12)
  t47 = jnp.where(t43 < 0.5e-12, t46, t42)
  t48 = br89_x(t47)
  t50 = jnp.exp(t48 / 0.3e1)
  t51 = jnp.exp(-t48)
  t58 = jnp.cbrt(6)
  t59 = t58 ** 2
  t60 = jnp.pi ** 2
  t61 = jnp.cbrt(t60)
  t62 = t61 ** 2
  t64 = 0.3e1 / 0.1e2 * t59 * t62
  t66 = tau0 * t24 * t28
  t67 = t64 - t66
  t68 = t64 + t66
  t71 = t67 ** 2
  t73 = t68 ** 2
  t78 = t71 ** 2
  t80 = t73 ** 2
  t91 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -t14 * t15 / t18 * t20 * t50 * (0.1e1 - t51 * (0.1e1 + t48 / 0.2e1)) / t48 * (0.1e1 + params.at * (t67 / t68 - 0.2e1 * t71 * t67 / t73 / t68 + t78 * t67 / t80 / t68)) / 0.4e1)
  res = 0.2e1 * t91
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