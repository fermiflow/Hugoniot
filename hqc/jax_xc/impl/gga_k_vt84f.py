"""Generated from gga_k_vt84f.mpl."""

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
  t3 = t2 ** 2
  t4 = jnp.cbrt(jnp.pi)
  t6 = t3 * t4 * jnp.pi
  t7 = r0 + r1
  t8 = 0.1e1 / t7
  t11 = 0.2e1 * r0 * t8 <= p.zeta_threshold
  t12 = p.zeta_threshold - 0.1e1
  t15 = 0.2e1 * r1 * t8 <= p.zeta_threshold
  t16 = -t12
  t18 = (r0 - r1) * t8
  t19 = jnp.where(t15, t16, t18)
  t20 = jnp.where(t11, t12, t19)
  t21 = 0.1e1 + t20
  t23 = jnp.cbrt(p.zeta_threshold)
  t24 = t23 ** 2
  t25 = t24 * p.zeta_threshold
  t26 = jnp.cbrt(t21)
  t27 = t26 ** 2
  t29 = jnp.where(t21 <= p.zeta_threshold, t25, t27 * t21)
  t30 = jnp.cbrt(t7)
  t31 = t30 ** 2
  t33 = jnp.cbrt(6)
  t34 = t33 ** 2
  t35 = jnp.pi ** 2
  t36 = jnp.cbrt(t35)
  t38 = t34 / t36
  t39 = jnp.sqrt(s0)
  t40 = jnp.cbrt(r0)
  t45 = t38 * t39 / t40 / r0 / 0.12e2
  t46 = jnp.sqrt(2.220446049250313e-16)
  t49 = (-params.mu + params.alpha + 0.5e1 / 0.3e1) * t33
  t50 = t36 ** 2
  t51 = 0.1e1 / t50
  t53 = r0 ** 2
  t54 = t40 ** 2
  t61 = params.mu ** 2
  t63 = (params.mu * params.alpha + t61 - params.alpha) * t34
  t65 = 0.1e1 / t36 / t35
  t66 = s0 ** 2
  t68 = t53 ** 2
  t77 = jnp.where(t46 < t45, t45, t46)
  t78 = t77 ** 2
  t79 = params.mu * t78
  t81 = jnp.exp(-params.alpha * t78)
  t86 = t78 ** 2
  t88 = jnp.exp(-params.alpha * t86)
  t95 = jnp.where(t45 <= t46, 0.1e1 + t49 * t51 * s0 / t54 / t53 / 0.24e2 + t63 * t65 * t66 / t40 / t68 / r0 / 0.576e3, 0.1e1 - t79 * t81 / (0.1e1 + t79) + (0.1e1 - t88) * (0.1e1 / t78 - 0.1e1) + 0.5e1 / 0.3e1 * t78)
  t99 = jnp.where(r0 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t29 * t31 * t95)
  t101 = jnp.where(t11, t16, -t18)
  t102 = jnp.where(t15, t12, t101)
  t103 = 0.1e1 + t102
  t105 = jnp.cbrt(t103)
  t106 = t105 ** 2
  t108 = jnp.where(t103 <= p.zeta_threshold, t25, t106 * t103)
  t110 = jnp.sqrt(s2)
  t111 = jnp.cbrt(r1)
  t116 = t38 * t110 / t111 / r1 / 0.12e2
  t119 = r1 ** 2
  t120 = t111 ** 2
  t126 = s2 ** 2
  t128 = t119 ** 2
  t137 = jnp.where(t46 < t116, t116, t46)
  t138 = t137 ** 2
  t139 = params.mu * t138
  t141 = jnp.exp(-params.alpha * t138)
  t146 = t138 ** 2
  t148 = jnp.exp(-params.alpha * t146)
  t155 = jnp.where(t116 <= t46, 0.1e1 + t49 * t51 * s2 / t120 / t119 / 0.24e2 + t63 * t65 * t126 / t111 / t128 / r1 / 0.576e3, 0.1e1 - t139 * t141 / (0.1e1 + t139) + (0.1e1 - t148) * (0.1e1 / t138 - 0.1e1) + 0.5e1 / 0.3e1 * t138)
  t159 = jnp.where(r1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t108 * t31 * t155)
  res = t99 + t159
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = jnp.cbrt(3)
  t4 = t3 ** 2
  t5 = jnp.cbrt(jnp.pi)
  t8 = 0.1e1 <= p.zeta_threshold
  t9 = p.zeta_threshold - 0.1e1
  t11 = jnp.where(t8, -t9, 0)
  t12 = jnp.where(t8, t9, t11)
  t13 = 0.1e1 + t12
  t15 = jnp.cbrt(p.zeta_threshold)
  t16 = t15 ** 2
  t18 = jnp.cbrt(t13)
  t19 = t18 ** 2
  t21 = jnp.where(t13 <= p.zeta_threshold, t16 * p.zeta_threshold, t19 * t13)
  t22 = jnp.cbrt(r0)
  t23 = t22 ** 2
  t25 = jnp.cbrt(6)
  t26 = t25 ** 2
  t27 = jnp.pi ** 2
  t28 = jnp.cbrt(t27)
  t31 = jnp.sqrt(s0)
  t32 = jnp.cbrt(2)
  t38 = t26 / t28 * t31 * t32 / t22 / r0 / 0.12e2
  t39 = jnp.sqrt(2.220446049250313e-16)
  t43 = t28 ** 2
  t46 = t32 ** 2
  t48 = r0 ** 2
  t55 = params.mu ** 2
  t61 = s0 ** 2
  t63 = t48 ** 2
  t72 = jnp.where(t39 < t38, t38, t39)
  t73 = t72 ** 2
  t74 = params.mu * t73
  t76 = jnp.exp(-params.alpha * t73)
  t81 = t73 ** 2
  t83 = jnp.exp(-params.alpha * t81)
  t90 = jnp.where(t38 <= t39, 0.1e1 + (-params.mu + params.alpha + 0.5e1 / 0.3e1) * t25 / t43 * s0 * t46 / t23 / t48 / 0.24e2 + (params.mu * params.alpha + t55 - params.alpha) * t26 / t28 / t27 * t61 * t32 / t22 / t63 / r0 / 0.288e3, 0.1e1 - t74 * t76 / (0.1e1 + t74) + (0.1e1 - t83) * (0.1e1 / t73 - 0.1e1) + 0.5e1 / 0.3e1 * t73)
  t94 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t4 * t5 * jnp.pi * t21 * t23 * t90)
  res = 0.2e1 * t94
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