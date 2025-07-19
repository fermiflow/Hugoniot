"""Generated from mgga_k_pc07.mpl."""

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
  t34 = jnp.pi ** 2
  t35 = jnp.cbrt(t34)
  t36 = t35 ** 2
  t38 = t33 / t36
  t39 = r0 ** 2
  t40 = jnp.cbrt(r0)
  t41 = t40 ** 2
  t45 = t38 * s0 / t41 / t39
  t46 = 0.5e1 / 0.72e2 * t45
  t53 = t33 ** 2
  t56 = t53 / t35 / t34
  t57 = l0 ** 2
  t63 = t56 * t57 / t40 / t39 / r0 / 0.5832e4
  t64 = t39 ** 2
  t70 = t56 * s0 / t40 / t64 * l0 / 0.5184e4
  t71 = s0 ** 2
  t77 = t56 * t71 / t40 / t64 / r0 / 0.17496e5
  t80 = (t63 - t70 + t77) ** 2
  t82 = (0.1e1 + t46) ** 2
  t86 = jnp.sqrt(0.1e1 + t80 / t82)
  t89 = (0.1e1 + 0.5e1 / 0.648e3 * t45 + 0.5e1 / 0.54e2 * t38 * l0 / t41 / r0 + t63 - t70 + t77) / t86 - t46
  t90 = params.a / 0.4e2
  t92 = 0.39e2 / 0.4e2 * params.a
  t94 = params.a * params.b
  t96 = jnp.where(t89 < t90, t90, t89)
  t98 = jnp.where(t96 < t92, t96, t92)
  t99 = 0.1e1 / t98
  t101 = jnp.exp(-t94 * t99)
  t105 = jnp.exp(-params.a / (params.a - t98))
  t107 = (0.1e1 + t105) ** params.b
  t110 = jnp.exp(-params.a * t99)
  t112 = (t110 + t105) ** params.b
  t115 = jnp.where(t92 <= t89, 1, t101 * t107 / t112)
  t116 = jnp.where(t89 <= t90, 0, t115)
  t122 = jnp.where(r0 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t29 * t31 * (t89 * t116 + t46))
  t124 = jnp.where(t11, t16, -t18)
  t125 = jnp.where(t15, t12, t124)
  t126 = 0.1e1 + t125
  t128 = jnp.cbrt(t126)
  t129 = t128 ** 2
  t131 = jnp.where(t126 <= p.zeta_threshold, t25, t129 * t126)
  t133 = r1 ** 2
  t134 = jnp.cbrt(r1)
  t135 = t134 ** 2
  t139 = t38 * s2 / t135 / t133
  t140 = 0.5e1 / 0.72e2 * t139
  t147 = l1 ** 2
  t153 = t56 * t147 / t134 / t133 / r1 / 0.5832e4
  t154 = t133 ** 2
  t160 = t56 * s2 / t134 / t154 * l1 / 0.5184e4
  t161 = s2 ** 2
  t167 = t56 * t161 / t134 / t154 / r1 / 0.17496e5
  t170 = (t153 - t160 + t167) ** 2
  t172 = (0.1e1 + t140) ** 2
  t176 = jnp.sqrt(0.1e1 + t170 / t172)
  t179 = (0.1e1 + 0.5e1 / 0.648e3 * t139 + 0.5e1 / 0.54e2 * t38 * l1 / t135 / r1 + t153 - t160 + t167) / t176 - t140
  t183 = jnp.where(t179 < t90, t90, t179)
  t185 = jnp.where(t183 < t92, t183, t92)
  t186 = 0.1e1 / t185
  t188 = jnp.exp(-t94 * t186)
  t192 = jnp.exp(-params.a / (params.a - t185))
  t194 = (0.1e1 + t192) ** params.b
  t197 = jnp.exp(-params.a * t186)
  t199 = (t197 + t192) ** params.b
  t202 = jnp.where(t92 <= t179, 1, t188 * t194 / t199)
  t203 = jnp.where(t179 <= t90, 0, t202)
  t209 = jnp.where(r1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t131 * t31 * (t179 * t203 + t140))
  res = t122 + t209
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
  t26 = jnp.pi ** 2
  t27 = jnp.cbrt(t26)
  t28 = t27 ** 2
  t30 = t25 / t28
  t31 = jnp.cbrt(2)
  t32 = t31 ** 2
  t34 = r0 ** 2
  t38 = t30 * s0 * t32 / t23 / t34
  t39 = 0.5e1 / 0.72e2 * t38
  t47 = t25 ** 2
  t50 = t47 / t27 / t26
  t51 = l0 ** 2
  t58 = t50 * t51 * t31 / t22 / t34 / r0 / 0.2916e4
  t60 = t34 ** 2
  t66 = t50 * s0 * t31 / t22 / t60 * l0 / 0.2592e4
  t67 = s0 ** 2
  t74 = t50 * t67 * t31 / t22 / t60 / r0 / 0.8748e4
  t77 = (t58 - t66 + t74) ** 2
  t79 = (0.1e1 + t39) ** 2
  t83 = jnp.sqrt(0.1e1 + t77 / t79)
  t86 = (0.1e1 + 0.5e1 / 0.648e3 * t38 + 0.5e1 / 0.54e2 * t30 * l0 * t32 / t23 / r0 + t58 - t66 + t74) / t83 - t39
  t87 = params.a / 0.4e2
  t89 = 0.39e2 / 0.4e2 * params.a
  t93 = jnp.where(t86 < t87, t87, t86)
  t95 = jnp.where(t93 < t89, t93, t89)
  t96 = 0.1e1 / t95
  t98 = jnp.exp(-params.a * params.b * t96)
  t102 = jnp.exp(-params.a / (params.a - t95))
  t104 = (0.1e1 + t102) ** params.b
  t107 = jnp.exp(-params.a * t96)
  t109 = (t107 + t102) ** params.b
  t112 = jnp.where(t89 <= t86, 1, t98 * t104 / t109)
  t113 = jnp.where(t86 <= t87, 0, t112)
  t119 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t4 * t5 * jnp.pi * t21 * t23 * (t86 * t113 + t39))
  res = 0.2e1 * t119
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