"""Generated from gga_x_htbs.mpl."""

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
  t29 = jnp.cbrt(6)
  t30 = t29 ** 2
  t31 = jnp.pi ** 2
  t32 = jnp.cbrt(t31)
  t34 = t30 / t32
  t35 = jnp.sqrt(s0)
  t36 = jnp.cbrt(r0)
  t40 = t34 * t35 / t36 / r0
  t41 = t40 / 0.12e2
  t43 = t32 ** 2
  t44 = 0.1e1 / t43
  t45 = t29 * t44
  t46 = r0 ** 2
  t47 = t36 ** 2
  t49 = 0.1e1 / t47 / t46
  t50 = s0 * t49
  t51 = t45 * t50
  t54 = jnp.exp(-t51 / 0.24e2)
  t60 = t30 / t32 / t31
  t61 = s0 ** 2
  t62 = t46 ** 2
  t67 = t60 * t61 / t36 / t62 / r0
  t70 = jnp.log(0.1e1 + 0.13780328706878157639e-4 * t67)
  t74 = 0.1804e1 - 0.646416 / (0.804 + 0.5e1 / 0.972e3 * t51 + 0.4002424276710846245e-2 * t45 * t50 * t54 + t70)
  t76 = t31 * t29
  t81 = jnp.exp(-0.11526490914032134466e-2 * t76 * t44 * s0 * t49)
  t83 = 0.1804e1 - 0.804 * t81
  t84 = 0.190125 * t40
  t85 = 0.195 * t51
  t86 = 0.1e1 / t31
  t91 = 0.86979166666666666668e-1 * t86 * t35 * s0 / t62
  t92 = 0.26041666666666666667e-2 * t67
  t95 = t29 / t43 / t31
  t102 = 0.16276041666666666667e-3 * t95 * t35 * t61 / t47 / t62 / t46
  t108 = jnp.where(0.26e1 < t41, t83, (-0.40608 + t84 - t85 + t91 - t92 + t102) * t83 + (0.140608e1 - t84 + t85 - t91 + t92 - t102) * t74)
  t109 = jnp.where(t41 < 0.6, t74, t108)
  t113 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * t109)
  t115 = jnp.where(t10, t15, -t17)
  t116 = jnp.where(t14, t11, t115)
  t117 = 0.1e1 + t116
  t119 = jnp.cbrt(t117)
  t121 = jnp.where(t117 <= p.zeta_threshold, t23, t119 * t117)
  t123 = jnp.sqrt(s2)
  t124 = jnp.cbrt(r1)
  t128 = t34 * t123 / t124 / r1
  t129 = t128 / 0.12e2
  t131 = r1 ** 2
  t132 = t124 ** 2
  t134 = 0.1e1 / t132 / t131
  t135 = s2 * t134
  t136 = t45 * t135
  t139 = jnp.exp(-t136 / 0.24e2)
  t143 = s2 ** 2
  t144 = t131 ** 2
  t149 = t60 * t143 / t124 / t144 / r1
  t152 = jnp.log(0.1e1 + 0.13780328706878157639e-4 * t149)
  t156 = 0.1804e1 - 0.646416 / (0.804 + 0.5e1 / 0.972e3 * t136 + 0.4002424276710846245e-2 * t45 * t135 * t139 + t152)
  t162 = jnp.exp(-0.11526490914032134466e-2 * t76 * t44 * s2 * t134)
  t164 = 0.1804e1 - 0.804 * t162
  t165 = 0.190125 * t128
  t166 = 0.195 * t136
  t171 = 0.86979166666666666668e-1 * t86 * t123 * s2 / t144
  t172 = 0.26041666666666666667e-2 * t149
  t179 = 0.16276041666666666667e-3 * t95 * t123 * t143 / t132 / t144 / t131
  t185 = jnp.where(0.26e1 < t129, t164, (-0.40608 + t165 - t166 + t171 - t172 + t179) * t164 + (0.140608e1 - t165 + t166 - t171 + t172 - t179) * t156)
  t186 = jnp.where(t129 < 0.6, t156, t185)
  t190 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t121 * t27 * t186)
  res = t113 + t190
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
  t21 = jnp.cbrt(6)
  t22 = t21 ** 2
  t23 = jnp.pi ** 2
  t24 = jnp.cbrt(t23)
  t27 = jnp.sqrt(s0)
  t28 = jnp.cbrt(2)
  t33 = t22 / t24 * t27 * t28 / t19 / r0
  t34 = t33 / 0.12e2
  t36 = t24 ** 2
  t37 = 0.1e1 / t36
  t38 = t21 * t37
  t39 = t28 ** 2
  t41 = r0 ** 2
  t42 = t19 ** 2
  t44 = 0.1e1 / t42 / t41
  t45 = s0 * t39 * t44
  t46 = t38 * t45
  t51 = jnp.exp(-t46 / 0.24e2)
  t58 = s0 ** 2
  t60 = t41 ** 2
  t65 = t22 / t24 / t23 * t58 * t28 / t19 / t60 / r0
  t68 = jnp.log(0.1e1 + 0.27560657413756315278e-4 * t65)
  t72 = 0.1804e1 - 0.646416 / (0.804 + 0.5e1 / 0.972e3 * t46 + 0.4002424276710846245e-2 * t38 * s0 * t39 * t44 * t51 + t68)
  t78 = jnp.exp(-0.11526490914032134466e-2 * t23 * t21 * t37 * t45)
  t80 = 0.1804e1 - 0.804 * t78
  t81 = 0.190125 * t33
  t82 = 0.195 * t46
  t88 = 0.17395833333333333334 / t23 * t27 * s0 / t60
  t89 = 0.52083333333333333334e-2 * t65
  t100 = 0.32552083333333333334e-3 * t21 / t36 / t23 * t27 * t58 * t39 / t42 / t60 / t41
  t106 = jnp.where(0.26e1 < t34, t80, (-0.40608 + t81 - t82 + t88 - t89 + t100) * t80 + (0.140608e1 - t81 + t82 - t88 + t89 - t100) * t72)
  t107 = jnp.where(t34 < 0.6, t72, t106)
  t111 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * t107)
  res = 0.2e1 * t111
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