"""Generated from mgga_x_m08.mpl."""

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
  t30 = jnp.pi ** 2
  t31 = jnp.cbrt(t30)
  t32 = t31 ** 2
  t34 = t29 / t32
  t35 = r0 ** 2
  t36 = jnp.cbrt(r0)
  t37 = t36 ** 2
  t41 = t34 * s0 / t37 / t35
  t47 = params.a[0]
  t48 = params.a[1]
  t49 = t29 ** 2
  t51 = 0.3e1 / 0.1e2 * t49 * t32
  t54 = tau0 / t37 / r0
  t55 = t51 - t54
  t57 = t51 + t54
  t58 = 0.1e1 / t57
  t60 = params.a[2]
  t61 = t55 ** 2
  t63 = t57 ** 2
  t64 = 0.1e1 / t63
  t66 = params.a[3]
  t67 = t61 * t55
  t69 = t63 * t57
  t70 = 0.1e1 / t69
  t72 = params.a[4]
  t73 = t61 ** 2
  t75 = t63 ** 2
  t76 = 0.1e1 / t75
  t78 = params.a[5]
  t79 = t73 * t55
  t82 = 0.1e1 / t75 / t57
  t84 = params.a[6]
  t85 = t73 * t61
  t88 = 0.1e1 / t75 / t63
  t90 = params.a[7]
  t91 = t73 * t67
  t94 = 0.1e1 / t75 / t69
  t96 = params.a[8]
  t97 = t73 ** 2
  t99 = t75 ** 2
  t100 = 0.1e1 / t99
  t102 = params.a[9]
  t103 = t97 * t55
  t106 = 0.1e1 / t99 / t57
  t108 = params.a[10]
  t109 = t97 * t61
  t112 = 0.1e1 / t99 / t63
  t114 = params.a[11]
  t115 = t97 * t67
  t118 = 0.1e1 / t99 / t69
  t120 = t47 + t48 * t55 * t58 + t60 * t61 * t64 + t66 * t67 * t70 + t72 * t73 * t76 + t78 * t79 * t82 + t84 * t85 * t88 + t90 * t91 * t94 + t96 * t97 * t100 + t102 * t103 * t106 + t108 * t109 * t112 + t114 * t115 * t118
  t123 = jnp.exp(-0.93189002206715572255e-2 * t41)
  t126 = params.b[0]
  t127 = params.b[1]
  t130 = params.b[2]
  t133 = params.b[3]
  t136 = params.b[4]
  t139 = params.b[5]
  t142 = params.b[6]
  t145 = params.b[7]
  t148 = params.b[8]
  t151 = params.b[9]
  t154 = params.b[10]
  t157 = params.b[11]
  t160 = t126 + t127 * t55 * t58 + t130 * t61 * t64 + t133 * t67 * t70 + t136 * t73 * t76 + t139 * t79 * t82 + t142 * t85 * t88 + t145 * t91 * t94 + t148 * t97 * t100 + t151 * t103 * t106 + t154 * t109 * t112 + t157 * t115 * t118
  t166 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * ((0.1804e1 - 0.646416 / (0.804 + 0.914625e-2 * t41)) * t120 + (0.1552e1 - 0.552 * t123) * t160))
  t168 = jnp.where(t10, t15, -t17)
  t169 = jnp.where(t14, t11, t168)
  t170 = 0.1e1 + t169
  t172 = jnp.cbrt(t170)
  t174 = jnp.where(t170 <= p.zeta_threshold, t23, t172 * t170)
  t176 = r1 ** 2
  t177 = jnp.cbrt(r1)
  t178 = t177 ** 2
  t182 = t34 * s2 / t178 / t176
  t190 = tau1 / t178 / r1
  t191 = t51 - t190
  t193 = t51 + t190
  t194 = 0.1e1 / t193
  t196 = t191 ** 2
  t198 = t193 ** 2
  t199 = 0.1e1 / t198
  t201 = t196 * t191
  t203 = t198 * t193
  t204 = 0.1e1 / t203
  t206 = t196 ** 2
  t208 = t198 ** 2
  t209 = 0.1e1 / t208
  t211 = t206 * t191
  t214 = 0.1e1 / t208 / t193
  t216 = t206 * t196
  t219 = 0.1e1 / t208 / t198
  t221 = t206 * t201
  t224 = 0.1e1 / t208 / t203
  t226 = t206 ** 2
  t228 = t208 ** 2
  t229 = 0.1e1 / t228
  t231 = t226 * t191
  t234 = 0.1e1 / t228 / t193
  t236 = t226 * t196
  t239 = 0.1e1 / t228 / t198
  t241 = t226 * t201
  t244 = 0.1e1 / t228 / t203
  t246 = t47 + t48 * t191 * t194 + t60 * t196 * t199 + t66 * t201 * t204 + t72 * t206 * t209 + t78 * t211 * t214 + t84 * t216 * t219 + t90 * t221 * t224 + t96 * t226 * t229 + t102 * t231 * t234 + t108 * t236 * t239 + t114 * t241 * t244
  t249 = jnp.exp(-0.93189002206715572255e-2 * t182)
  t274 = t126 + t127 * t191 * t194 + t130 * t196 * t199 + t133 * t201 * t204 + t136 * t206 * t209 + t139 * t211 * t214 + t142 * t216 * t219 + t145 * t221 * t224 + t148 * t226 * t229 + t151 * t231 * t234 + t154 * t236 * t239 + t157 * t241 * t244
  t280 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t174 * t27 * ((0.1804e1 - 0.646416 / (0.804 + 0.914625e-2 * t182)) * t246 + (0.1552e1 - 0.552 * t249) * t274))
  res = t166 + t280
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
  t22 = jnp.pi ** 2
  t23 = jnp.cbrt(t22)
  t24 = t23 ** 2
  t27 = jnp.cbrt(2)
  t28 = t27 ** 2
  t30 = r0 ** 2
  t31 = t19 ** 2
  t35 = t21 / t24 * s0 * t28 / t31 / t30
  t43 = t21 ** 2
  t45 = 0.3e1 / 0.1e2 * t43 * t24
  t49 = tau0 * t28 / t31 / r0
  t50 = t45 - t49
  t52 = t45 + t49
  t53 = 0.1e1 / t52
  t56 = t50 ** 2
  t58 = t52 ** 2
  t59 = 0.1e1 / t58
  t62 = t56 * t50
  t64 = t58 * t52
  t65 = 0.1e1 / t64
  t68 = t56 ** 2
  t70 = t58 ** 2
  t71 = 0.1e1 / t70
  t74 = t68 * t50
  t77 = 0.1e1 / t70 / t52
  t80 = t68 * t56
  t83 = 0.1e1 / t70 / t58
  t86 = t68 * t62
  t89 = 0.1e1 / t70 / t64
  t92 = t68 ** 2
  t94 = t70 ** 2
  t95 = 0.1e1 / t94
  t98 = t92 * t50
  t101 = 0.1e1 / t94 / t52
  t104 = t92 * t56
  t107 = 0.1e1 / t94 / t58
  t110 = t92 * t62
  t113 = 0.1e1 / t94 / t64
  t115 = params.a[0] + params.a[1] * t50 * t53 + params.a[2] * t56 * t59 + params.a[3] * t62 * t65 + params.a[4] * t68 * t71 + params.a[5] * t74 * t77 + params.a[6] * t80 * t83 + params.a[7] * t86 * t89 + params.a[8] * t92 * t95 + params.a[9] * t98 * t101 + params.a[10] * t104 * t107 + params.a[11] * t110 * t113
  t118 = jnp.exp(-0.93189002206715572255e-2 * t35)
  t155 = params.b[0] + params.b[1] * t50 * t53 + params.b[2] * t56 * t59 + params.b[3] * t62 * t65 + params.b[4] * t68 * t71 + params.b[5] * t74 * t77 + params.b[6] * t80 * t83 + params.b[7] * t86 * t89 + params.b[8] * t92 * t95 + params.b[9] * t98 * t101 + params.b[10] * t104 * t107 + params.b[11] * t110 * t113
  t161 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * ((0.1804e1 - 0.646416 / (0.804 + 0.914625e-2 * t35)) * t115 + (0.1552e1 - 0.552 * t118) * t155))
  res = 0.2e1 * t161
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