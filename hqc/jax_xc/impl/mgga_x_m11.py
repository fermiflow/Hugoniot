"""Generated from mgga_x_m11.mpl."""

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
  t29 = jnp.cbrt(9)
  t30 = t29 ** 2
  t32 = jnp.cbrt(0.1e1 / jnp.pi)
  t33 = t32 ** 2
  t35 = t30 * t33 * p.cam_omega
  t37 = t2 / t28
  t39 = 0.1e1 + t17 <= p.zeta_threshold
  t41 = 0.1e1 - t17 <= p.zeta_threshold
  t42 = jnp.where(t41, t15, t17)
  t43 = jnp.where(t39, t11, t42)
  t44 = 0.1e1 + t43
  t46 = jnp.cbrt(t44)
  t47 = jnp.where(t44 <= p.zeta_threshold, t22, t46)
  t51 = t35 * t37 / t47 / 0.18e2
  t53 = 0.135e1 < t51
  t54 = jnp.where(t53, t51, 0.135e1)
  t55 = t54 ** 2
  t58 = t55 ** 2
  t61 = t58 * t55
  t64 = t58 ** 2
  t76 = t64 ** 2
  t80 = jnp.where(t53, 0.135e1, t51)
  t81 = jnp.sqrt(jnp.pi)
  t84 = jax.lax.erf(0.1e1 / t80 / 0.2e1)
  t86 = t80 ** 2
  t89 = jnp.exp(-0.1e1 / t86 / 0.4e1)
  t100 = jnp.where(0.135e1 <= t51, 0.1e1 / t55 / 0.36e2 - 0.1e1 / t58 / 0.96e3 + 0.1e1 / t61 / 0.2688e5 - 0.1e1 / t64 / 0.82944e6 + 0.1e1 / t64 / t55 / 0.2838528e8 - 0.1e1 / t64 / t58 / 0.107347968e10 + 0.1e1 / t64 / t61 / 0.445906944e11 - 0.1e1 / t76 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t80 * (t81 * t84 + 0.2e1 * t80 * (t89 - 0.3e1 / 0.2e1 - 0.2e1 * t86 * (t89 - 0.1e1))))
  t102 = jnp.cbrt(6)
  t103 = jnp.pi ** 2
  t104 = jnp.cbrt(t103)
  t105 = t104 ** 2
  t107 = t102 / t105
  t108 = r0 ** 2
  t109 = jnp.cbrt(r0)
  t110 = t109 ** 2
  t114 = t107 * s0 / t110 / t108
  t120 = params.a[0]
  t121 = params.a[1]
  t122 = t102 ** 2
  t124 = 0.3e1 / 0.1e2 * t122 * t105
  t127 = tau0 / t110 / r0
  t128 = t124 - t127
  t130 = t124 + t127
  t131 = 0.1e1 / t130
  t133 = params.a[2]
  t134 = t128 ** 2
  t136 = t130 ** 2
  t137 = 0.1e1 / t136
  t139 = params.a[3]
  t140 = t134 * t128
  t142 = t136 * t130
  t143 = 0.1e1 / t142
  t145 = params.a[4]
  t146 = t134 ** 2
  t148 = t136 ** 2
  t149 = 0.1e1 / t148
  t151 = params.a[5]
  t152 = t146 * t128
  t155 = 0.1e1 / t148 / t130
  t157 = params.a[6]
  t158 = t146 * t134
  t161 = 0.1e1 / t148 / t136
  t163 = params.a[7]
  t164 = t146 * t140
  t167 = 0.1e1 / t148 / t142
  t169 = params.a[8]
  t170 = t146 ** 2
  t172 = t148 ** 2
  t173 = 0.1e1 / t172
  t175 = params.a[9]
  t176 = t170 * t128
  t179 = 0.1e1 / t172 / t130
  t181 = params.a[10]
  t182 = t170 * t134
  t185 = 0.1e1 / t172 / t136
  t187 = params.a[11]
  t188 = t170 * t140
  t191 = 0.1e1 / t172 / t142
  t193 = t120 + t121 * t128 * t131 + t133 * t134 * t137 + t139 * t140 * t143 + t145 * t146 * t149 + t151 * t152 * t155 + t157 * t158 * t161 + t163 * t164 * t167 + t169 * t170 * t173 + t175 * t176 * t179 + t181 * t182 * t185 + t187 * t188 * t191
  t196 = jnp.exp(-0.93189002206715572255e-2 * t114)
  t199 = params.b[0]
  t200 = params.b[1]
  t203 = params.b[2]
  t206 = params.b[3]
  t209 = params.b[4]
  t212 = params.b[5]
  t215 = params.b[6]
  t218 = params.b[7]
  t221 = params.b[8]
  t224 = params.b[9]
  t227 = params.b[10]
  t230 = params.b[11]
  t233 = t199 + t200 * t128 * t131 + t203 * t134 * t137 + t206 * t140 * t143 + t209 * t146 * t149 + t212 * t152 * t155 + t215 * t158 * t161 + t218 * t164 * t167 + t221 * t170 * t173 + t224 * t176 * t179 + t227 * t182 * t185 + t230 * t188 * t191
  t239 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * t100 * ((0.1804e1 - 0.646416 / (0.804 + 0.914625e-2 * t114)) * t193 + (0.1552e1 - 0.552 * t196) * t233))
  t241 = jnp.where(t10, t15, -t17)
  t242 = jnp.where(t14, t11, t241)
  t243 = 0.1e1 + t242
  t245 = jnp.cbrt(t243)
  t247 = jnp.where(t243 <= p.zeta_threshold, t23, t245 * t243)
  t249 = jnp.where(t39, t15, -t17)
  t250 = jnp.where(t41, t11, t249)
  t251 = 0.1e1 + t250
  t253 = jnp.cbrt(t251)
  t254 = jnp.where(t251 <= p.zeta_threshold, t22, t253)
  t258 = t35 * t37 / t254 / 0.18e2
  t260 = 0.135e1 < t258
  t261 = jnp.where(t260, t258, 0.135e1)
  t262 = t261 ** 2
  t265 = t262 ** 2
  t268 = t265 * t262
  t271 = t265 ** 2
  t283 = t271 ** 2
  t287 = jnp.where(t260, 0.135e1, t258)
  t290 = jax.lax.erf(0.1e1 / t287 / 0.2e1)
  t292 = t287 ** 2
  t295 = jnp.exp(-0.1e1 / t292 / 0.4e1)
  t306 = jnp.where(0.135e1 <= t258, 0.1e1 / t262 / 0.36e2 - 0.1e1 / t265 / 0.96e3 + 0.1e1 / t268 / 0.2688e5 - 0.1e1 / t271 / 0.82944e6 + 0.1e1 / t271 / t262 / 0.2838528e8 - 0.1e1 / t271 / t265 / 0.107347968e10 + 0.1e1 / t271 / t268 / 0.445906944e11 - 0.1e1 / t283 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t287 * (t81 * t290 + 0.2e1 * t287 * (t295 - 0.3e1 / 0.2e1 - 0.2e1 * t292 * (t295 - 0.1e1))))
  t308 = r1 ** 2
  t309 = jnp.cbrt(r1)
  t310 = t309 ** 2
  t314 = t107 * s2 / t310 / t308
  t322 = tau1 / t310 / r1
  t323 = t124 - t322
  t325 = t124 + t322
  t326 = 0.1e1 / t325
  t328 = t323 ** 2
  t330 = t325 ** 2
  t331 = 0.1e1 / t330
  t333 = t328 * t323
  t335 = t330 * t325
  t336 = 0.1e1 / t335
  t338 = t328 ** 2
  t340 = t330 ** 2
  t341 = 0.1e1 / t340
  t343 = t338 * t323
  t346 = 0.1e1 / t340 / t325
  t348 = t338 * t328
  t351 = 0.1e1 / t340 / t330
  t353 = t338 * t333
  t356 = 0.1e1 / t340 / t335
  t358 = t338 ** 2
  t360 = t340 ** 2
  t361 = 0.1e1 / t360
  t363 = t358 * t323
  t366 = 0.1e1 / t360 / t325
  t368 = t358 * t328
  t371 = 0.1e1 / t360 / t330
  t373 = t358 * t333
  t376 = 0.1e1 / t360 / t335
  t378 = t120 + t121 * t323 * t326 + t133 * t328 * t331 + t139 * t333 * t336 + t145 * t338 * t341 + t151 * t343 * t346 + t157 * t348 * t351 + t163 * t353 * t356 + t169 * t358 * t361 + t175 * t363 * t366 + t181 * t368 * t371 + t187 * t373 * t376
  t381 = jnp.exp(-0.93189002206715572255e-2 * t314)
  t406 = t199 + t200 * t323 * t326 + t203 * t328 * t331 + t206 * t333 * t336 + t209 * t338 * t341 + t212 * t343 * t346 + t215 * t348 * t351 + t218 * t353 * t356 + t221 * t358 * t361 + t224 * t363 * t366 + t227 * t368 * t371 + t230 * t373 * t376
  t412 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t247 * t28 * t306 * ((0.1804e1 - 0.646416 / (0.804 + 0.914625e-2 * t314)) * t378 + (0.1552e1 - 0.552 * t381) * t406))
  res = t239 + t412
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
  t13 = t12 <= p.zeta_threshold
  t14 = jnp.cbrt(p.zeta_threshold)
  t16 = jnp.cbrt(t12)
  t18 = jnp.where(t13, t14 * p.zeta_threshold, t16 * t12)
  t20 = jnp.cbrt(r0)
  t21 = jnp.cbrt(9)
  t22 = t21 ** 2
  t24 = jnp.cbrt(0.1e1 / jnp.pi)
  t25 = t24 ** 2
  t30 = jnp.where(t13, t14, t16)
  t34 = t22 * t25 * p.cam_omega * t3 / t20 / t30 / 0.18e2
  t36 = 0.135e1 < t34
  t37 = jnp.where(t36, t34, 0.135e1)
  t38 = t37 ** 2
  t41 = t38 ** 2
  t44 = t41 * t38
  t47 = t41 ** 2
  t59 = t47 ** 2
  t63 = jnp.where(t36, 0.135e1, t34)
  t64 = jnp.sqrt(jnp.pi)
  t67 = jax.lax.erf(0.1e1 / t63 / 0.2e1)
  t69 = t63 ** 2
  t72 = jnp.exp(-0.1e1 / t69 / 0.4e1)
  t83 = jnp.where(0.135e1 <= t34, 0.1e1 / t38 / 0.36e2 - 0.1e1 / t41 / 0.96e3 + 0.1e1 / t44 / 0.2688e5 - 0.1e1 / t47 / 0.82944e6 + 0.1e1 / t47 / t38 / 0.2838528e8 - 0.1e1 / t47 / t41 / 0.107347968e10 + 0.1e1 / t47 / t44 / 0.445906944e11 - 0.1e1 / t59 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t63 * (t64 * t67 + 0.2e1 * t63 * (t72 - 0.3e1 / 0.2e1 - 0.2e1 * t69 * (t72 - 0.1e1))))
  t85 = jnp.cbrt(6)
  t86 = jnp.pi ** 2
  t87 = jnp.cbrt(t86)
  t88 = t87 ** 2
  t91 = jnp.cbrt(2)
  t92 = t91 ** 2
  t94 = r0 ** 2
  t95 = t20 ** 2
  t99 = t85 / t88 * s0 * t92 / t95 / t94
  t107 = t85 ** 2
  t109 = 0.3e1 / 0.1e2 * t107 * t88
  t113 = tau0 * t92 / t95 / r0
  t114 = t109 - t113
  t116 = t109 + t113
  t117 = 0.1e1 / t116
  t120 = t114 ** 2
  t122 = t116 ** 2
  t123 = 0.1e1 / t122
  t126 = t120 * t114
  t128 = t122 * t116
  t129 = 0.1e1 / t128
  t132 = t120 ** 2
  t134 = t122 ** 2
  t135 = 0.1e1 / t134
  t138 = t132 * t114
  t141 = 0.1e1 / t134 / t116
  t144 = t132 * t120
  t147 = 0.1e1 / t134 / t122
  t150 = t132 * t126
  t153 = 0.1e1 / t134 / t128
  t156 = t132 ** 2
  t158 = t134 ** 2
  t159 = 0.1e1 / t158
  t162 = t156 * t114
  t165 = 0.1e1 / t158 / t116
  t168 = t156 * t120
  t171 = 0.1e1 / t158 / t122
  t174 = t156 * t126
  t177 = 0.1e1 / t158 / t128
  t179 = params.a[0] + params.a[1] * t114 * t117 + params.a[2] * t120 * t123 + params.a[3] * t126 * t129 + params.a[4] * t132 * t135 + params.a[5] * t138 * t141 + params.a[6] * t144 * t147 + params.a[7] * t150 * t153 + params.a[8] * t156 * t159 + params.a[9] * t162 * t165 + params.a[10] * t168 * t171 + params.a[11] * t174 * t177
  t182 = jnp.exp(-0.93189002206715572255e-2 * t99)
  t219 = params.b[0] + params.b[1] * t114 * t117 + params.b[2] * t120 * t123 + params.b[3] * t126 * t129 + params.b[4] * t132 * t135 + params.b[5] * t138 * t141 + params.b[6] * t144 * t147 + params.b[7] * t150 * t153 + params.b[8] * t156 * t159 + params.b[9] * t162 * t165 + params.b[10] * t168 * t171 + params.b[11] * t174 * t177
  t225 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * t83 * ((0.1804e1 - 0.646416 / (0.804 + 0.914625e-2 * t99)) * t179 + (0.1552e1 - 0.552 * t182) * t219))
  res = 0.2e1 * t225
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