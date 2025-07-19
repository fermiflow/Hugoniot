"""Generated from mgga_x_m11_l.mpl."""

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
  t29 = jnp.cbrt(9)
  t30 = t29 ** 2
  t32 = jnp.cbrt(0.1e1 / jnp.pi)
  t33 = t32 ** 2
  t35 = t30 * t33 * p.cam_omega
  t37 = t2 / t27
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
  t101 = jnp.cbrt(6)
  t102 = jnp.pi ** 2
  t103 = jnp.cbrt(t102)
  t104 = t103 ** 2
  t106 = t101 / t104
  t107 = r0 ** 2
  t108 = jnp.cbrt(r0)
  t109 = t108 ** 2
  t113 = t106 * s0 / t109 / t107
  t118 = 0.1804e1 - 0.646416 / (0.804 + 0.914625e-2 * t113)
  t119 = params.a[0]
  t120 = params.a[1]
  t121 = t101 ** 2
  t123 = 0.3e1 / 0.1e2 * t121 * t104
  t126 = tau0 / t109 / r0
  t127 = t123 - t126
  t129 = t123 + t126
  t130 = 0.1e1 / t129
  t132 = params.a[2]
  t133 = t127 ** 2
  t135 = t129 ** 2
  t136 = 0.1e1 / t135
  t138 = params.a[3]
  t139 = t133 * t127
  t141 = t135 * t129
  t142 = 0.1e1 / t141
  t144 = params.a[4]
  t145 = t133 ** 2
  t147 = t135 ** 2
  t148 = 0.1e1 / t147
  t150 = params.a[5]
  t151 = t145 * t127
  t154 = 0.1e1 / t147 / t129
  t156 = params.a[6]
  t157 = t145 * t133
  t160 = 0.1e1 / t147 / t135
  t162 = params.a[7]
  t163 = t145 * t139
  t166 = 0.1e1 / t147 / t141
  t168 = params.a[8]
  t169 = t145 ** 2
  t171 = t147 ** 2
  t172 = 0.1e1 / t171
  t174 = params.a[9]
  t175 = t169 * t127
  t178 = 0.1e1 / t171 / t129
  t180 = params.a[10]
  t181 = t169 * t133
  t184 = 0.1e1 / t171 / t135
  t186 = params.a[11]
  t187 = t169 * t139
  t190 = 0.1e1 / t171 / t141
  t192 = t119 + t120 * t127 * t130 + t132 * t133 * t136 + t138 * t139 * t142 + t144 * t145 * t148 + t150 * t151 * t154 + t156 * t157 * t160 + t162 * t163 * t166 + t168 * t169 * t172 + t174 * t175 * t178 + t180 * t181 * t184 + t186 * t187 * t190
  t195 = jnp.exp(-0.93189002206715572255e-2 * t113)
  t197 = 0.1552e1 - 0.552 * t195
  t198 = params.b[0]
  t199 = params.b[1]
  t202 = params.b[2]
  t205 = params.b[3]
  t208 = params.b[4]
  t211 = params.b[5]
  t214 = params.b[6]
  t217 = params.b[7]
  t220 = params.b[8]
  t223 = params.b[9]
  t226 = params.b[10]
  t229 = params.b[11]
  t232 = t198 + t199 * t127 * t130 + t202 * t133 * t136 + t205 * t139 * t142 + t208 * t145 * t148 + t211 * t151 * t154 + t214 * t157 * t160 + t217 * t163 * t166 + t220 * t169 * t172 + t223 * t175 * t178 + t226 * t181 * t184 + t229 * t187 * t190
  t237 = params.c[0]
  t238 = params.c[1]
  t241 = params.c[2]
  t244 = params.c[3]
  t247 = params.c[4]
  t250 = params.c[5]
  t253 = params.c[6]
  t256 = params.c[7]
  t259 = params.c[8]
  t262 = params.c[9]
  t265 = params.c[10]
  t268 = params.c[11]
  t271 = t237 + t238 * t127 * t130 + t241 * t133 * t136 + t244 * t139 * t142 + t247 * t145 * t148 + t250 * t151 * t154 + t253 * t157 * t160 + t256 * t163 * t166 + t259 * t169 * t172 + t262 * t175 * t178 + t265 * t181 * t184 + t268 * t187 * t190
  t273 = params.d[0]
  t274 = params.d[1]
  t277 = params.d[2]
  t280 = params.d[3]
  t283 = params.d[4]
  t286 = params.d[5]
  t289 = params.d[6]
  t292 = params.d[7]
  t295 = params.d[8]
  t298 = params.d[9]
  t301 = params.d[10]
  t304 = params.d[11]
  t307 = t273 + t274 * t127 * t130 + t277 * t133 * t136 + t280 * t139 * t142 + t283 * t145 * t148 + t286 * t151 * t154 + t289 * t157 * t160 + t292 * t163 * t166 + t295 * t169 * t172 + t298 * t175 * t178 + t301 * t181 * t184 + t304 * t187 * t190
  t315 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (t100 * (t118 * t192 + t197 * t232) + (0.1e1 - t100) * (t118 * t271 + t197 * t307)))
  t317 = jnp.where(t10, t15, -t17)
  t318 = jnp.where(t14, t11, t317)
  t319 = 0.1e1 + t318
  t321 = jnp.cbrt(t319)
  t323 = jnp.where(t319 <= p.zeta_threshold, t23, t321 * t319)
  t325 = jnp.where(t39, t15, -t17)
  t326 = jnp.where(t41, t11, t325)
  t327 = 0.1e1 + t326
  t329 = jnp.cbrt(t327)
  t330 = jnp.where(t327 <= p.zeta_threshold, t22, t329)
  t334 = t35 * t37 / t330 / 0.18e2
  t336 = 0.135e1 < t334
  t337 = jnp.where(t336, t334, 0.135e1)
  t338 = t337 ** 2
  t341 = t338 ** 2
  t344 = t341 * t338
  t347 = t341 ** 2
  t359 = t347 ** 2
  t363 = jnp.where(t336, 0.135e1, t334)
  t366 = jax.lax.erf(0.1e1 / t363 / 0.2e1)
  t368 = t363 ** 2
  t371 = jnp.exp(-0.1e1 / t368 / 0.4e1)
  t382 = jnp.where(0.135e1 <= t334, 0.1e1 / t338 / 0.36e2 - 0.1e1 / t341 / 0.96e3 + 0.1e1 / t344 / 0.2688e5 - 0.1e1 / t347 / 0.82944e6 + 0.1e1 / t347 / t338 / 0.2838528e8 - 0.1e1 / t347 / t341 / 0.107347968e10 + 0.1e1 / t347 / t344 / 0.445906944e11 - 0.1e1 / t359 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t363 * (t81 * t366 + 0.2e1 * t363 * (t371 - 0.3e1 / 0.2e1 - 0.2e1 * t368 * (t371 - 0.1e1))))
  t383 = r1 ** 2
  t384 = jnp.cbrt(r1)
  t385 = t384 ** 2
  t389 = t106 * s2 / t385 / t383
  t394 = 0.1804e1 - 0.646416 / (0.804 + 0.914625e-2 * t389)
  t397 = tau1 / t385 / r1
  t398 = t123 - t397
  t400 = t123 + t397
  t401 = 0.1e1 / t400
  t403 = t398 ** 2
  t405 = t400 ** 2
  t406 = 0.1e1 / t405
  t408 = t403 * t398
  t410 = t405 * t400
  t411 = 0.1e1 / t410
  t413 = t403 ** 2
  t415 = t405 ** 2
  t416 = 0.1e1 / t415
  t418 = t413 * t398
  t421 = 0.1e1 / t415 / t400
  t423 = t413 * t403
  t426 = 0.1e1 / t415 / t405
  t428 = t413 * t408
  t431 = 0.1e1 / t415 / t410
  t433 = t413 ** 2
  t435 = t415 ** 2
  t436 = 0.1e1 / t435
  t438 = t433 * t398
  t441 = 0.1e1 / t435 / t400
  t443 = t433 * t403
  t446 = 0.1e1 / t435 / t405
  t448 = t433 * t408
  t451 = 0.1e1 / t435 / t410
  t453 = t119 + t120 * t398 * t401 + t132 * t403 * t406 + t138 * t408 * t411 + t144 * t413 * t416 + t150 * t418 * t421 + t156 * t423 * t426 + t162 * t428 * t431 + t168 * t433 * t436 + t174 * t438 * t441 + t180 * t443 * t446 + t186 * t448 * t451
  t456 = jnp.exp(-0.93189002206715572255e-2 * t389)
  t458 = 0.1552e1 - 0.552 * t456
  t481 = t198 + t199 * t398 * t401 + t202 * t403 * t406 + t205 * t408 * t411 + t208 * t413 * t416 + t211 * t418 * t421 + t214 * t423 * t426 + t217 * t428 * t431 + t220 * t433 * t436 + t223 * t438 * t441 + t226 * t443 * t446 + t229 * t448 * t451
  t508 = t237 + t238 * t398 * t401 + t241 * t403 * t406 + t244 * t408 * t411 + t247 * t413 * t416 + t250 * t418 * t421 + t253 * t423 * t426 + t256 * t428 * t431 + t259 * t433 * t436 + t262 * t438 * t441 + t265 * t443 * t446 + t268 * t448 * t451
  t532 = t273 + t274 * t398 * t401 + t277 * t403 * t406 + t280 * t408 * t411 + t283 * t413 * t416 + t286 * t418 * t421 + t289 * t423 * t426 + t292 * t428 * t431 + t295 * t433 * t436 + t298 * t438 * t441 + t301 * t443 * t446 + t304 * t448 * t451
  t540 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t323 * t27 * (t382 * (t394 * t453 + t458 * t481) + (0.1e1 - t382) * (t394 * t508 + t458 * t532)))
  res = t315 + t540
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
  t19 = jnp.cbrt(r0)
  t21 = jnp.cbrt(9)
  t22 = t21 ** 2
  t24 = jnp.cbrt(0.1e1 / jnp.pi)
  t25 = t24 ** 2
  t30 = jnp.where(t13, t14, t16)
  t34 = t22 * t25 * p.cam_omega * t3 / t19 / t30 / 0.18e2
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
  t84 = jnp.cbrt(6)
  t85 = jnp.pi ** 2
  t86 = jnp.cbrt(t85)
  t87 = t86 ** 2
  t90 = jnp.cbrt(2)
  t91 = t90 ** 2
  t93 = r0 ** 2
  t94 = t19 ** 2
  t98 = t84 / t87 * s0 * t91 / t94 / t93
  t103 = 0.1804e1 - 0.646416 / (0.804 + 0.914625e-2 * t98)
  t106 = t84 ** 2
  t108 = 0.3e1 / 0.1e2 * t106 * t87
  t112 = tau0 * t91 / t94 / r0
  t113 = t108 - t112
  t115 = t108 + t112
  t116 = 0.1e1 / t115
  t119 = t113 ** 2
  t121 = t115 ** 2
  t122 = 0.1e1 / t121
  t125 = t119 * t113
  t127 = t121 * t115
  t128 = 0.1e1 / t127
  t131 = t119 ** 2
  t133 = t121 ** 2
  t134 = 0.1e1 / t133
  t137 = t131 * t113
  t140 = 0.1e1 / t133 / t115
  t143 = t131 * t119
  t146 = 0.1e1 / t133 / t121
  t149 = t131 * t125
  t152 = 0.1e1 / t133 / t127
  t155 = t131 ** 2
  t157 = t133 ** 2
  t158 = 0.1e1 / t157
  t161 = t155 * t113
  t164 = 0.1e1 / t157 / t115
  t167 = t155 * t119
  t170 = 0.1e1 / t157 / t121
  t173 = t155 * t125
  t176 = 0.1e1 / t157 / t127
  t178 = params.a[0] + params.a[1] * t113 * t116 + params.a[2] * t119 * t122 + params.a[3] * t125 * t128 + params.a[4] * t131 * t134 + params.a[5] * t137 * t140 + params.a[6] * t143 * t146 + params.a[7] * t149 * t152 + params.a[8] * t155 * t158 + params.a[9] * t161 * t164 + params.a[10] * t167 * t170 + params.a[11] * t173 * t176
  t181 = jnp.exp(-0.93189002206715572255e-2 * t98)
  t183 = 0.1552e1 - 0.552 * t181
  t218 = params.b[0] + params.b[1] * t113 * t116 + params.b[2] * t119 * t122 + params.b[3] * t125 * t128 + params.b[4] * t131 * t134 + params.b[5] * t137 * t140 + params.b[6] * t143 * t146 + params.b[7] * t149 * t152 + params.b[8] * t155 * t158 + params.b[9] * t161 * t164 + params.b[10] * t167 * t170 + params.b[11] * t173 * t176
  t257 = params.c[0] + params.c[1] * t113 * t116 + params.c[2] * t119 * t122 + params.c[3] * t125 * t128 + params.c[4] * t131 * t134 + params.c[5] * t137 * t140 + params.c[6] * t143 * t146 + params.c[7] * t149 * t152 + params.c[8] * t155 * t158 + params.c[9] * t161 * t164 + params.c[10] * t167 * t170 + params.c[11] * t173 * t176
  t293 = params.d[0] + params.d[1] * t113 * t116 + params.d[2] * t119 * t122 + params.d[3] * t125 * t128 + params.d[4] * t131 * t134 + params.d[5] * t137 * t140 + params.d[6] * t143 * t146 + params.d[7] * t149 * t152 + params.d[8] * t155 * t158 + params.d[9] * t161 * t164 + params.d[10] * t167 * t170 + params.d[11] * t173 * t176
  t301 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (t83 * (t103 * t178 + t183 * t218) + (0.1e1 - t83) * (t103 * t257 + t183 * t293)))
  res = 0.2e1 * t301
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