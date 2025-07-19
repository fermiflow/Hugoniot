"""Generated from hyb_mgga_xc_wb97mv.mpl."""

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
  t2 = r0 - r1
  t3 = r0 + r1
  t5 = t2 / t3
  t6 = 0.1e1 + t5
  t7 = t6 <= p.zeta_threshold
  t8 = jnp.logical_or(r0 <= p.dens_threshold, t7)
  t9 = jnp.cbrt(3)
  t12 = jnp.cbrt(0.1e1 / jnp.pi)
  t13 = jnp.cbrt(4)
  t14 = t13 ** 2
  t16 = jnp.cbrt(2)
  t17 = t12 * t14 * t16
  t19 = 0.2e1 <= p.zeta_threshold
  t20 = jnp.cbrt(p.zeta_threshold)
  t21 = t20 * p.zeta_threshold
  t23 = jnp.where(t19, t21, 0.2e1 * t16)
  t24 = jnp.cbrt(t3)
  t25 = t23 * t24
  t27 = jnp.cbrt(0.1e1 / t6)
  t29 = jnp.cbrt(9)
  t30 = t29 ** 2
  t31 = t12 ** 2
  t34 = t30 * t31 * p.cam_omega * t9
  t35 = 0.1e1 / t24
  t36 = t35 * t16
  t37 = jnp.where(t19, t20, t16)
  t38 = 0.1e1 / t37
  t42 = t34 * t36 * t27 * t38 / 0.18e2
  t44 = 0.135e1 < t42
  t45 = jnp.where(t44, t42, 0.135e1)
  t46 = t45 ** 2
  t49 = t46 ** 2
  t52 = t49 * t46
  t55 = t49 ** 2
  t67 = t55 ** 2
  t71 = jnp.where(t44, 0.135e1, t42)
  t72 = jnp.sqrt(jnp.pi)
  t75 = jax.lax.erf(0.1e1 / t71 / 0.2e1)
  t77 = t71 ** 2
  t80 = jnp.exp(-0.1e1 / t77 / 0.4e1)
  t91 = jnp.where(0.135e1 <= t42, 0.1e1 / t46 / 0.36e2 - 0.1e1 / t49 / 0.96e3 + 0.1e1 / t52 / 0.2688e5 - 0.1e1 / t55 / 0.82944e6 + 0.1e1 / t55 / t46 / 0.2838528e8 - 0.1e1 / t55 / t49 / 0.107347968e10 + 0.1e1 / t55 / t52 / 0.445906944e11 - 0.1e1 / t67 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t71 * (t72 * t75 + 0.2e1 * t71 * (t80 - 0.3e1 / 0.2e1 - 0.2e1 * t77 * (t80 - 0.1e1))))
  t93 = params.c_x[0]
  t94 = params.c_x[1]
  t96 = r0 ** 2
  t97 = jnp.cbrt(r0)
  t98 = t97 ** 2
  t100 = 0.1e1 / t98 / t96
  t101 = s0 * t100
  t108 = params.c_x[2]
  t109 = jnp.cbrt(6)
  t110 = t109 ** 2
  t111 = jnp.pi ** 2
  t112 = jnp.cbrt(t111)
  t113 = t112 ** 2
  t114 = t110 * t113
  t115 = 0.3e1 / 0.1e2 * t114
  t118 = tau0 / t98 / r0
  t119 = t115 - t118
  t121 = t115 + t118
  t122 = 0.1e1 / t121
  t129 = jnp.where(t8, 0, -0.3e1 / 0.64e2 * t6 * t9 * t17 * t25 / t27 * t91 * (t93 + 0.4e-2 * t94 * s0 * t100 / (0.1e1 + 0.4e-2 * t101) + t108 * t119 * t122))
  t131 = 0.1e1 - t5
  t132 = t131 <= p.zeta_threshold
  t133 = jnp.logical_or(r1 <= p.dens_threshold, t132)
  t137 = jnp.cbrt(0.1e1 / t131)
  t142 = t34 * t36 * t137 * t38 / 0.18e2
  t144 = 0.135e1 < t142
  t145 = jnp.where(t144, t142, 0.135e1)
  t146 = t145 ** 2
  t149 = t146 ** 2
  t152 = t149 * t146
  t155 = t149 ** 2
  t167 = t155 ** 2
  t171 = jnp.where(t144, 0.135e1, t142)
  t174 = jax.lax.erf(0.1e1 / t171 / 0.2e1)
  t176 = t171 ** 2
  t179 = jnp.exp(-0.1e1 / t176 / 0.4e1)
  t190 = jnp.where(0.135e1 <= t142, 0.1e1 / t146 / 0.36e2 - 0.1e1 / t149 / 0.96e3 + 0.1e1 / t152 / 0.2688e5 - 0.1e1 / t155 / 0.82944e6 + 0.1e1 / t155 / t146 / 0.2838528e8 - 0.1e1 / t155 / t149 / 0.107347968e10 + 0.1e1 / t155 / t152 / 0.445906944e11 - 0.1e1 / t167 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t171 * (t72 * t174 + 0.2e1 * t171 * (t179 - 0.3e1 / 0.2e1 - 0.2e1 * t176 * (t179 - 0.1e1))))
  t193 = r1 ** 2
  t194 = jnp.cbrt(r1)
  t195 = t194 ** 2
  t197 = 0.1e1 / t195 / t193
  t198 = s2 * t197
  t207 = tau1 / t195 / r1
  t208 = t115 - t207
  t210 = t115 + t207
  t211 = 0.1e1 / t210
  t218 = jnp.where(t133, 0, -0.3e1 / 0.64e2 * t131 * t9 * t17 * t25 / t137 * t190 * (t93 + 0.4e-2 * t94 * s2 * t197 / (0.1e1 + 0.4e-2 * t198) + t108 * t208 * t211))
  t219 = jnp.where(t7, p.zeta_threshold, t6)
  t220 = t9 * t12
  t221 = t220 * t14
  t222 = 0.1e1 / t20
  t223 = jnp.cbrt(t6)
  t225 = jnp.where(t7, t222, 0.1e1 / t223)
  t227 = t221 * t36 * t225
  t230 = jnp.sqrt(t227)
  t233 = t227 ** 0.15e1
  t235 = t9 ** 2
  t236 = t235 * t31
  t237 = t236 * t13
  t238 = t24 ** 2
  t239 = 0.1e1 / t238
  t240 = t16 ** 2
  t241 = t239 * t240
  t242 = t225 ** 2
  t244 = t237 * t241 * t242
  t250 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t230 + 0.8969 * t227 + 0.204775 * t233 + 0.123235 * t244))
  t252 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t227) * t250
  t254 = jnp.where(0. <= p.zeta_threshold, t21, 0)
  t258 = 0.1e1 / (0.2e1 * t16 - 0.2e1)
  t259 = (t23 + t254 - 0.2e1) * t258
  t270 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t230 + 0.1549425e1 * t227 + 0.420775 * t233 + 0.1562925 * t244))
  t283 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t230 + 0.905775 * t227 + 0.1100325 * t233 + 0.1241775 * t244))
  t284 = (0.1e1 + 0.278125e-1 * t227) * t283
  t293 = jnp.where(t8, 0, t219 * (-t252 + t259 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t227) * t270 + t252 - 0.19751673498613801407e-1 * t284) + 0.19751673498613801407e-1 * t259 * t284) / 0.2e1)
  t294 = params.c_ss[0]
  t295 = params.c_ss[1]
  t296 = s0 ** 2
  t297 = t296 ** 2
  t299 = t96 ** 2
  t300 = t299 ** 2
  t305 = 0.1e1 + 0.2 * t101
  t306 = t305 ** 2
  t307 = t306 ** 2
  t312 = params.c_ss[2]
  t315 = params.c_ss[3]
  t316 = t119 ** 2
  t318 = t121 ** 2
  t321 = params.c_ss[4]
  t322 = t316 ** 2
  t324 = t318 ** 2
  t337 = jnp.where(t132, p.zeta_threshold, t131)
  t338 = jnp.cbrt(t131)
  t340 = jnp.where(t132, t222, 0.1e1 / t338)
  t342 = t221 * t36 * t340
  t345 = jnp.sqrt(t342)
  t348 = t342 ** 0.15e1
  t350 = t340 ** 2
  t352 = t237 * t241 * t350
  t358 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t345 + 0.8969 * t342 + 0.204775 * t348 + 0.123235 * t352))
  t360 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t342) * t358
  t371 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t345 + 0.1549425e1 * t342 + 0.420775 * t348 + 0.1562925 * t352))
  t384 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t345 + 0.905775 * t342 + 0.1100325 * t348 + 0.1241775 * t352))
  t385 = (0.1e1 + 0.278125e-1 * t342) * t384
  t394 = jnp.where(t133, 0, t337 * (-t360 + t259 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t342) * t371 + t360 - 0.19751673498613801407e-1 * t385) + 0.19751673498613801407e-1 * t259 * t385) / 0.2e1)
  t395 = s2 ** 2
  t396 = t395 ** 2
  t398 = t193 ** 2
  t399 = t398 ** 2
  t404 = 0.1e1 + 0.2 * t198
  t405 = t404 ** 2
  t406 = t405 ** 2
  t413 = t208 ** 2
  t415 = t210 ** 2
  t418 = t413 ** 2
  t420 = t415 ** 2
  t434 = t220 * t14 * t35
  t437 = jnp.sqrt(t434)
  t440 = t434 ** 0.15e1
  t443 = t236 * t13 * t239
  t449 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t437 + 0.8969 * t434 + 0.204775 * t440 + 0.123235 * t443))
  t451 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t434) * t449
  t452 = t2 ** 2
  t453 = t452 ** 2
  t454 = t3 ** 2
  t455 = t454 ** 2
  t459 = jnp.where(t7, t21, t223 * t6)
  t461 = jnp.where(t132, t21, t338 * t131)
  t463 = (t459 + t461 - 0.2e1) * t258
  t474 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t437 + 0.1549425e1 * t434 + 0.420775 * t440 + 0.1562925 * t443))
  t487 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t437 + 0.905775 * t434 + 0.1100325 * t440 + 0.1241775 * t443))
  t488 = (0.1e1 + 0.278125e-1 * t434) * t487
  t500 = 0.3e1 / 0.1e2 * t114 * (t118 + t207)
  t502 = 0.2e1 * t118 * t207
  t503 = t500 - t502
  t505 = t500 + t502
  t509 = t503 ** 2
  t511 = t505 ** 2
  t512 = 0.1e1 / t511
  t516 = t101 + t198
  t521 = 0.1e1 / (0.1e1 + 0.3e-2 * t101 + 0.3e-2 * t198)
  t526 = t509 ** 2
  t527 = t526 * t509
  t529 = t511 ** 2
  t531 = 0.1e1 / t529 / t511
  res = t129 + t218 + t293 * (t294 + 0.16e-2 * t295 * t297 / t98 / t300 / t96 / t307 + t312 * t119 * t122 + t315 * t316 / t318 + 0.8e-2 * t321 * t322 / t324 * t296 * s0 / t300 / t306 / t305) + t394 * (t294 + 0.16e-2 * t295 * t396 / t195 / t399 / t193 / t406 + t312 * t208 * t211 + t315 * t413 / t415 + 0.8e-2 * t321 * t418 / t420 * t395 * s2 / t399 / t405 / t404) + (-t451 + t453 / t455 * t463 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t434) * t474 + t451 - 0.19751673498613801407e-1 * t488) + 0.19751673498613801407e-1 * t463 * t488 - t293 - t394) * (params.c_os[0] + params.c_os[1] * t503 / t505 + params.c_os[2] * t509 * t512 + 0.3e-2 * params.c_os[3] * t509 * t512 * t516 * t521 + params.c_os[4] * t527 * t531 + 0.3e-2 * params.c_os[5] * t527 * t531 * t516 * t521)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = 0.1e1 <= p.zeta_threshold
  t4 = jnp.logical_or(r0 / 0.2e1 <= p.dens_threshold, t3)
  t5 = jnp.cbrt(3)
  t7 = jnp.cbrt(0.1e1 / jnp.pi)
  t8 = t5 * t7
  t9 = jnp.cbrt(4)
  t10 = t9 ** 2
  t11 = jnp.cbrt(2)
  t14 = 0.2e1 <= p.zeta_threshold
  t15 = jnp.cbrt(p.zeta_threshold)
  t16 = t15 * p.zeta_threshold
  t18 = jnp.where(t14, t16, 0.2e1 * t11)
  t19 = jnp.cbrt(r0)
  t21 = jnp.cbrt(9)
  t22 = t21 ** 2
  t23 = t7 ** 2
  t26 = 0.1e1 / t19
  t28 = jnp.where(t14, t15, t11)
  t33 = t22 * t23 * p.cam_omega * t5 * t26 * t11 / t28 / 0.18e2
  t35 = 0.135e1 < t33
  t36 = jnp.where(t35, t33, 0.135e1)
  t37 = t36 ** 2
  t40 = t37 ** 2
  t43 = t40 * t37
  t46 = t40 ** 2
  t58 = t46 ** 2
  t62 = jnp.where(t35, 0.135e1, t33)
  t63 = jnp.sqrt(jnp.pi)
  t66 = jax.lax.erf(0.1e1 / t62 / 0.2e1)
  t68 = t62 ** 2
  t71 = jnp.exp(-0.1e1 / t68 / 0.4e1)
  t82 = jnp.where(0.135e1 <= t33, 0.1e1 / t37 / 0.36e2 - 0.1e1 / t40 / 0.96e3 + 0.1e1 / t43 / 0.2688e5 - 0.1e1 / t46 / 0.82944e6 + 0.1e1 / t46 / t37 / 0.2838528e8 - 0.1e1 / t46 / t40 / 0.107347968e10 + 0.1e1 / t46 / t43 / 0.445906944e11 - 0.1e1 / t58 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t62 * (t63 * t66 + 0.2e1 * t62 * (t71 - 0.3e1 / 0.2e1 - 0.2e1 * t68 * (t71 - 0.1e1))))
  t86 = t11 ** 2
  t87 = r0 ** 2
  t88 = t19 ** 2
  t90 = 0.1e1 / t88 / t87
  t92 = s0 * t86
  t93 = t92 * t90
  t101 = jnp.cbrt(6)
  t102 = t101 ** 2
  t103 = jnp.pi ** 2
  t104 = jnp.cbrt(t103)
  t105 = t104 ** 2
  t106 = t102 * t105
  t107 = 0.3e1 / 0.1e2 * t106
  t111 = tau0 * t86 / t88 / r0
  t112 = t107 - t111
  t114 = t107 + t111
  t115 = 0.1e1 / t114
  t122 = jnp.where(t4, 0, -0.3e1 / 0.64e2 * t8 * t10 * t11 * t18 * t19 * t82 * (params.c_x[0] + 0.4e-2 * params.c_x[1] * s0 * t86 * t90 / (0.1e1 + 0.4e-2 * t93) + params.c_x[2] * t112 * t115))
  t124 = jnp.where(t3, p.zeta_threshold, 1)
  t128 = jnp.where(t3, 0.1e1 / t15, 1)
  t130 = t8 * t10 * t26 * t11 * t128
  t133 = jnp.sqrt(t130)
  t136 = t130 ** 0.15e1
  t138 = t5 ** 2
  t139 = t138 * t23
  t141 = 0.1e1 / t88
  t143 = t128 ** 2
  t145 = t139 * t9 * t141 * t86 * t143
  t151 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t133 + 0.8969 * t130 + 0.204775 * t136 + 0.123235 * t145))
  t153 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t130) * t151
  t155 = jnp.where(0. <= p.zeta_threshold, t16, 0)
  t159 = 0.1e1 / (0.2e1 * t11 - 0.2e1)
  t160 = (t18 + t155 - 0.2e1) * t159
  t171 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t133 + 0.1549425e1 * t130 + 0.420775 * t136 + 0.1562925 * t145))
  t184 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t133 + 0.905775 * t130 + 0.1100325 * t136 + 0.1241775 * t145))
  t185 = (0.1e1 + 0.278125e-1 * t130) * t184
  t194 = jnp.where(t4, 0, t124 * (-t153 + t160 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t130) * t171 + t153 - 0.19751673498613801407e-1 * t185) + 0.19751673498613801407e-1 * t160 * t185) / 0.2e1)
  t197 = s0 ** 2
  t198 = t197 ** 2
  t200 = t87 ** 2
  t201 = t200 ** 2
  t207 = 0.1e1 + 0.2 * t93
  t208 = t207 ** 2
  t209 = t208 ** 2
  t218 = t112 ** 2
  t220 = t114 ** 2
  t224 = t218 ** 2
  t226 = t220 ** 2
  t241 = t8 * t10 * t26
  t244 = jnp.sqrt(t241)
  t247 = t241 ** 0.15e1
  t250 = t139 * t9 * t141
  t256 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t244 + 0.8969 * t241 + 0.204775 * t247 + 0.123235 * t250))
  t259 = jnp.where(t3, t16, 1)
  t273 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t244 + 0.905775 * t241 + 0.1100325 * t247 + 0.1241775 * t250))
  t282 = 0.3e1 / 0.5e1 * t106 * t111
  t283 = tau0 ** 2
  t289 = 0.4e1 * t283 * t11 / t19 / t87 / r0
  t290 = t282 - t289
  t292 = t282 + t289
  t296 = t290 ** 2
  t298 = t292 ** 2
  t299 = 0.1e1 / t298
  t308 = t92 * t90 / (0.1e1 + 0.6e-2 * t93)
  t312 = t296 ** 2
  t313 = t312 * t296
  t315 = t298 ** 2
  t317 = 0.1e1 / t315 / t298
  res = 0.2e1 * t122 + 0.2e1 * t194 * (params.c_ss[0] + 0.64e-2 * params.c_ss[1] * t198 * t86 / t88 / t201 / t87 / t209 + params.c_ss[2] * t112 * t115 + params.c_ss[3] * t218 / t220 + 0.32e-1 * params.c_ss[4] * t224 / t226 * t197 * s0 / t201 / t208 / t207) + (-0.621814e-1 * (0.1e1 + 0.53425e-1 * t241) * t256 + 0.19751673498613801407e-1 * (0.2e1 * t259 - 0.2e1) * t159 * (0.1e1 + 0.278125e-1 * t241) * t273 - 0.2e1 * t194) * (params.c_os[0] + params.c_os[1] * t290 / t292 + params.c_os[2] * t296 * t299 + 0.6e-2 * params.c_os[3] * t296 * t299 * t308 + params.c_os[4] * t313 * t317 + 0.6e-2 * params.c_os[5] * t313 * t317 * t308)
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