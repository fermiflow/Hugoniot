"""Generated from mgga_xc_b97mv.mpl."""

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
  t1 = r0 - r1
  t2 = r0 + r1
  t4 = t1 / t2
  t5 = 0.1e1 + t4
  t6 = t5 <= p.zeta_threshold
  t7 = jnp.where(t6, p.zeta_threshold, t5)
  t8 = r0 <= p.dens_threshold
  t9 = jnp.cbrt(3)
  t10 = jnp.cbrt(jnp.pi)
  t12 = t9 / t10
  t14 = jnp.cbrt(p.zeta_threshold)
  t15 = t14 * p.zeta_threshold
  t16 = jnp.cbrt(2)
  t18 = jnp.where(0.2e1 <= p.zeta_threshold, t15, 0.2e1 * t16)
  t19 = jnp.cbrt(t2)
  t20 = t18 * t19
  t23 = jnp.cbrt(t2 / r0)
  t24 = 0.1e1 / t23
  t28 = jnp.where(t8, 0, -0.3e1 / 0.8e1 * t12 * t20 * t24)
  t29 = 0. <= p.dens_threshold
  t31 = jnp.where(0. <= p.zeta_threshold, t15, 0)
  t32 = t31 * t19
  t36 = jnp.where(t29, 0, -0.3e1 / 0.8e1 * t12 * t32 * t24)
  t39 = params.c_x[0]
  t40 = params.c_x[1]
  t42 = r0 ** 2
  t43 = jnp.cbrt(r0)
  t44 = t43 ** 2
  t46 = 0.1e1 / t44 / t42
  t47 = s0 * t46
  t49 = 0.1e1 + 0.4e-2 * t47
  t50 = 0.1e1 / t49
  t54 = params.c_x[2]
  t55 = s0 ** 2
  t57 = t42 ** 2
  t60 = 0.1e1 / t43 / t57 / r0
  t61 = t49 ** 2
  t66 = params.c_x[3]
  t67 = jnp.cbrt(6)
  t68 = t67 ** 2
  t69 = jnp.pi ** 2
  t70 = jnp.cbrt(t69)
  t71 = t70 ** 2
  t72 = t68 * t71
  t73 = 0.3e1 / 0.1e2 * t72
  t76 = tau0 / t44 / r0
  t77 = t73 - t76
  t79 = t73 + t76
  t80 = 0.1e1 / t79
  t82 = params.c_x[4]
  t91 = 0.1e1 - t4
  t92 = t91 <= p.zeta_threshold
  t93 = jnp.where(t92, p.zeta_threshold, t91)
  t94 = r1 <= p.dens_threshold
  t97 = jnp.cbrt(t2 / r1)
  t98 = 0.1e1 / t97
  t102 = jnp.where(t94, 0, -0.3e1 / 0.8e1 * t12 * t20 * t98)
  t106 = jnp.where(t29, 0, -0.3e1 / 0.8e1 * t12 * t32 * t98)
  t110 = r1 ** 2
  t111 = jnp.cbrt(r1)
  t112 = t111 ** 2
  t114 = 0.1e1 / t112 / t110
  t115 = s2 * t114
  t117 = 0.1e1 + 0.4e-2 * t115
  t118 = 0.1e1 / t117
  t122 = s2 ** 2
  t124 = t110 ** 2
  t127 = 0.1e1 / t111 / t124 / r1
  t128 = t117 ** 2
  t135 = tau1 / t112 / r1
  t136 = t73 - t135
  t138 = t73 + t135
  t139 = 0.1e1 / t138
  t149 = jnp.logical_or(t8, t6)
  t151 = jnp.cbrt(0.1e1 / jnp.pi)
  t152 = t9 * t151
  t153 = jnp.cbrt(4)
  t154 = t153 ** 2
  t155 = t152 * t154
  t156 = 0.1e1 / t19
  t157 = t156 * t16
  t158 = 0.1e1 / t14
  t159 = jnp.cbrt(t5)
  t161 = jnp.where(t6, t158, 0.1e1 / t159)
  t163 = t155 * t157 * t161
  t166 = jnp.sqrt(t163)
  t169 = t163 ** 0.15e1
  t171 = t9 ** 2
  t172 = t151 ** 2
  t173 = t171 * t172
  t174 = t173 * t153
  t175 = t19 ** 2
  t176 = 0.1e1 / t175
  t177 = t16 ** 2
  t178 = t176 * t177
  t179 = t161 ** 2
  t181 = t174 * t178 * t179
  t187 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t166 + 0.8969 * t163 + 0.204775 * t169 + 0.123235 * t181))
  t189 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t163) * t187
  t193 = 0.1e1 / (0.2e1 * t16 - 0.2e1)
  t194 = (t18 + t31 - 0.2e1) * t193
  t205 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t166 + 0.1549425e1 * t163 + 0.420775 * t169 + 0.1562925 * t181))
  t218 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t166 + 0.905775 * t163 + 0.1100325 * t169 + 0.1241775 * t181))
  t219 = (0.1e1 + 0.278125e-1 * t163) * t218
  t228 = jnp.where(t149, 0, t7 * (-t189 + t194 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t163) * t205 + t189 - 0.19751673498613801407e-1 * t219) + 0.19751673498613801407e-1 * t194 * t219) / 0.2e1)
  t229 = params.c_ss[0]
  t230 = params.c_ss[1]
  t234 = (0.1e1 + 0.2 * t47) ** 2
  t235 = 0.1e1 / t234
  t239 = params.c_ss[2]
  t242 = params.c_ss[3]
  t243 = t77 ** 2
  t246 = t79 ** 2
  t251 = t55 * t60 * t235
  t254 = params.c_ss[4]
  t255 = t243 ** 2
  t257 = t246 ** 2
  t264 = jnp.logical_or(t94, t92)
  t265 = jnp.cbrt(t91)
  t267 = jnp.where(t92, t158, 0.1e1 / t265)
  t269 = t155 * t157 * t267
  t272 = jnp.sqrt(t269)
  t275 = t269 ** 0.15e1
  t277 = t267 ** 2
  t279 = t174 * t178 * t277
  t285 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t272 + 0.8969 * t269 + 0.204775 * t275 + 0.123235 * t279))
  t287 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t269) * t285
  t298 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t272 + 0.1549425e1 * t269 + 0.420775 * t275 + 0.1562925 * t279))
  t311 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t272 + 0.905775 * t269 + 0.1100325 * t275 + 0.1241775 * t279))
  t312 = (0.1e1 + 0.278125e-1 * t269) * t311
  t321 = jnp.where(t264, 0, t93 * (-t287 + t194 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t269) * t298 + t287 - 0.19751673498613801407e-1 * t312) + 0.19751673498613801407e-1 * t194 * t312) / 0.2e1)
  t325 = (0.1e1 + 0.2 * t115) ** 2
  t326 = 0.1e1 / t325
  t332 = t136 ** 2
  t335 = t138 ** 2
  t340 = t122 * t127 * t326
  t343 = t332 ** 2
  t345 = t335 ** 2
  t353 = t152 * t154 * t156
  t356 = jnp.sqrt(t353)
  t359 = t353 ** 0.15e1
  t362 = t173 * t153 * t176
  t368 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t356 + 0.8969 * t353 + 0.204775 * t359 + 0.123235 * t362))
  t370 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t353) * t368
  t371 = t1 ** 2
  t372 = t371 ** 2
  t373 = t2 ** 2
  t374 = t373 ** 2
  t378 = jnp.where(t6, t15, t159 * t5)
  t380 = jnp.where(t92, t15, t265 * t91)
  t382 = (t378 + t380 - 0.2e1) * t193
  t393 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t356 + 0.1549425e1 * t353 + 0.420775 * t359 + 0.1562925 * t362))
  t406 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t356 + 0.905775 * t353 + 0.1100325 * t359 + 0.1241775 * t362))
  t407 = (0.1e1 + 0.278125e-1 * t353) * t406
  t417 = t47 + t115
  t421 = 0.1e1 + 0.3e-2 * t47 + 0.3e-2 * t115
  t426 = t417 ** 2
  t429 = t421 ** 2
  t437 = 0.3e1 / 0.1e2 * t72 * (t76 + t135)
  t439 = 0.2e1 * t76 * t135
  t440 = t437 - t439
  t442 = t437 + t439
  t446 = t440 ** 2
  t449 = t442 ** 2
  res = t7 * (t28 + t36) * (t39 + 0.4e-2 * t40 * s0 * t46 * t50 + 0.16e-4 * t54 * t55 * t60 / t61 + t66 * t77 * t80 + 0.4e-2 * t82 * t77 * t80 * t47 * t50) / 0.2e1 + t93 * (t102 + t106) * (t39 + 0.4e-2 * t40 * s2 * t114 * t118 + 0.16e-4 * t54 * t122 * t127 / t128 + t66 * t136 * t139 + 0.4e-2 * t82 * t136 * t139 * t115 * t118) / 0.2e1 + t228 * (t229 + 0.4e-1 * t230 * t55 * t60 * t235 + t239 * t77 * t80 + 0.4e-1 * t242 * t243 * t77 / t246 / t79 * t251 + 0.4e-1 * t254 * t255 / t257 * t251) + t321 * (t229 + 0.4e-1 * t230 * t122 * t127 * t326 + t239 * t136 * t139 + 0.4e-1 * t242 * t332 * t136 / t335 / t138 * t340 + 0.4e-1 * t254 * t343 / t345 * t340) + (-t370 + t372 / t374 * t382 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t353) * t393 + t370 - 0.19751673498613801407e-1 * t407) + 0.19751673498613801407e-1 * t382 * t407 - t228 - t321) * (params.c_os[0] + 0.3e-2 * params.c_os[1] * t417 / t421 + 0.27e-7 * params.c_os[2] * t426 * t417 / t429 / t421 + params.c_os[3] * t440 / t442 + 0.9e-5 * params.c_os[4] * t446 * t440 / t449 / t442 * t426 / t429)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = 0.1e1 <= p.zeta_threshold
  t2 = jnp.where(t1, p.zeta_threshold, 1)
  t4 = r0 / 0.2e1 <= p.dens_threshold
  t5 = jnp.cbrt(3)
  t6 = jnp.cbrt(jnp.pi)
  t8 = t5 / t6
  t9 = jnp.cbrt(2)
  t10 = t9 ** 2
  t12 = jnp.cbrt(p.zeta_threshold)
  t13 = t12 * p.zeta_threshold
  t15 = jnp.where(0.2e1 <= p.zeta_threshold, t13, 0.2e1 * t9)
  t17 = jnp.cbrt(r0)
  t21 = jnp.where(t4, 0, -0.3e1 / 0.16e2 * t8 * t10 * t15 * t17)
  t24 = jnp.where(0. <= p.zeta_threshold, t13, 0)
  t29 = jnp.where(0. <= p.dens_threshold, 0, -0.3e1 / 0.16e2 * t8 * t10 * t24 * t17)
  t35 = r0 ** 2
  t36 = t17 ** 2
  t38 = 0.1e1 / t36 / t35
  t39 = t10 * t38
  t40 = s0 * t10
  t41 = t40 * t38
  t43 = 0.1e1 + 0.4e-2 * t41
  t44 = 0.1e1 / t43
  t49 = s0 ** 2
  t51 = t35 ** 2
  t54 = 0.1e1 / t17 / t51 / r0
  t55 = t9 * t54
  t56 = t43 ** 2
  t62 = jnp.cbrt(6)
  t63 = t62 ** 2
  t64 = jnp.pi ** 2
  t65 = jnp.cbrt(t64)
  t66 = t65 ** 2
  t67 = t63 * t66
  t68 = 0.3e1 / 0.1e2 * t67
  t72 = tau0 * t10 / t36 / r0
  t73 = t68 - t72
  t75 = t68 + t72
  t76 = 0.1e1 / t75
  t87 = jnp.logical_or(t4, t1)
  t89 = jnp.cbrt(0.1e1 / jnp.pi)
  t90 = t5 * t89
  t91 = jnp.cbrt(4)
  t92 = t91 ** 2
  t94 = 0.1e1 / t17
  t97 = jnp.where(t1, 0.1e1 / t12, 1)
  t99 = t90 * t92 * t94 * t9 * t97
  t102 = jnp.sqrt(t99)
  t105 = t99 ** 0.15e1
  t107 = t5 ** 2
  t108 = t89 ** 2
  t109 = t107 * t108
  t111 = 0.1e1 / t36
  t113 = t97 ** 2
  t115 = t109 * t91 * t111 * t10 * t113
  t121 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t102 + 0.8969 * t99 + 0.204775 * t105 + 0.123235 * t115))
  t123 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t99) * t121
  t127 = 0.1e1 / (0.2e1 * t9 - 0.2e1)
  t128 = (t15 + t24 - 0.2e1) * t127
  t139 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t102 + 0.1549425e1 * t99 + 0.420775 * t105 + 0.1562925 * t115))
  t152 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t102 + 0.905775 * t99 + 0.1100325 * t105 + 0.1241775 * t115))
  t153 = (0.1e1 + 0.278125e-1 * t99) * t152
  t162 = jnp.where(t87, 0, t2 * (-t123 + t128 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t99) * t139 + t123 - 0.19751673498613801407e-1 * t153) + 0.19751673498613801407e-1 * t128 * t153) / 0.2e1)
  t168 = (0.1e1 + 0.2 * t41) ** 2
  t169 = 0.1e1 / t168
  t177 = t73 ** 2
  t180 = t75 ** 2
  t184 = t49 * t9
  t186 = t184 * t54 * t169
  t190 = t177 ** 2
  t192 = t180 ** 2
  t201 = t90 * t92 * t94
  t204 = jnp.sqrt(t201)
  t207 = t201 ** 0.15e1
  t210 = t109 * t91 * t111
  t216 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t204 + 0.8969 * t201 + 0.204775 * t207 + 0.123235 * t210))
  t219 = jnp.where(t1, t13, 1)
  t233 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t204 + 0.905775 * t201 + 0.1100325 * t207 + 0.1241775 * t210))
  t243 = 0.1e1 + 0.6e-2 * t41
  t251 = t51 ** 2
  t253 = t243 ** 2
  t261 = 0.3e1 / 0.5e1 * t67 * t72
  t262 = tau0 ** 2
  t268 = 0.4e1 * t262 * t9 / t17 / t35 / r0
  t269 = t261 - t268
  t271 = t261 + t268
  t275 = t269 ** 2
  t278 = t271 ** 2
  res = t2 * (t21 + t29) * (params.c_x[0] + 0.4e-2 * params.c_x[1] * s0 * t39 * t44 + 0.32e-4 * params.c_x[2] * t49 * t55 / t56 + params.c_x[3] * t73 * t76 + 0.4e-2 * params.c_x[4] * t73 * t76 * t40 * t38 * t44) + 0.2e1 * t162 * (params.c_ss[0] + 0.8e-1 * params.c_ss[1] * t49 * t55 * t169 + params.c_ss[2] * t73 * t76 + 0.8e-1 * params.c_ss[3] * t177 * t73 / t180 / t75 * t186 + 0.8e-1 * params.c_ss[4] * t190 / t192 * t186) + (-0.621814e-1 * (0.1e1 + 0.53425e-1 * t201) * t216 + 0.19751673498613801407e-1 * (0.2e1 * t219 - 0.2e1) * t127 * (0.1e1 + 0.278125e-1 * t201) * t233 - 0.2e1 * t162) * (params.c_os[0] + 0.6e-2 * params.c_os[1] * s0 * t39 / t243 + 0.864e-6 * params.c_os[2] * t49 * s0 / t251 / t253 / t243 + params.c_os[3] * t269 / t271 + 0.72e-4 * params.c_os[4] * t275 * t269 / t278 / t271 * t184 * t54 / t253)
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