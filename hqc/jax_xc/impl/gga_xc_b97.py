"""Generated from gga_xc_b97.mpl."""

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
  t4 = 0.1e1 / t3
  t5 = t2 * t4
  t6 = 0.1e1 + t5
  t7 = t6 <= p.zeta_threshold
  t8 = jnp.logical_or(r0 <= p.dens_threshold, t7)
  t9 = jnp.where(t7, p.zeta_threshold, t6)
  t10 = r0 * t4
  t13 = jnp.cbrt(p.zeta_threshold)
  t14 = 0.1e1 / t13
  t15 = jnp.cbrt(2)
  t16 = t15 ** 2
  t17 = jnp.cbrt(t10)
  t21 = jnp.where(0.2e1 * t10 <= p.zeta_threshold, t14, t16 / t17 / 0.2e1)
  t22 = t21 ** 2
  t28 = jnp.cbrt(3)
  t29 = jnp.cbrt(jnp.pi)
  t32 = t28 / t29 * t16
  t34 = t13 * p.zeta_threshold
  t36 = jnp.where(0.2e1 <= p.zeta_threshold, t34, 0.2e1 * t15)
  t37 = jnp.cbrt(t3)
  t38 = t36 * t37
  t39 = 0.1e1 / t21
  t43 = jnp.where(t3 / t22 / t21 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.16e2 * t32 * t38 * t39)
  t44 = 0. <= p.dens_threshold
  t46 = jnp.where(0. <= p.zeta_threshold, t34, 0)
  t47 = t46 * t37
  t51 = jnp.where(t44, 0, -0.3e1 / 0.16e2 * t32 * t47 * t39)
  t55 = jnp.where(t8, 0, t9 * (t43 + t51) / 0.2e1)
  t56 = params.c_x[0]
  t57 = params.c_x[1]
  t59 = r0 ** 2
  t60 = jnp.cbrt(r0)
  t61 = t60 ** 2
  t63 = 0.1e1 / t61 / t59
  t64 = s0 * t63
  t66 = 0.1e1 + 0.4e-2 * t64
  t71 = params.c_x[2]
  t72 = s0 ** 2
  t74 = t59 ** 2
  t77 = 0.1e1 / t60 / t74 / r0
  t78 = t66 ** 2
  t83 = params.c_x[3]
  t84 = t72 * s0
  t86 = t74 ** 2
  t87 = 0.1e1 / t86
  t93 = params.c_x[4]
  t94 = t72 ** 2
  t98 = 0.1e1 / t61 / t86 / t59
  t99 = t78 ** 2
  t107 = 0.1e1 - t5
  t108 = t107 <= p.zeta_threshold
  t109 = jnp.logical_or(r1 <= p.dens_threshold, t108)
  t110 = jnp.where(t108, p.zeta_threshold, t107)
  t111 = r1 * t4
  t114 = jnp.cbrt(t111)
  t118 = jnp.where(0.2e1 * t111 <= p.zeta_threshold, t14, t16 / t114 / 0.2e1)
  t119 = t118 ** 2
  t125 = 0.1e1 / t118
  t129 = jnp.where(t3 / t119 / t118 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.16e2 * t32 * t38 * t125)
  t133 = jnp.where(t44, 0, -0.3e1 / 0.16e2 * t32 * t47 * t125)
  t137 = jnp.where(t109, 0, t110 * (t129 + t133) / 0.2e1)
  t139 = r1 ** 2
  t140 = jnp.cbrt(r1)
  t141 = t140 ** 2
  t143 = 0.1e1 / t141 / t139
  t144 = s2 * t143
  t146 = 0.1e1 + 0.4e-2 * t144
  t151 = s2 ** 2
  t153 = t139 ** 2
  t156 = 0.1e1 / t140 / t153 / r1
  t157 = t146 ** 2
  t162 = t151 * s2
  t164 = t153 ** 2
  t165 = 0.1e1 / t164
  t171 = t151 ** 2
  t175 = 0.1e1 / t141 / t164 / t139
  t176 = t157 ** 2
  t184 = jnp.cbrt(0.1e1 / jnp.pi)
  t185 = t28 * t184
  t186 = jnp.cbrt(4)
  t187 = t186 ** 2
  t188 = t185 * t187
  t189 = 0.1e1 / t37
  t190 = t189 * t15
  t191 = jnp.cbrt(t6)
  t193 = jnp.where(t7, t14, 0.1e1 / t191)
  t195 = t188 * t190 * t193
  t198 = jnp.sqrt(t195)
  t201 = t195 ** 0.15e1
  t203 = t28 ** 2
  t204 = t184 ** 2
  t205 = t203 * t204
  t206 = t205 * t186
  t207 = t37 ** 2
  t208 = 0.1e1 / t207
  t209 = t208 * t16
  t210 = t193 ** 2
  t212 = t206 * t209 * t210
  t218 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t198 + 0.8969 * t195 + 0.204775 * t201 + 0.123235 * t212))
  t220 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t195) * t218
  t224 = 0.1e1 / (0.2e1 * t15 - 0.2e1)
  t225 = (t36 + t46 - 0.2e1) * t224
  t236 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t198 + 0.1549425e1 * t195 + 0.420775 * t201 + 0.1562925 * t212))
  t249 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t198 + 0.905775 * t195 + 0.1100325 * t201 + 0.1241775 * t212))
  t250 = (0.1e1 + 0.278125e-1 * t195) * t249
  t259 = jnp.where(t8, 0, t9 * (-t220 + t225 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t195) * t236 + t220 - 0.19751789702565206229e-1 * t250) + 0.19751789702565206229e-1 * t225 * t250) / 0.2e1)
  t260 = params.c_ss[0]
  t261 = params.c_ss[1]
  t264 = 0.1e1 + 0.2 * t64
  t269 = params.c_ss[2]
  t271 = t264 ** 2
  t276 = params.c_ss[3]
  t283 = params.c_ss[4]
  t285 = t271 ** 2
  t292 = jnp.cbrt(t107)
  t294 = jnp.where(t108, t14, 0.1e1 / t292)
  t296 = t188 * t190 * t294
  t299 = jnp.sqrt(t296)
  t302 = t296 ** 0.15e1
  t304 = t294 ** 2
  t306 = t206 * t209 * t304
  t312 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t299 + 0.8969 * t296 + 0.204775 * t302 + 0.123235 * t306))
  t314 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t296) * t312
  t325 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t299 + 0.1549425e1 * t296 + 0.420775 * t302 + 0.1562925 * t306))
  t338 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t299 + 0.905775 * t296 + 0.1100325 * t302 + 0.1241775 * t306))
  t339 = (0.1e1 + 0.278125e-1 * t296) * t338
  t348 = jnp.where(t109, 0, t110 * (-t314 + t225 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t296) * t325 + t314 - 0.19751789702565206229e-1 * t339) + 0.19751789702565206229e-1 * t225 * t339) / 0.2e1)
  t351 = 0.1e1 + 0.2 * t144
  t357 = t351 ** 2
  t369 = t357 ** 2
  t377 = t185 * t187 * t189
  t380 = jnp.sqrt(t377)
  t383 = t377 ** 0.15e1
  t386 = t205 * t186 * t208
  t392 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t380 + 0.8969 * t377 + 0.204775 * t383 + 0.123235 * t386))
  t394 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t377) * t392
  t395 = t2 ** 2
  t396 = t395 ** 2
  t397 = t3 ** 2
  t398 = t397 ** 2
  t402 = jnp.where(t7, t34, t191 * t6)
  t404 = jnp.where(t108, t34, t292 * t107)
  t406 = (t402 + t404 - 0.2e1) * t224
  t417 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t380 + 0.1549425e1 * t377 + 0.420775 * t383 + 0.1562925 * t386))
  t430 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t380 + 0.905775 * t377 + 0.1100325 * t383 + 0.1241775 * t386))
  t431 = (0.1e1 + 0.278125e-1 * t377) * t430
  t441 = t64 + t144
  t445 = 0.1e1 + 0.3e-2 * t64 + 0.3e-2 * t144
  t450 = t441 ** 2
  t452 = t445 ** 2
  t464 = t450 ** 2
  t466 = t452 ** 2
  res = t55 * (t56 + 0.4e-2 * t57 * s0 * t63 / t66 + 0.16e-4 * t71 * t72 * t77 / t78 + 0.64e-7 * t83 * t84 * t87 / t78 / t66 + 0.256e-9 * t93 * t94 * t98 / t99) + t137 * (t56 + 0.4e-2 * t57 * s2 * t143 / t146 + 0.16e-4 * t71 * t151 * t156 / t157 + 0.64e-7 * t83 * t162 * t165 / t157 / t146 + 0.256e-9 * t93 * t171 * t175 / t176) + t259 * (t260 + 0.2 * t261 * s0 * t63 / t264 + 0.4e-1 * t269 * t72 * t77 / t271 + 0.8e-2 * t276 * t84 * t87 / t271 / t264 + 0.16e-2 * t283 * t94 * t98 / t285) + t348 * (t260 + 0.2 * t261 * s2 * t143 / t351 + 0.4e-1 * t269 * t151 * t156 / t357 + 0.8e-2 * t276 * t162 * t165 / t357 / t351 + 0.16e-2 * t283 * t171 * t175 / t369) + (-t394 + t396 / t398 * t406 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t377) * t417 + t394 - 0.19751789702565206229e-1 * t431) + 0.19751789702565206229e-1 * t406 * t431 - t259 - t348) * (params.c_ab[0] + 0.3e-2 * params.c_ab[1] * t441 / t445 + 0.9e-5 * params.c_ab[2] * t450 / t452 + 0.27e-7 * params.c_ab[3] * t450 * t441 / t452 / t445 + 0.81e-10 * params.c_ab[4] * t464 / t466)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = 0.1e1 <= p.zeta_threshold
  t4 = jnp.logical_or(r0 / 0.2e1 <= p.dens_threshold, t3)
  t5 = jnp.where(t3, p.zeta_threshold, 1)
  t6 = jnp.cbrt(p.zeta_threshold)
  t8 = jnp.where(t3, 0.1e1 / t6, 1)
  t9 = t8 ** 2
  t15 = jnp.cbrt(3)
  t16 = jnp.cbrt(jnp.pi)
  t19 = jnp.cbrt(2)
  t20 = t19 ** 2
  t21 = t15 / t16 * t20
  t23 = t6 * p.zeta_threshold
  t25 = jnp.where(0.2e1 <= p.zeta_threshold, t23, 0.2e1 * t19)
  t26 = jnp.cbrt(r0)
  t28 = 0.1e1 / t8
  t32 = jnp.where(r0 / t9 / t8 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.16e2 * t21 * t25 * t26 * t28)
  t35 = jnp.where(0. <= p.zeta_threshold, t23, 0)
  t40 = jnp.where(0. <= p.dens_threshold, 0, -0.3e1 / 0.16e2 * t21 * t35 * t26 * t28)
  t44 = jnp.where(t4, 0, t5 * (t32 + t40) / 0.2e1)
  t48 = r0 ** 2
  t49 = t26 ** 2
  t51 = 0.1e1 / t49 / t48
  t52 = t20 * t51
  t54 = s0 * t20 * t51
  t56 = 0.1e1 + 0.4e-2 * t54
  t62 = s0 ** 2
  t64 = t48 ** 2
  t68 = t19 / t26 / t64 / r0
  t69 = t56 ** 2
  t75 = t62 * s0
  t77 = t64 ** 2
  t78 = 0.1e1 / t77
  t85 = t62 ** 2
  t90 = t20 / t49 / t77 / t48
  t91 = t69 ** 2
  t100 = jnp.cbrt(0.1e1 / jnp.pi)
  t101 = t15 * t100
  t102 = jnp.cbrt(4)
  t103 = t102 ** 2
  t105 = 0.1e1 / t26
  t108 = t101 * t103 * t105 * t19 * t8
  t111 = jnp.sqrt(t108)
  t114 = t108 ** 0.15e1
  t116 = t15 ** 2
  t117 = t100 ** 2
  t118 = t116 * t117
  t120 = 0.1e1 / t49
  t123 = t118 * t102 * t120 * t20 * t9
  t129 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t111 + 0.8969 * t108 + 0.204775 * t114 + 0.123235 * t123))
  t131 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t108) * t129
  t135 = 0.1e1 / (0.2e1 * t19 - 0.2e1)
  t136 = (t25 + t35 - 0.2e1) * t135
  t147 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t111 + 0.1549425e1 * t108 + 0.420775 * t114 + 0.1562925 * t123))
  t160 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t111 + 0.905775 * t108 + 0.1100325 * t114 + 0.1241775 * t123))
  t161 = (0.1e1 + 0.278125e-1 * t108) * t160
  t170 = jnp.where(t4, 0, t5 * (-t131 + t136 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t108) * t147 + t131 - 0.19751789702565206229e-1 * t161) + 0.19751789702565206229e-1 * t136 * t161) / 0.2e1)
  t175 = 0.1e1 + 0.2 * t54
  t182 = t175 ** 2
  t196 = t182 ** 2
  t205 = t101 * t103 * t105
  t208 = jnp.sqrt(t205)
  t211 = t205 ** 0.15e1
  t214 = t118 * t102 * t120
  t220 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t208 + 0.8969 * t205 + 0.204775 * t211 + 0.123235 * t214))
  t223 = jnp.where(t3, t23, 1)
  t237 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t208 + 0.905775 * t205 + 0.1100325 * t211 + 0.1241775 * t214))
  t247 = 0.1e1 + 0.6e-2 * t54
  t254 = t247 ** 2
  t268 = t254 ** 2
  res = 0.2e1 * t44 * (params.c_x[0] + 0.4e-2 * params.c_x[1] * s0 * t52 / t56 + 0.32e-4 * params.c_x[2] * t62 * t68 / t69 + 0.256e-6 * params.c_x[3] * t75 * t78 / t69 / t56 + 0.1024e-8 * params.c_x[4] * t85 * t90 / t91) + 0.2e1 * t170 * (params.c_ss[0] + 0.2 * params.c_ss[1] * s0 * t52 / t175 + 0.8e-1 * params.c_ss[2] * t62 * t68 / t182 + 0.32e-1 * params.c_ss[3] * t75 * t78 / t182 / t175 + 0.64e-2 * params.c_ss[4] * t85 * t90 / t196) + (-0.62182e-1 * (0.1e1 + 0.53425e-1 * t205) * t220 + 0.19751789702565206229e-1 * (0.2e1 * t223 - 0.2e1) * t135 * (0.1e1 + 0.278125e-1 * t205) * t237 - 0.2e1 * t170) * (params.c_ab[0] + 0.6e-2 * params.c_ab[1] * s0 * t52 / t247 + 0.72e-4 * params.c_ab[2] * t62 * t68 / t254 + 0.864e-6 * params.c_ab[3] * t75 * t78 / t254 / t247 + 0.5184e-8 * params.c_ab[4] * t85 * t90 / t268)
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