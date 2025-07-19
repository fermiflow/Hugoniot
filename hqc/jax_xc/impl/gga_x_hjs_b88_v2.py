"""Generated from gga_x_hjs_b88_v2.mpl."""

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
  t29 = t2 ** 2
  t30 = p.cam_omega * t29
  t31 = jnp.pi ** 2
  t32 = jnp.cbrt(t31)
  t33 = 0.1e1 / t32
  t34 = t30 * t33
  t36 = 0.1e1 + t17 <= p.zeta_threshold
  t38 = 0.1e1 - t17 <= p.zeta_threshold
  t39 = jnp.where(t38, t15, t17)
  t40 = jnp.where(t36, t11, t39)
  t41 = 0.1e1 + t40
  t43 = jnp.cbrt(t41)
  t44 = jnp.where(t41 <= p.zeta_threshold, t22, t43)
  t45 = 0.1e1 / t44
  t46 = 0.1e1 / t27
  t47 = t45 * t46
  t48 = jnp.cbrt(6)
  t49 = t48 ** 2
  t50 = t49 * t33
  t51 = jnp.sqrt(s0)
  t52 = jnp.cbrt(r0)
  t58 = jnp.exp(-t50 * t51 / t52 / r0 / 0.12e2)
  t59 = jnp.exp(20)
  t61 = 0.1e1 / (t59 - 0.1e1)
  t64 = 0.1e1 / (0.1e1 + t61)
  t66 = jnp.log((t58 + t61) * t64)
  t67 = t66 ** 2
  t68 = params.a[0]
  t70 = params.a[1]
  t71 = t67 * t66
  t73 = params.a[2]
  t74 = t67 ** 2
  t76 = params.a[3]
  t77 = t74 * t66
  t79 = params.a[4]
  t80 = t74 * t67
  t82 = params.a[5]
  t83 = t74 * t71
  t87 = params.b[0]
  t89 = params.b[1]
  t91 = params.b[2]
  t93 = params.b[3]
  t95 = params.b[4]
  t97 = params.b[5]
  t99 = params.b[6]
  t101 = params.b[7]
  t102 = t74 ** 2
  t104 = params.b[8]
  t109 = t67 * (t68 * t67 - t70 * t71 + t73 * t74 - t76 * t77 + t79 * t80 - t82 * t83) / (-t104 * t102 * t66 + t101 * t102 - t87 * t66 + t89 * t67 - t91 * t71 + t93 * t74 - t95 * t77 + t97 * t80 - t99 * t83 + 0.1e1)
  t111 = jnp.where(0.1e-9 < t109, t109, 0.1e-9)
  t112 = p.cam_omega ** 2
  t113 = t112 * t2
  t114 = t32 ** 2
  t115 = 0.1e1 / t114
  t116 = t44 ** 2
  t119 = t27 ** 2
  t120 = 0.1e1 / t119
  t122 = t113 * t115 / t116 * t120
  t124 = 0.60965 + t111 + t122 / 0.3e1
  t125 = jnp.sqrt(t124)
  t128 = t34 * t47 / t125
  t131 = 0.60965 + t111
  t141 = 0.1e1 + 0.31215633538451261314 * t67 / (0.1e1 + t67 / 0.4e1) + 0.42141105276909202774e1 * t111
  t144 = t112 * p.cam_omega / t31
  t151 = t144 / t116 / t44 * t7 / t125 / t124
  t155 = t131 ** 2
  t162 = t155 * t131
  t164 = jnp.sqrt(t131)
  t166 = jnp.sqrt(jnp.pi)
  t167 = 0.4e1 / 0.5e1 * t166
  t168 = jnp.sqrt(t111)
  t173 = jnp.where(0. < 0.7572109999 + t111, 0.757211 + t111, 0.1e-9)
  t174 = jnp.sqrt(t173)
  t181 = t112 ** 2
  t186 = t181 * p.cam_omega * t2 / t114 / t31
  t187 = t116 ** 2
  t191 = 0.1e1 / t119 / t6
  t193 = t124 ** 2
  t205 = 0.3e1 * t122
  t207 = jnp.sqrt(0.9e1 * t111 + t205)
  t210 = jnp.sqrt(0.9e1 * t173 + t205)
  t218 = t30 * t33 * t45 * t46
  t223 = 0.1e1 / (t218 / 0.3e1 + t125)
  t225 = jnp.log((t218 / 0.3e1 + t207 / 0.3e1) * t223)
  t231 = jnp.log((t218 / 0.3e1 + t210 / 0.3e1) * t223)
  t238 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.757211 + 0.47272888888888888889e-1 * (0.1e1 - t128 / 0.3e1) / t131 + 0.26366444444444444444e-1 * t141 * (0.2e1 - t128 + t151 / 0.3e1) / t155 - (0.474596e-1 * t141 * t131 + 0.28363733333333333333e-1 * t155 - 0.9086532 * t162 - t164 * t162 * (t167 + 0.12e2 / 0.5e1 * t168 - 0.12e2 / 0.5e1 * t174)) * (0.8e1 - 0.5e1 * t128 + 0.1e2 / 0.3e1 * t151 - t186 / t187 / t44 * t191 / t125 / t193 / 0.3e1) / t162 / 0.9e1 + 0.2e1 / 0.3e1 * t34 * t47 * (t207 / 0.3e1 - t210 / 0.3e1) + 0.2e1 * t111 * t225 - 0.2e1 * t173 * t231))
  t240 = jnp.where(t10, t15, -t17)
  t241 = jnp.where(t14, t11, t240)
  t242 = 0.1e1 + t241
  t244 = jnp.cbrt(t242)
  t246 = jnp.where(t242 <= p.zeta_threshold, t23, t244 * t242)
  t248 = jnp.where(t36, t15, -t17)
  t249 = jnp.where(t38, t11, t248)
  t250 = 0.1e1 + t249
  t252 = jnp.cbrt(t250)
  t253 = jnp.where(t250 <= p.zeta_threshold, t22, t252)
  t254 = 0.1e1 / t253
  t255 = t254 * t46
  t256 = jnp.sqrt(s2)
  t257 = jnp.cbrt(r1)
  t263 = jnp.exp(-t50 * t256 / t257 / r1 / 0.12e2)
  t266 = jnp.log((t263 + t61) * t64)
  t267 = t266 ** 2
  t269 = t267 * t266
  t271 = t267 ** 2
  t273 = t271 * t266
  t275 = t271 * t267
  t277 = t271 * t269
  t288 = t271 ** 2
  t294 = t267 * (t68 * t267 - t70 * t269 + t73 * t271 - t76 * t273 + t79 * t275 - t82 * t277) / (-t104 * t288 * t266 + t101 * t288 - t87 * t266 + t89 * t267 - t91 * t269 + t93 * t271 - t95 * t273 + t97 * t275 - t99 * t277 + 0.1e1)
  t296 = jnp.where(0.1e-9 < t294, t294, 0.1e-9)
  t297 = t253 ** 2
  t301 = t113 * t115 / t297 * t120
  t303 = 0.60965 + t296 + t301 / 0.3e1
  t304 = jnp.sqrt(t303)
  t307 = t34 * t255 / t304
  t310 = 0.60965 + t296
  t320 = 0.1e1 + 0.31215633538451261314 * t267 / (0.1e1 + t267 / 0.4e1) + 0.42141105276909202774e1 * t296
  t327 = t144 / t297 / t253 * t7 / t304 / t303
  t331 = t310 ** 2
  t338 = t331 * t310
  t340 = jnp.sqrt(t310)
  t342 = jnp.sqrt(t296)
  t347 = jnp.where(0. < 0.7572109999 + t296, 0.757211 + t296, 0.1e-9)
  t348 = jnp.sqrt(t347)
  t355 = t297 ** 2
  t359 = t303 ** 2
  t371 = 0.3e1 * t301
  t373 = jnp.sqrt(0.9e1 * t296 + t371)
  t376 = jnp.sqrt(0.9e1 * t347 + t371)
  t384 = t30 * t33 * t254 * t46
  t389 = 0.1e1 / (t384 / 0.3e1 + t304)
  t391 = jnp.log((t384 / 0.3e1 + t373 / 0.3e1) * t389)
  t397 = jnp.log((t384 / 0.3e1 + t376 / 0.3e1) * t389)
  t404 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t246 * t27 * (0.757211 + 0.47272888888888888889e-1 * (0.1e1 - t307 / 0.3e1) / t310 + 0.26366444444444444444e-1 * t320 * (0.2e1 - t307 + t327 / 0.3e1) / t331 - (0.474596e-1 * t320 * t310 + 0.28363733333333333333e-1 * t331 - 0.9086532 * t338 - t340 * t338 * (t167 + 0.12e2 / 0.5e1 * t342 - 0.12e2 / 0.5e1 * t348)) * (0.8e1 - 0.5e1 * t307 + 0.1e2 / 0.3e1 * t327 - t186 / t355 / t253 * t191 / t304 / t359 / 0.3e1) / t338 / 0.9e1 + 0.2e1 / 0.3e1 * t34 * t255 * (t373 / 0.3e1 - t376 / 0.3e1) + 0.2e1 * t296 * t391 - 0.2e1 * t347 * t397))
  res = t238 + t404
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
  t21 = t3 ** 2
  t22 = p.cam_omega * t21
  t23 = jnp.pi ** 2
  t24 = jnp.cbrt(t23)
  t25 = 0.1e1 / t24
  t26 = t22 * t25
  t27 = jnp.where(t13, t14, t16)
  t28 = 0.1e1 / t27
  t29 = 0.1e1 / t19
  t30 = t28 * t29
  t31 = jnp.cbrt(6)
  t32 = t31 ** 2
  t34 = jnp.sqrt(s0)
  t35 = jnp.cbrt(2)
  t42 = jnp.exp(-t32 * t25 * t34 * t35 / t19 / r0 / 0.12e2)
  t43 = jnp.exp(20)
  t45 = 0.1e1 / (t43 - 0.1e1)
  t50 = jnp.log((t42 + t45) / (0.1e1 + t45))
  t51 = t50 ** 2
  t55 = t51 * t50
  t58 = t51 ** 2
  t61 = t58 * t50
  t64 = t58 * t51
  t67 = t58 * t55
  t86 = t58 ** 2
  t93 = t51 * (params.a[0] * t51 - params.a[1] * t55 + params.a[2] * t58 - params.a[3] * t61 + params.a[4] * t64 - params.a[5] * t67) / (-params.b[8] * t86 * t50 - params.b[0] * t50 + params.b[1] * t51 - params.b[2] * t55 + params.b[3] * t58 - params.b[4] * t61 + params.b[5] * t64 - params.b[6] * t67 + params.b[7] * t86 + 0.1e1)
  t95 = jnp.where(0.1e-9 < t93, t93, 0.1e-9)
  t96 = p.cam_omega ** 2
  t98 = t24 ** 2
  t100 = t27 ** 2
  t103 = t19 ** 2
  t106 = t96 * t3 / t98 / t100 / t103
  t108 = 0.60965 + t95 + t106 / 0.3e1
  t109 = jnp.sqrt(t108)
  t112 = t26 * t30 / t109
  t115 = 0.60965 + t95
  t125 = 0.1e1 + 0.31215633538451261314 * t51 / (0.1e1 + t51 / 0.4e1) + 0.42141105276909202774e1 * t95
  t136 = t96 * p.cam_omega / t23 / t100 / t27 / r0 / t109 / t108
  t140 = t115 ** 2
  t147 = t140 * t115
  t149 = jnp.sqrt(t115)
  t151 = jnp.sqrt(jnp.pi)
  t153 = jnp.sqrt(t95)
  t158 = jnp.where(0. < 0.7572109999 + t95, 0.757211 + t95, 0.1e-9)
  t159 = jnp.sqrt(t158)
  t166 = t96 ** 2
  t172 = t100 ** 2
  t178 = t108 ** 2
  t190 = 0.3e1 * t106
  t192 = jnp.sqrt(0.9e1 * t95 + t190)
  t195 = jnp.sqrt(0.9e1 * t158 + t190)
  t203 = t22 * t25 * t28 * t29
  t208 = 0.1e1 / (t203 / 0.3e1 + t109)
  t210 = jnp.log((t203 / 0.3e1 + t192 / 0.3e1) * t208)
  t216 = jnp.log((t203 / 0.3e1 + t195 / 0.3e1) * t208)
  t223 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.757211 + 0.47272888888888888889e-1 * (0.1e1 - t112 / 0.3e1) / t115 + 0.26366444444444444444e-1 * t125 * (0.2e1 - t112 + t136 / 0.3e1) / t140 - (0.474596e-1 * t125 * t115 + 0.28363733333333333333e-1 * t140 - 0.9086532 * t147 - t149 * t147 * (0.4e1 / 0.5e1 * t151 + 0.12e2 / 0.5e1 * t153 - 0.12e2 / 0.5e1 * t159)) * (0.8e1 - 0.5e1 * t112 + 0.1e2 / 0.3e1 * t136 - t166 * p.cam_omega * t3 / t98 / t23 / t172 / t27 / t103 / r0 / t109 / t178 / 0.3e1) / t147 / 0.9e1 + 0.2e1 / 0.3e1 * t26 * t30 * (t192 / 0.3e1 - t195 / 0.3e1) + 0.2e1 * t95 * t210 - 0.2e1 * t158 * t216))
  res = 0.2e1 * t223
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