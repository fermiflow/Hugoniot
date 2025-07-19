"""Generated from gga_x_hjs.mpl."""

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
  t49 = t32 ** 2
  t50 = 0.1e1 / t49
  t51 = t48 * t50
  t53 = r0 ** 2
  t54 = jnp.cbrt(r0)
  t55 = t54 ** 2
  t57 = 0.1e1 / t55 / t53
  t59 = params.a[0] * t48
  t61 = t50 * s0 * t57
  t65 = 0.1e1 / t31
  t66 = params.a[1] * t65
  t67 = jnp.sqrt(s0)
  t69 = t53 ** 2
  t71 = t67 * s0 / t69
  t75 = t48 ** 2
  t76 = params.a[2] * t75
  t78 = 0.1e1 / t32 / t31
  t79 = s0 ** 2
  t84 = t78 * t79 / t54 / t69 / r0
  t88 = params.a[3] * t48
  t90 = 0.1e1 / t49 / t31
  t96 = t90 * t67 * t79 / t55 / t69 / t53
  t100 = t31 ** 2
  t101 = 0.1e1 / t100
  t102 = params.a[4] * t101
  t103 = t79 * s0
  t104 = t69 ** 2
  t106 = t103 / t104
  t110 = params.a[5] * t75
  t112 = 0.1e1 / t32 / t100
  t118 = t112 * t67 * t103 / t54 / t104 / r0
  t124 = params.b[0] * t75
  t132 = params.b[1] * t48
  t136 = params.b[2] * t65
  t140 = params.b[3] * t75
  t144 = params.b[4] * t48
  t148 = params.b[5] * t101
  t152 = params.b[6] * t75
  t156 = params.b[7] * t48
  t158 = 0.1e1 / t49 / t100
  t159 = t79 ** 2
  t170 = params.b[8] / t100 / t31
  t181 = t51 * s0 * t57 * (t59 * t61 / 0.24e2 + t66 * t71 / 0.48e2 + t76 * t84 / 0.576e3 + t88 * t96 / 0.1152e4 + t102 * t106 / 0.2304e4 + t110 * t118 / 0.27648e5) / (0.1e1 + t124 * t33 * t67 / t54 / r0 / 0.12e2 + t132 * t61 / 0.24e2 + t136 * t71 / 0.48e2 + t140 * t84 / 0.576e3 + t144 * t96 / 0.1152e4 + t148 * t106 / 0.2304e4 + t152 * t118 / 0.27648e5 + t156 * t158 * t159 / t55 / t104 / t53 / 0.55296e5 + t170 * t67 * t159 / t104 / t69 / 0.110592e6) / 0.24e2
  t183 = jnp.where(0.1e-9 < t181, t181, 0.1e-9)
  t184 = p.cam_omega ** 2
  t185 = t184 * t2
  t186 = t44 ** 2
  t189 = t27 ** 2
  t190 = 0.1e1 / t189
  t192 = t185 * t50 / t186 * t190
  t194 = 0.60965 + t183 + t192 / 0.3e1
  t195 = jnp.sqrt(t194)
  t198 = t34 * t47 / t195
  t201 = 0.60965 + t183
  t205 = s0 * t57
  t214 = 0.1e1 + 0.13006513974354692214e-1 * t51 * t205 / (0.1e1 + t51 * t205 / 0.96e2) + 0.42141105276909202774e1 * t183
  t216 = t184 * p.cam_omega * t65
  t223 = t216 / t186 / t44 * t7 / t195 / t194
  t227 = t201 ** 2
  t234 = t227 * t201
  t236 = jnp.sqrt(t201)
  t238 = jnp.sqrt(jnp.pi)
  t239 = 0.4e1 / 0.5e1 * t238
  t240 = jnp.sqrt(t183)
  t245 = jnp.where(0. < 0.7572109999 + t183, 0.757211 + t183, 0.1e-9)
  t246 = jnp.sqrt(t245)
  t253 = t184 ** 2
  t256 = t253 * p.cam_omega * t2 * t90
  t257 = t186 ** 2
  t261 = 0.1e1 / t189 / t6
  t263 = t194 ** 2
  t275 = 0.3e1 * t192
  t277 = jnp.sqrt(0.9e1 * t183 + t275)
  t280 = jnp.sqrt(0.9e1 * t245 + t275)
  t288 = t30 * t33 * t45 * t46
  t293 = 0.1e1 / (t288 / 0.3e1 + t195)
  t295 = jnp.log((t288 / 0.3e1 + t277 / 0.3e1) * t293)
  t301 = jnp.log((t288 / 0.3e1 + t280 / 0.3e1) * t293)
  t308 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.757211 + 0.47272888888888888889e-1 * (0.1e1 - t198 / 0.3e1) / t201 + 0.26366444444444444444e-1 * t214 * (0.2e1 - t198 + t223 / 0.3e1) / t227 - (0.474596e-1 * t214 * t201 + 0.28363733333333333333e-1 * t227 - 0.9086532 * t234 - t236 * t234 * (t239 + 0.12e2 / 0.5e1 * t240 - 0.12e2 / 0.5e1 * t246)) * (0.8e1 - 0.5e1 * t198 + 0.1e2 / 0.3e1 * t223 - t256 / t257 / t44 * t261 / t195 / t263 / 0.3e1) / t234 / 0.9e1 + 0.2e1 / 0.3e1 * t34 * t47 * (t277 / 0.3e1 - t280 / 0.3e1) + 0.2e1 * t183 * t295 - 0.2e1 * t245 * t301))
  t310 = jnp.where(t10, t15, -t17)
  t311 = jnp.where(t14, t11, t310)
  t312 = 0.1e1 + t311
  t314 = jnp.cbrt(t312)
  t316 = jnp.where(t312 <= p.zeta_threshold, t23, t314 * t312)
  t318 = jnp.where(t36, t15, -t17)
  t319 = jnp.where(t38, t11, t318)
  t320 = 0.1e1 + t319
  t322 = jnp.cbrt(t320)
  t323 = jnp.where(t320 <= p.zeta_threshold, t22, t322)
  t324 = 0.1e1 / t323
  t325 = t324 * t46
  t327 = r1 ** 2
  t328 = jnp.cbrt(r1)
  t329 = t328 ** 2
  t331 = 0.1e1 / t329 / t327
  t333 = t50 * s2 * t331
  t336 = jnp.sqrt(s2)
  t338 = t327 ** 2
  t340 = t336 * s2 / t338
  t343 = s2 ** 2
  t348 = t78 * t343 / t328 / t338 / r1
  t356 = t90 * t336 * t343 / t329 / t338 / t327
  t359 = t343 * s2
  t360 = t338 ** 2
  t362 = t359 / t360
  t370 = t112 * t336 * t359 / t328 / t360 / r1
  t393 = t343 ** 2
  t411 = t51 * s2 * t331 * (t59 * t333 / 0.24e2 + t66 * t340 / 0.48e2 + t76 * t348 / 0.576e3 + t88 * t356 / 0.1152e4 + t102 * t362 / 0.2304e4 + t110 * t370 / 0.27648e5) / (0.1e1 + t124 * t33 * t336 / t328 / r1 / 0.12e2 + t132 * t333 / 0.24e2 + t136 * t340 / 0.48e2 + t140 * t348 / 0.576e3 + t144 * t356 / 0.1152e4 + t148 * t362 / 0.2304e4 + t152 * t370 / 0.27648e5 + t156 * t158 * t393 / t329 / t360 / t327 / 0.55296e5 + t170 * t336 * t393 / t360 / t338 / 0.110592e6) / 0.24e2
  t413 = jnp.where(0.1e-9 < t411, t411, 0.1e-9)
  t414 = t323 ** 2
  t418 = t185 * t50 / t414 * t190
  t420 = 0.60965 + t413 + t418 / 0.3e1
  t421 = jnp.sqrt(t420)
  t424 = t34 * t325 / t421
  t427 = 0.60965 + t413
  t431 = s2 * t331
  t440 = 0.1e1 + 0.13006513974354692214e-1 * t51 * t431 / (0.1e1 + t51 * t431 / 0.96e2) + 0.42141105276909202774e1 * t413
  t447 = t216 / t414 / t323 * t7 / t421 / t420
  t451 = t427 ** 2
  t458 = t451 * t427
  t460 = jnp.sqrt(t427)
  t462 = jnp.sqrt(t413)
  t467 = jnp.where(0. < 0.7572109999 + t413, 0.757211 + t413, 0.1e-9)
  t468 = jnp.sqrt(t467)
  t475 = t414 ** 2
  t479 = t420 ** 2
  t491 = 0.3e1 * t418
  t493 = jnp.sqrt(0.9e1 * t413 + t491)
  t496 = jnp.sqrt(0.9e1 * t467 + t491)
  t504 = t30 * t33 * t324 * t46
  t509 = 0.1e1 / (t504 / 0.3e1 + t421)
  t511 = jnp.log((t504 / 0.3e1 + t493 / 0.3e1) * t509)
  t517 = jnp.log((t504 / 0.3e1 + t496 / 0.3e1) * t509)
  t524 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t316 * t27 * (0.757211 + 0.47272888888888888889e-1 * (0.1e1 - t424 / 0.3e1) / t427 + 0.26366444444444444444e-1 * t440 * (0.2e1 - t424 + t447 / 0.3e1) / t451 - (0.474596e-1 * t440 * t427 + 0.28363733333333333333e-1 * t451 - 0.9086532 * t458 - t460 * t458 * (t239 + 0.12e2 / 0.5e1 * t462 - 0.12e2 / 0.5e1 * t468)) * (0.8e1 - 0.5e1 * t424 + 0.1e2 / 0.3e1 * t447 - t256 / t475 / t323 * t261 / t421 / t479 / 0.3e1) / t458 / 0.9e1 + 0.2e1 / 0.3e1 * t34 * t325 * (t493 / 0.3e1 - t496 / 0.3e1) + 0.2e1 * t413 * t511 - 0.2e1 * t467 * t517))
  res = t308 + t524
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
  t32 = t24 ** 2
  t33 = 0.1e1 / t32
  t34 = t31 * t33
  t35 = t34 * s0
  t36 = jnp.cbrt(2)
  t37 = t36 ** 2
  t38 = r0 ** 2
  t39 = t19 ** 2
  t41 = 0.1e1 / t39 / t38
  t42 = t37 * t41
  t47 = s0 * t37 * t41
  t51 = 0.1e1 / t23
  t53 = jnp.sqrt(s0)
  t55 = t38 ** 2
  t57 = t53 * s0 / t55
  t61 = t31 ** 2
  t64 = 0.1e1 / t24 / t23
  t66 = s0 ** 2
  t71 = t66 * t36 / t19 / t55 / r0
  t77 = 0.1e1 / t32 / t23
  t84 = t53 * t66 * t37 / t39 / t55 / t38
  t88 = t23 ** 2
  t89 = 0.1e1 / t88
  t91 = t66 * s0
  t92 = t55 ** 2
  t94 = t91 / t92
  t100 = 0.1e1 / t24 / t88
  t107 = t53 * t91 * t36 / t19 / t92 / r0
  t153 = t66 ** 2
  t176 = t35 * t42 * (params.a[0] * t31 * t33 * t47 / 0.24e2 + params.a[1] * t51 * t57 / 0.24e2 + params.a[2] * t61 * t64 * t71 / 0.288e3 + params.a[3] * t31 * t77 * t84 / 0.576e3 + params.a[4] * t89 * t94 / 0.576e3 + params.a[5] * t61 * t100 * t107 / 0.6912e4) / (0.1e1 + params.b[0] * t61 * t25 * t53 * t36 / t19 / r0 / 0.12e2 + params.b[1] * t31 * t33 * t47 / 0.24e2 + params.b[2] * t51 * t57 / 0.24e2 + params.b[3] * t61 * t64 * t71 / 0.288e3 + params.b[4] * t31 * t77 * t84 / 0.576e3 + params.b[5] * t89 * t94 / 0.576e3 + params.b[6] * t61 * t100 * t107 / 0.6912e4 + params.b[7] * t31 / t32 / t88 * t153 * t37 / t39 / t92 / t38 / 0.13824e5 + params.b[8] / t88 / t23 * t53 * t153 / t92 / t55 / 0.13824e5) / 0.24e2
  t178 = jnp.where(0.1e-9 < t176, t176, 0.1e-9)
  t179 = p.cam_omega ** 2
  t181 = t27 ** 2
  t186 = t179 * t3 * t33 / t181 / t39
  t188 = 0.60965 + t178 + t186 / 0.3e1
  t189 = jnp.sqrt(t188)
  t192 = t26 * t30 / t189
  t195 = 0.60965 + t178
  t207 = 0.1e1 + 0.13006513974354692214e-1 * t35 * t42 / (0.1e1 + t34 * t47 / 0.96e2) + 0.42141105276909202774e1 * t178
  t217 = t179 * p.cam_omega * t51 / t181 / t27 / r0 / t189 / t188
  t221 = t195 ** 2
  t228 = t221 * t195
  t230 = jnp.sqrt(t195)
  t232 = jnp.sqrt(jnp.pi)
  t234 = jnp.sqrt(t178)
  t239 = jnp.where(0. < 0.7572109999 + t178, 0.757211 + t178, 0.1e-9)
  t240 = jnp.sqrt(t239)
  t247 = t179 ** 2
  t251 = t181 ** 2
  t257 = t188 ** 2
  t269 = 0.3e1 * t186
  t271 = jnp.sqrt(0.9e1 * t178 + t269)
  t274 = jnp.sqrt(0.9e1 * t239 + t269)
  t282 = t22 * t25 * t28 * t29
  t287 = 0.1e1 / (t282 / 0.3e1 + t189)
  t289 = jnp.log((t282 / 0.3e1 + t271 / 0.3e1) * t287)
  t295 = jnp.log((t282 / 0.3e1 + t274 / 0.3e1) * t287)
  t302 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.757211 + 0.47272888888888888889e-1 * (0.1e1 - t192 / 0.3e1) / t195 + 0.26366444444444444444e-1 * t207 * (0.2e1 - t192 + t217 / 0.3e1) / t221 - (0.474596e-1 * t207 * t195 + 0.28363733333333333333e-1 * t221 - 0.9086532 * t228 - t230 * t228 * (0.4e1 / 0.5e1 * t232 + 0.12e2 / 0.5e1 * t234 - 0.12e2 / 0.5e1 * t240)) * (0.8e1 - 0.5e1 * t192 + 0.1e2 / 0.3e1 * t217 - t247 * p.cam_omega * t3 * t77 / t251 / t27 / t39 / r0 / t189 / t257 / 0.3e1) / t228 / 0.9e1 + 0.2e1 / 0.3e1 * t26 * t30 * (t271 / 0.3e1 - t274 / 0.3e1) + 0.2e1 * t178 * t289 - 0.2e1 * t239 * t295))
  res = 0.2e1 * t302
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