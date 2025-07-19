"""Generated from hyb_gga_xc_wb97.mpl."""

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
  t103 = 0.1e1 + 0.4e-2 * t101
  t108 = params.c_x[2]
  t109 = s0 ** 2
  t111 = t96 ** 2
  t114 = 0.1e1 / t97 / t111 / r0
  t115 = t103 ** 2
  t120 = params.c_x[3]
  t121 = t109 * s0
  t123 = t111 ** 2
  t124 = 0.1e1 / t123
  t130 = params.c_x[4]
  t131 = t109 ** 2
  t135 = 0.1e1 / t98 / t123 / t96
  t136 = t115 ** 2
  t146 = jnp.where(t8, 0, -0.3e1 / 0.64e2 * t6 * t9 * t17 * t25 / t27 * t91 * (t93 + 0.4e-2 * t94 * s0 * t100 / t103 + 0.16e-4 * t108 * t109 * t114 / t115 + 0.64e-7 * t120 * t121 * t124 / t115 / t103 + 0.256e-9 * t130 * t131 * t135 / t136))
  t148 = 0.1e1 - t5
  t149 = t148 <= p.zeta_threshold
  t150 = jnp.logical_or(r1 <= p.dens_threshold, t149)
  t154 = jnp.cbrt(0.1e1 / t148)
  t159 = t34 * t36 * t154 * t38 / 0.18e2
  t161 = 0.135e1 < t159
  t162 = jnp.where(t161, t159, 0.135e1)
  t163 = t162 ** 2
  t166 = t163 ** 2
  t169 = t166 * t163
  t172 = t166 ** 2
  t184 = t172 ** 2
  t188 = jnp.where(t161, 0.135e1, t159)
  t191 = jax.lax.erf(0.1e1 / t188 / 0.2e1)
  t193 = t188 ** 2
  t196 = jnp.exp(-0.1e1 / t193 / 0.4e1)
  t207 = jnp.where(0.135e1 <= t159, 0.1e1 / t163 / 0.36e2 - 0.1e1 / t166 / 0.96e3 + 0.1e1 / t169 / 0.2688e5 - 0.1e1 / t172 / 0.82944e6 + 0.1e1 / t172 / t163 / 0.2838528e8 - 0.1e1 / t172 / t166 / 0.107347968e10 + 0.1e1 / t172 / t169 / 0.445906944e11 - 0.1e1 / t184 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t188 * (t72 * t191 + 0.2e1 * t188 * (t196 - 0.3e1 / 0.2e1 - 0.2e1 * t193 * (t196 - 0.1e1))))
  t210 = r1 ** 2
  t211 = jnp.cbrt(r1)
  t212 = t211 ** 2
  t214 = 0.1e1 / t212 / t210
  t215 = s2 * t214
  t217 = 0.1e1 + 0.4e-2 * t215
  t222 = s2 ** 2
  t224 = t210 ** 2
  t227 = 0.1e1 / t211 / t224 / r1
  t228 = t217 ** 2
  t233 = t222 * s2
  t235 = t224 ** 2
  t236 = 0.1e1 / t235
  t242 = t222 ** 2
  t246 = 0.1e1 / t212 / t235 / t210
  t247 = t228 ** 2
  t257 = jnp.where(t150, 0, -0.3e1 / 0.64e2 * t148 * t9 * t17 * t25 / t154 * t207 * (t93 + 0.4e-2 * t94 * s2 * t214 / t217 + 0.16e-4 * t108 * t222 * t227 / t228 + 0.64e-7 * t120 * t233 * t236 / t228 / t217 + 0.256e-9 * t130 * t242 * t246 / t247))
  t258 = jnp.where(t7, p.zeta_threshold, t6)
  t259 = t9 * t12
  t260 = t259 * t14
  t261 = 0.1e1 / t20
  t262 = jnp.cbrt(t6)
  t264 = jnp.where(t7, t261, 0.1e1 / t262)
  t266 = t260 * t36 * t264
  t269 = jnp.sqrt(t266)
  t272 = t266 ** 0.15e1
  t274 = t9 ** 2
  t275 = t274 * t31
  t276 = t275 * t13
  t277 = t24 ** 2
  t278 = 0.1e1 / t277
  t279 = t16 ** 2
  t280 = t278 * t279
  t281 = t264 ** 2
  t283 = t276 * t280 * t281
  t289 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t269 + 0.8969 * t266 + 0.204775 * t272 + 0.123235 * t283))
  t291 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t266) * t289
  t293 = jnp.where(0. <= p.zeta_threshold, t21, 0)
  t297 = 0.1e1 / (0.2e1 * t16 - 0.2e1)
  t298 = (t23 + t293 - 0.2e1) * t297
  t309 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t269 + 0.1549425e1 * t266 + 0.420775 * t272 + 0.1562925 * t283))
  t322 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t269 + 0.905775 * t266 + 0.1100325 * t272 + 0.1241775 * t283))
  t323 = (0.1e1 + 0.278125e-1 * t266) * t322
  t332 = jnp.where(t8, 0, t258 * (-t291 + t298 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t266) * t309 + t291 - 0.19751789702565206229e-1 * t323) + 0.19751789702565206229e-1 * t298 * t323) / 0.2e1)
  t333 = params.c_ss[0]
  t334 = params.c_ss[1]
  t337 = 0.1e1 + 0.2 * t101
  t342 = params.c_ss[2]
  t344 = t337 ** 2
  t349 = params.c_ss[3]
  t356 = params.c_ss[4]
  t358 = t344 ** 2
  t365 = jnp.where(t149, p.zeta_threshold, t148)
  t366 = jnp.cbrt(t148)
  t368 = jnp.where(t149, t261, 0.1e1 / t366)
  t370 = t260 * t36 * t368
  t373 = jnp.sqrt(t370)
  t376 = t370 ** 0.15e1
  t378 = t368 ** 2
  t380 = t276 * t280 * t378
  t386 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t373 + 0.8969 * t370 + 0.204775 * t376 + 0.123235 * t380))
  t388 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t370) * t386
  t399 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t373 + 0.1549425e1 * t370 + 0.420775 * t376 + 0.1562925 * t380))
  t412 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t373 + 0.905775 * t370 + 0.1100325 * t376 + 0.1241775 * t380))
  t413 = (0.1e1 + 0.278125e-1 * t370) * t412
  t422 = jnp.where(t150, 0, t365 * (-t388 + t298 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t370) * t399 + t388 - 0.19751789702565206229e-1 * t413) + 0.19751789702565206229e-1 * t298 * t413) / 0.2e1)
  t425 = 0.1e1 + 0.2 * t215
  t431 = t425 ** 2
  t443 = t431 ** 2
  t451 = t259 * t14 * t35
  t454 = jnp.sqrt(t451)
  t457 = t451 ** 0.15e1
  t460 = t275 * t13 * t278
  t466 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t454 + 0.8969 * t451 + 0.204775 * t457 + 0.123235 * t460))
  t468 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t451) * t466
  t469 = t2 ** 2
  t470 = t469 ** 2
  t471 = t3 ** 2
  t472 = t471 ** 2
  t476 = jnp.where(t7, t21, t262 * t6)
  t478 = jnp.where(t149, t21, t366 * t148)
  t480 = (t476 + t478 - 0.2e1) * t297
  t491 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t454 + 0.1549425e1 * t451 + 0.420775 * t457 + 0.1562925 * t460))
  t504 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t454 + 0.905775 * t451 + 0.1100325 * t457 + 0.1241775 * t460))
  t505 = (0.1e1 + 0.278125e-1 * t451) * t504
  t515 = t101 + t215
  t519 = 0.1e1 + 0.3e-2 * t101 + 0.3e-2 * t215
  t524 = t515 ** 2
  t526 = t519 ** 2
  t538 = t524 ** 2
  t540 = t526 ** 2
  res = t146 + t257 + t332 * (t333 + 0.2 * t334 * s0 * t100 / t337 + 0.4e-1 * t342 * t109 * t114 / t344 + 0.8e-2 * t349 * t121 * t124 / t344 / t337 + 0.16e-2 * t356 * t131 * t135 / t358) + t422 * (t333 + 0.2 * t334 * s2 * t214 / t425 + 0.4e-1 * t342 * t222 * t227 / t431 + 0.8e-2 * t349 * t233 * t236 / t431 / t425 + 0.16e-2 * t356 * t242 * t246 / t443) + (-t468 + t470 / t472 * t480 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t451) * t491 + t468 - 0.19751789702565206229e-1 * t505) + 0.19751789702565206229e-1 * t480 * t505 - t332 - t422) * (params.c_ab[0] + 0.3e-2 * params.c_ab[1] * t515 / t519 + 0.9e-5 * params.c_ab[2] * t524 / t526 + 0.27e-7 * params.c_ab[3] * t524 * t515 / t526 / t519 + 0.81e-10 * params.c_ab[4] * t538 / t540)
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
  t91 = t86 * t90
  t93 = s0 * t86 * t90
  t95 = 0.1e1 + 0.4e-2 * t93
  t101 = s0 ** 2
  t103 = t87 ** 2
  t107 = t11 / t19 / t103 / r0
  t108 = t95 ** 2
  t114 = t101 * s0
  t116 = t103 ** 2
  t117 = 0.1e1 / t116
  t124 = t101 ** 2
  t129 = t86 / t88 / t116 / t87
  t130 = t108 ** 2
  t140 = jnp.where(t4, 0, -0.3e1 / 0.64e2 * t8 * t10 * t11 * t18 * t19 * t82 * (params.c_x[0] + 0.4e-2 * params.c_x[1] * s0 * t91 / t95 + 0.32e-4 * params.c_x[2] * t101 * t107 / t108 + 0.256e-6 * params.c_x[3] * t114 * t117 / t108 / t95 + 0.1024e-8 * params.c_x[4] * t124 * t129 / t130))
  t142 = jnp.where(t3, p.zeta_threshold, 1)
  t146 = jnp.where(t3, 0.1e1 / t15, 1)
  t148 = t8 * t10 * t26 * t11 * t146
  t151 = jnp.sqrt(t148)
  t154 = t148 ** 0.15e1
  t156 = t5 ** 2
  t157 = t156 * t23
  t159 = 0.1e1 / t88
  t161 = t146 ** 2
  t163 = t157 * t9 * t159 * t86 * t161
  t169 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t151 + 0.8969 * t148 + 0.204775 * t154 + 0.123235 * t163))
  t171 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t148) * t169
  t173 = jnp.where(0. <= p.zeta_threshold, t16, 0)
  t177 = 0.1e1 / (0.2e1 * t11 - 0.2e1)
  t178 = (t18 + t173 - 0.2e1) * t177
  t189 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t151 + 0.1549425e1 * t148 + 0.420775 * t154 + 0.1562925 * t163))
  t202 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t151 + 0.905775 * t148 + 0.1100325 * t154 + 0.1241775 * t163))
  t203 = (0.1e1 + 0.278125e-1 * t148) * t202
  t212 = jnp.where(t4, 0, t142 * (-t171 + t178 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t148) * t189 + t171 - 0.19751789702565206229e-1 * t203) + 0.19751789702565206229e-1 * t178 * t203) / 0.2e1)
  t217 = 0.1e1 + 0.2 * t93
  t224 = t217 ** 2
  t238 = t224 ** 2
  t247 = t8 * t10 * t26
  t250 = jnp.sqrt(t247)
  t253 = t247 ** 0.15e1
  t256 = t157 * t9 * t159
  t262 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t250 + 0.8969 * t247 + 0.204775 * t253 + 0.123235 * t256))
  t265 = jnp.where(t3, t16, 1)
  t279 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t250 + 0.905775 * t247 + 0.1100325 * t253 + 0.1241775 * t256))
  t289 = 0.1e1 + 0.6e-2 * t93
  t296 = t289 ** 2
  t310 = t296 ** 2
  res = 0.2e1 * t140 + 0.2e1 * t212 * (params.c_ss[0] + 0.2 * params.c_ss[1] * s0 * t91 / t217 + 0.8e-1 * params.c_ss[2] * t101 * t107 / t224 + 0.32e-1 * params.c_ss[3] * t114 * t117 / t224 / t217 + 0.64e-2 * params.c_ss[4] * t124 * t129 / t238) + (-0.62182e-1 * (0.1e1 + 0.53425e-1 * t247) * t262 + 0.19751789702565206229e-1 * (0.2e1 * t265 - 0.2e1) * t177 * (0.1e1 + 0.278125e-1 * t247) * t279 - 0.2e1 * t212) * (params.c_ab[0] + 0.6e-2 * params.c_ab[1] * s0 * t91 / t289 + 0.72e-4 * params.c_ab[2] * t101 * t107 / t296 + 0.864e-6 * params.c_ab[3] * t114 * t117 / t296 / t289 + 0.5184e-8 * params.c_ab[4] * t124 * t129 / t310)
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