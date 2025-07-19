"""Generated from mgga_c_m06l.mpl."""

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
  t9 = jnp.where(t7, p.zeta_threshold, t6)
  t10 = jnp.cbrt(3)
  t12 = jnp.cbrt(0.1e1 / jnp.pi)
  t13 = t10 * t12
  t14 = jnp.cbrt(4)
  t15 = t14 ** 2
  t16 = t13 * t15
  t17 = jnp.cbrt(t3)
  t18 = 0.1e1 / t17
  t19 = jnp.cbrt(2)
  t20 = t18 * t19
  t21 = jnp.cbrt(p.zeta_threshold)
  t22 = 0.1e1 / t21
  t23 = jnp.cbrt(t6)
  t25 = jnp.where(t7, t22, 0.1e1 / t23)
  t27 = t16 * t20 * t25
  t30 = jnp.sqrt(t27)
  t33 = t27 ** 0.15e1
  t35 = t10 ** 2
  t36 = t12 ** 2
  t37 = t35 * t36
  t38 = t37 * t14
  t39 = t17 ** 2
  t40 = 0.1e1 / t39
  t41 = t19 ** 2
  t42 = t40 * t41
  t43 = t25 ** 2
  t45 = t38 * t42 * t43
  t51 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t30 + 0.8969 * t27 + 0.204775 * t33 + 0.123235 * t45))
  t53 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t27) * t51
  t55 = t21 * p.zeta_threshold
  t57 = jnp.where(0.2e1 <= p.zeta_threshold, t55, 0.2e1 * t19)
  t59 = jnp.where(0. <= p.zeta_threshold, t55, 0)
  t63 = 0.1e1 / (0.2e1 * t19 - 0.2e1)
  t64 = (t57 + t59 - 0.2e1) * t63
  t75 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t30 + 0.1549425e1 * t27 + 0.420775 * t33 + 0.1562925 * t45))
  t88 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t30 + 0.905775 * t27 + 0.1100325 * t33 + 0.1241775 * t45))
  t89 = (0.1e1 + 0.278125e-1 * t27) * t88
  t98 = jnp.where(t8, 0, t9 * (-t53 + t64 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t27) * t75 + t53 - 0.19751673498613801407e-1 * t89) + 0.19751673498613801407e-1 * t64 * t89) / 0.2e1)
  t99 = params.css[0]
  t101 = params.css[1] * params.gamma_ss
  t102 = r0 ** 2
  t103 = jnp.cbrt(r0)
  t104 = t103 ** 2
  t106 = 0.1e1 / t104 / t102
  t107 = s0 * t106
  t110 = params.gamma_ss * s0 * t106 + 0.1e1
  t115 = params.gamma_ss ** 2
  t116 = params.css[2] * t115
  t117 = s0 ** 2
  t118 = t102 ** 2
  t121 = 0.1e1 / t103 / t118 / r0
  t123 = t110 ** 2
  t129 = params.css[3] * t115 * params.gamma_ss
  t131 = t118 ** 2
  t139 = t115 ** 2
  t140 = params.css[4] * t139
  t141 = t117 ** 2
  t146 = t123 ** 2
  t157 = 0.1e1 - s0 / r0 / tau0 / 0.8e1
  t158 = tau0 ** 2
  t163 = params.Fermi_D_cnst ** 2
  t164 = 0.1e1 / t163
  t167 = jnp.exp(-0.4e1 * t158 / t103 / t102 / r0 * t164)
  t172 = 0.1e1 - t5
  t173 = t172 <= p.zeta_threshold
  t174 = jnp.logical_or(r1 <= p.dens_threshold, t173)
  t175 = jnp.where(t173, p.zeta_threshold, t172)
  t176 = jnp.cbrt(t172)
  t178 = jnp.where(t173, t22, 0.1e1 / t176)
  t180 = t16 * t20 * t178
  t183 = jnp.sqrt(t180)
  t186 = t180 ** 0.15e1
  t188 = t178 ** 2
  t190 = t38 * t42 * t188
  t196 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t183 + 0.8969 * t180 + 0.204775 * t186 + 0.123235 * t190))
  t198 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t180) * t196
  t209 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t183 + 0.1549425e1 * t180 + 0.420775 * t186 + 0.1562925 * t190))
  t222 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t183 + 0.905775 * t180 + 0.1100325 * t186 + 0.1241775 * t190))
  t223 = (0.1e1 + 0.278125e-1 * t180) * t222
  t232 = jnp.where(t174, 0, t175 * (-t198 + t64 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t180) * t209 + t198 - 0.19751673498613801407e-1 * t223) + 0.19751673498613801407e-1 * t64 * t223) / 0.2e1)
  t233 = r1 ** 2
  t234 = jnp.cbrt(r1)
  t235 = t234 ** 2
  t237 = 0.1e1 / t235 / t233
  t238 = s2 * t237
  t241 = params.gamma_ss * s2 * t237 + 0.1e1
  t245 = s2 ** 2
  t246 = t233 ** 2
  t249 = 0.1e1 / t234 / t246 / r1
  t251 = t241 ** 2
  t256 = t246 ** 2
  t263 = t245 ** 2
  t268 = t251 ** 2
  t279 = 0.1e1 - s2 / r1 / tau1 / 0.8e1
  t280 = tau1 ** 2
  t287 = jnp.exp(-0.4e1 * t280 / t234 / t233 / r1 * t164)
  t292 = t13 * t15 * t18
  t295 = jnp.sqrt(t292)
  t298 = t292 ** 0.15e1
  t301 = t37 * t14 * t40
  t307 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t295 + 0.8969 * t292 + 0.204775 * t298 + 0.123235 * t301))
  t309 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t292) * t307
  t310 = t2 ** 2
  t311 = t310 ** 2
  t312 = t3 ** 2
  t313 = t312 ** 2
  t317 = jnp.where(t7, t55, t23 * t6)
  t319 = jnp.where(t173, t55, t176 * t172)
  t321 = (t317 + t319 - 0.2e1) * t63
  t332 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t295 + 0.1549425e1 * t292 + 0.420775 * t298 + 0.1562925 * t301))
  t345 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t295 + 0.905775 * t292 + 0.1100325 * t298 + 0.1241775 * t301))
  t346 = (0.1e1 + 0.278125e-1 * t292) * t345
  t353 = -t309 + t311 / t313 * t321 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t292) * t332 + t309 - 0.19751673498613801407e-1 * t346) + 0.19751673498613801407e-1 * t321 * t346 - t98 - t232
  t357 = t107 + t238
  t359 = params.gamma_ab * t357 + 0.1e1
  t364 = params.gamma_ab ** 2
  t366 = t357 ** 2
  t367 = t359 ** 2
  t380 = t364 ** 2
  t382 = t366 ** 2
  t383 = t367 ** 2
  t389 = params.dss[0]
  t393 = 0.2e1 * tau0 / t104 / r0
  t394 = jnp.cbrt(6)
  t395 = t394 ** 2
  t396 = jnp.pi ** 2
  t397 = jnp.cbrt(t396)
  t398 = t397 ** 2
  t399 = t395 * t398
  t400 = 0.3e1 / 0.5e1 * t399
  t403 = 0.1e1 + params.alpha_ss * (t107 + t393 - t400)
  t406 = params.dss[1]
  t409 = params.dss[2]
  t410 = t393 - t400
  t413 = t403 ** 2
  t416 = params.dss[3]
  t419 = params.dss[4]
  t423 = params.dss[5]
  t424 = t410 ** 2
  t436 = 0.2e1 * tau1 / t235 / r1
  t439 = 0.1e1 + params.alpha_ss * (t238 + t436 - t400)
  t444 = t436 - t400
  t447 = t439 ** 2
  t455 = t444 ** 2
  t465 = 0.6e1 / 0.5e1 * t399
  t468 = 0.1e1 + params.alpha_ab * (t107 + t238 + t393 + t436 - t465)
  t474 = t393 + t436 - t465
  t477 = t468 ** 2
  t486 = t474 ** 2
  res = t98 * (t99 + t101 * t107 / t110 + t116 * t117 * t121 / t123 + t129 * t117 * s0 / t131 / t123 / t110 + t140 * t141 / t104 / t131 / t102 / t146) * t157 * (0.1e1 - t167) + t232 * (t99 + t101 * t238 / t241 + t116 * t245 * t249 / t251 + t129 * t245 * s2 / t256 / t251 / t241 + t140 * t263 / t235 / t256 / t233 / t268) * t279 * (0.1e1 - t287) + t353 * (params.cab[0] + params.cab[1] * params.gamma_ab * t357 / t359 + params.cab[2] * t364 * t366 / t367 + params.cab[3] * t364 * params.gamma_ab * t366 * t357 / t367 / t359 + params.cab[4] * t380 * t382 / t383) + t98 * (t389 / t403 + (t406 * s0 * t106 + t409 * t410) / t413 + (t419 * s0 * t106 * t410 + t416 * t117 * t121 + t423 * t424) / t413 / t403) * t157 + t232 * (t389 / t439 + (t406 * s2 * t237 + t409 * t444) / t447 + (t419 * s2 * t237 * t444 + t416 * t245 * t249 + t423 * t455) / t447 / t439) * t279 + t353 * (params.dab[0] / t468 + (params.dab[1] * t357 + params.dab[2] * t474) / t477 + (params.dab[4] * t357 * t474 + params.dab[3] * t366 + params.dab[5] * t486) / t477 / t468)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = 0.1e1 <= p.zeta_threshold
  t4 = jnp.logical_or(r0 / 0.2e1 <= p.dens_threshold, t3)
  t5 = jnp.where(t3, p.zeta_threshold, 1)
  t6 = jnp.cbrt(3)
  t8 = jnp.cbrt(0.1e1 / jnp.pi)
  t9 = t6 * t8
  t10 = jnp.cbrt(4)
  t11 = t10 ** 2
  t13 = jnp.cbrt(r0)
  t14 = 0.1e1 / t13
  t15 = jnp.cbrt(2)
  t17 = jnp.cbrt(p.zeta_threshold)
  t19 = jnp.where(t3, 0.1e1 / t17, 1)
  t21 = t9 * t11 * t14 * t15 * t19
  t24 = jnp.sqrt(t21)
  t27 = t21 ** 0.15e1
  t29 = t6 ** 2
  t30 = t8 ** 2
  t31 = t29 * t30
  t33 = t13 ** 2
  t34 = 0.1e1 / t33
  t35 = t15 ** 2
  t37 = t19 ** 2
  t39 = t31 * t10 * t34 * t35 * t37
  t45 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t24 + 0.8969 * t21 + 0.204775 * t27 + 0.123235 * t39))
  t47 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t21) * t45
  t49 = t17 * p.zeta_threshold
  t51 = jnp.where(0.2e1 <= p.zeta_threshold, t49, 0.2e1 * t15)
  t53 = jnp.where(0. <= p.zeta_threshold, t49, 0)
  t57 = 0.1e1 / (0.2e1 * t15 - 0.2e1)
  t58 = (t51 + t53 - 0.2e1) * t57
  t69 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t24 + 0.1549425e1 * t21 + 0.420775 * t27 + 0.1562925 * t39))
  t82 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t24 + 0.905775 * t21 + 0.1100325 * t27 + 0.1241775 * t39))
  t83 = (0.1e1 + 0.278125e-1 * t21) * t82
  t92 = jnp.where(t4, 0, t5 * (-t47 + t58 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t21) * t69 + t47 - 0.19751673498613801407e-1 * t83) + 0.19751673498613801407e-1 * t58 * t83) / 0.2e1)
  t97 = r0 ** 2
  t99 = 0.1e1 / t33 / t97
  t100 = t35 * t99
  t103 = params.gamma_ss * s0 * t100 + 0.1e1
  t108 = params.gamma_ss ** 2
  t110 = s0 ** 2
  t112 = t97 ** 2
  t116 = t15 / t13 / t112 / r0
  t117 = t103 ** 2
  t126 = t112 ** 2
  t128 = t110 * s0 / t126
  t135 = t108 ** 2
  t137 = t110 ** 2
  t142 = t35 / t33 / t126 / t97
  t143 = t117 ** 2
  t155 = 0.1e1 - s0 / r0 / tau0 / 0.8e1
  t156 = tau0 ** 2
  t161 = params.Fermi_D_cnst ** 2
  t166 = jnp.exp(-0.8e1 * t156 * t15 / t13 / t97 / r0 / t161)
  t172 = t9 * t11 * t14
  t175 = jnp.sqrt(t172)
  t178 = t172 ** 0.15e1
  t181 = t31 * t10 * t34
  t187 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t175 + 0.8969 * t172 + 0.204775 * t178 + 0.123235 * t181))
  t190 = jnp.where(t3, t49, 1)
  t204 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t175 + 0.905775 * t172 + 0.1100325 * t178 + 0.1241775 * t181))
  t209 = -0.621814e-1 * (0.1e1 + 0.53425e-1 * t172) * t187 + 0.19751673498613801407e-1 * (0.2e1 * t190 - 0.2e1) * t57 * (0.1e1 + 0.278125e-1 * t172) * t204 - 0.2e1 * t92
  t217 = 0.2e1 * params.gamma_ab * s0 * t100 + 0.1e1
  t223 = params.gamma_ab ** 2
  t226 = t217 ** 2
  t240 = t223 ** 2
  t243 = t226 ** 2
  t252 = s0 * t35 * t99
  t256 = tau0 * t35 / t33 / r0
  t257 = 0.2e1 * t256
  t258 = jnp.cbrt(6)
  t259 = t258 ** 2
  t260 = jnp.pi ** 2
  t261 = jnp.cbrt(t260)
  t262 = t261 ** 2
  t263 = t259 * t262
  t264 = 0.3e1 / 0.5e1 * t263
  t267 = 0.1e1 + params.alpha_ss * (t252 + t257 - t264)
  t274 = t257 - t264
  t277 = t267 ** 2
  t289 = t274 ** 2
  t301 = 0.4e1 * t256
  t302 = 0.6e1 / 0.5e1 * t263
  t305 = 0.1e1 + params.alpha_ab * (0.2e1 * t252 + t301 - t302)
  t313 = t301 - t302
  t316 = t305 ** 2
  t329 = t313 ** 2
  res = 0.2e1 * t92 * (params.css[0] + params.css[1] * params.gamma_ss * s0 * t100 / t103 + 0.2e1 * params.css[2] * t108 * t110 * t116 / t117 + 0.4e1 * params.css[3] * t108 * params.gamma_ss * t128 / t117 / t103 + 0.4e1 * params.css[4] * t135 * t137 * t142 / t143) * t155 * (0.1e1 - t166) + t209 * (params.cab[0] + 0.2e1 * params.cab[1] * params.gamma_ab * s0 * t100 / t217 + 0.8e1 * params.cab[2] * t223 * t110 * t116 / t226 + 0.32e2 * params.cab[3] * t223 * params.gamma_ab * t128 / t226 / t217 + 0.64e2 * params.cab[4] * t240 * t137 * t142 / t243) + 0.2e1 * t92 * (params.dss[0] / t267 + (params.dss[1] * s0 * t100 + params.dss[2] * t274) / t277 + (params.dss[4] * s0 * t100 * t274 + 0.2e1 * params.dss[3] * t110 * t116 + params.dss[5] * t289) / t277 / t267) * t155 + t209 * (params.dab[0] / t305 + (0.2e1 * params.dab[1] * s0 * t100 + params.dab[2] * t313) / t316 + (0.2e1 * params.dab[4] * s0 * t100 * t313 + 0.8e1 * params.dab[3] * t110 * t116 + params.dab[5] * t329) / t316 / t305)
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