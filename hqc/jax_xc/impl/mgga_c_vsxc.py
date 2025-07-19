"""Generated from mgga_c_vsxc.mpl."""

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
  t99 = params.dss[0]
  t100 = r0 ** 2
  t101 = jnp.cbrt(r0)
  t102 = t101 ** 2
  t104 = 0.1e1 / t102 / t100
  t105 = s0 * t104
  t109 = 0.2e1 * tau0 / t102 / r0
  t110 = jnp.cbrt(6)
  t111 = t110 ** 2
  t112 = jnp.pi ** 2
  t113 = jnp.cbrt(t112)
  t114 = t113 ** 2
  t115 = t111 * t114
  t116 = 0.3e1 / 0.5e1 * t115
  t119 = 0.1e1 + params.alpha_ss * (t105 + t109 - t116)
  t122 = params.dss[1]
  t125 = params.dss[2]
  t126 = t109 - t116
  t129 = t119 ** 2
  t132 = params.dss[3]
  t133 = s0 ** 2
  t135 = t100 ** 2
  t140 = params.dss[4]
  t144 = params.dss[5]
  t145 = t126 ** 2
  t161 = 0.1e1 - t5
  t162 = t161 <= p.zeta_threshold
  t163 = jnp.logical_or(r1 <= p.dens_threshold, t162)
  t164 = jnp.where(t162, p.zeta_threshold, t161)
  t165 = jnp.cbrt(t161)
  t167 = jnp.where(t162, t22, 0.1e1 / t165)
  t169 = t16 * t20 * t167
  t172 = jnp.sqrt(t169)
  t175 = t169 ** 0.15e1
  t177 = t167 ** 2
  t179 = t38 * t42 * t177
  t185 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t172 + 0.8969 * t169 + 0.204775 * t175 + 0.123235 * t179))
  t187 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t169) * t185
  t198 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t172 + 0.1549425e1 * t169 + 0.420775 * t175 + 0.1562925 * t179))
  t211 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t172 + 0.905775 * t169 + 0.1100325 * t175 + 0.1241775 * t179))
  t212 = (0.1e1 + 0.278125e-1 * t169) * t211
  t221 = jnp.where(t163, 0, t164 * (-t187 + t64 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t169) * t198 + t187 - 0.19751673498613801407e-1 * t212) + 0.19751673498613801407e-1 * t64 * t212) / 0.2e1)
  t222 = r1 ** 2
  t223 = jnp.cbrt(r1)
  t224 = t223 ** 2
  t226 = 0.1e1 / t224 / t222
  t227 = s2 * t226
  t231 = 0.2e1 * tau1 / t224 / r1
  t234 = 0.1e1 + params.alpha_ss * (t227 + t231 - t116)
  t239 = t231 - t116
  t242 = t234 ** 2
  t245 = s2 ** 2
  t247 = t222 ** 2
  t255 = t239 ** 2
  t271 = t13 * t15 * t18
  t274 = jnp.sqrt(t271)
  t277 = t271 ** 0.15e1
  t280 = t37 * t14 * t40
  t286 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t274 + 0.8969 * t271 + 0.204775 * t277 + 0.123235 * t280))
  t288 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t271) * t286
  t289 = t2 ** 2
  t290 = t289 ** 2
  t291 = t3 ** 2
  t292 = t291 ** 2
  t296 = jnp.where(t7, t55, t23 * t6)
  t298 = jnp.where(t162, t55, t165 * t161)
  t300 = (t296 + t298 - 0.2e1) * t63
  t311 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t274 + 0.1549425e1 * t271 + 0.420775 * t277 + 0.1562925 * t280))
  t324 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t274 + 0.905775 * t271 + 0.1100325 * t277 + 0.1241775 * t280))
  t325 = (0.1e1 + 0.278125e-1 * t271) * t324
  t334 = 0.6e1 / 0.5e1 * t115
  t337 = 0.1e1 + params.alpha_ab * (t105 + t227 + t109 + t231 - t334)
  t341 = t105 + t227
  t344 = t109 + t231 - t334
  t347 = t337 ** 2
  t351 = t341 ** 2
  t357 = t344 ** 2
  res = t98 * (t99 / t119 + (t122 * s0 * t104 + t125 * t126) / t129 + (t132 * t133 / t101 / t135 / r0 + t140 * s0 * t104 * t126 + t144 * t145) / t129 / t119) * (0.1e1 - s0 / r0 / tau0 / 0.8e1) + t221 * (t99 / t234 + (t122 * s2 * t226 + t125 * t239) / t242 + (t132 * t245 / t223 / t247 / r1 + t140 * s2 * t226 * t239 + t144 * t255) / t242 / t234) * (0.1e1 - s2 / r1 / tau1 / 0.8e1) + (-t288 + t290 / t292 * t300 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t271) * t311 + t288 - 0.19751673498613801407e-1 * t325) + 0.19751673498613801407e-1 * t300 * t325 - t98 - t221) * (params.dab[0] / t337 + (params.dab[1] * t341 + params.dab[2] * t344) / t347 + (params.dab[4] * t341 * t344 + params.dab[3] * t351 + params.dab[5] * t357) / t347 / t337)
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
  t95 = r0 ** 2
  t97 = 0.1e1 / t33 / t95
  t98 = s0 * t35 * t97
  t102 = tau0 * t35 / t33 / r0
  t103 = 0.2e1 * t102
  t104 = jnp.cbrt(6)
  t105 = t104 ** 2
  t106 = jnp.pi ** 2
  t107 = jnp.cbrt(t106)
  t108 = t107 ** 2
  t109 = t105 * t108
  t110 = 0.3e1 / 0.5e1 * t109
  t113 = 0.1e1 + params.alpha_ss * (t98 + t103 - t110)
  t118 = t35 * t97
  t121 = t103 - t110
  t124 = t113 ** 2
  t128 = s0 ** 2
  t130 = t95 ** 2
  t134 = t15 / t13 / t130 / r0
  t142 = t121 ** 2
  t159 = t9 * t11 * t14
  t162 = jnp.sqrt(t159)
  t165 = t159 ** 0.15e1
  t168 = t31 * t10 * t34
  t174 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t162 + 0.8969 * t159 + 0.204775 * t165 + 0.123235 * t168))
  t177 = jnp.where(t3, t49, 1)
  t191 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t162 + 0.905775 * t159 + 0.1100325 * t165 + 0.1241775 * t168))
  t199 = 0.4e1 * t102
  t200 = 0.6e1 / 0.5e1 * t109
  t203 = 0.1e1 + params.alpha_ab * (0.2e1 * t98 + t199 - t200)
  t211 = t199 - t200
  t214 = t203 ** 2
  t227 = t211 ** 2
  res = 0.2e1 * t92 * (params.dss[0] / t113 + (params.dss[1] * s0 * t118 + params.dss[2] * t121) / t124 + (params.dss[4] * s0 * t118 * t121 + 0.2e1 * params.dss[3] * t128 * t134 + params.dss[5] * t142) / t124 / t113) * (0.1e1 - s0 / r0 / tau0 / 0.8e1) + (-0.621814e-1 * (0.1e1 + 0.53425e-1 * t159) * t174 + 0.19751673498613801407e-1 * (0.2e1 * t177 - 0.2e1) * t57 * (0.1e1 + 0.278125e-1 * t159) * t191 - 0.2e1 * t92) * (params.dab[0] / t203 + (0.2e1 * params.dab[1] * s0 * t118 + params.dab[2] * t211) / t214 + (0.2e1 * params.dab[4] * s0 * t118 * t211 + 0.8e1 * params.dab[3] * t128 * t134 + params.dab[5] * t227) / t214 / t203)
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