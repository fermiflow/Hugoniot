"""Generated from gga_c_q2d.mpl."""

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
  t2 = s0 + 0.2e1 * s1 + s2
  t3 = t2 ** 2
  t4 = r0 + r1
  t5 = t4 ** 2
  t6 = t5 ** 2
  t7 = jnp.cbrt(t4)
  t8 = t7 ** 2
  t10 = 0.1e1 / t8 / t6
  t11 = t3 * t10
  t12 = jnp.cbrt(2)
  t13 = t12 ** 2
  t14 = r0 - r1
  t15 = 0.1e1 / t4
  t16 = t14 * t15
  t17 = 0.1e1 + t16
  t18 = t17 <= p.zeta_threshold
  t19 = jnp.cbrt(p.zeta_threshold)
  t20 = t19 ** 2
  t21 = jnp.cbrt(t17)
  t22 = t21 ** 2
  t23 = jnp.where(t18, t20, t22)
  t24 = 0.1e1 - t16
  t25 = t24 <= p.zeta_threshold
  t26 = jnp.cbrt(t24)
  t27 = t26 ** 2
  t28 = jnp.where(t25, t20, t27)
  t30 = t23 / 0.2e1 + t28 / 0.2e1
  t31 = t30 ** 2
  t32 = t31 ** 2
  t34 = t13 / t32
  t36 = jnp.cbrt(3)
  t37 = 0.1e1 / jnp.pi
  t38 = jnp.cbrt(t37)
  t39 = t38 ** 2
  t40 = 0.1e1 / t39
  t41 = t36 * t40
  t42 = jnp.cbrt(4)
  t43 = t42 ** 2
  t49 = t36 ** 2
  t55 = t2 / t7 / t5 * t12 / t31 * t49 / t38 * t42 / 0.96e2
  t56 = 0.1e1 + t55
  t69 = 0.1e1 / (0.1e7 + t3 * t2 / t6 / t5 / t4 / t32 / t31 * jnp.pi / 0.12288e5)
  t76 = 0.1e1 / t7
  t78 = t36 * t38 * t43 * t76
  t81 = jnp.sqrt(t78)
  t84 = t78 ** 0.15e1
  t89 = t49 * t39 * t42 / t8
  t95 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t81 + 0.8969 * t78 + 0.204775 * t84 + 0.123235 * t89))
  t97 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t78) * t95
  t98 = t14 ** 2
  t99 = t98 ** 2
  t100 = 0.1e1 / t6
  t101 = t99 * t100
  t102 = t19 * p.zeta_threshold
  t104 = jnp.where(t18, t102, t21 * t17)
  t106 = jnp.where(t25, t102, t26 * t24)
  t111 = (t104 + t106 - 0.2e1) / (0.2e1 * t12 - 0.2e1)
  t122 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t81 + 0.1549425e1 * t78 + 0.420775 * t84 + 0.1562925 * t89))
  t135 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t81 + 0.905775 * t78 + 0.1100325 * t84 + 0.1241775 * t89))
  t136 = (0.1e1 + 0.278125e-1 * t78) * t135
  t140 = t101 * t111 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t78) * t122 + t97 - 0.19751673498613801407e-1 * t136)
  t142 = 0.19751673498613801407e-1 * t111 * t136
  t143 = jnp.log(0.2e1)
  t144 = 0.1e1 - t143
  t145 = jnp.pi ** 2
  t148 = t31 * t30
  t149 = 0.1e1 / t144
  t150 = t149 * t145
  t156 = jnp.exp(-(-t97 + t140 + t142) * t149 * t145 / t148)
  t158 = 0.1e1 / (t156 - 0.1e1)
  t166 = t55 + 0.21720231316129303386e-4 * t150 * t158 * t3 * t10 * t34 * t41 * t43
  t177 = jnp.log(0.1e1 + 0.6672455060314922e-1 * t166 * t149 * t145 / (0.1e1 + 0.6672455060314922e-1 * t150 * t158 * t166))
  t186 = jnp.sqrt(0.3e1)
  t188 = jnp.cbrt(6)
  t189 = t188 ** 2
  t190 = jnp.cbrt(t145)
  t191 = 0.1e1 / t190
  t193 = jnp.sqrt(t2)
  t197 = t189 * t191 * t193 / t7 / t4
  t198 = jnp.sqrt(t197)
  t199 = t186 * t76 * t198
  t201 = 0.1e1 / t5
  t204 = t201 * t189 * t191 * t193
  t208 = t186 * t15 * t198 * t197
  t212 = t199 ** 0.15e1
  t219 = jnp.log(0.1e1 + 0.1e1 / (0.2846248 * t199 - 0.31313960595450713442e-2 * t212 + 0.8226186096e-1 * t204 + 0.120051939264e-2 * t208))
  t231 = jnp.log(0.1e1 + 0.1e1 / (0.1173772 * t199 + 0.161747623056e-1 * t204 + 0.535938794688e-4 * t208))
  t245 = jnp.log(0.1e1 + 0.1e1 / (0.404501484 * t199 + 0.79926897828288e-1 * t208))
  t251 = jnp.exp(-0.3801624 * t199)
  t253 = jnp.sqrt(0.2e1)
  t258 = jnp.sqrt(p.zeta_threshold)
  t259 = t258 * p.zeta_threshold
  t260 = jnp.sqrt(t17)
  t262 = jnp.where(t18, t259, t260 * t17)
  t264 = jnp.sqrt(t24)
  t266 = jnp.where(t25, t259, t264 * t24)
  res = (0.1e1 - t11 * t34 * t41 * t43 * t56 * t69 / 0.3072e4) * (-t97 + t140 + t142 + t144 / t145 * t148 * t177) + t11 * t34 * t36 * t40 * t43 * t56 * t69 * (-0.1925 + (0.245130624e-1 * t199 + 0.138498611712e-1 * t204 + 0.2310999830832e-3 * t208) * t219 + (0.117331 + (-0.963896e-2 * t199 - 0.18553259352e-2 * t204 - 0.62882234719537728e-5 * t208) * t231) * t98 * t201 + (0.234188e-1 + (-0.10534412e-1 * t199 + 0.39590320224e-2 * t204 - 0.18717920348611110144e-2 * t208) * t245) * t99 * t100 - 0.15649452269170579029e1 * (t251 - 0.1e1) * t253 * t37 * t186 * t7 / t198 * (t262 / 0.2e1 + t266 / 0.2e1 - 0.1e1 - 0.3e1 / 0.8e1 * t98 * t201 - 0.3e1 / 0.128e3 * t101)) / 0.3072e4
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = s0 ** 2
  t2 = r0 ** 2
  t3 = t2 ** 2
  t4 = jnp.cbrt(r0)
  t5 = t4 ** 2
  t7 = 0.1e1 / t5 / t3
  t8 = t1 * t7
  t9 = jnp.cbrt(2)
  t10 = t9 ** 2
  t11 = 0.1e1 <= p.zeta_threshold
  t12 = jnp.cbrt(p.zeta_threshold)
  t13 = t12 ** 2
  t14 = jnp.where(t11, t13, 1)
  t15 = t14 ** 2
  t16 = t15 ** 2
  t18 = t10 / t16
  t20 = jnp.cbrt(3)
  t21 = 0.1e1 / jnp.pi
  t22 = jnp.cbrt(t21)
  t23 = t22 ** 2
  t24 = 0.1e1 / t23
  t25 = t20 * t24
  t26 = jnp.cbrt(4)
  t27 = t26 ** 2
  t33 = t20 ** 2
  t39 = s0 / t4 / t2 * t9 / t15 * t33 / t22 * t26 / 0.96e2
  t40 = 0.1e1 + t39
  t53 = 0.1e1 / (0.1e7 + t1 * s0 / t3 / t2 / r0 / t16 / t15 * jnp.pi / 0.12288e5)
  t60 = 0.1e1 / t4
  t62 = t20 * t22 * t27 * t60
  t65 = jnp.sqrt(t62)
  t68 = t62 ** 0.15e1
  t73 = t33 * t23 * t26 / t5
  t79 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t65 + 0.8969 * t62 + 0.204775 * t68 + 0.123235 * t73))
  t81 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t62) * t79
  t83 = jnp.where(t11, t12 * p.zeta_threshold, 1)
  t100 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t65 + 0.905775 * t62 + 0.1100325 * t68 + 0.1241775 * t73))
  t103 = 0.19751673498613801407e-1 * (0.2e1 * t83 - 0.2e1) / (0.2e1 * t9 - 0.2e1) * (0.1e1 + 0.278125e-1 * t62) * t100
  t104 = jnp.log(0.2e1)
  t105 = 0.1e1 - t104
  t106 = jnp.pi ** 2
  t109 = t15 * t14
  t110 = 0.1e1 / t105
  t111 = t110 * t106
  t117 = jnp.exp(-(-t81 + t103) * t110 * t106 / t109)
  t119 = 0.1e1 / (t117 - 0.1e1)
  t127 = t39 + 0.21720231316129303386e-4 * t111 * t119 * t1 * t7 * t18 * t25 * t27
  t138 = jnp.log(0.1e1 + 0.6672455060314922e-1 * t127 * t110 * t106 / (0.1e1 + 0.6672455060314922e-1 * t111 * t119 * t127))
  t147 = jnp.sqrt(0.3e1)
  t149 = jnp.cbrt(6)
  t150 = t149 ** 2
  t151 = jnp.cbrt(t106)
  t152 = 0.1e1 / t151
  t154 = jnp.sqrt(s0)
  t158 = t150 * t152 * t154 / t4 / r0
  t159 = jnp.sqrt(t158)
  t160 = t147 * t60 * t159
  t165 = 0.1e1 / t2 * t150 * t152 * t154
  t170 = t147 / r0 * t159 * t158
  t174 = t160 ** 0.15e1
  t181 = jnp.log(0.1e1 + 0.1e1 / (0.2846248 * t160 - 0.31313960595450713442e-2 * t174 + 0.8226186096e-1 * t165 + 0.120051939264e-2 * t170))
  t184 = jnp.exp(-0.3801624 * t160)
  t186 = jnp.sqrt(0.2e1)
  t191 = jnp.sqrt(p.zeta_threshold)
  t193 = jnp.where(t11, t191 * p.zeta_threshold, 1)
  res = (0.1e1 - t8 * t18 * t25 * t27 * t40 * t53 / 0.3072e4) * (-t81 + t103 + t105 / t106 * t109 * t138) + t8 * t18 * t20 * t24 * t27 * t40 * t53 * (-0.1925 + (0.245130624e-1 * t160 + 0.138498611712e-1 * t165 + 0.2310999830832e-3 * t170) * t181 - 0.15649452269170579029e1 * (t184 - 0.1e1) * t186 * t21 * t147 * t4 / t159 * (t193 - 0.1e1)) / 0.3072e4
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