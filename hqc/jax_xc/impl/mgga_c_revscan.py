"""Generated from mgga_c_revscan.mpl."""

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
  t1 = jnp.cbrt(3)
  t3 = jnp.cbrt(0.1e1 / jnp.pi)
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = r0 + r1
  t8 = jnp.cbrt(t7)
  t11 = t1 * t3 * t6 / t8
  t14 = jnp.sqrt(t11)
  t17 = t11 ** 0.15e1
  t19 = t1 ** 2
  t20 = t3 ** 2
  t22 = t8 ** 2
  t25 = t19 * t20 * t5 / t22
  t31 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t14 + 0.8969 * t11 + 0.204775 * t17 + 0.123235 * t25))
  t33 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t11) * t31
  t34 = r0 - r1
  t35 = t34 ** 2
  t36 = t35 ** 2
  t37 = t7 ** 2
  t38 = t37 ** 2
  t42 = t34 / t7
  t43 = 0.1e1 + t42
  t44 = t43 <= p.zeta_threshold
  t45 = jnp.cbrt(p.zeta_threshold)
  t46 = t45 * p.zeta_threshold
  t47 = jnp.cbrt(t43)
  t49 = jnp.where(t44, t46, t47 * t43)
  t50 = 0.1e1 - t42
  t51 = t50 <= p.zeta_threshold
  t52 = jnp.cbrt(t50)
  t54 = jnp.where(t51, t46, t52 * t50)
  t55 = t49 + t54 - 0.2e1
  t56 = jnp.cbrt(2)
  t57 = t56 - 0.1e1
  t59 = 0.1e1 / t57 / 0.2e1
  t60 = t55 * t59
  t71 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t14 + 0.1549425e1 * t11 + 0.420775 * t17 + 0.1562925 * t25))
  t84 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t14 + 0.905775 * t11 + 0.1100325 * t17 + 0.1241775 * t25))
  t85 = (0.1e1 + 0.278125e-1 * t11) * t84
  t89 = t36 / t38 * t60 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t11) * t71 + t33 - 0.19751673498613801407e-1 * t85)
  t91 = 0.19751673498613801407e-1 * t60 * t85
  t92 = jnp.log(0.2e1)
  t93 = 0.1e1 - t92
  t94 = jnp.pi ** 2
  t97 = t45 ** 2
  t98 = t47 ** 2
  t99 = jnp.where(t44, t97, t98)
  t100 = t52 ** 2
  t101 = jnp.where(t51, t97, t100)
  t103 = t99 / 0.2e1 + t101 / 0.2e1
  t104 = t103 ** 2
  t105 = t104 * t103
  t107 = 0.1e1 + 0.25e-1 * t11
  t109 = 0.1e1 + 0.4445e-1 * t11
  t112 = 0.1e1 / t93
  t119 = jnp.exp(-(-t33 + t89 + t91) * t112 * t94 / t105)
  t120 = t119 - 0.1e1
  t124 = s0 + 0.2e1 * s1 + s2
  t139 = (0.1e1 + 0.55603792169291016666e-2 * t107 / t109 * t112 * t94 / t120 * t124 / t8 / t37 * t56 / t104 * t19 / t3 * t5) ** (0.1e1 / 0.4e1)
  t142 = t107 ** 2
  t143 = t109 ** 2
  t146 = t93 ** 2
  t149 = t94 ** 2
  t150 = t120 ** 2
  t153 = t124 ** 2
  t158 = t56 ** 2
  t160 = t104 ** 2
  t170 = (0.1e1 + 0.11594181388521408694e-3 * t142 / t143 / t146 * t149 / t150 * t153 / t22 / t38 * t158 / t160 * t1 / t20 * t6) ** (0.1e1 / 0.8e1)
  t177 = jnp.log(0.1e1 + 0.1e1 * (0.1e1 - 0.1e1 / t139 / 0.2e1 - 0.1e1 / t170 / 0.2e1) * t120)
  t179 = t93 / t94 * t105 * t177
  t180 = jnp.cbrt(r0)
  t181 = t180 ** 2
  t185 = t43 / 0.2e1
  t186 = jnp.cbrt(t185)
  t187 = t186 ** 2
  t188 = t187 * t185
  t190 = jnp.cbrt(r1)
  t191 = t190 ** 2
  t195 = t50 / 0.2e1
  t196 = jnp.cbrt(t195)
  t197 = t196 ** 2
  t198 = t197 * t195
  t201 = 0.1e1 / t22 / t37
  t205 = jnp.cbrt(6)
  t207 = jnp.cbrt(t94)
  t208 = t207 ** 2
  t209 = 0.1e1 / t208
  t214 = 0.5e1 / 0.9e1 * (tau0 / t181 / r0 * t188 + tau1 / t191 / r1 * t198 - t124 * t201 / 0.8e1) * t205 * t209 / (t188 + t198)
  t216 = jnp.log(2.220446049250313e-16)
  t219 = t216 / (-t216 + 0.1131e1)
  t222 = jnp.where(t214 < -t219, t214, -t219)
  t227 = jnp.exp(-0.1131e1 * t222 / (0.1e1 - t222))
  t228 = jnp.where(-t219 < t214, 0, t227)
  t230 = jnp.log(0.72992700729927007299 * 2.220446049250313e-16)
  t233 = (-t230 + 0.17e1) / t230
  t234 = t214 < -t233
  t235 = jnp.where(t234, -t233, t214)
  t239 = jnp.exp(0.17e1 / (0.1e1 - t235))
  t241 = jnp.where(t234, 0, -0.137e1 * t239)
  t242 = jnp.where(t214 <= 0.1e1, t228, t241)
  t246 = 0.1e1 / (0.1e1 - 0.33115e-1 * t14 + 0.4168e-1 * t11)
  t249 = jnp.exp(0.1e1 * t246)
  t257 = (0.1e1 + 0.21337642104376358333e-1 * t205 * t209 * t158 * t124 * t201) ** (0.1e1 / 0.4e1)
  t260 = t205 ** 2
  t272 = (0.1e1 + 0.45529497057445474566e-2 * t260 / t207 / t94 * t56 * t153 / t8 / t38 / t7) ** (0.1e1 / 0.8e1)
  t277 = jnp.log(0.1e1 + (t249 - 0.1e1) * (0.1e1 - 0.1e1 / t257 / 0.2e1 - t272 / 0.2e1))
  t285 = t36 ** 2
  t287 = t38 ** 2
  res = -t33 + t89 + t91 + t179 + t242 * ((-0.30197e-1 * t246 + 0.30197e-1 * t277) * (0.1e1 - 0.2363e1 * t57 * t55 * t59) * (0.1e1 - t285 * t36 / t287 / t38) + t33 - t89 - t91 - t179)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(3)
  t3 = jnp.cbrt(0.1e1 / jnp.pi)
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = jnp.cbrt(r0)
  t10 = t1 * t3 * t6 / t7
  t13 = jnp.sqrt(t10)
  t16 = t10 ** 0.15e1
  t18 = t1 ** 2
  t19 = t3 ** 2
  t21 = t7 ** 2
  t24 = t18 * t19 * t5 / t21
  t30 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t13 + 0.8969 * t10 + 0.204775 * t16 + 0.123235 * t24))
  t32 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t10) * t30
  t33 = 0.1e1 <= p.zeta_threshold
  t34 = jnp.cbrt(p.zeta_threshold)
  t36 = jnp.where(t33, t34 * p.zeta_threshold, 1)
  t38 = 0.2e1 * t36 - 0.2e1
  t39 = jnp.cbrt(2)
  t40 = t39 - 0.1e1
  t42 = 0.1e1 / t40 / 0.2e1
  t54 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t13 + 0.905775 * t10 + 0.1100325 * t16 + 0.1241775 * t24))
  t57 = 0.19751673498613801407e-1 * t38 * t42 * (0.1e1 + 0.278125e-1 * t10) * t54
  t58 = jnp.log(0.2e1)
  t59 = 0.1e1 - t58
  t60 = jnp.pi ** 2
  t63 = t34 ** 2
  t64 = jnp.where(t33, t63, 1)
  t65 = t64 ** 2
  t66 = t65 * t64
  t68 = 0.1e1 + 0.25e-1 * t10
  t70 = 0.1e1 + 0.4445e-1 * t10
  t73 = 0.1e1 / t59
  t80 = jnp.exp(-(-t32 + t57) * t73 * t60 / t66)
  t81 = t80 - 0.1e1
  t86 = r0 ** 2
  t99 = (0.1e1 + 0.55603792169291016666e-2 * t68 / t70 * t73 * t60 / t81 * s0 / t7 / t86 * t39 / t65 * t18 / t3 * t5) ** (0.1e1 / 0.4e1)
  t102 = t68 ** 2
  t103 = t70 ** 2
  t106 = t59 ** 2
  t109 = t60 ** 2
  t110 = t81 ** 2
  t113 = s0 ** 2
  t116 = t86 ** 2
  t119 = t39 ** 2
  t121 = t65 ** 2
  t131 = (0.1e1 + 0.11594181388521408694e-3 * t102 / t103 / t106 * t109 / t110 * t113 / t21 / t116 * t119 / t121 * t1 / t19 * t6) ** (0.1e1 / 0.8e1)
  t138 = jnp.log(0.1e1 + 0.1e1 * (0.1e1 - 0.1e1 / t99 / 0.2e1 - 0.1e1 / t131 / 0.2e1) * t81)
  t140 = t59 / t60 * t66 * t138
  t145 = 0.1e1 / t21 / t86
  t149 = jnp.cbrt(6)
  t151 = jnp.cbrt(t60)
  t152 = t151 ** 2
  t153 = 0.1e1 / t152
  t156 = 0.5e1 / 0.9e1 * (tau0 / t21 / r0 - s0 * t145 / 0.8e1) * t149 * t153 * t119
  t158 = jnp.log(2.220446049250313e-16)
  t161 = t158 / (-t158 + 0.1131e1)
  t164 = jnp.where(t156 < -t161, t156, -t161)
  t169 = jnp.exp(-0.1131e1 * t164 / (0.1e1 - t164))
  t170 = jnp.where(-t161 < t156, 0, t169)
  t172 = jnp.log(0.72992700729927007299 * 2.220446049250313e-16)
  t175 = (-t172 + 0.17e1) / t172
  t176 = t156 < -t175
  t177 = jnp.where(t176, -t175, t156)
  t181 = jnp.exp(0.17e1 / (0.1e1 - t177))
  t183 = jnp.where(t176, 0, -0.137e1 * t181)
  t184 = jnp.where(t156 <= 0.1e1, t170, t183)
  t188 = 0.1e1 / (0.1e1 - 0.33115e-1 * t13 + 0.4168e-1 * t10)
  t191 = jnp.exp(0.1e1 * t188)
  t199 = (0.1e1 + 0.21337642104376358333e-1 * t149 * t153 * t119 * s0 * t145) ** (0.1e1 / 0.4e1)
  t202 = t149 ** 2
  t214 = (0.1e1 + 0.45529497057445474566e-2 * t202 / t151 / t60 * t39 * t113 / t7 / t116 / r0) ** (0.1e1 / 0.8e1)
  t219 = jnp.log(0.1e1 + (t191 - 0.1e1) * (0.1e1 - 0.1e1 / t199 / 0.2e1 - t214 / 0.2e1))
  res = -t32 + t57 + t140 + t184 * ((-0.30197e-1 * t188 + 0.30197e-1 * t219) * (0.1e1 - 0.2363e1 * t40 * t38 * t42) + t32 - t57 - t140)
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