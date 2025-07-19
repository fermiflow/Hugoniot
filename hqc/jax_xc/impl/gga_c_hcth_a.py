"""Generated from gga_c_hcth_a.mpl."""

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
  t28 = t27 / 0.4e1
  t29 = jnp.sqrt(t27)
  t32 = 0.1e1 / (t28 + 0.186372e1 * t29 + 0.129352e2)
  t37 = jnp.log(t16 * t20 * t25 * t32 / 0.4e1)
  t38 = 0.310907e-1 * t37
  t42 = jnp.arctan(0.61519908197590802322e1 / (t29 + 0.372744e1))
  t43 = 0.38783294878113014393e-1 * t42
  t44 = t29 / 0.2e1
  t46 = (t44 + 0.10498) ** 2
  t48 = jnp.log(t46 * t32)
  t49 = 0.96902277115443742139e-3 * t48
  t52 = 0.1e1 / (t28 + 0.353021e1 * t29 + 0.180578e2)
  t57 = jnp.log(t16 * t20 * t25 * t52 / 0.4e1)
  t62 = jnp.arctan(0.473092690956011283e1 / (t29 + 0.706042e1))
  t65 = (t44 + 0.325) ** 2
  t67 = jnp.log(t65 * t52)
  t71 = t21 * p.zeta_threshold
  t73 = jnp.where(0.2e1 <= p.zeta_threshold, t71, 0.2e1 * t19)
  t75 = jnp.where(0. <= p.zeta_threshold, t71, 0)
  t76 = t73 + t75 - 0.2e1
  t78 = t19 - 0.1e1
  t80 = 0.1e1 / t78 / 0.2e1
  t85 = jnp.where(t8, 0, t9 * (t38 + t43 + t49 + (0.1554535e-1 * t57 + 0.52491393169780936218e-1 * t62 + 0.22478670955426118383e-2 * t67 - t38 - t43 - t49) * t76 * t80) / 0.2e1)
  t86 = r0 ** 2
  t87 = jnp.cbrt(r0)
  t88 = t87 ** 2
  t91 = s0 / t88 / t86
  t93 = 0.1e1 + 0.2 * t91
  t97 = s0 ** 2
  t98 = t86 ** 2
  t103 = t93 ** 2
  t108 = t98 ** 2
  t118 = 0.1e1 - t5
  t119 = t118 <= p.zeta_threshold
  t120 = jnp.logical_or(r1 <= p.dens_threshold, t119)
  t121 = jnp.where(t119, p.zeta_threshold, t118)
  t122 = jnp.cbrt(t118)
  t124 = jnp.where(t119, t22, 0.1e1 / t122)
  t126 = t16 * t20 * t124
  t127 = t126 / 0.4e1
  t128 = jnp.sqrt(t126)
  t131 = 0.1e1 / (t127 + 0.186372e1 * t128 + 0.129352e2)
  t136 = jnp.log(t16 * t20 * t124 * t131 / 0.4e1)
  t137 = 0.310907e-1 * t136
  t141 = jnp.arctan(0.61519908197590802322e1 / (t128 + 0.372744e1))
  t142 = 0.38783294878113014393e-1 * t141
  t143 = t128 / 0.2e1
  t145 = (t143 + 0.10498) ** 2
  t147 = jnp.log(t145 * t131)
  t148 = 0.96902277115443742139e-3 * t147
  t151 = 0.1e1 / (t127 + 0.353021e1 * t128 + 0.180578e2)
  t156 = jnp.log(t16 * t20 * t124 * t151 / 0.4e1)
  t161 = jnp.arctan(0.473092690956011283e1 / (t128 + 0.706042e1))
  t164 = (t143 + 0.325) ** 2
  t166 = jnp.log(t164 * t151)
  t174 = jnp.where(t120, 0, t121 * (t137 + t142 + t148 + (0.1554535e-1 * t156 + 0.52491393169780936218e-1 * t161 + 0.22478670955426118383e-2 * t166 - t137 - t142 - t148) * t76 * t80) / 0.2e1)
  t175 = r1 ** 2
  t176 = jnp.cbrt(r1)
  t177 = t176 ** 2
  t180 = s2 / t177 / t175
  t182 = 0.1e1 + 0.2 * t180
  t186 = s2 ** 2
  t187 = t175 ** 2
  t192 = t182 ** 2
  t197 = t187 ** 2
  t206 = t15 * t18
  t207 = t13 * t206
  t208 = t207 / 0.4e1
  t209 = jnp.sqrt(t207)
  t212 = 0.1e1 / (t208 + 0.186372e1 * t209 + 0.129352e2)
  t216 = jnp.log(t13 * t206 * t212 / 0.4e1)
  t217 = 0.310907e-1 * t216
  t221 = jnp.arctan(0.61519908197590802322e1 / (t209 + 0.372744e1))
  t222 = 0.38783294878113014393e-1 * t221
  t223 = t209 / 0.2e1
  t225 = (t223 + 0.10498) ** 2
  t227 = jnp.log(t225 * t212)
  t228 = 0.96902277115443742139e-3 * t227
  t229 = jnp.pi ** 2
  t233 = 0.1e1 / (t208 + 0.565535 * t209 + 0.130045e2)
  t237 = jnp.log(t13 * t206 * t233 / 0.4e1)
  t241 = jnp.arctan(0.71231089178181179908e1 / (t209 + 0.113107e1))
  t244 = (t223 + 0.47584e-2) ** 2
  t246 = jnp.log(t244 * t233)
  t251 = jnp.where(t7, t71, t23 * t6)
  t253 = jnp.where(t119, t71, t122 * t118)
  t254 = t251 + t253 - 0.2e1
  t256 = t2 ** 2
  t257 = t256 ** 2
  t258 = t3 ** 2
  t259 = t258 ** 2
  t260 = 0.1e1 / t259
  t270 = 0.1e1 / (t208 + 0.353021e1 * t209 + 0.180578e2)
  t274 = jnp.log(t13 * t206 * t270 / 0.4e1)
  t279 = jnp.arctan(0.473092690956011283e1 / (t209 + 0.706042e1))
  t282 = (t223 + 0.325) ** 2
  t284 = jnp.log(t282 * t270)
  t292 = t91 + t180
  t295 = 0.1e1 + 0.3e-2 * t91 + 0.3e-2 * t180
  t299 = t292 ** 2
  t300 = t295 ** 2
  res = t85 * (0.136823e-1 + 0.53784e-1 * t91 / t93 - 0.2203076e-1 * t97 / t87 / t98 / r0 / t103 + 0.831576e-2 * t97 * s0 / t108 / t103 / t93) + t174 * (0.136823e-1 + 0.53784e-1 * t180 / t182 - 0.2203076e-1 * t186 / t176 / t187 / r1 / t192 + 0.831576e-2 * t186 * s2 / t197 / t192 / t182) + (t217 + t222 + t228 - 0.3e1 / 0.8e1 / t229 * (t237 + 0.317708004743941464 * t241 + 0.41403379428206274608e-3 * t246) * t254 * t80 * (-t257 * t260 + 0.1e1) * t78 + (0.1554535e-1 * t274 + 0.52491393169780936218e-1 * t279 + 0.22478670955426118383e-2 * t284 - t217 - t222 - t228) * t254 * t80 * t257 * t260 - t85 - t174) * (0.836897 + 0.516153e-2 * t292 / t295 - 0.2506482e-4 * t299 / t300 - 0.12352608e-6 * t299 * t292 / t300 / t295)
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
  t12 = t9 * t11
  t13 = jnp.cbrt(r0)
  t14 = 0.1e1 / t13
  t15 = jnp.cbrt(2)
  t16 = t14 * t15
  t17 = jnp.cbrt(p.zeta_threshold)
  t19 = jnp.where(t3, 0.1e1 / t17, 1)
  t21 = t12 * t16 * t19
  t22 = t21 / 0.4e1
  t23 = jnp.sqrt(t21)
  t26 = 0.1e1 / (t22 + 0.186372e1 * t23 + 0.129352e2)
  t31 = jnp.log(t12 * t16 * t19 * t26 / 0.4e1)
  t32 = 0.310907e-1 * t31
  t36 = jnp.arctan(0.61519908197590802322e1 / (t23 + 0.372744e1))
  t37 = 0.38783294878113014393e-1 * t36
  t38 = t23 / 0.2e1
  t40 = (t38 + 0.10498) ** 2
  t42 = jnp.log(t40 * t26)
  t43 = 0.96902277115443742139e-3 * t42
  t46 = 0.1e1 / (t22 + 0.353021e1 * t23 + 0.180578e2)
  t51 = jnp.log(t12 * t16 * t19 * t46 / 0.4e1)
  t56 = jnp.arctan(0.473092690956011283e1 / (t23 + 0.706042e1))
  t59 = (t38 + 0.325) ** 2
  t61 = jnp.log(t59 * t46)
  t65 = t17 * p.zeta_threshold
  t67 = jnp.where(0.2e1 <= p.zeta_threshold, t65, 0.2e1 * t15)
  t69 = jnp.where(0. <= p.zeta_threshold, t65, 0)
  t72 = t15 - 0.1e1
  t74 = 0.1e1 / t72 / 0.2e1
  t79 = jnp.where(t4, 0, t5 * (t32 + t37 + t43 + (0.1554535e-1 * t51 + 0.52491393169780936218e-1 * t56 + 0.22478670955426118383e-2 * t61 - t32 - t37 - t43) * (t67 + t69 - 0.2e1) * t74) / 0.2e1)
  t80 = t15 ** 2
  t81 = s0 * t80
  t82 = r0 ** 2
  t83 = t13 ** 2
  t85 = 0.1e1 / t83 / t82
  t86 = t81 * t85
  t88 = 0.1e1 + 0.2 * t86
  t93 = s0 ** 2
  t94 = t93 * t15
  t95 = t82 ** 2
  t98 = 0.1e1 / t13 / t95 / r0
  t99 = t88 ** 2
  t105 = t95 ** 2
  t107 = t93 * s0 / t105
  t115 = t11 * t14
  t116 = t9 * t115
  t117 = t116 / 0.4e1
  t118 = jnp.sqrt(t116)
  t121 = 0.1e1 / (t117 + 0.186372e1 * t118 + 0.129352e2)
  t125 = jnp.log(t9 * t115 * t121 / 0.4e1)
  t130 = jnp.arctan(0.61519908197590802322e1 / (t118 + 0.372744e1))
  t132 = t118 / 0.2e1
  t134 = (t132 + 0.10498) ** 2
  t136 = jnp.log(t134 * t121)
  t138 = jnp.pi ** 2
  t142 = 0.1e1 / (t117 + 0.565535 * t118 + 0.130045e2)
  t146 = jnp.log(t9 * t115 * t142 / 0.4e1)
  t150 = jnp.arctan(0.71231089178181179908e1 / (t118 + 0.113107e1))
  t153 = (t132 + 0.47584e-2) ** 2
  t155 = jnp.log(t153 * t142)
  t159 = jnp.where(t3, t65, 1)
  t170 = 0.1e1 + 0.6e-2 * t86
  t175 = t170 ** 2
  res = 0.2e1 * t79 * (0.136823e-1 + 0.53784e-1 * t81 * t85 / t88 - 0.4406152e-1 * t94 * t98 / t99 + 0.3326304e-1 * t107 / t99 / t88) + (0.310907e-1 * t125 + 0.38783294878113014393e-1 * t130 + 0.96902277115443742139e-3 * t136 - 0.3e1 / 0.8e1 / t138 * (t146 + 0.317708004743941464 * t150 + 0.41403379428206274608e-3 * t155) * (0.2e1 * t159 - 0.2e1) * t74 * t72 - 0.2e1 * t79) * (0.836897 + 0.1032306e-1 * t81 * t85 / t170 - 0.20051856e-3 * t94 * t98 / t175 - 0.395283456e-5 * t107 / t175 / t170)
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