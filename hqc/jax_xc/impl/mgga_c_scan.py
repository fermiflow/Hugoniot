"""Generated from mgga_c_scan.mpl."""

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
  t112 = 0.1e1 / t93
  t119 = jnp.exp(-(-t33 + t89 + t91) * t112 * t94 / t105)
  t120 = t119 - 0.1e1
  t124 = s0 + 0.2e1 * s1 + s2
  t139 = (0.1e1 + 0.27801896084645508333e-2 * (0.1e1 + 0.25e-1 * t11) / (0.1e1 + 0.4445e-1 * t11) * t112 * t94 / t120 * t124 / t8 / t37 * t56 / t104 * t19 / t3 * t5) ** (0.1e1 / 0.4e1)
  t145 = jnp.log(0.1e1 + 0.1e1 * (0.1e1 - 0.1e1 / t139) * t120)
  t147 = t93 / t94 * t105 * t145
  t148 = jnp.cbrt(r0)
  t149 = t148 ** 2
  t153 = t43 / 0.2e1
  t154 = jnp.cbrt(t153)
  t155 = t154 ** 2
  t156 = t155 * t153
  t158 = jnp.cbrt(r1)
  t159 = t158 ** 2
  t163 = t50 / 0.2e1
  t164 = jnp.cbrt(t163)
  t165 = t164 ** 2
  t166 = t165 * t163
  t169 = 0.1e1 / t22 / t37
  t173 = jnp.cbrt(6)
  t175 = jnp.cbrt(t94)
  t176 = t175 ** 2
  t177 = 0.1e1 / t176
  t182 = 0.5e1 / 0.9e1 * (tau0 / t149 / r0 * t156 + tau1 / t159 / r1 * t166 - t124 * t169 / 0.8e1) * t173 * t177 / (t156 + t166)
  t184 = jnp.log(2.220446049250313e-16)
  t187 = t184 / (-t184 + 0.64)
  t190 = jnp.where(t182 < -t187, t182, -t187)
  t195 = jnp.exp(-0.64 * t190 / (0.1e1 - t190))
  t196 = jnp.where(-t187 < t182, 0, t195)
  t198 = jnp.log(0.14285714285714285714e1 * 2.220446049250313e-16)
  t201 = (-t198 + 0.15e1) / t198
  t202 = t182 < -t201
  t203 = jnp.where(t202, -t201, t182)
  t207 = jnp.exp(0.15e1 / (0.1e1 - t203))
  t209 = jnp.where(t202, 0, -0.7 * t207)
  t210 = jnp.where(t182 <= 0.1e1, t196, t209)
  t214 = 0.1e1 / (0.1e1 + 0.4445e-1 * t14 + 0.3138525e-1 * t11)
  t217 = jnp.exp(0.1e1 * t214)
  t220 = t56 ** 2
  t226 = (0.1e1 + 0.21337642104376358333e-1 * t173 * t177 * t220 * t124 * t169) ** (0.1e1 / 0.4e1)
  t231 = jnp.log(0.1e1 + (t217 - 0.1e1) * (0.1e1 - 0.1e1 / t226))
  t239 = t36 ** 2
  t241 = t38 ** 2
  res = -t33 + t89 + t91 + t147 + t210 * ((-0.285764e-1 * t214 + 0.285764e-1 * t231) * (0.1e1 - 0.2363e1 * t57 * t55 * t59) * (0.1e1 - t239 * t36 / t241 / t38) + t33 - t89 - t91 - t147)
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
  t73 = 0.1e1 / t59
  t80 = jnp.exp(-(-t32 + t57) * t73 * t60 / t66)
  t81 = t80 - 0.1e1
  t86 = r0 ** 2
  t99 = (0.1e1 + 0.27801896084645508333e-2 * (0.1e1 + 0.25e-1 * t10) / (0.1e1 + 0.4445e-1 * t10) * t73 * t60 / t81 * s0 / t7 / t86 * t39 / t65 * t18 / t3 * t5) ** (0.1e1 / 0.4e1)
  t105 = jnp.log(0.1e1 + 0.1e1 * (0.1e1 - 0.1e1 / t99) * t81)
  t107 = t59 / t60 * t66 * t105
  t112 = 0.1e1 / t21 / t86
  t116 = jnp.cbrt(6)
  t118 = jnp.cbrt(t60)
  t119 = t118 ** 2
  t120 = 0.1e1 / t119
  t121 = t39 ** 2
  t124 = 0.5e1 / 0.9e1 * (tau0 / t21 / r0 - s0 * t112 / 0.8e1) * t116 * t120 * t121
  t126 = jnp.log(2.220446049250313e-16)
  t129 = t126 / (-t126 + 0.64)
  t132 = jnp.where(t124 < -t129, t124, -t129)
  t137 = jnp.exp(-0.64 * t132 / (0.1e1 - t132))
  t138 = jnp.where(-t129 < t124, 0, t137)
  t140 = jnp.log(0.14285714285714285714e1 * 2.220446049250313e-16)
  t143 = (-t140 + 0.15e1) / t140
  t144 = t124 < -t143
  t145 = jnp.where(t144, -t143, t124)
  t149 = jnp.exp(0.15e1 / (0.1e1 - t145))
  t151 = jnp.where(t144, 0, -0.7 * t149)
  t152 = jnp.where(t124 <= 0.1e1, t138, t151)
  t156 = 0.1e1 / (0.1e1 + 0.4445e-1 * t13 + 0.3138525e-1 * t10)
  t159 = jnp.exp(0.1e1 * t156)
  t167 = (0.1e1 + 0.21337642104376358333e-1 * t116 * t120 * t121 * s0 * t112) ** (0.1e1 / 0.4e1)
  t172 = jnp.log(0.1e1 + (t159 - 0.1e1) * (0.1e1 - 0.1e1 / t167))
  res = -t32 + t57 + t107 + t152 * ((-0.285764e-1 * t156 + 0.285764e-1 * t172) * (0.1e1 - 0.2363e1 * t40 * t38 * t42) + t32 - t57 - t107)
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