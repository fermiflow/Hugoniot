"""Generated from mgga_c_rscan.mpl."""

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
  t149 = jnp.cbrt(r0)
  t150 = t149 ** 2
  t154 = t43 / 0.2e1
  t155 = jnp.cbrt(t154)
  t156 = t155 ** 2
  t157 = t156 * t154
  t159 = jnp.cbrt(r1)
  t160 = t159 ** 2
  t164 = t50 / 0.2e1
  t165 = jnp.cbrt(t164)
  t166 = t165 ** 2
  t167 = t166 * t164
  t170 = 0.1e1 / t22 / t37
  t173 = tau0 / t150 / r0 * t157 + tau1 / t160 / r1 * t167 - t124 * t170 / 0.8e1
  t175 = jnp.where(0. < t173, t173, 0)
  t176 = t175 ** 2
  t179 = jnp.cbrt(6)
  t180 = t179 ** 2
  t181 = jnp.cbrt(t94)
  t182 = t181 ** 2
  t187 = t56 ** 2
  t189 = 0.3e1 / 0.1e2 * t180 * t182 * t22 * t7 + 0.1e-3 * t187
  t190 = t189 ** 2
  t193 = t157 + t167
  t194 = t193 ** 2
  t208 = t38 * t7 * t176 * t175 / t190 / t189 / t194 / t193 / (t8 * t37 * t7 * t176 / t190 / t194 + 0.1e-2)
  t210 = 0.25e1 < t208
  t211 = jnp.where(t210, 0.25e1, t208)
  t213 = t211 ** 2
  t215 = t213 * t211
  t217 = t213 ** 2
  t226 = jnp.where(t210, t208, 0.25e1)
  t230 = jnp.exp(0.15e1 / (0.1e1 - t226))
  t232 = jnp.where(t208 <= 0.25e1, 0.1e1 - 0.64 * t211 - 0.4352 * t213 - 0.1535685604549e1 * t215 + 0.3061560252175e1 * t217 - 0.1915710236206e1 * t217 * t211 + 0.516884468372 * t217 * t213 - 0.51848879792e-1 * t217 * t215, -0.7 * t230)
  t236 = 0.1e1 / (0.1e1 + 0.4445e-1 * t14 + 0.3138525e-1 * t11)
  t239 = jnp.exp(0.1e1 * t236)
  t248 = (0.1e1 + 0.21337642104376358333e-1 * t179 / t182 * t187 * t124 * t170) ** (0.1e1 / 0.4e1)
  t253 = jnp.log(0.1e1 + (t239 - 0.1e1) * (0.1e1 - 0.1e1 / t248))
  t261 = t36 ** 2
  t263 = t38 ** 2
  res = -t33 + t89 + t91 + t147 + t232 * ((-0.285764e-1 * t236 + 0.285764e-1 * t253) * (0.1e1 - 0.2363e1 * t57 * t55 * t59) * (0.1e1 - t261 * t36 / t263 / t38) + t33 - t89 - t91 - t147)
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
  t108 = t86 ** 2
  t110 = t21 * r0
  t114 = 0.1e1 / t21 / t86
  t117 = tau0 / t110 - s0 * t114 / 0.8e1
  t119 = jnp.where(0. < t117, t117, 0)
  t120 = t119 ** 2
  t123 = jnp.cbrt(6)
  t124 = t123 ** 2
  t125 = jnp.cbrt(t60)
  t126 = t125 ** 2
  t130 = t39 ** 2
  t132 = 0.3e1 / 0.1e2 * t124 * t126 * t110 + 0.1e-3 * t130
  t133 = t132 ** 2
  t147 = 0.4e1 * t108 * r0 * t120 * t119 / t133 / t132 / (0.2e1 * t7 * t86 * r0 * t120 / t133 * t39 + 0.1e-2)
  t149 = 0.25e1 < t147
  t150 = jnp.where(t149, 0.25e1, t147)
  t152 = t150 ** 2
  t154 = t152 * t150
  t156 = t152 ** 2
  t165 = jnp.where(t149, t147, 0.25e1)
  t169 = jnp.exp(0.15e1 / (0.1e1 - t165))
  t171 = jnp.where(t147 <= 0.25e1, 0.1e1 - 0.64 * t150 - 0.4352 * t152 - 0.1535685604549e1 * t154 + 0.3061560252175e1 * t156 - 0.1915710236206e1 * t156 * t150 + 0.516884468372 * t156 * t152 - 0.51848879792e-1 * t156 * t154, -0.7 * t169)
  t175 = 0.1e1 / (0.1e1 + 0.4445e-1 * t13 + 0.3138525e-1 * t10)
  t178 = jnp.exp(0.1e1 * t175)
  t187 = (0.1e1 + 0.21337642104376358333e-1 * t123 / t126 * t130 * s0 * t114) ** (0.1e1 / 0.4e1)
  t192 = jnp.log(0.1e1 + (t178 - 0.1e1) * (0.1e1 - 0.1e1 / t187))
  res = -t32 + t57 + t107 + t171 * ((-0.285764e-1 * t175 + 0.285764e-1 * t192) * (0.1e1 - 0.2363e1 * t40 * t38 * t42) + t32 - t57 - t107)
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