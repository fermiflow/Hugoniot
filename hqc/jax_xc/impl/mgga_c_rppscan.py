"""Generated from mgga_c_rppscan.mpl."""

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
  t174 = t173 ** 2
  t175 = jnp.cbrt(t94)
  t176 = t175 ** 2
  t186 = (tau0 / t149 / r0 * t156 + tau1 / t159 / r1 * t166 - t124 * t169 / 0.8e1) / (0.3e1 / 0.1e2 * t174 * t176 * (t156 + t166) + params.eta * t124 * t169 / 0.8e1)
  t188 = 0.25e1 < t186
  t189 = jnp.where(t188, 0.25e1, t186)
  t191 = t189 ** 2
  t193 = t191 * t189
  t195 = t191 ** 2
  t204 = jnp.where(t188, t186, 0.25e1)
  t208 = jnp.exp(0.15e1 / (0.1e1 - t204))
  t210 = jnp.where(t186 <= 0.25e1, 0.1e1 - 0.64 * t189 - 0.4352 * t191 - 0.1535685604549e1 * t193 + 0.3061560252175e1 * t195 - 0.1915710236206e1 * t195 * t189 + 0.516884468372 * t195 * t191 - 0.51848879792e-1 * t195 * t193, -0.7 * t208)
  t214 = 0.1e1 / (0.1e1 + 0.4445e-1 * t14 + 0.3138525e-1 * t11)
  t217 = jnp.exp(0.1e1 * t214)
  t221 = t56 ** 2
  t227 = (0.1e1 + 0.21337642104376358333e-1 * t173 / t176 * t221 * t124 * t169) ** (0.1e1 / 0.4e1)
  t232 = jnp.log(0.1e1 + (t217 - 0.1e1) * (0.1e1 - 0.1e1 / t227))
  t240 = t36 ** 2
  t242 = t38 ** 2
  res = -t33 + t89 + t91 + t147 + t210 * ((-0.285764e-1 * t214 + 0.285764e-1 * t232) * (0.1e1 - 0.2363e1 * t57 * t55 * t59) * (0.1e1 - t240 * t36 / t242 / t38) + t33 - t89 - t91 - t147)
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
  t117 = t116 ** 2
  t118 = jnp.cbrt(t60)
  t119 = t118 ** 2
  t128 = (tau0 / t21 / r0 - s0 * t112 / 0.8e1) / (0.3e1 / 0.2e2 * t117 * t119 * t39 + params.eta * s0 * t112 / 0.8e1)
  t130 = 0.25e1 < t128
  t131 = jnp.where(t130, 0.25e1, t128)
  t133 = t131 ** 2
  t135 = t133 * t131
  t137 = t133 ** 2
  t146 = jnp.where(t130, t128, 0.25e1)
  t150 = jnp.exp(0.15e1 / (0.1e1 - t146))
  t152 = jnp.where(t128 <= 0.25e1, 0.1e1 - 0.64 * t131 - 0.4352 * t133 - 0.1535685604549e1 * t135 + 0.3061560252175e1 * t137 - 0.1915710236206e1 * t137 * t131 + 0.516884468372 * t137 * t133 - 0.51848879792e-1 * t137 * t135, -0.7 * t150)
  t156 = 0.1e1 / (0.1e1 + 0.4445e-1 * t13 + 0.3138525e-1 * t10)
  t159 = jnp.exp(0.1e1 * t156)
  t163 = t39 ** 2
  t169 = (0.1e1 + 0.21337642104376358333e-1 * t116 / t119 * t163 * s0 * t112) ** (0.1e1 / 0.4e1)
  t174 = jnp.log(0.1e1 + (t159 - 0.1e1) * (0.1e1 - 0.1e1 / t169))
  res = -t32 + t57 + t107 + t152 * ((-0.285764e-1 * t156 + 0.285764e-1 * t174) * (0.1e1 - 0.2363e1 * t40 * t38 * t42) + t32 - t57 - t107)
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