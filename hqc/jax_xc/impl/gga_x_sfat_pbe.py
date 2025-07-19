"""Generated from gga_x_sfat_pbe.mpl."""

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
  t28 = jnp.cbrt(t6)
  t29 = t2 ** 2
  t30 = jnp.pi * t29
  t32 = jnp.cbrt(0.1e1 / jnp.pi)
  t34 = jnp.cbrt(4)
  t35 = 0.1e1 / t32 * t34
  t36 = jnp.cbrt(6)
  t37 = jnp.pi ** 2
  t38 = jnp.cbrt(t37)
  t39 = t38 ** 2
  t41 = t36 / t39
  t42 = r0 ** 2
  t43 = jnp.cbrt(r0)
  t44 = t43 ** 2
  t53 = 0.1804e1 - 0.646416 / (0.804 + 0.91464571985215458336e-2 * t41 * s0 / t44 / t42)
  t57 = jnp.sqrt(t30 * t35 / t53)
  t60 = jnp.cbrt(2)
  t62 = jnp.cbrt(t20 * t6)
  t66 = p.cam_omega / t57 * t60 / t62 / 0.2e1
  t68 = 0.192e1 < t66
  t69 = jnp.where(t68, t66, 0.192e1)
  t70 = t69 ** 2
  t71 = t70 ** 2
  t74 = t71 * t70
  t77 = t71 ** 2
  t80 = t77 * t70
  t83 = t77 * t71
  t86 = t77 * t74
  t89 = t77 ** 2
  t113 = t89 ** 2
  t124 = -0.1e1 / t71 / 0.3e2 + 0.1e1 / t74 / 0.7e2 - 0.1e1 / t77 / 0.135e3 + 0.1e1 / t80 / 0.231e3 - 0.1e1 / t83 / 0.364e3 + 0.1e1 / t86 / 0.54e3 - 0.1e1 / t89 / 0.765e3 + 0.1e1 / t89 / t70 / 0.1045e4 - 0.1e1 / t89 / t71 / 0.1386e4 + 0.1e1 / t89 / t74 / 0.1794e4 - 0.1e1 / t89 / t77 / 0.2275e4 + 0.1e1 / t89 / t80 / 0.2835e4 - 0.1e1 / t89 / t83 / 0.348e4 + 0.1e1 / t89 / t86 / 0.4216e4 - 0.1e1 / t113 / 0.5049e4 + 0.1e1 / t113 / t70 / 0.5985e4 - 0.1e1 / t113 / t71 / 0.703e4 + 0.1e1 / t70 / 0.9e1
  t125 = jnp.where(t68, 0.192e1, t66)
  t126 = jnp.arctan2(0.1e1, t125)
  t127 = t125 ** 2
  t131 = jnp.log(0.1e1 + 0.1e1 / t127)
  t140 = jnp.where(0.192e1 <= t66, t124, 0.1e1 - 0.8e1 / 0.3e1 * t125 * (t126 + t125 * (0.1e1 - (t127 + 0.3e1) * t131) / 0.4e1))
  t145 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * t140 * t53)
  t147 = jnp.where(t10, t15, -t17)
  t148 = jnp.where(t14, t11, t147)
  t149 = 0.1e1 + t148
  t151 = jnp.cbrt(t149)
  t153 = jnp.where(t149 <= p.zeta_threshold, t23, t151 * t149)
  t155 = r1 ** 2
  t156 = jnp.cbrt(r1)
  t157 = t156 ** 2
  t166 = 0.1804e1 - 0.646416 / (0.804 + 0.91464571985215458336e-2 * t41 * s2 / t157 / t155)
  t170 = jnp.sqrt(t30 * t35 / t166)
  t174 = jnp.cbrt(t149 * t6)
  t178 = p.cam_omega / t170 * t60 / t174 / 0.2e1
  t180 = 0.192e1 < t178
  t181 = jnp.where(t180, t178, 0.192e1)
  t182 = t181 ** 2
  t183 = t182 ** 2
  t186 = t183 * t182
  t189 = t183 ** 2
  t192 = t189 * t182
  t195 = t189 * t183
  t198 = t189 * t186
  t201 = t189 ** 2
  t225 = t201 ** 2
  t236 = -0.1e1 / t183 / 0.3e2 + 0.1e1 / t186 / 0.7e2 - 0.1e1 / t189 / 0.135e3 + 0.1e1 / t192 / 0.231e3 - 0.1e1 / t195 / 0.364e3 + 0.1e1 / t198 / 0.54e3 - 0.1e1 / t201 / 0.765e3 + 0.1e1 / t201 / t182 / 0.1045e4 - 0.1e1 / t201 / t183 / 0.1386e4 + 0.1e1 / t201 / t186 / 0.1794e4 - 0.1e1 / t201 / t189 / 0.2275e4 + 0.1e1 / t201 / t192 / 0.2835e4 - 0.1e1 / t201 / t195 / 0.348e4 + 0.1e1 / t201 / t198 / 0.4216e4 - 0.1e1 / t225 / 0.5049e4 + 0.1e1 / t225 / t182 / 0.5985e4 - 0.1e1 / t225 / t183 / 0.703e4 + 0.1e1 / t182 / 0.9e1
  t237 = jnp.where(t180, 0.192e1, t178)
  t238 = jnp.arctan2(0.1e1, t237)
  t239 = t237 ** 2
  t243 = jnp.log(0.1e1 + 0.1e1 / t239)
  t252 = jnp.where(0.192e1 <= t178, t236, 0.1e1 - 0.8e1 / 0.3e1 * t237 * (t238 + t237 * (0.1e1 - (t239 + 0.3e1) * t243) / 0.4e1))
  t257 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t153 * t28 * t252 * t166)
  res = t145 + t257
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
  t14 = jnp.cbrt(p.zeta_threshold)
  t16 = jnp.cbrt(t12)
  t18 = jnp.where(t12 <= p.zeta_threshold, t14 * p.zeta_threshold, t16 * t12)
  t20 = jnp.cbrt(r0)
  t21 = t3 ** 2
  t24 = jnp.cbrt(0.1e1 / jnp.pi)
  t26 = jnp.cbrt(4)
  t28 = jnp.cbrt(6)
  t29 = jnp.pi ** 2
  t30 = jnp.cbrt(t29)
  t31 = t30 ** 2
  t34 = jnp.cbrt(2)
  t35 = t34 ** 2
  t37 = r0 ** 2
  t38 = t20 ** 2
  t47 = 0.1804e1 - 0.646416 / (0.804 + 0.91464571985215458336e-2 * t28 / t31 * s0 * t35 / t38 / t37)
  t51 = jnp.sqrt(jnp.pi * t21 / t24 * t26 / t47)
  t55 = jnp.cbrt(t12 * r0)
  t59 = p.cam_omega / t51 * t34 / t55 / 0.2e1
  t61 = 0.192e1 < t59
  t62 = jnp.where(t61, t59, 0.192e1)
  t63 = t62 ** 2
  t64 = t63 ** 2
  t67 = t64 * t63
  t70 = t64 ** 2
  t73 = t70 * t63
  t76 = t70 * t64
  t79 = t70 * t67
  t82 = t70 ** 2
  t106 = t82 ** 2
  t117 = -0.1e1 / t64 / 0.3e2 + 0.1e1 / t67 / 0.7e2 - 0.1e1 / t70 / 0.135e3 + 0.1e1 / t73 / 0.231e3 - 0.1e1 / t76 / 0.364e3 + 0.1e1 / t79 / 0.54e3 - 0.1e1 / t82 / 0.765e3 + 0.1e1 / t82 / t63 / 0.1045e4 - 0.1e1 / t82 / t64 / 0.1386e4 + 0.1e1 / t82 / t67 / 0.1794e4 - 0.1e1 / t82 / t70 / 0.2275e4 + 0.1e1 / t82 / t73 / 0.2835e4 - 0.1e1 / t82 / t76 / 0.348e4 + 0.1e1 / t82 / t79 / 0.4216e4 - 0.1e1 / t106 / 0.5049e4 + 0.1e1 / t106 / t63 / 0.5985e4 - 0.1e1 / t106 / t64 / 0.703e4 + 0.1e1 / t63 / 0.9e1
  t118 = jnp.where(t61, 0.192e1, t59)
  t119 = jnp.arctan2(0.1e1, t118)
  t120 = t118 ** 2
  t124 = jnp.log(0.1e1 + 0.1e1 / t120)
  t133 = jnp.where(0.192e1 <= t59, t117, 0.1e1 - 0.8e1 / 0.3e1 * t118 * (t119 + t118 * (0.1e1 - (t120 + 0.3e1) * t124) / 0.4e1))
  t138 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * t133 * t47)
  res = 0.2e1 * t138
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