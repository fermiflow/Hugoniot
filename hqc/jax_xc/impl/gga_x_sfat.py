"""Generated from gga_x_sfat.mpl."""

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
  t33 = 0.1e1 / t32
  t34 = jnp.cbrt(4)
  t35 = t33 * t34
  t37 = t29 * t33 * t34
  t38 = r0 ** 2
  t39 = jnp.cbrt(r0)
  t40 = t39 ** 2
  t44 = jnp.sqrt(s0)
  t47 = t44 / t39 / r0
  t48 = jnp.arcsinh(t47)
  t56 = 0.1e1 + 0.93333333333333333332e-3 * t37 * s0 / t40 / t38 / (0.1e1 + 0.252e-1 * t47 * t48)
  t60 = jnp.sqrt(t30 * t35 / t56)
  t63 = jnp.cbrt(2)
  t65 = jnp.cbrt(t20 * t6)
  t69 = p.cam_omega / t60 * t63 / t65 / 0.2e1
  t71 = 0.192e1 < t69
  t72 = jnp.where(t71, t69, 0.192e1)
  t73 = t72 ** 2
  t74 = t73 ** 2
  t77 = t74 * t73
  t80 = t74 ** 2
  t83 = t80 * t73
  t86 = t80 * t74
  t89 = t80 * t77
  t92 = t80 ** 2
  t116 = t92 ** 2
  t127 = -0.1e1 / t74 / 0.3e2 + 0.1e1 / t77 / 0.7e2 - 0.1e1 / t80 / 0.135e3 + 0.1e1 / t83 / 0.231e3 - 0.1e1 / t86 / 0.364e3 + 0.1e1 / t89 / 0.54e3 - 0.1e1 / t92 / 0.765e3 + 0.1e1 / t92 / t73 / 0.1045e4 - 0.1e1 / t92 / t74 / 0.1386e4 + 0.1e1 / t92 / t77 / 0.1794e4 - 0.1e1 / t92 / t80 / 0.2275e4 + 0.1e1 / t92 / t83 / 0.2835e4 - 0.1e1 / t92 / t86 / 0.348e4 + 0.1e1 / t92 / t89 / 0.4216e4 - 0.1e1 / t116 / 0.5049e4 + 0.1e1 / t116 / t73 / 0.5985e4 - 0.1e1 / t116 / t74 / 0.703e4 + 0.1e1 / t73 / 0.9e1
  t128 = jnp.where(t71, 0.192e1, t69)
  t129 = jnp.arctan2(0.1e1, t128)
  t130 = t128 ** 2
  t134 = jnp.log(0.1e1 + 0.1e1 / t130)
  t143 = jnp.where(0.192e1 <= t69, t127, 0.1e1 - 0.8e1 / 0.3e1 * t128 * (t129 + t128 * (0.1e1 - (t130 + 0.3e1) * t134) / 0.4e1))
  t148 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * t143 * t56)
  t150 = jnp.where(t10, t15, -t17)
  t151 = jnp.where(t14, t11, t150)
  t152 = 0.1e1 + t151
  t154 = jnp.cbrt(t152)
  t156 = jnp.where(t152 <= p.zeta_threshold, t23, t154 * t152)
  t158 = r1 ** 2
  t159 = jnp.cbrt(r1)
  t160 = t159 ** 2
  t164 = jnp.sqrt(s2)
  t167 = t164 / t159 / r1
  t168 = jnp.arcsinh(t167)
  t176 = 0.1e1 + 0.93333333333333333332e-3 * t37 * s2 / t160 / t158 / (0.1e1 + 0.252e-1 * t167 * t168)
  t180 = jnp.sqrt(t30 * t35 / t176)
  t184 = jnp.cbrt(t152 * t6)
  t188 = p.cam_omega / t180 * t63 / t184 / 0.2e1
  t190 = 0.192e1 < t188
  t191 = jnp.where(t190, t188, 0.192e1)
  t192 = t191 ** 2
  t193 = t192 ** 2
  t196 = t193 * t192
  t199 = t193 ** 2
  t202 = t199 * t192
  t205 = t199 * t193
  t208 = t199 * t196
  t211 = t199 ** 2
  t235 = t211 ** 2
  t246 = -0.1e1 / t193 / 0.3e2 + 0.1e1 / t196 / 0.7e2 - 0.1e1 / t199 / 0.135e3 + 0.1e1 / t202 / 0.231e3 - 0.1e1 / t205 / 0.364e3 + 0.1e1 / t208 / 0.54e3 - 0.1e1 / t211 / 0.765e3 + 0.1e1 / t211 / t192 / 0.1045e4 - 0.1e1 / t211 / t193 / 0.1386e4 + 0.1e1 / t211 / t196 / 0.1794e4 - 0.1e1 / t211 / t199 / 0.2275e4 + 0.1e1 / t211 / t202 / 0.2835e4 - 0.1e1 / t211 / t205 / 0.348e4 + 0.1e1 / t211 / t208 / 0.4216e4 - 0.1e1 / t235 / 0.5049e4 + 0.1e1 / t235 / t192 / 0.5985e4 - 0.1e1 / t235 / t193 / 0.703e4 + 0.1e1 / t192 / 0.9e1
  t247 = jnp.where(t190, 0.192e1, t188)
  t248 = jnp.arctan2(0.1e1, t247)
  t249 = t247 ** 2
  t253 = jnp.log(0.1e1 + 0.1e1 / t249)
  t262 = jnp.where(0.192e1 <= t188, t246, 0.1e1 - 0.8e1 / 0.3e1 * t247 * (t248 + t247 * (0.1e1 - (t249 + 0.3e1) * t253) / 0.4e1))
  t267 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t156 * t28 * t262 * t176)
  res = t148 + t267
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
  t25 = 0.1e1 / t24
  t26 = jnp.cbrt(4)
  t30 = jnp.cbrt(2)
  t31 = t30 ** 2
  t33 = r0 ** 2
  t34 = t20 ** 2
  t37 = jnp.sqrt(s0)
  t38 = t37 * t30
  t40 = 0.1e1 / t20 / r0
  t42 = jnp.arcsinh(t38 * t40)
  t52 = 0.1e1 + 0.93333333333333333332e-3 * t21 * t25 * t26 * s0 * t31 / t34 / t33 / (0.1e1 + 0.252e-1 * t38 * t40 * t42)
  t56 = jnp.sqrt(jnp.pi * t21 * t25 * t26 / t52)
  t60 = jnp.cbrt(t12 * r0)
  t64 = p.cam_omega / t56 * t30 / t60 / 0.2e1
  t66 = 0.192e1 < t64
  t67 = jnp.where(t66, t64, 0.192e1)
  t68 = t67 ** 2
  t69 = t68 ** 2
  t72 = t69 * t68
  t75 = t69 ** 2
  t78 = t75 * t68
  t81 = t75 * t69
  t84 = t75 * t72
  t87 = t75 ** 2
  t111 = t87 ** 2
  t122 = -0.1e1 / t69 / 0.3e2 + 0.1e1 / t72 / 0.7e2 - 0.1e1 / t75 / 0.135e3 + 0.1e1 / t78 / 0.231e3 - 0.1e1 / t81 / 0.364e3 + 0.1e1 / t84 / 0.54e3 - 0.1e1 / t87 / 0.765e3 + 0.1e1 / t87 / t68 / 0.1045e4 - 0.1e1 / t87 / t69 / 0.1386e4 + 0.1e1 / t87 / t72 / 0.1794e4 - 0.1e1 / t87 / t75 / 0.2275e4 + 0.1e1 / t87 / t78 / 0.2835e4 - 0.1e1 / t87 / t81 / 0.348e4 + 0.1e1 / t87 / t84 / 0.4216e4 - 0.1e1 / t111 / 0.5049e4 + 0.1e1 / t111 / t68 / 0.5985e4 - 0.1e1 / t111 / t69 / 0.703e4 + 0.1e1 / t68 / 0.9e1
  t123 = jnp.where(t66, 0.192e1, t64)
  t124 = jnp.arctan2(0.1e1, t123)
  t125 = t123 ** 2
  t129 = jnp.log(0.1e1 + 0.1e1 / t125)
  t138 = jnp.where(0.192e1 <= t64, t122, 0.1e1 - 0.8e1 / 0.3e1 * t123 * (t124 + t123 * (0.1e1 - (t125 + 0.3e1) * t129) / 0.4e1))
  t143 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * t138 * t52)
  res = 0.2e1 * t143
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