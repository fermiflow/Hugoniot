"""Generated from mgga_c_b94.mpl."""

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
  t1 = r0 - r1
  t2 = t1 ** 2
  t3 = r0 + r1
  t4 = t3 ** 2
  t9 = r0 <= p.dens_threshold
  t10 = jnp.cbrt(2)
  t11 = 0.1e1 / t3
  t14 = 0.2e1 * r0 * t11 <= p.zeta_threshold
  t15 = p.zeta_threshold - 0.1e1
  t18 = 0.2e1 * r1 * t11 <= p.zeta_threshold
  t19 = -t15
  t20 = t1 * t11
  t21 = jnp.where(t18, t19, t20)
  t22 = jnp.where(t14, t15, t21)
  t24 = (0.1e1 + t22) * t3
  t25 = jnp.cbrt(t24)
  t26 = 0.1e1 / t25
  t28 = jnp.cbrt(jnp.pi)
  t29 = 0.1e1 / t28
  t31 = jnp.cbrt(r0)
  t32 = t31 ** 2
  t34 = 0.1e1 / t32 / r0
  t41 = r0 ** 2
  t43 = 0.1e1 / t32 / t41
  t46 = l0 * t34 / 0.6e1 - 0.2e1 / 0.3e1 * params.gamma * tau0 * t34 + params.gamma * s0 * t43 / 0.12e2
  t47 = jnp.abs(t46)
  t50 = jnp.where(0. < t46, 0.5e-12, -0.5e-12)
  t51 = jnp.where(t47 < 0.5e-12, t50, t46)
  t52 = br89_x(t51)
  t54 = jnp.exp(t52 / 0.3e1)
  t55 = 0.1e1 / t54
  t56 = jnp.exp(-t52)
  t60 = 0.1e1 - t56 * (0.1e1 + t52 / 0.2e1)
  t61 = 0.1e1 / t60
  t66 = jnp.where(t9, 0, t10 * t26 * t29 * t55 * t61 * t52 / 0.2e1)
  t67 = r1 <= p.dens_threshold
  t68 = jnp.where(t14, t19, -t20)
  t69 = jnp.where(t18, t15, t68)
  t71 = (0.1e1 + t69) * t3
  t72 = jnp.cbrt(t71)
  t73 = 0.1e1 / t72
  t76 = jnp.cbrt(r1)
  t77 = t76 ** 2
  t79 = 0.1e1 / t77 / r1
  t86 = r1 ** 2
  t88 = 0.1e1 / t77 / t86
  t91 = l1 * t79 / 0.6e1 - 0.2e1 / 0.3e1 * params.gamma * tau1 * t79 + params.gamma * s2 * t88 / 0.12e2
  t92 = jnp.abs(t91)
  t95 = jnp.where(0. < t91, 0.5e-12, -0.5e-12)
  t96 = jnp.where(t92 < 0.5e-12, t95, t91)
  t97 = br89_x(t96)
  t99 = jnp.exp(t97 / 0.3e1)
  t100 = 0.1e1 / t99
  t101 = jnp.exp(-t97)
  t105 = 0.1e1 - t101 * (0.1e1 + t97 / 0.2e1)
  t106 = 0.1e1 / t105
  t111 = jnp.where(t67, 0, t10 * t73 * t29 * t100 * t106 * t97 / 0.2e1)
  t113 = params.cab * (t66 + t111)
  t115 = jnp.log(0.1e1 + t113)
  t121 = 0.1e1 + t20 <= p.zeta_threshold
  t123 = 0.1e1 - t20 <= p.zeta_threshold
  t124 = jnp.where(t123, t19, t20)
  t125 = jnp.where(t121, t15, t124)
  t126 = 0.1e1 + t125
  t127 = t126 ** 2
  t128 = jnp.cbrt(t126)
  t129 = t128 ** 2
  t131 = t10 ** 2
  t133 = jnp.cbrt(t3)
  t134 = t133 ** 2
  t135 = t134 * t3
  t142 = params.css ** 2
  t143 = t142 ** 2
  t149 = 0.1e1 / t28 / jnp.pi
  t151 = t54 ** 2
  t152 = t151 ** 2
  t155 = t60 ** 2
  t156 = t155 ** 2
  t158 = t52 ** 2
  t159 = t158 ** 2
  t161 = params.css * t10
  t169 = jnp.log(0.1e1 + t161 * t26 * t29 * t55 * t61 * t52 / 0.2e1)
  t170 = 0.1e1 / params.css
  t184 = jnp.where(t9, 0, -0.25e-2 * t129 * t127 * t131 * t135 * (0.2e1 * tau0 * t34 - s0 * t43 / 0.4e1) * t143 / t25 / t24 * t149 / t152 / t156 * t159 * (0.1e1 - t169 * t170 * t131 * t25 * t28 * t54 * t60 / t52))
  t185 = jnp.where(t121, t19, -t20)
  t186 = jnp.where(t123, t15, t185)
  t187 = 0.1e1 + t186
  t188 = t187 ** 2
  t189 = jnp.cbrt(t187)
  t190 = t189 ** 2
  t204 = t99 ** 2
  t205 = t204 ** 2
  t208 = t105 ** 2
  t209 = t208 ** 2
  t211 = t97 ** 2
  t212 = t211 ** 2
  t221 = jnp.log(0.1e1 + t161 * t73 * t29 * t100 * t106 * t97 / 0.2e1)
  t235 = jnp.where(t67, 0, -0.25e-2 * t190 * t188 * t131 * t135 * (0.2e1 * tau1 * t79 - s2 * t88 / 0.4e1) * t143 / t72 / t71 * t149 / t205 / t209 * t212 * (0.1e1 - t221 * t170 * t131 * t72 * t28 * t99 * t105 / t97))
  res = -0.2 * (0.1e1 - t2 / t4) * t3 * t113 * (t113 - t115) + t184 + t235
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = r0 / 0.2e1 <= p.dens_threshold
  t4 = jnp.cbrt(2)
  t5 = 0.1e1 <= p.zeta_threshold
  t6 = p.zeta_threshold - 0.1e1
  t8 = jnp.where(t5, -t6, 0)
  t9 = jnp.where(t5, t6, t8)
  t10 = 0.1e1 + t9
  t11 = t10 * r0
  t12 = jnp.cbrt(t11)
  t13 = 0.1e1 / t12
  t15 = jnp.cbrt(jnp.pi)
  t16 = 0.1e1 / t15
  t18 = t4 ** 2
  t20 = jnp.cbrt(r0)
  t21 = t20 ** 2
  t22 = t21 * r0
  t23 = 0.1e1 / t22
  t31 = r0 ** 2
  t33 = 0.1e1 / t21 / t31
  t37 = l0 * t18 * t23 / 0.6e1 - 0.2e1 / 0.3e1 * params.gamma * tau0 * t18 * t23 + params.gamma * s0 * t18 * t33 / 0.12e2
  t38 = jnp.abs(t37)
  t41 = jnp.where(0. < t37, 0.5e-12, -0.5e-12)
  t42 = jnp.where(t38 < 0.5e-12, t41, t37)
  t43 = br89_x(t42)
  t45 = jnp.exp(t43 / 0.3e1)
  t46 = 0.1e1 / t45
  t47 = jnp.exp(-t43)
  t51 = 0.1e1 - t47 * (0.1e1 + t43 / 0.2e1)
  t52 = 0.1e1 / t51
  t57 = jnp.where(t3, 0, t4 * t13 * t16 * t46 * t52 * t43 / 0.2e1)
  t59 = 0.2e1 * params.cab * t57
  t61 = jnp.log(0.1e1 + t59)
  t66 = t10 ** 2
  t67 = jnp.cbrt(t10)
  t68 = t67 ** 2
  t79 = params.css ** 2
  t80 = t79 ** 2
  t88 = t45 ** 2
  t89 = t88 ** 2
  t92 = t51 ** 2
  t93 = t92 ** 2
  t95 = t43 ** 2
  t96 = t95 ** 2
  t106 = jnp.log(0.1e1 + params.css * t4 * t13 * t16 * t46 * t52 * t43 / 0.2e1)
  t121 = jnp.where(t3, 0, -0.25e-2 * t68 * t66 * t18 * t22 * (0.2e1 * tau0 * t18 * t23 - s0 * t18 * t33 / 0.4e1) * t80 / t12 / t11 / t15 / jnp.pi / t89 / t93 * t96 * (0.1e1 - t106 / params.css * t18 * t12 * t15 * t45 * t51 / t43))
  res = -0.4 * r0 * params.cab * t57 * (t59 - t61) + 0.2e1 * t121
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