"""Generated from mgga_x_r2scan.mpl."""

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
  t30 = 0.2e2 / 0.27e2 + 0.5e1 / 0.3e1 * params.eta
  t31 = jnp.cbrt(6)
  t32 = t31 ** 2
  t33 = jnp.pi ** 2
  t34 = jnp.cbrt(t33)
  t37 = t32 / t34 / t33
  t38 = s0 ** 2
  t39 = r0 ** 2
  t40 = t39 ** 2
  t42 = jnp.cbrt(r0)
  t46 = params.dp2 ** 2
  t47 = t46 ** 2
  t48 = 0.1e1 / t47
  t52 = jnp.exp(-t37 * t38 / t42 / t40 / r0 * t48 / 0.576e3)
  t57 = t34 ** 2
  t58 = 0.1e1 / t57
  t60 = t42 ** 2
  t62 = 0.1e1 / t60 / t39
  t70 = params.k1 * (0.1e1 - params.k1 / (params.k1 + (-0.162742215233874 * t30 * t52 + 0.1e2 / 0.81e2) * t31 * t58 * s0 * t62 / 0.24e2))
  t78 = 0.3e1 / 0.1e2 * t32 * t57
  t84 = (tau0 / t60 / r0 - s0 * t62 / 0.8e1) / (t78 + params.eta * s0 * t62 / 0.8e1)
  t87 = jnp.where(0. < t84, 0, t84)
  t92 = jnp.exp(-params.c1 * t87 / (0.1e1 - t87))
  t94 = 0.25e1 < t84
  t95 = jnp.where(t94, 0.25e1, t84)
  t97 = t95 ** 2
  t99 = t97 * t95
  t101 = t97 ** 2
  t110 = jnp.where(t94, t84, 0.25e1)
  t114 = jnp.exp(params.c2 / (0.1e1 - t110))
  t116 = jnp.where(t84 <= 0.25e1, 0.1e1 - 0.667 * t95 - 0.4445555 * t97 - 0.663086601049 * t99 + 0.145129704449e1 * t101 - 0.887998041597 * t101 * t95 + 0.234528941479 * t101 * t97 - 0.23185843322e-1 * t101 * t99, -params.d * t114)
  t117 = jnp.where(t84 <= 0., t92, t116)
  t122 = jnp.sqrt(0.3e1)
  t124 = t32 / t34
  t125 = jnp.sqrt(s0)
  t130 = jnp.sqrt(t124 * t125 / t42 / r0)
  t134 = jnp.exp(-0.98958e1 * t122 / t130)
  t139 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * (0.1e1 + t70 + t117 * (0.174 - t70)) * (0.1e1 - t134))
  t141 = jnp.where(t10, t15, -t17)
  t142 = jnp.where(t14, t11, t141)
  t143 = 0.1e1 + t142
  t145 = jnp.cbrt(t143)
  t147 = jnp.where(t143 <= p.zeta_threshold, t23, t145 * t143)
  t149 = s2 ** 2
  t150 = r1 ** 2
  t151 = t150 ** 2
  t153 = jnp.cbrt(r1)
  t160 = jnp.exp(-t37 * t149 / t153 / t151 / r1 * t48 / 0.576e3)
  t166 = t153 ** 2
  t168 = 0.1e1 / t166 / t150
  t176 = params.k1 * (0.1e1 - params.k1 / (params.k1 + (-0.162742215233874 * t30 * t160 + 0.1e2 / 0.81e2) * t31 * t58 * s2 * t168 / 0.24e2))
  t188 = (tau1 / t166 / r1 - s2 * t168 / 0.8e1) / (t78 + params.eta * s2 * t168 / 0.8e1)
  t191 = jnp.where(0. < t188, 0, t188)
  t196 = jnp.exp(-params.c1 * t191 / (0.1e1 - t191))
  t198 = 0.25e1 < t188
  t199 = jnp.where(t198, 0.25e1, t188)
  t201 = t199 ** 2
  t203 = t201 * t199
  t205 = t201 ** 2
  t214 = jnp.where(t198, t188, 0.25e1)
  t218 = jnp.exp(params.c2 / (0.1e1 - t214))
  t220 = jnp.where(t188 <= 0.25e1, 0.1e1 - 0.667 * t199 - 0.4445555 * t201 - 0.663086601049 * t203 + 0.145129704449e1 * t205 - 0.887998041597 * t205 * t199 + 0.234528941479 * t205 * t201 - 0.23185843322e-1 * t205 * t203, -params.d * t218)
  t221 = jnp.where(t188 <= 0., t196, t220)
  t226 = jnp.sqrt(s2)
  t231 = jnp.sqrt(t124 * t226 / t153 / r1)
  t235 = jnp.exp(-0.98958e1 * t122 / t231)
  t240 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t147 * t28 * (0.1e1 + t176 + t221 * (0.174 - t176)) * (0.1e1 - t235))
  res = t139 + t240
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
  t23 = jnp.cbrt(6)
  t24 = t23 ** 2
  t25 = jnp.pi ** 2
  t26 = jnp.cbrt(t25)
  t30 = s0 ** 2
  t32 = jnp.cbrt(2)
  t33 = r0 ** 2
  t34 = t33 ** 2
  t39 = params.dp2 ** 2
  t40 = t39 ** 2
  t45 = jnp.exp(-t24 / t26 / t25 * t30 * t32 / t20 / t34 / r0 / t40 / 0.288e3)
  t50 = t26 ** 2
  t53 = t32 ** 2
  t55 = t20 ** 2
  t57 = 0.1e1 / t55 / t33
  t58 = s0 * t53 * t57
  t65 = params.k1 * (0.1e1 - params.k1 / (params.k1 + (-0.162742215233874 * (0.2e2 / 0.27e2 + 0.5e1 / 0.3e1 * params.eta) * t45 + 0.1e2 / 0.81e2) * t23 / t50 * t58 / 0.24e2))
  t80 = (tau0 * t53 / t55 / r0 - t58 / 0.8e1) / (0.3e1 / 0.1e2 * t24 * t50 + params.eta * s0 * t53 * t57 / 0.8e1)
  t83 = jnp.where(0. < t80, 0, t80)
  t88 = jnp.exp(-params.c1 * t83 / (0.1e1 - t83))
  t90 = 0.25e1 < t80
  t91 = jnp.where(t90, 0.25e1, t80)
  t93 = t91 ** 2
  t95 = t93 * t91
  t97 = t93 ** 2
  t106 = jnp.where(t90, t80, 0.25e1)
  t110 = jnp.exp(params.c2 / (0.1e1 - t106))
  t112 = jnp.where(t80 <= 0.25e1, 0.1e1 - 0.667 * t91 - 0.4445555 * t93 - 0.663086601049 * t95 + 0.145129704449e1 * t97 - 0.887998041597 * t97 * t91 + 0.234528941479 * t97 * t93 - 0.23185843322e-1 * t97 * t95, -params.d * t110)
  t113 = jnp.where(t80 <= 0., t88, t112)
  t118 = jnp.sqrt(0.3e1)
  t121 = jnp.sqrt(s0)
  t127 = jnp.sqrt(t24 / t26 * t121 * t32 / t20 / r0)
  t131 = jnp.exp(-0.98958e1 * t118 / t127)
  t136 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * (0.1e1 + t65 + t113 * (0.174 - t65)) * (0.1e1 - t131))
  res = 0.2e1 * t136
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