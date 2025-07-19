"""Generated from mgga_x_rtpss.mpl."""

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
  t27 = jnp.cbrt(t6)
  t29 = s0 ** 2
  t31 = r0 ** 2
  t32 = 0.1e1 / t31
  t33 = tau0 ** 2
  t34 = 0.1e1 / t33
  t35 = t32 * t34
  t37 = t29 * t32 * t34
  t40 = (0.1e1 + t37 / 0.64e2) ** 2
  t46 = jnp.cbrt(6)
  t48 = jnp.pi ** 2
  t49 = jnp.cbrt(t48)
  t50 = t49 ** 2
  t51 = 0.1e1 / t50
  t53 = jnp.cbrt(r0)
  t54 = t53 ** 2
  t56 = 0.1e1 / t54 / t31
  t57 = t51 * s0 * t56
  t63 = s0 * t56
  t65 = tau0 / t54 / r0 - t63 / 0.8e1
  t69 = 0.5e1 / 0.9e1 * t65 * t46 * t51 - 0.1e1
  t71 = t46 * t51
  t76 = jnp.sqrt(0.5e1 * params.b * t65 * t71 * t69 + 0.9e1)
  t82 = 0.27e2 / 0.2e2 * t69 / t76 + t71 * t63 / 0.36e2
  t83 = t82 ** 2
  t86 = t46 ** 2
  t88 = 0.1e1 / t49 / t48
  t89 = t86 * t88
  t90 = t31 ** 2
  t93 = 0.1e1 / t53 / t90 / r0
  t98 = jnp.sqrt(0.5e2 * t89 * t29 * t93 + 0.162e3 * t37)
  t101 = 0.1e1 / params.kappa
  t102 = t101 * t86
  t107 = jnp.sqrt(params.e)
  t111 = params.e * params.mu
  t112 = t48 ** 2
  t113 = 0.1e1 / t112
  t116 = t90 ** 2
  t122 = t107 * t46
  t126 = (0.1e1 + t122 * t57 / 0.24e2) ** 2
  t130 = jnp.exp(-((0.1e2 / 0.81e2 + params.c * t29 * t35 / t40 / 0.64e2) * t46 * t57 / 0.24e2 + 0.146e3 / 0.2025e4 * t83 - 0.73e2 / 0.972e5 * t82 * t98 + 0.25e2 / 0.944784e6 * t102 * t88 * t29 * t93 + t107 * t29 * t35 / 0.72e3 + t111 * t113 * t29 * s0 / t116 / 0.2304e4) / t126 * t101)
  t137 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + params.kappa * (0.1e1 - t130)))
  t139 = jnp.where(t10, t15, -t17)
  t140 = jnp.where(t14, t11, t139)
  t141 = 0.1e1 + t140
  t143 = jnp.cbrt(t141)
  t145 = jnp.where(t141 <= p.zeta_threshold, t23, t143 * t141)
  t147 = s2 ** 2
  t149 = r1 ** 2
  t150 = 0.1e1 / t149
  t151 = tau1 ** 2
  t152 = 0.1e1 / t151
  t153 = t150 * t152
  t155 = t147 * t150 * t152
  t158 = (0.1e1 + t155 / 0.64e2) ** 2
  t166 = jnp.cbrt(r1)
  t167 = t166 ** 2
  t169 = 0.1e1 / t167 / t149
  t170 = t51 * s2 * t169
  t176 = s2 * t169
  t178 = tau1 / t167 / r1 - t176 / 0.8e1
  t182 = 0.5e1 / 0.9e1 * t178 * t46 * t51 - 0.1e1
  t188 = jnp.sqrt(0.5e1 * params.b * t178 * t71 * t182 + 0.9e1)
  t194 = 0.27e2 / 0.2e2 * t182 / t188 + t71 * t176 / 0.36e2
  t195 = t194 ** 2
  t198 = t149 ** 2
  t201 = 0.1e1 / t166 / t198 / r1
  t206 = jnp.sqrt(0.5e2 * t89 * t147 * t201 + 0.162e3 * t155)
  t218 = t198 ** 2
  t227 = (0.1e1 + t122 * t170 / 0.24e2) ** 2
  t231 = jnp.exp(-((0.1e2 / 0.81e2 + params.c * t147 * t153 / t158 / 0.64e2) * t46 * t170 / 0.24e2 + 0.146e3 / 0.2025e4 * t195 - 0.73e2 / 0.972e5 * t194 * t206 + 0.25e2 / 0.944784e6 * t102 * t88 * t147 * t201 + t107 * t147 * t153 / 0.72e3 + t111 * t113 * t147 * s2 / t218 / 0.2304e4) / t227 * t101)
  t238 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t145 * t27 * (0.1e1 + params.kappa * (0.1e1 - t231)))
  res = t137 + t238
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
  t19 = jnp.cbrt(r0)
  t21 = s0 ** 2
  t23 = r0 ** 2
  t24 = 0.1e1 / t23
  t25 = tau0 ** 2
  t26 = 0.1e1 / t25
  t27 = t24 * t26
  t29 = t21 * t24 * t26
  t32 = (0.1e1 + t29 / 0.64e2) ** 2
  t38 = jnp.cbrt(6)
  t40 = jnp.pi ** 2
  t41 = jnp.cbrt(t40)
  t42 = t41 ** 2
  t43 = 0.1e1 / t42
  t45 = jnp.cbrt(2)
  t46 = t45 ** 2
  t48 = t19 ** 2
  t51 = s0 * t46 / t48 / t23
  t59 = tau0 * t46 / t48 / r0 - t51 / 0.8e1
  t63 = 0.5e1 / 0.9e1 * t59 * t38 * t43 - 0.1e1
  t65 = t38 * t43
  t70 = jnp.sqrt(0.5e1 * params.b * t59 * t65 * t63 + 0.9e1)
  t76 = 0.27e2 / 0.2e2 * t63 / t70 + t65 * t51 / 0.36e2
  t77 = t76 ** 2
  t80 = t38 ** 2
  t82 = 0.1e1 / t41 / t40
  t85 = t23 ** 2
  t89 = t21 * t45 / t19 / t85 / r0
  t93 = jnp.sqrt(0.1e3 * t80 * t82 * t89 + 0.162e3 * t29)
  t96 = 0.1e1 / params.kappa
  t101 = jnp.sqrt(params.e)
  t106 = t40 ** 2
  t110 = t85 ** 2
  t121 = (0.1e1 + t101 * t38 * t43 * t51 / 0.24e2) ** 2
  t125 = jnp.exp(-((0.1e2 / 0.81e2 + params.c * t21 * t27 / t32 / 0.64e2) * t38 * t43 * t51 / 0.24e2 + 0.146e3 / 0.2025e4 * t77 - 0.73e2 / 0.972e5 * t76 * t93 + 0.25e2 / 0.472392e6 * t96 * t80 * t82 * t89 + t101 * t21 * t27 / 0.72e3 + params.e * params.mu / t106 * t21 * s0 / t110 / 0.576e3) / t121 * t96)
  t132 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + params.kappa * (0.1e1 - t125)))
  res = 0.2e1 * t132
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