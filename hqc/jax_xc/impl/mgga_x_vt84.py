"""Generated from mgga_x_vt84.mpl."""

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
  t30 = t29 * s0
  t31 = r0 ** 2
  t35 = tau0 ** 2
  t41 = t29 / t31 / t35
  t44 = (0.1e1 + t41 / 0.64e2) ** 2
  t50 = jnp.cbrt(6)
  t52 = jnp.pi ** 2
  t53 = jnp.cbrt(t52)
  t54 = t53 ** 2
  t55 = 0.1e1 / t54
  t57 = jnp.cbrt(r0)
  t58 = t57 ** 2
  t60 = 0.1e1 / t58 / t31
  t67 = s0 * t60
  t70 = (tau0 / t58 / r0 - t67 / 0.8e1) * t50
  t73 = 0.5e1 / 0.9e1 * t70 * t55 - 0.1e1
  t78 = jnp.sqrt(0.1e1 + 0.22222222222222222222 * t70 * t55 * t73)
  t82 = t50 * t55
  t83 = t82 * t67
  t85 = 0.9e1 / 0.2e2 * t73 / t78 + t83 / 0.36e2
  t86 = t85 ** 2
  t89 = t50 ** 2
  t92 = t89 / t53 / t52
  t93 = t31 ** 2
  t98 = t92 * t29 / t57 / t93 / r0
  t101 = jnp.sqrt(0.162e3 * t41 + 0.5e2 * t98)
  t106 = t52 ** 2
  t107 = 0.1e1 / t106
  t109 = t93 ** 2
  t113 = (0.1e2 / 0.81e2 + 0.419826171875e-2 * t30 / t31 / r0 / t35 / tau0 / t44) * t50 * t55 * s0 * t60 / 0.24e2 + 0.146e3 / 0.2025e4 * t86 - 0.73e2 / 0.972e5 * t85 * t101 + 0.26505934954444613795e-4 * t98 + 0.19577914932045745128e-2 * t41 + 0.10647076474622770919e-3 * t107 * t30 / t109
  t116 = (0.1e1 + 0.58733744796137235383e-1 * t83) ** 2
  t118 = t113 / t116
  t120 = jnp.exp(-0.1863e-3 * t118)
  t125 = t113 ** 2
  t126 = t116 ** 2
  t130 = jnp.exp(-0.150903e-2 * t125 / t126)
  t141 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + t118 * t120 / (0.1e1 + t118) + (0.1e1 - t130) * (0.1e2 / 0.81e2 / t113 * t116 - 0.1e1)))
  t143 = jnp.where(t10, t15, -t17)
  t144 = jnp.where(t14, t11, t143)
  t145 = 0.1e1 + t144
  t147 = jnp.cbrt(t145)
  t149 = jnp.where(t145 <= p.zeta_threshold, t23, t147 * t145)
  t151 = s2 ** 2
  t152 = t151 * s2
  t153 = r1 ** 2
  t157 = tau1 ** 2
  t163 = t151 / t153 / t157
  t166 = (0.1e1 + t163 / 0.64e2) ** 2
  t174 = jnp.cbrt(r1)
  t175 = t174 ** 2
  t177 = 0.1e1 / t175 / t153
  t184 = s2 * t177
  t187 = (tau1 / t175 / r1 - t184 / 0.8e1) * t50
  t190 = 0.5e1 / 0.9e1 * t187 * t55 - 0.1e1
  t195 = jnp.sqrt(0.1e1 + 0.22222222222222222222 * t187 * t55 * t190)
  t199 = t82 * t184
  t201 = 0.9e1 / 0.2e2 * t190 / t195 + t199 / 0.36e2
  t202 = t201 ** 2
  t205 = t153 ** 2
  t210 = t92 * t151 / t174 / t205 / r1
  t213 = jnp.sqrt(0.162e3 * t163 + 0.5e2 * t210)
  t219 = t205 ** 2
  t223 = (0.1e2 / 0.81e2 + 0.419826171875e-2 * t152 / t153 / r1 / t157 / tau1 / t166) * t50 * t55 * s2 * t177 / 0.24e2 + 0.146e3 / 0.2025e4 * t202 - 0.73e2 / 0.972e5 * t201 * t213 + 0.26505934954444613795e-4 * t210 + 0.19577914932045745128e-2 * t163 + 0.10647076474622770919e-3 * t107 * t152 / t219
  t226 = (0.1e1 + 0.58733744796137235383e-1 * t199) ** 2
  t228 = t223 / t226
  t230 = jnp.exp(-0.1863e-3 * t228)
  t235 = t223 ** 2
  t236 = t226 ** 2
  t240 = jnp.exp(-0.150903e-2 * t235 / t236)
  t251 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t149 * t27 * (0.1e1 + t228 * t230 / (0.1e1 + t228) + (0.1e1 - t240) * (0.1e2 / 0.81e2 / t223 * t226 - 0.1e1)))
  res = t141 + t251
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
  t22 = t21 * s0
  t23 = r0 ** 2
  t27 = tau0 ** 2
  t33 = t21 / t23 / t27
  t36 = (0.1e1 + t33 / 0.64e2) ** 2
  t42 = jnp.cbrt(6)
  t44 = jnp.pi ** 2
  t45 = jnp.cbrt(t44)
  t46 = t45 ** 2
  t47 = 0.1e1 / t46
  t49 = jnp.cbrt(2)
  t50 = t49 ** 2
  t52 = t19 ** 2
  t55 = s0 * t50 / t52 / t23
  t64 = (tau0 * t50 / t52 / r0 - t55 / 0.8e1) * t42
  t67 = 0.5e1 / 0.9e1 * t64 * t47 - 0.1e1
  t72 = jnp.sqrt(0.1e1 + 0.22222222222222222222 * t64 * t47 * t67)
  t77 = t42 * t47 * t55
  t79 = 0.9e1 / 0.2e2 * t67 / t72 + t77 / 0.36e2
  t80 = t79 ** 2
  t83 = t42 ** 2
  t88 = t23 ** 2
  t93 = t83 / t45 / t44 * t21 * t49 / t19 / t88 / r0
  t96 = jnp.sqrt(0.162e3 * t33 + 0.1e3 * t93)
  t101 = t44 ** 2
  t104 = t88 ** 2
  t108 = (0.1e2 / 0.81e2 + 0.419826171875e-2 * t22 / t23 / r0 / t27 / tau0 / t36) * t42 * t47 * t55 / 0.24e2 + 0.146e3 / 0.2025e4 * t80 - 0.73e2 / 0.972e5 * t79 * t96 + 0.5301186990888922759e-4 * t93 + 0.19577914932045745128e-2 * t33 + 0.42588305898491083676e-3 / t101 * t22 / t104
  t111 = (0.1e1 + 0.58733744796137235383e-1 * t77) ** 2
  t113 = t108 / t111
  t115 = jnp.exp(-0.1863e-3 * t113)
  t120 = t108 ** 2
  t121 = t111 ** 2
  t125 = jnp.exp(-0.150903e-2 * t120 / t121)
  t136 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + t113 * t115 / (0.1e1 + t113) + (0.1e1 - t125) * (0.1e2 / 0.81e2 / t108 * t111 - 0.1e1)))
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