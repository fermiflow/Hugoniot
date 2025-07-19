"""Generated from mgga_k_rda.mpl."""

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
  t3 = t2 ** 2
  t4 = jnp.cbrt(jnp.pi)
  t6 = t3 * t4 * jnp.pi
  t7 = r0 + r1
  t8 = 0.1e1 / t7
  t11 = 0.2e1 * r0 * t8 <= p.zeta_threshold
  t12 = p.zeta_threshold - 0.1e1
  t15 = 0.2e1 * r1 * t8 <= p.zeta_threshold
  t16 = -t12
  t18 = (r0 - r1) * t8
  t19 = jnp.where(t15, t16, t18)
  t20 = jnp.where(t11, t12, t19)
  t21 = 0.1e1 + t20
  t23 = jnp.cbrt(p.zeta_threshold)
  t24 = t23 ** 2
  t25 = t24 * p.zeta_threshold
  t26 = jnp.cbrt(t21)
  t27 = t26 ** 2
  t29 = jnp.where(t21 <= p.zeta_threshold, t25, t27 * t21)
  t30 = jnp.cbrt(t7)
  t31 = t30 ** 2
  t33 = jnp.cbrt(6)
  t34 = jnp.pi ** 2
  t35 = jnp.cbrt(t34)
  t36 = t35 ** 2
  t37 = 0.1e1 / t36
  t38 = t33 * t37
  t39 = r0 ** 2
  t40 = jnp.cbrt(r0)
  t41 = t40 ** 2
  t45 = t38 * s0 / t41 / t39
  t47 = t33 ** 2
  t49 = 0.1e1 / t35 / t34
  t50 = t47 * t49
  t51 = s0 ** 2
  t52 = t39 ** 2
  t57 = t50 * t51 / t40 / t52 / r0
  t58 = params.a * t47
  t59 = l0 ** 2
  t64 = t49 * t59 / t40 / t39 / r0
  t66 = t58 * t64 + t57
  t68 = jnp.sqrt(t66)
  t72 = (0.1e1 + params.beta1 * t68 / 0.24e2) ** 2
  t76 = params.b * t47
  t78 = t76 * t64 + t57
  t79 = t78 ** 2
  t81 = jnp.sqrt(t78)
  t85 = (0.1e1 + params.beta2 * t81 / 0.24e2) ** 2
  t86 = t85 ** 2
  t90 = params.c * t33
  t97 = t45 / 0.24e2 + t90 * t37 * l0 / t41 / r0 / 0.24e2
  t107 = jnp.where(r0 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t29 * t31 * (0.5e1 / 0.72e2 * t45 + params.A0 + params.A1 * t66 / t72 / 0.576e3 + params.A2 * t79 / t86 / 0.331776e6 + params.A3 * t97 / (params.beta3 * t97 + 0.1e1)))
  t109 = jnp.where(t11, t16, -t18)
  t110 = jnp.where(t15, t12, t109)
  t111 = 0.1e1 + t110
  t113 = jnp.cbrt(t111)
  t114 = t113 ** 2
  t116 = jnp.where(t111 <= p.zeta_threshold, t25, t114 * t111)
  t118 = r1 ** 2
  t119 = jnp.cbrt(r1)
  t120 = t119 ** 2
  t124 = t38 * s2 / t120 / t118
  t126 = s2 ** 2
  t127 = t118 ** 2
  t132 = t50 * t126 / t119 / t127 / r1
  t133 = l1 ** 2
  t138 = t49 * t133 / t119 / t118 / r1
  t140 = t58 * t138 + t132
  t142 = jnp.sqrt(t140)
  t146 = (0.1e1 + params.beta1 * t142 / 0.24e2) ** 2
  t151 = t76 * t138 + t132
  t152 = t151 ** 2
  t154 = jnp.sqrt(t151)
  t158 = (0.1e1 + params.beta2 * t154 / 0.24e2) ** 2
  t159 = t158 ** 2
  t169 = t124 / 0.24e2 + t90 * t37 * l1 / t120 / r1 / 0.24e2
  t179 = jnp.where(r1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t116 * t31 * (0.5e1 / 0.72e2 * t124 + params.A0 + params.A1 * t140 / t146 / 0.576e3 + params.A2 * t152 / t159 / 0.331776e6 + params.A3 * t169 / (params.beta3 * t169 + 0.1e1)))
  res = t107 + t179
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = jnp.cbrt(3)
  t4 = t3 ** 2
  t5 = jnp.cbrt(jnp.pi)
  t8 = 0.1e1 <= p.zeta_threshold
  t9 = p.zeta_threshold - 0.1e1
  t11 = jnp.where(t8, -t9, 0)
  t12 = jnp.where(t8, t9, t11)
  t13 = 0.1e1 + t12
  t15 = jnp.cbrt(p.zeta_threshold)
  t16 = t15 ** 2
  t18 = jnp.cbrt(t13)
  t19 = t18 ** 2
  t21 = jnp.where(t13 <= p.zeta_threshold, t16 * p.zeta_threshold, t19 * t13)
  t22 = jnp.cbrt(r0)
  t23 = t22 ** 2
  t25 = jnp.cbrt(6)
  t26 = jnp.pi ** 2
  t27 = jnp.cbrt(t26)
  t28 = t27 ** 2
  t29 = 0.1e1 / t28
  t31 = jnp.cbrt(2)
  t32 = t31 ** 2
  t34 = r0 ** 2
  t38 = t25 * t29 * s0 * t32 / t23 / t34
  t40 = t25 ** 2
  t42 = 0.1e1 / t27 / t26
  t44 = s0 ** 2
  t46 = t34 ** 2
  t51 = t40 * t42 * t44 * t31 / t22 / t46 / r0
  t54 = l0 ** 2
  t59 = t54 * t31 / t22 / t34 / r0
  t62 = 0.2e1 * params.a * t40 * t42 * t59 + 0.2e1 * t51
  t64 = jnp.sqrt(t62)
  t68 = (0.1e1 + params.beta1 * t64 / 0.24e2) ** 2
  t76 = 0.2e1 * params.b * t40 * t42 * t59 + 0.2e1 * t51
  t77 = t76 ** 2
  t79 = jnp.sqrt(t76)
  t83 = (0.1e1 + params.beta2 * t79 / 0.24e2) ** 2
  t84 = t83 ** 2
  t96 = t38 / 0.24e2 + params.c * t25 * t29 * l0 * t32 / t23 / r0 / 0.24e2
  t106 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t4 * t5 * jnp.pi * t21 * t23 * (0.5e1 / 0.72e2 * t38 + params.A0 + params.A1 * t62 / t68 / 0.576e3 + params.A2 * t77 / t84 / 0.331776e6 + params.A3 * t96 / (params.beta3 * t96 + 0.1e1)))
  res = 0.2e1 * t106
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