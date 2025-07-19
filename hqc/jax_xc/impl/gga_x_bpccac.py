"""Generated from gga_x_bpccac.mpl."""

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
  t29 = jnp.sqrt(s0)
  t30 = jnp.cbrt(r0)
  t33 = t29 / t30 / r0
  t35 = jnp.exp(-t33 + 0.19e2)
  t37 = 0.1e1 / (0.1e1 + t35)
  t39 = jnp.cbrt(6)
  t40 = jnp.pi ** 2
  t41 = jnp.cbrt(t40)
  t42 = t41 ** 2
  t43 = 0.1e1 / t42
  t44 = t39 * t43
  t45 = r0 ** 2
  t46 = t30 ** 2
  t48 = 0.1e1 / t46 / t45
  t50 = t44 * s0 * t48
  t58 = jnp.exp(-0.25e2 / 0.6e1 * t50)
  t66 = t39 ** 2
  t69 = t66 / t41 / t40
  t70 = s0 ** 2
  t71 = t45 ** 2
  t77 = 0.69444444444444444444e-5 * t69 * t70 / t30 / t71 / r0
  t80 = t66 / t41
  t83 = jnp.arcsinh(0.64963333333333333333 * t80 * t33)
  t96 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * ((0.1e1 - t37) * (0.2227e1 - 0.1505529e1 / (0.1227e1 + 0.91464571985215458336e-2 * t50)) + t37 * (0.1e1 + ((0.2743 - 0.1508 * t58) * t39 * t43 * s0 * t48 / 0.24e2 - t77) / (0.1e1 + 0.16370833333333333333e-1 * t80 * t33 * t83 + t77))))
  t98 = jnp.where(t10, t15, -t17)
  t99 = jnp.where(t14, t11, t98)
  t100 = 0.1e1 + t99
  t102 = jnp.cbrt(t100)
  t104 = jnp.where(t100 <= p.zeta_threshold, t23, t102 * t100)
  t106 = jnp.sqrt(s2)
  t107 = jnp.cbrt(r1)
  t110 = t106 / t107 / r1
  t112 = jnp.exp(-t110 + 0.19e2)
  t114 = 0.1e1 / (0.1e1 + t112)
  t116 = r1 ** 2
  t117 = t107 ** 2
  t119 = 0.1e1 / t117 / t116
  t121 = t44 * s2 * t119
  t129 = jnp.exp(-0.25e2 / 0.6e1 * t121)
  t137 = s2 ** 2
  t138 = t116 ** 2
  t144 = 0.69444444444444444444e-5 * t69 * t137 / t107 / t138 / r1
  t148 = jnp.arcsinh(0.64963333333333333333 * t80 * t110)
  t161 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t104 * t27 * ((0.1e1 - t114) * (0.2227e1 - 0.1505529e1 / (0.1227e1 + 0.91464571985215458336e-2 * t121)) + t114 * (0.1e1 + ((0.2743 - 0.1508 * t129) * t39 * t43 * s2 * t119 / 0.24e2 - t144) / (0.1e1 + 0.16370833333333333333e-1 * t80 * t110 * t148 + t144))))
  res = t96 + t161
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
  t21 = jnp.sqrt(s0)
  t22 = jnp.cbrt(2)
  t25 = 0.1e1 / t19 / r0
  t26 = t21 * t22 * t25
  t28 = jnp.exp(-t26 + 0.19e2)
  t30 = 0.1e1 / (0.1e1 + t28)
  t32 = jnp.cbrt(6)
  t33 = jnp.pi ** 2
  t34 = jnp.cbrt(t33)
  t35 = t34 ** 2
  t36 = 0.1e1 / t35
  t38 = t22 ** 2
  t40 = r0 ** 2
  t41 = t19 ** 2
  t44 = s0 * t38 / t41 / t40
  t45 = t32 * t36 * t44
  t53 = jnp.exp(-0.25e2 / 0.6e1 * t45)
  t60 = t32 ** 2
  t64 = s0 ** 2
  t66 = t40 ** 2
  t72 = 0.13888888888888888889e-4 * t60 / t34 / t33 * t64 * t22 / t19 / t66 / r0
  t75 = t60 / t34
  t80 = jnp.arcsinh(0.64963333333333333333 * t75 * t26)
  t93 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * ((0.1e1 - t30) * (0.2227e1 - 0.1505529e1 / (0.1227e1 + 0.91464571985215458336e-2 * t45)) + t30 * (0.1e1 + ((0.2743 - 0.1508 * t53) * t32 * t36 * t44 / 0.24e2 - t72) / (0.1e1 + 0.16370833333333333333e-1 * t75 * t21 * t22 * t25 * t80 + t72))))
  res = 0.2e1 * t93
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