"""Generated from mgga_k_lk.mpl."""

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
  t38 = t33 / t36
  t39 = r0 ** 2
  t40 = jnp.cbrt(r0)
  t41 = t40 ** 2
  t43 = 0.1e1 / t41 / t39
  t47 = t33 ** 2
  t50 = t47 / t35 / t34
  t51 = l0 ** 2
  t57 = t50 * t51 / t40 / t39 / r0 / 0.5832e4
  t58 = t39 ** 2
  t64 = t50 * s0 / t40 / t58 * l0 / 0.5184e4
  t65 = s0 ** 2
  t69 = t65 / t40 / t58 / r0
  t71 = t50 * t69 / 0.17496e5
  t72 = 0.1e1 / params.kappa
  t86 = t34 ** 2
  t87 = 0.1e1 / t86
  t90 = t58 ** 2
  t92 = params.kappa ** 2
  t93 = 0.1e1 / t92
  t107 = jnp.where(r0 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t29 * t31 * (0.1e1 + params.kappa * (0.2e1 - 0.1e1 / (0.1e1 + (0.5e1 / 0.648e3 * t38 * s0 * t43 + t57 - t64 + t71 + 0.25e2 / 0.419904e6 * t50 * t69 * t72) * t72) - 0.1e1 / (0.1e1 + (0.5e1 / 0.324e3 * t38 * s0 * t43 * (t57 - t64 + t71) * t72 + 0.125e3 / 0.45349632e8 * t87 * t65 * s0 / t90 * t93) * t72))))
  t109 = jnp.where(t11, t16, -t18)
  t110 = jnp.where(t15, t12, t109)
  t111 = 0.1e1 + t110
  t113 = jnp.cbrt(t111)
  t114 = t113 ** 2
  t116 = jnp.where(t111 <= p.zeta_threshold, t25, t114 * t111)
  t118 = r1 ** 2
  t119 = jnp.cbrt(r1)
  t120 = t119 ** 2
  t122 = 0.1e1 / t120 / t118
  t126 = l1 ** 2
  t132 = t50 * t126 / t119 / t118 / r1 / 0.5832e4
  t133 = t118 ** 2
  t139 = t50 * s2 / t119 / t133 * l1 / 0.5184e4
  t140 = s2 ** 2
  t144 = t140 / t119 / t133 / r1
  t146 = t50 * t144 / 0.17496e5
  t162 = t133 ** 2
  t177 = jnp.where(r1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t116 * t31 * (0.1e1 + params.kappa * (0.2e1 - 0.1e1 / (0.1e1 + (0.5e1 / 0.648e3 * t38 * s2 * t122 + t132 - t139 + t146 + 0.25e2 / 0.419904e6 * t50 * t144 * t72) * t72) - 0.1e1 / (0.1e1 + (0.5e1 / 0.324e3 * t38 * s2 * t122 * (t132 - t139 + t146) * t72 + 0.125e3 / 0.45349632e8 * t87 * t140 * s2 / t162 * t93) * t72))))
  res = t107 + t177
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
  t30 = t25 / t28
  t31 = jnp.cbrt(2)
  t32 = t31 ** 2
  t34 = r0 ** 2
  t36 = 0.1e1 / t23 / t34
  t40 = t25 ** 2
  t43 = t40 / t27 / t26
  t44 = l0 ** 2
  t51 = t43 * t44 * t31 / t22 / t34 / r0 / 0.2916e4
  t53 = t34 ** 2
  t59 = t43 * s0 * t31 / t22 / t53 * l0 / 0.2592e4
  t60 = s0 ** 2
  t64 = 0.1e1 / t22 / t53 / r0
  t67 = t43 * t60 * t31 * t64 / 0.8748e4
  t70 = 0.1e1 / params.kappa
  t85 = t26 ** 2
  t89 = t53 ** 2
  t91 = params.kappa ** 2
  t106 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t4 * t5 * jnp.pi * t21 * t23 * (0.1e1 + params.kappa * (0.2e1 - 0.1e1 / (0.1e1 + (0.5e1 / 0.648e3 * t30 * s0 * t32 * t36 + t51 - t59 + t67 + 0.25e2 / 0.209952e6 * t43 * t60 * t31 * t64 * t70) * t70) - 0.1e1 / (0.1e1 + (0.5e1 / 0.324e3 * t30 * s0 * t32 * t36 * (t51 - t59 + t67) * t70 + 0.125e3 / 0.11337408e8 / t85 * t60 * s0 / t89 / t91) * t70))))
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