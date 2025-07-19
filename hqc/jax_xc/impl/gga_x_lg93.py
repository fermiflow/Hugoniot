"""Generated from gga_x_lg93.mpl."""

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
  t29 = jnp.cbrt(6)
  t30 = jnp.pi ** 2
  t31 = jnp.cbrt(t30)
  t32 = t31 ** 2
  t34 = t29 / t32
  t35 = r0 ** 2
  t36 = jnp.cbrt(r0)
  t37 = t36 ** 2
  t41 = t34 * s0 / t37 / t35
  t43 = t29 ** 2
  t46 = t43 / t31 / t30
  t47 = s0 ** 2
  t48 = t35 ** 2
  t49 = t48 * r0
  t55 = t30 ** 2
  t56 = 0.1e1 / t55
  t59 = t48 ** 2
  t65 = t29 / t32 / t55
  t66 = t47 ** 2
  t76 = t43 / t31 / t55 / t30
  t84 = t55 ** 2
  t85 = 0.1e1 / t84
  t88 = t59 ** 2
  t93 = (0.1e1 + 0.20588079936467259283 * t41 + 0.51718749999999999998e-1 * t46 * t47 / t36 / t49 + 0.97296006944444444444e-2 * t56 * t47 * s0 / t59 + 0.21916594328703703703e-3 * t65 * t66 / t37 / t59 / t35 + 0.11831024546682098765e-2 * t76 * t66 * s0 / t36 / t59 / t49 + 0.10538736979166666667e-4 * t85 * t66 * t47 / t88) ** 0.24974e-1
  t101 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * t93 / (0.1e1 + 0.41666666666666666666e-9 * t41))
  t103 = jnp.where(t10, t15, -t17)
  t104 = jnp.where(t14, t11, t103)
  t105 = 0.1e1 + t104
  t107 = jnp.cbrt(t105)
  t109 = jnp.where(t105 <= p.zeta_threshold, t23, t107 * t105)
  t111 = r1 ** 2
  t112 = jnp.cbrt(r1)
  t113 = t112 ** 2
  t117 = t34 * s2 / t113 / t111
  t119 = s2 ** 2
  t120 = t111 ** 2
  t121 = t120 * r1
  t129 = t120 ** 2
  t133 = t119 ** 2
  t149 = t129 ** 2
  t154 = (0.1e1 + 0.20588079936467259283 * t117 + 0.51718749999999999998e-1 * t46 * t119 / t112 / t121 + 0.97296006944444444444e-2 * t56 * t119 * s2 / t129 + 0.21916594328703703703e-3 * t65 * t133 / t113 / t129 / t111 + 0.11831024546682098765e-2 * t76 * t133 * s2 / t112 / t129 / t121 + 0.10538736979166666667e-4 * t85 * t133 * t119 / t149) ** 0.24974e-1
  t162 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t109 * t28 * t154 / (0.1e1 + 0.41666666666666666666e-9 * t117))
  res = t101 + t162
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
  t21 = jnp.cbrt(6)
  t22 = jnp.pi ** 2
  t23 = jnp.cbrt(t22)
  t24 = t23 ** 2
  t27 = jnp.cbrt(2)
  t28 = t27 ** 2
  t30 = r0 ** 2
  t31 = t20 ** 2
  t35 = t21 / t24 * s0 * t28 / t31 / t30
  t37 = t21 ** 2
  t41 = s0 ** 2
  t43 = t30 ** 2
  t44 = t43 * r0
  t50 = t22 ** 2
  t54 = t43 ** 2
  t61 = t41 ** 2
  t81 = t50 ** 2
  t85 = t54 ** 2
  t90 = (0.1e1 + 0.20588079936467259283 * t35 + 0.1034375 * t37 / t23 / t22 * t41 * t27 / t20 / t44 + 0.38918402777777777778e-1 / t50 * t41 * s0 / t54 + 0.87666377314814814812e-3 * t21 / t24 / t50 * t61 * t28 / t31 / t54 / t30 + 0.9464819637345679012e-2 * t37 / t23 / t50 / t22 * t61 * s0 * t27 / t20 / t54 / t44 + 0.16861979166666666667e-3 / t81 * t61 * t41 / t85) ** 0.24974e-1
  t98 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * t90 / (0.1e1 + 0.41666666666666666666e-9 * t35))
  res = 0.2e1 * t98
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