"""Generated from gga_k_mpbe.mpl."""

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
  t35 = jnp.pi ** 2
  t36 = jnp.cbrt(t35)
  t37 = t36 ** 2
  t38 = 0.1e1 / t37
  t39 = params.c1 * t33 * t38
  t40 = r0 ** 2
  t41 = jnp.cbrt(r0)
  t42 = t41 ** 2
  t44 = 0.1e1 / t42 / t40
  t46 = params.a * t33
  t51 = 0.1e1 + t46 * t38 * s0 * t44 / 0.24e2
  t56 = t33 ** 2
  t60 = params.c2 * t56 / t36 / t35
  t61 = s0 ** 2
  t62 = t40 ** 2
  t67 = t51 ** 2
  t72 = t35 ** 2
  t74 = params.c3 / t72
  t76 = t62 ** 2
  t88 = jnp.where(r0 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t29 * t31 * (0.1e1 + t39 * s0 * t44 / t51 / 0.24e2 + t60 * t61 / t41 / t62 / r0 / t67 / 0.576e3 + t74 * t61 * s0 / t76 / t67 / t51 / 0.2304e4))
  t90 = jnp.where(t11, t16, -t18)
  t91 = jnp.where(t15, t12, t90)
  t92 = 0.1e1 + t91
  t94 = jnp.cbrt(t92)
  t95 = t94 ** 2
  t97 = jnp.where(t92 <= p.zeta_threshold, t25, t95 * t92)
  t99 = r1 ** 2
  t100 = jnp.cbrt(r1)
  t101 = t100 ** 2
  t103 = 0.1e1 / t101 / t99
  t109 = 0.1e1 + t46 * t38 * s2 * t103 / 0.24e2
  t114 = s2 ** 2
  t115 = t99 ** 2
  t120 = t109 ** 2
  t126 = t115 ** 2
  t138 = jnp.where(r1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t97 * t31 * (0.1e1 + t39 * s2 * t103 / t109 / 0.24e2 + t60 * t114 / t100 / t115 / r1 / t120 / 0.576e3 + t74 * t114 * s2 / t126 / t120 / t109 / 0.2304e4))
  res = t88 + t138
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
  t27 = jnp.pi ** 2
  t28 = jnp.cbrt(t27)
  t29 = t28 ** 2
  t30 = 0.1e1 / t29
  t32 = jnp.cbrt(2)
  t33 = t32 ** 2
  t34 = s0 * t33
  t35 = r0 ** 2
  t37 = 0.1e1 / t23 / t35
  t43 = 0.1e1 + params.a * t25 * t30 * t34 * t37 / 0.24e2
  t49 = t25 ** 2
  t54 = s0 ** 2
  t56 = t35 ** 2
  t60 = t43 ** 2
  t66 = t27 ** 2
  t70 = t56 ** 2
  t82 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t4 * t5 * jnp.pi * t21 * t23 * (0.1e1 + params.c1 * t25 * t30 * t34 * t37 / t43 / 0.24e2 + params.c2 * t49 / t28 / t27 * t54 * t32 / t22 / t56 / r0 / t60 / 0.288e3 + params.c3 / t66 * t54 * s0 / t70 / t60 / t43 / 0.576e3))
  res = 0.2e1 * t82
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