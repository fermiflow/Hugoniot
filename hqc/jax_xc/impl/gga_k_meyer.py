"""Generated from gga_k_meyer.mpl."""

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
  t48 = t33 ** 2
  t50 = t48 / t35
  t51 = jnp.sqrt(s0)
  t52 = t40 * r0
  t56 = t50 * t51 / t52 / 0.72e2
  t59 = jnp.abs(0.1e1 - t56)
  t62 = jnp.log((0.1e1 + t56) / t59)
  t69 = 0.3e1 * (0.1e1 - t38 * s0 / t41 / t39 / 0.864e3) * t62 * t33 * t35 / t51 * t52
  t79 = jnp.where(r0 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t29 * t31 * (0.1e1 + 0.2e2 * (0.1e1 / 0.2e1 - t69) / (0.1e1 / 0.2e1 + t69)))
  t81 = jnp.where(t11, t16, -t18)
  t82 = jnp.where(t15, t12, t81)
  t83 = 0.1e1 + t82
  t85 = jnp.cbrt(t83)
  t86 = t85 ** 2
  t88 = jnp.where(t83 <= p.zeta_threshold, t25, t86 * t83)
  t90 = r1 ** 2
  t91 = jnp.cbrt(r1)
  t92 = t91 ** 2
  t99 = jnp.sqrt(s2)
  t100 = t91 * r1
  t104 = t50 * t99 / t100 / 0.72e2
  t107 = jnp.abs(0.1e1 - t104)
  t110 = jnp.log((0.1e1 + t104) / t107)
  t117 = 0.3e1 * (0.1e1 - t38 * s2 / t92 / t90 / 0.864e3) * t110 * t33 * t35 / t99 * t100
  t127 = jnp.where(r1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t88 * t31 * (0.1e1 + 0.2e2 * (0.1e1 / 0.2e1 - t117) / (0.1e1 / 0.2e1 + t117)))
  res = t79 + t127
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
  t31 = jnp.cbrt(2)
  t32 = t31 ** 2
  t34 = r0 ** 2
  t41 = t25 ** 2
  t44 = jnp.sqrt(s0)
  t46 = t22 * r0
  t50 = t41 / t27 * t44 * t31 / t46 / 0.72e2
  t53 = jnp.abs(0.1e1 - t50)
  t56 = jnp.log((0.1e1 + t50) / t53)
  t64 = 0.3e1 / 0.2e1 * (0.1e1 - t25 / t28 * s0 * t32 / t23 / t34 / 0.864e3) * t56 * t25 * t27 / t44 * t32 * t46
  t74 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t4 * t5 * jnp.pi * t21 * t23 * (0.1e1 + 0.2e2 * (0.1e1 / 0.2e1 - t64) / (0.1e1 / 0.2e1 + t64)))
  res = 0.2e1 * t74
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