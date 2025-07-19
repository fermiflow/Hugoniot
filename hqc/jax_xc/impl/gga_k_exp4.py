"""Generated from gga_k_exp4.mpl."""

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
  t47 = jnp.exp(-0.83254166666666666664e1 * t38 * s0 / t41 / t39)
  t49 = t33 ** 2
  t52 = t49 / t35 / t34
  t53 = s0 ** 2
  t54 = t39 ** 2
  t61 = jnp.exp(-0.75479166666666666666e-2 * t52 * t53 / t40 / t54 / r0)
  t67 = jnp.where(r0 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t29 * t31 * (0.20788e1 - 0.8524 * t47 - 0.12264e1 * t61))
  t69 = jnp.where(t11, t16, -t18)
  t70 = jnp.where(t15, t12, t69)
  t71 = 0.1e1 + t70
  t73 = jnp.cbrt(t71)
  t74 = t73 ** 2
  t76 = jnp.where(t71 <= p.zeta_threshold, t25, t74 * t71)
  t78 = r1 ** 2
  t79 = jnp.cbrt(r1)
  t80 = t79 ** 2
  t86 = jnp.exp(-0.83254166666666666664e1 * t38 * s2 / t80 / t78)
  t88 = s2 ** 2
  t89 = t78 ** 2
  t96 = jnp.exp(-0.75479166666666666666e-2 * t52 * t88 / t79 / t89 / r1)
  t102 = jnp.where(r1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t76 * t31 * (0.20788e1 - 0.8524 * t86 - 0.12264e1 * t96))
  res = t67 + t102
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
  t40 = jnp.exp(-0.83254166666666666664e1 * t25 / t28 * s0 * t32 / t23 / t34)
  t42 = t25 ** 2
  t46 = s0 ** 2
  t48 = t34 ** 2
  t55 = jnp.exp(-0.15095833333333333333e-1 * t42 / t27 / t26 * t46 * t31 / t22 / t48 / r0)
  t61 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t4 * t5 * jnp.pi * t21 * t23 * (0.20788e1 - 0.8524 * t40 - 0.12264e1 * t55))
  res = 0.2e1 * t61
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