"""Generated from gga_k_rational_p.mpl."""

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
  t35 = jnp.cbrt(6)
  t36 = params.C2 / params.p * t35
  t37 = jnp.pi ** 2
  t38 = jnp.cbrt(t37)
  t39 = t38 ** 2
  t40 = 0.1e1 / t39
  t42 = r0 ** 2
  t43 = jnp.cbrt(r0)
  t44 = t43 ** 2
  t51 = (0.1e1 + t36 * t40 * s0 / t44 / t42 / 0.24e2) ** (-params.p)
  t55 = jnp.where(r0 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t29 * t31 * t51)
  t57 = jnp.where(t11, t16, -t18)
  t58 = jnp.where(t15, t12, t57)
  t59 = 0.1e1 + t58
  t61 = jnp.cbrt(t59)
  t62 = t61 ** 2
  t64 = jnp.where(t59 <= p.zeta_threshold, t25, t62 * t59)
  t67 = r1 ** 2
  t68 = jnp.cbrt(r1)
  t69 = t68 ** 2
  t76 = (0.1e1 + t36 * t40 * s2 / t69 / t67 / 0.24e2) ** (-params.p)
  t80 = jnp.where(r1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t64 * t31 * t76)
  res = t55 + t80
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
  t27 = jnp.cbrt(6)
  t29 = jnp.pi ** 2
  t30 = jnp.cbrt(t29)
  t31 = t30 ** 2
  t34 = jnp.cbrt(2)
  t35 = t34 ** 2
  t36 = r0 ** 2
  t44 = (0.1e1 + params.C2 / params.p * t27 / t31 * s0 * t35 / t23 / t36 / 0.24e2) ** (-params.p)
  t48 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t4 * t5 * jnp.pi * t21 * t23 * t44)
  res = 0.2e1 * t48
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