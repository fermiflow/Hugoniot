"""Generated from gga_k_ol2.mpl."""

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
  t34 = r0 ** 2
  t35 = jnp.cbrt(r0)
  t36 = t35 ** 2
  t41 = jnp.sqrt(s0)
  t44 = 0.1e1 / t35 / r0
  t45 = jnp.cbrt(2)
  t56 = jnp.where(r0 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t29 * t31 * (params.aa + 0.13888888888888888889e-1 * params.bb * s0 / t36 / t34 + params.cc * t41 * t44 / (0.4e1 * t41 * t44 + t45)))
  t58 = jnp.where(t11, t16, -t18)
  t59 = jnp.where(t15, t12, t58)
  t60 = 0.1e1 + t59
  t62 = jnp.cbrt(t60)
  t63 = t62 ** 2
  t65 = jnp.where(t60 <= p.zeta_threshold, t25, t63 * t60)
  t68 = r1 ** 2
  t69 = jnp.cbrt(r1)
  t70 = t69 ** 2
  t75 = jnp.sqrt(s2)
  t78 = 0.1e1 / t69 / r1
  t89 = jnp.where(r1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t65 * t31 * (params.aa + 0.13888888888888888889e-1 * params.bb * s2 / t70 / t68 + params.cc * t75 * t78 / (0.4e1 * t75 * t78 + t45)))
  res = t56 + t89
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
  t26 = jnp.cbrt(2)
  t27 = t26 ** 2
  t28 = r0 ** 2
  t34 = jnp.sqrt(s0)
  t37 = 0.1e1 / t22 / r0
  t50 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t4 * t5 * jnp.pi * t21 * t23 * (params.aa + 0.13888888888888888889e-1 * params.bb * s0 * t27 / t23 / t28 + params.cc * t34 * t26 * t37 / (0.4e1 * t34 * t26 * t37 + t26)))
  res = 0.2e1 * t50
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