"""Generated from gga_x_ol2.mpl."""

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
  t30 = r0 ** 2
  t31 = jnp.cbrt(r0)
  t32 = t31 ** 2
  t37 = jnp.sqrt(s0)
  t40 = 0.1e1 / t31 / r0
  t41 = jnp.cbrt(2)
  t52 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (params.aa + 0.13888888888888888889e-1 * params.bb * s0 / t32 / t30 + params.cc * t37 * t40 / (0.4e1 * t37 * t40 + t41)))
  t54 = jnp.where(t10, t15, -t17)
  t55 = jnp.where(t14, t11, t54)
  t56 = 0.1e1 + t55
  t58 = jnp.cbrt(t56)
  t60 = jnp.where(t56 <= p.zeta_threshold, t23, t58 * t56)
  t63 = r1 ** 2
  t64 = jnp.cbrt(r1)
  t65 = t64 ** 2
  t70 = jnp.sqrt(s2)
  t73 = 0.1e1 / t64 / r1
  t84 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t60 * t27 * (params.aa + 0.13888888888888888889e-1 * params.bb * s2 / t65 / t63 + params.cc * t70 * t73 / (0.4e1 * t70 * t73 + t41)))
  res = t52 + t84
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
  t22 = jnp.cbrt(2)
  t23 = t22 ** 2
  t24 = r0 ** 2
  t25 = t19 ** 2
  t31 = jnp.sqrt(s0)
  t34 = 0.1e1 / t19 / r0
  t47 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (params.aa + 0.13888888888888888889e-1 * params.bb * s0 * t23 / t25 / t24 + params.cc * t31 * t22 * t34 / (0.4e1 * t31 * t22 * t34 + t22)))
  res = 0.2e1 * t47
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