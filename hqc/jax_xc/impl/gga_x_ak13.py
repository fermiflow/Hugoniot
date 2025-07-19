"""Generated from gga_x_ak13.mpl."""

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
  t29 = jnp.cbrt(6)
  t30 = t29 ** 2
  t32 = jnp.pi ** 2
  t33 = jnp.cbrt(t32)
  t34 = 0.1e1 / t33
  t35 = params.B1 * t30 * t34
  t36 = jnp.sqrt(s0)
  t37 = jnp.cbrt(r0)
  t40 = t36 / t37 / r0
  t41 = t30 * t34
  t45 = jnp.log(0.1e1 + t41 * t40 / 0.12e2)
  t50 = params.B2 * t30 * t34
  t52 = jnp.log(0.1e1 + t45)
  t60 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + t35 * t40 * t45 / 0.12e2 + t50 * t40 * t52 / 0.12e2))
  t62 = jnp.where(t10, t15, -t17)
  t63 = jnp.where(t14, t11, t62)
  t64 = 0.1e1 + t63
  t66 = jnp.cbrt(t64)
  t68 = jnp.where(t64 <= p.zeta_threshold, t23, t66 * t64)
  t70 = jnp.sqrt(s2)
  t71 = jnp.cbrt(r1)
  t74 = t70 / t71 / r1
  t78 = jnp.log(0.1e1 + t41 * t74 / 0.12e2)
  t83 = jnp.log(0.1e1 + t78)
  t91 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t68 * t27 * (0.1e1 + t35 * t74 * t78 / 0.12e2 + t50 * t74 * t83 / 0.12e2))
  res = t60 + t91
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
  t21 = jnp.cbrt(6)
  t22 = t21 ** 2
  t24 = jnp.pi ** 2
  t25 = jnp.cbrt(t24)
  t26 = 0.1e1 / t25
  t28 = jnp.sqrt(s0)
  t29 = jnp.cbrt(2)
  t30 = t28 * t29
  t32 = 0.1e1 / t19 / r0
  t38 = jnp.log(0.1e1 + t22 * t26 * t30 * t32 / 0.12e2)
  t46 = jnp.log(0.1e1 + t38)
  t55 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + params.B1 * t22 * t26 * t30 * t32 * t38 / 0.12e2 + params.B2 * t22 * t26 * t30 * t32 * t46 / 0.12e2))
  res = 0.2e1 * t55
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