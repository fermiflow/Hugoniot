"""Generated from gga_x_lsrpbe.mpl."""

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
  t31 = jnp.pi ** 2
  t32 = jnp.cbrt(t31)
  t33 = t32 ** 2
  t34 = 0.1e1 / t33
  t35 = params.mu * t29 * t34
  t36 = r0 ** 2
  t37 = jnp.cbrt(r0)
  t38 = t37 ** 2
  t40 = 0.1e1 / t38 / t36
  t42 = 0.1e1 / params.kappa
  t46 = jnp.exp(-t35 * s0 * t40 * t42 / 0.24e2)
  t49 = params.kappa + 0.1e1
  t50 = params.alpha * t29
  t55 = jnp.exp(-t50 * t34 * s0 * t40 / 0.24e2)
  t62 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + params.kappa * (0.1e1 - t46) - t49 * (0.1e1 - t55)))
  t64 = jnp.where(t10, t15, -t17)
  t65 = jnp.where(t14, t11, t64)
  t66 = 0.1e1 + t65
  t68 = jnp.cbrt(t66)
  t70 = jnp.where(t66 <= p.zeta_threshold, t23, t68 * t66)
  t72 = r1 ** 2
  t73 = jnp.cbrt(r1)
  t74 = t73 ** 2
  t76 = 0.1e1 / t74 / t72
  t81 = jnp.exp(-t35 * s2 * t76 * t42 / 0.24e2)
  t88 = jnp.exp(-t50 * t34 * s2 * t76 / 0.24e2)
  t95 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t70 * t27 * (0.1e1 + params.kappa * (0.1e1 - t81) - t49 * (0.1e1 - t88)))
  res = t62 + t95
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
  t23 = jnp.pi ** 2
  t24 = jnp.cbrt(t23)
  t25 = t24 ** 2
  t26 = 0.1e1 / t25
  t28 = jnp.cbrt(2)
  t29 = t28 ** 2
  t30 = s0 * t29
  t31 = r0 ** 2
  t32 = t19 ** 2
  t34 = 0.1e1 / t32 / t31
  t40 = jnp.exp(-params.mu * t21 * t26 * t30 * t34 / params.kappa / 0.24e2)
  t49 = jnp.exp(-params.alpha * t21 * t26 * t30 * t34 / 0.24e2)
  t56 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + params.kappa * (0.1e1 - t40) - (params.kappa + 0.1e1) * (0.1e1 - t49)))
  res = 0.2e1 * t56
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