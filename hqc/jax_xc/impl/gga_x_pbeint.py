"""Generated from gga_x_pbeint.mpl."""

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
  t31 = jnp.cbrt(6)
  t32 = (params.muPBE - params.muGE) * params.alpha * t31
  t33 = jnp.pi ** 2
  t34 = jnp.cbrt(t33)
  t35 = t34 ** 2
  t36 = 0.1e1 / t35
  t37 = t36 * s0
  t38 = r0 ** 2
  t39 = jnp.cbrt(r0)
  t40 = t39 ** 2
  t42 = 0.1e1 / t40 / t38
  t43 = params.alpha * t31
  t44 = t37 * t42
  t66 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + params.kappa * (0.1e1 - params.kappa / (params.kappa + (params.muGE + t32 * t37 * t42 / (0.1e1 + t43 * t44 / 0.24e2) / 0.24e2) * t31 * t44 / 0.24e2))))
  t68 = jnp.where(t10, t15, -t17)
  t69 = jnp.where(t14, t11, t68)
  t70 = 0.1e1 + t69
  t72 = jnp.cbrt(t70)
  t74 = jnp.where(t70 <= p.zeta_threshold, t23, t72 * t70)
  t76 = t36 * s2
  t77 = r1 ** 2
  t78 = jnp.cbrt(r1)
  t79 = t78 ** 2
  t81 = 0.1e1 / t79 / t77
  t82 = t76 * t81
  t104 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t74 * t27 * (0.1e1 + params.kappa * (0.1e1 - params.kappa / (params.kappa + (params.muGE + t32 * t76 * t81 / (0.1e1 + t43 * t82 / 0.24e2) / 0.24e2) * t31 * t82 / 0.24e2))))
  res = t66 + t104
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
  t23 = jnp.cbrt(6)
  t24 = jnp.pi ** 2
  t25 = jnp.cbrt(t24)
  t26 = t25 ** 2
  t27 = 0.1e1 / t26
  t30 = jnp.cbrt(2)
  t31 = t30 ** 2
  t32 = s0 * t31
  t33 = r0 ** 2
  t34 = t19 ** 2
  t36 = 0.1e1 / t34 / t33
  t39 = t32 * t36
  t62 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + params.kappa * (0.1e1 - params.kappa / (params.kappa + (params.muGE + (params.muPBE - params.muGE) * params.alpha * t23 * t27 * t32 * t36 / (0.1e1 + params.alpha * t23 * t27 * t39 / 0.24e2) / 0.24e2) * t23 * t27 * t39 / 0.24e2))))
  res = 0.2e1 * t62
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