"""Generated from gga_c_lyp.mpl."""

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
  t1 = r0 - r1
  t2 = t1 ** 2
  t3 = r0 + r1
  t4 = t3 ** 2
  t7 = 0.1e1 - t2 / t4
  t8 = jnp.cbrt(t3)
  t9 = 0.1e1 / t8
  t12 = 0.1e1 / (params.d * t9 + 0.1e1)
  t15 = jnp.exp(-params.c * t9)
  t19 = t8 ** 2
  t25 = (params.d * t12 + params.c) * t9
  t32 = jnp.cbrt(3)
  t33 = t32 ** 2
  t34 = jnp.pi ** 2
  t35 = jnp.cbrt(t34)
  t36 = t35 ** 2
  t39 = t1 / t3
  t40 = 0.1e1 + t39
  t41 = t40 <= p.zeta_threshold
  t42 = p.zeta_threshold ** 2
  t43 = jnp.cbrt(p.zeta_threshold)
  t44 = t43 ** 2
  t45 = t44 * t42
  t46 = t40 ** 2
  t47 = jnp.cbrt(t40)
  t48 = t47 ** 2
  t50 = jnp.where(t41, t45, t48 * t46)
  t51 = 0.1e1 - t39
  t52 = t51 <= p.zeta_threshold
  t53 = t51 ** 2
  t54 = jnp.cbrt(t51)
  t55 = t54 ** 2
  t57 = jnp.where(t52, t45, t55 * t53)
  t62 = jnp.cbrt(2)
  t63 = t62 * t7
  t66 = r0 ** 2
  t67 = jnp.cbrt(r0)
  t68 = t67 ** 2
  t70 = 0.1e1 / t68 / t66
  t71 = s0 * t70
  t72 = t71 * t50
  t73 = r1 ** 2
  t74 = jnp.cbrt(r1)
  t75 = t74 ** 2
  t77 = 0.1e1 / t75 / t73
  t78 = s2 * t77
  t79 = t78 * t57
  t86 = t44 * t42 * p.zeta_threshold
  t89 = jnp.where(t41, t86, t48 * t46 * t40)
  t93 = jnp.where(t52, t86, t55 * t53 * t51)
  t101 = jnp.where(t41, t42, t46)
  t106 = jnp.where(t52, t42, t53)
  res = params.a * (-t7 * t12 + params.b * t15 * t12 * (-(s0 + 0.2e1 * s1 + s2) / t19 / t4 * (t7 * (0.47e2 - 0.7e1 * t25) / 0.72e2 - 0.2e1 / 0.3e1) - 0.3e1 / 0.2e2 * t33 * t36 * t7 * (t50 + t57) + t63 * (0.5e1 / 0.2e1 - t25 / 0.18e2) * (t72 + t79) / 0.32e2 + t63 * (t25 - 0.11e2) * (t71 * t89 + t78 * t93) / 0.576e3 - t62 * (0.2e1 / 0.3e1 * t72 + 0.2e1 / 0.3e1 * t79 - t101 * s2 * t77 * t57 / 0.4e1 - t106 * s0 * t70 * t50 / 0.4e1) / 0.8e1))
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(r0)
  t2 = 0.1e1 / t1
  t5 = 0.1e1 / (params.d * t2 + 0.1e1)
  t7 = jnp.exp(-params.c * t2)
  t9 = r0 ** 2
  t10 = t1 ** 2
  t12 = 0.1e1 / t10 / t9
  t16 = (params.d * t5 + params.c) * t2
  t20 = jnp.cbrt(3)
  t21 = t20 ** 2
  t22 = jnp.pi ** 2
  t23 = jnp.cbrt(t22)
  t24 = t23 ** 2
  t26 = 0.1e1 <= p.zeta_threshold
  t27 = p.zeta_threshold ** 2
  t28 = jnp.cbrt(p.zeta_threshold)
  t29 = t28 ** 2
  t31 = jnp.where(t26, t29 * t27, 1)
  t37 = t12 * t31
  t44 = jnp.where(t26, t29 * t27 * p.zeta_threshold, 1)
  t48 = jnp.cbrt(2)
  t49 = t48 ** 2
  t53 = jnp.where(t26, t27, 1)
  res = params.a * (-t5 + params.b * t7 * t5 * (-s0 * t12 * (-0.1e1 / 0.72e2 - 0.7e1 / 0.72e2 * t16) - 0.3e1 / 0.1e2 * t21 * t24 * t31 + (0.5e1 / 0.2e1 - t16 / 0.18e2) * s0 * t37 / 0.8e1 + (t16 - 0.11e2) * s0 * t12 * t44 / 0.144e3 - t48 * (0.4e1 / 0.3e1 * s0 * t49 * t37 - t53 * s0 * t49 * t12 * t31 / 0.2e1) / 0.8e1))
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