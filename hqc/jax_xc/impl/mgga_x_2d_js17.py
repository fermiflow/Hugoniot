"""Generated from mgga_x_2d_js17.mpl."""

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
  t2 = jnp.sqrt(jnp.pi)
  t3 = 0.1e1 / t2
  t4 = r0 + r1
  t5 = 0.1e1 / t4
  t8 = 0.2e1 * r0 * t5 <= p.zeta_threshold
  t9 = p.zeta_threshold - 0.1e1
  t12 = 0.2e1 * r1 * t5 <= p.zeta_threshold
  t13 = -t9
  t15 = (r0 - r1) * t5
  t16 = jnp.where(t12, t13, t15)
  t17 = jnp.where(t8, t9, t16)
  t18 = 0.1e1 + t17
  t20 = jnp.sqrt(p.zeta_threshold)
  t21 = t20 * p.zeta_threshold
  t22 = jnp.sqrt(t18)
  t24 = jnp.where(t18 <= p.zeta_threshold, t21, t22 * t18)
  t26 = jnp.sqrt(0.2e1)
  t27 = jnp.sqrt(t4)
  t28 = t26 * t27
  t29 = 0.1e1 / jnp.pi
  t31 = r0 ** 2
  t34 = t29 * s0 / t31 / r0
  t36 = jnp.pi ** 2
  t37 = 0.1e1 / t36
  t38 = s0 ** 2
  t40 = t31 ** 2
  t45 = 0.1e1 + 0.1296e1 * t34 + 0.62208e-2 * t37 * t38 / t40 / t31
  t46 = t45 ** (0.1e1 / 0.15e2)
  t52 = 0.36912e1 * jnp.pi
  t57 = t45 ** (0.1e1 / 0.5e1)
  t65 = jnp.where(r0 <= p.dens_threshold, 0, -0.2e1 / 0.3e1 * t3 * t24 * t28 * (0.1e1 / t46 + 0.2e1 / 0.5e1 * (0.1e1 + 0.87771428571428571429e-1 * t34 + (-0.772e-1 * tau0 / t31 - t52) * t29 / 0.4e1) / t57))
  t67 = jnp.where(t8, t13, -t15)
  t68 = jnp.where(t12, t9, t67)
  t69 = 0.1e1 + t68
  t71 = jnp.sqrt(t69)
  t73 = jnp.where(t69 <= p.zeta_threshold, t21, t71 * t69)
  t76 = r1 ** 2
  t79 = t29 * s2 / t76 / r1
  t81 = s2 ** 2
  t83 = t76 ** 2
  t88 = 0.1e1 + 0.1296e1 * t79 + 0.62208e-2 * t37 * t81 / t83 / t76
  t89 = t88 ** (0.1e1 / 0.15e2)
  t99 = t88 ** (0.1e1 / 0.5e1)
  t107 = jnp.where(r1 <= p.dens_threshold, 0, -0.2e1 / 0.3e1 * t3 * t73 * t28 * (0.1e1 / t89 + 0.2e1 / 0.5e1 * (0.1e1 + 0.87771428571428571429e-1 * t79 + (-0.772e-1 * tau1 / t76 - t52) * t29 / 0.4e1) / t99))
  res = t65 + t107
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = jnp.sqrt(jnp.pi)
  t5 = 0.1e1 <= p.zeta_threshold
  t6 = p.zeta_threshold - 0.1e1
  t8 = jnp.where(t5, -t6, 0)
  t9 = jnp.where(t5, t6, t8)
  t10 = 0.1e1 + t9
  t12 = jnp.sqrt(p.zeta_threshold)
  t14 = jnp.sqrt(t10)
  t16 = jnp.where(t10 <= p.zeta_threshold, t12 * p.zeta_threshold, t14 * t10)
  t18 = jnp.sqrt(0.2e1)
  t19 = jnp.sqrt(r0)
  t21 = 0.1e1 / jnp.pi
  t23 = r0 ** 2
  t26 = t21 * s0 / t23 / r0
  t28 = jnp.pi ** 2
  t30 = s0 ** 2
  t32 = t23 ** 2
  t37 = 0.1e1 + 0.2592e1 * t26 + 0.248832e-1 / t28 * t30 / t32 / t23
  t38 = t37 ** (0.1e1 / 0.15e2)
  t49 = t37 ** (0.1e1 / 0.5e1)
  t57 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.2e1 / 0.3e1 / t3 * t16 * t18 * t19 * (0.1e1 / t38 + 0.2e1 / 0.5e1 * (0.1e1 + 0.17554285714285714286 * t26 + (-0.1544 * tau0 / t23 - 0.36912e1 * jnp.pi) * t21 / 0.4e1) / t49))
  res = 0.2e1 * t57
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