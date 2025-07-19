"""Generated from gga_x_2d_b86_mgc.mpl."""

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
  t30 = r0 ** 2
  t32 = 0.1e1 / t30 / r0
  t36 = (0.1e1 + 0.8323e-2 * s0 * t32) ** (0.1e1 / 0.4e1)
  t37 = t36 ** 2
  t47 = jnp.where(r0 <= p.dens_threshold, 0, -0.2e1 / 0.3e1 * t3 * t24 * t28 * (0.1e1 + 0.1243875e-2 * t2 * s0 * t32 / t37 / t36))
  t49 = jnp.where(t8, t13, -t15)
  t50 = jnp.where(t12, t9, t49)
  t51 = 0.1e1 + t50
  t53 = jnp.sqrt(t51)
  t55 = jnp.where(t51 <= p.zeta_threshold, t21, t53 * t51)
  t58 = r1 ** 2
  t60 = 0.1e1 / t58 / r1
  t64 = (0.1e1 + 0.8323e-2 * s2 * t60) ** (0.1e1 / 0.4e1)
  t65 = t64 ** 2
  t75 = jnp.where(r1 <= p.dens_threshold, 0, -0.2e1 / 0.3e1 * t3 * t55 * t28 * (0.1e1 + 0.1243875e-2 * t2 * s2 * t60 / t65 / t64))
  res = t47 + t75
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
  t22 = r0 ** 2
  t24 = 0.1e1 / t22 / r0
  t28 = (0.1e1 + 0.16646e-1 * s0 * t24) ** (0.1e1 / 0.4e1)
  t29 = t28 ** 2
  t39 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.2e1 / 0.3e1 / t3 * t16 * t18 * t19 * (0.1e1 + 0.248775e-2 * t3 * s0 * t24 / t29 / t28))
  res = 0.2e1 * t39
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