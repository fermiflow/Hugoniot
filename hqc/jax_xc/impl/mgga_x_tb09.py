"""Generated from mgga_x_tb09.mpl."""

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
  t1 = jnp.cbrt(jnp.pi)
  t3 = jnp.cbrt(r0)
  t4 = t3 ** 2
  t6 = 0.1e1 / t4 / r0
  t9 = tau0 * t6
  t11 = r0 ** 2
  t13 = 0.1e1 / t4 / t11
  t16 = l0 * t6 / 0.6e1 - 0.53333333333333333333 * t9 + 0.66666666666666666667e-1 * s0 * t13
  t17 = jnp.abs(t16)
  t20 = jnp.where(0. < t16, 0.5e-12, -0.5e-12)
  t21 = jnp.where(t17 < 0.5e-12, t20, t16)
  t22 = br89_x(t21)
  t24 = jnp.exp(t22 / 0.3e1)
  t25 = jnp.exp(-t22)
  t37 = jnp.sqrt(0.15e2)
  t40 = jnp.sqrt(0.2e1)
  t45 = t9 - params.alpha * s0 * t13 / 0.8e1
  t47 = jnp.where(0.1e-9 < t45, t45, 0.1e-9)
  t48 = jnp.sqrt(t47)
  res = (-0.2e1 * params.c * t1 * t24 * (0.1e1 - t25 * (0.1e1 + t22 / 0.2e1)) / t22 + (0.3e1 * params.c - 0.2e1) * t37 / jnp.pi * t40 * t48 / 0.6e1) * t3
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(jnp.pi)
  t3 = jnp.cbrt(2)
  t4 = t3 ** 2
  t6 = jnp.cbrt(r0)
  t7 = t6 ** 2
  t9 = 0.1e1 / t7 / r0
  t13 = tau0 * t4 * t9
  t16 = r0 ** 2
  t18 = 0.1e1 / t7 / t16
  t21 = l0 * t4 * t9 / 0.6e1 - 0.53333333333333333333 * t13 + 0.66666666666666666667e-1 * s0 * t4 * t18
  t22 = jnp.abs(t21)
  t25 = jnp.where(0. < t21, 0.5e-12, -0.5e-12)
  t26 = jnp.where(t22 < 0.5e-12, t25, t21)
  t27 = br89_x(t26)
  t29 = jnp.exp(t27 / 0.3e1)
  t30 = jnp.exp(-t27)
  t42 = jnp.sqrt(0.15e2)
  t45 = jnp.sqrt(0.2e1)
  t51 = t13 - params.alpha * s0 * t4 * t18 / 0.8e1
  t53 = jnp.where(0.1e-9 < t51, t51, 0.1e-9)
  t54 = jnp.sqrt(t53)
  res = (-0.2e1 * params.c * t1 * t29 * (0.1e1 - t30 * (0.1e1 + t27 / 0.2e1)) / t27 + (0.3e1 * params.c - 0.2e1) * t42 / jnp.pi * t45 * t54 / 0.6e1) * t4 * t6 / 0.2e1
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