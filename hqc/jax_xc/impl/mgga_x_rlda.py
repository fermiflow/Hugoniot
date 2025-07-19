"""Generated from mgga_x_rlda.mpl."""

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
  t2 = jnp.cbrt(jnp.pi)
  t3 = t2 ** 2
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
  t20 = jnp.cbrt(p.zeta_threshold)
  t21 = t20 * p.zeta_threshold
  t22 = jnp.cbrt(t18)
  t24 = jnp.where(t18 <= p.zeta_threshold, t21, t22 * t18)
  t26 = jnp.cbrt(t4)
  t29 = jnp.cbrt(0.1e1 / jnp.pi)
  t31 = params.prefactor / t29
  t32 = jnp.cbrt(4)
  t33 = jnp.cbrt(r0)
  t34 = t33 ** 2
  t36 = 0.1e1 / t34 / r0
  t47 = jnp.where(r0 <= p.dens_threshold, 0, -0.15e2 / 0.16e2 * t3 * t24 * t26 * t31 * t32 / (0.2e1 * tau0 * t36 - l0 * t36 / 0.4e1))
  t49 = jnp.where(t8, t13, -t15)
  t50 = jnp.where(t12, t9, t49)
  t51 = 0.1e1 + t50
  t53 = jnp.cbrt(t51)
  t55 = jnp.where(t51 <= p.zeta_threshold, t21, t53 * t51)
  t58 = jnp.cbrt(r1)
  t59 = t58 ** 2
  t61 = 0.1e1 / t59 / r1
  t72 = jnp.where(r1 <= p.dens_threshold, 0, -0.15e2 / 0.16e2 * t3 * t55 * t26 * t31 * t32 / (0.2e1 * tau1 * t61 - l1 * t61 / 0.4e1))
  res = t47 + t72
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = jnp.cbrt(jnp.pi)
  t4 = t3 ** 2
  t5 = 0.1e1 <= p.zeta_threshold
  t6 = p.zeta_threshold - 0.1e1
  t8 = jnp.where(t5, -t6, 0)
  t9 = jnp.where(t5, t6, t8)
  t10 = 0.1e1 + t9
  t12 = jnp.cbrt(p.zeta_threshold)
  t14 = jnp.cbrt(t10)
  t16 = jnp.where(t10 <= p.zeta_threshold, t12 * p.zeta_threshold, t14 * t10)
  t18 = jnp.cbrt(r0)
  t21 = jnp.cbrt(0.1e1 / jnp.pi)
  t24 = jnp.cbrt(4)
  t25 = jnp.cbrt(2)
  t26 = t25 ** 2
  t28 = t18 ** 2
  t30 = 0.1e1 / t28 / r0
  t42 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.15e2 / 0.16e2 * t4 * t16 * t18 * params.prefactor / t21 * t24 / (0.2e1 * tau0 * t26 * t30 - l0 * t26 * t30 / 0.4e1))
  res = 0.2e1 * t42
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