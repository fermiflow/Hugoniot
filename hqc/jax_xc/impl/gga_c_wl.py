"""Generated from gga_c_wl.mpl."""

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
  t2 = (r0 - r1) ** 2
  t3 = r0 + r1
  t4 = t3 ** 2
  t8 = jnp.sqrt(0.1e1 - t2 / t4)
  t11 = jnp.sqrt(s0 + 0.2e1 * s1 + s2)
  t12 = jnp.cbrt(t3)
  t19 = jnp.sqrt(s0)
  t20 = jnp.cbrt(r0)
  t25 = jnp.sqrt(s2)
  t26 = jnp.cbrt(r1)
  t31 = jnp.cbrt(3)
  t33 = jnp.cbrt(0.1e1 / jnp.pi)
  t35 = jnp.cbrt(4)
  t36 = t35 ** 2
  res = t8 * (-0.7486 + 0.6001e-1 * t11 / t12 / t3) / (0.360073e1 + 0.9 * t19 / t20 / r0 + 0.9 * t25 / t26 / r1 + t31 * t33 * t36 / t12 / 0.4e1)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.sqrt(s0)
  t2 = jnp.cbrt(r0)
  t4 = 0.1e1 / t2 / r0
  t8 = jnp.cbrt(2)
  t12 = jnp.cbrt(3)
  t14 = jnp.cbrt(0.1e1 / jnp.pi)
  t16 = jnp.cbrt(4)
  t17 = t16 ** 2
  res = (-0.7486 + 0.6001e-1 * t1 * t4) / (0.360073e1 + 0.18e1 * t1 * t8 * t4 + t12 * t14 * t17 / t2 / 0.4e1)
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