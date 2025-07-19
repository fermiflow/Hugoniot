"""Generated from gga_c_w94.mpl."""

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
  t2 = r0 + r1
  t4 = (r0 - r1) / t2
  t6 = jnp.where(0. < t4, t4, -t4)
  t8 = jnp.where(0.1e-9 < t6, t6, 0.1e-9)
  t9 = jnp.cbrt(t8)
  t10 = t9 ** 2
  t13 = jnp.sqrt(-t10 * t8 + 0.1e1)
  t15 = s0 + 0.2e1 * s1 + s2
  t16 = jnp.sqrt(t15)
  t18 = t2 ** 2
  t19 = t18 ** 2
  t22 = jnp.cbrt(t2)
  t26 = (t16 / t22 / t2) ** (0.1e1 / 0.16e2)
  t27 = t26 ** 2
  t35 = jnp.cbrt(3)
  t37 = jnp.cbrt(0.1e1 / jnp.pi)
  t39 = jnp.cbrt(4)
  t40 = t39 ** 2
  res = -t13 / (0.118e2 + 0.15067 * t27 * t26 * t16 * t15 / t19 + 0.1102e-1 * t15 / t18 / t2 + t35 * t37 * t40 / t22 / 0.4e1)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t2 = jnp.where(0 < 0, 0, 0)
  t4 = jnp.where(0.1e-9 < t2, t2, 0.1e-9)
  t5 = jnp.cbrt(t4)
  t6 = t5 ** 2
  t9 = jnp.sqrt(-t6 * t4 + 0.1e1)
  t10 = jnp.sqrt(s0)
  t12 = r0 ** 2
  t13 = t12 ** 2
  t16 = jnp.cbrt(r0)
  t20 = (t10 / t16 / r0) ** (0.1e1 / 0.16e2)
  t21 = t20 ** 2
  t29 = jnp.cbrt(3)
  t31 = jnp.cbrt(0.1e1 / jnp.pi)
  t33 = jnp.cbrt(4)
  t34 = t33 ** 2
  res = -t9 / (0.118e2 + 0.15067 * t21 * t20 * t10 * s0 / t13 + 0.1102e-1 * s0 / t12 / r0 + t29 * t31 * t34 / t16 / 0.4e1)
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