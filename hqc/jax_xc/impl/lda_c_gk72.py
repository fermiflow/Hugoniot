"""Generated from lda_c_gk72.mpl."""

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
  t1 = jnp.cbrt(3)
  t3 = jnp.cbrt(0.1e1 / jnp.pi)
  t4 = t1 * t3
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t8 = jnp.cbrt(r0 + r1)
  t10 = t6 / t8
  t11 = t4 * t10
  t12 = t11 / 0.4e1
  t14 = jnp.log(t12)
  t24 = t1 ** 2
  t30 = jnp.sqrt(0.4e1)
  t31 = jnp.sqrt(t11)
  t36 = t3 ** 2
  t39 = t8 ** 2
  t53 = jnp.where(t12 < 0.1e2, -0.6156e-1 + 0.1898e-1 * t14, 0.146 * t24 / t3 * t5 * t8 + 0.53e1 * t30 / t31 / t11 - 0.49 * t1 / t36 * t6 * t39 - 0.16e1 * t30 / t31 / t24 / t36 / t5 * t39)
  res = jnp.where(t12 < 0.7, 0.311e-1 * t14 - 0.48e-1 + 0.225e-2 * t4 * t10 * t14 - 0.425e-2 * t11, t53)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(3)
  t3 = jnp.cbrt(0.1e1 / jnp.pi)
  t4 = t1 * t3
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = jnp.cbrt(r0)
  t9 = t6 / t7
  t10 = t4 * t9
  t11 = t10 / 0.4e1
  t13 = jnp.log(t11)
  t23 = t1 ** 2
  t29 = jnp.sqrt(0.4e1)
  t30 = jnp.sqrt(t10)
  t35 = t3 ** 2
  t38 = t7 ** 2
  t52 = jnp.where(t11 < 0.1e2, -0.6156e-1 + 0.1898e-1 * t13, 0.146 * t23 / t3 * t5 * t7 + 0.53e1 * t29 / t30 / t10 - 0.49 * t1 / t35 * t6 * t38 - 0.16e1 * t29 / t30 / t23 / t35 / t5 * t38)
  res = jnp.where(t11 < 0.7, 0.311e-1 * t13 - 0.48e-1 + 0.225e-2 * t4 * t9 * t13 - 0.425e-2 * t10, t52)
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