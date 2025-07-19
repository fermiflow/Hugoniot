"""Generated from lda_k_zlp.mpl."""

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
  t2 = t1 ** 2
  t4 = jnp.cbrt(0.1e1 / jnp.pi)
  t7 = jnp.cbrt(4)
  t10 = r0 + r1
  t12 = (r0 - r1) / t10
  t13 = 0.1e1 + t12
  t15 = jnp.cbrt(p.zeta_threshold)
  t16 = t15 ** 2
  t17 = t16 * p.zeta_threshold
  t18 = jnp.cbrt(t13)
  t19 = t18 ** 2
  t21 = jnp.where(t13 <= p.zeta_threshold, t17, t19 * t13)
  t22 = 0.1e1 - t12
  t24 = jnp.cbrt(t22)
  t25 = t24 ** 2
  t27 = jnp.where(t22 <= p.zeta_threshold, t17, t25 * t22)
  t30 = jnp.cbrt(t10)
  t31 = t30 ** 2
  t36 = jnp.log(0.1e1 + 0.5102040816326530612e3 / t30)
  res = 0.10790666666666666667e1 * t2 / t4 * t7 * (t21 / 0.2e1 + t27 / 0.2e1) * t31 * (0.1e1 - 0.196e-2 * t30 * t36)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(3)
  t2 = t1 ** 2
  t4 = jnp.cbrt(0.1e1 / jnp.pi)
  t7 = jnp.cbrt(4)
  t10 = jnp.cbrt(p.zeta_threshold)
  t11 = t10 ** 2
  t13 = jnp.where(0.1e1 <= p.zeta_threshold, t11 * p.zeta_threshold, 1)
  t14 = jnp.cbrt(r0)
  t15 = t14 ** 2
  t20 = jnp.log(0.1e1 + 0.5102040816326530612e3 / t14)
  res = 0.10790666666666666667e1 * t2 / t4 * t7 * t13 * t15 * (0.1e1 - 0.196e-2 * t14 * t20)
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