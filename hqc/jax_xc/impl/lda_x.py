"""Generated from lda_x.mpl."""

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
  t8 = r0 * t7
  t11 = jnp.cbrt(p.zeta_threshold)
  t12 = t11 * p.zeta_threshold
  t13 = jnp.cbrt(2)
  t15 = jnp.cbrt(t8)
  t19 = jnp.where(0.2e1 * t8 <= p.zeta_threshold, t12, 0.2e1 * t13 * r0 * t7 * t15)
  t20 = jnp.cbrt(t6)
  t24 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t19 * t20)
  t27 = r1 * t7
  t31 = jnp.cbrt(t27)
  t35 = jnp.where(0.2e1 * t27 <= p.zeta_threshold, t12, 0.2e1 * t13 * r1 * t7 * t31)
  t39 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t35 * t20)
  res = params.alpha * t24 + params.alpha * t39
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = jnp.cbrt(3)
  t4 = jnp.cbrt(jnp.pi)
  t8 = jnp.cbrt(p.zeta_threshold)
  t10 = jnp.where(0.1e1 <= p.zeta_threshold, t8 * p.zeta_threshold, 1)
  t11 = jnp.cbrt(r0)
  t15 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t10 * t11)
  res = 0.2e1 * params.alpha * t15
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