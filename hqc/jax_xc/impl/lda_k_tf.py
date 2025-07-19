"""Generated from lda_k_tf.mpl."""

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
  t5 = 0.1e1 + t4
  t7 = jnp.cbrt(p.zeta_threshold)
  t8 = t7 ** 2
  t9 = t8 * p.zeta_threshold
  t10 = jnp.cbrt(t5)
  t11 = t10 ** 2
  t13 = jnp.where(t5 <= p.zeta_threshold, t9, t11 * t5)
  t14 = 0.1e1 - t4
  t16 = jnp.cbrt(t14)
  t17 = t16 ** 2
  t19 = jnp.where(t14 <= p.zeta_threshold, t9, t17 * t14)
  t23 = jnp.cbrt(3)
  t26 = jnp.cbrt(0.1e1 / jnp.pi)
  t27 = t26 ** 2
  t29 = jnp.cbrt(4)
  t30 = t29 ** 2
  t32 = jnp.cbrt(t2)
  t33 = t32 ** 2
  res = params.ax * (t13 / 0.2e1 + t19 / 0.2e1) * t23 / t27 * t30 * t33 / 0.3e1
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t2 = jnp.cbrt(p.zeta_threshold)
  t3 = t2 ** 2
  t5 = jnp.where(0.1e1 <= p.zeta_threshold, t3 * p.zeta_threshold, 1)
  t7 = jnp.cbrt(3)
  t10 = jnp.cbrt(0.1e1 / jnp.pi)
  t11 = t10 ** 2
  t13 = jnp.cbrt(4)
  t14 = t13 ** 2
  t16 = jnp.cbrt(r0)
  t17 = t16 ** 2
  res = params.ax * t5 * t7 / t11 * t14 * t17 / 0.3e1
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