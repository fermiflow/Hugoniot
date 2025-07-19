"""Generated from lda_xc_tih.mpl."""

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
  t4 = jnp.tanh(0.10953e1 + 0.334789e-1 * r0 + 0.334789e-1 * r1)
  t9 = jnp.tanh(-0.414661 + 0.152399 * r0 + 0.152399 * r1)
  t14 = jnp.tanh(-0.354691 + 0.390837e-1 * r0 + 0.390837e-1 * r1)
  t19 = jnp.tanh(0.748531e-1 + 0.136598 * r0 + 0.136598 * r1)
  t24 = jnp.tanh(-0.141063e1 + 0.496577e-2 * r0 + 0.496577e-2 * r1)
  t29 = jnp.tanh(0.48315 + 0.402905e1 * r0 + 0.402905e1 * r1)
  t34 = jnp.tanh(-0.420166 + 0.104352e-1 * r0 + 0.104352e-1 * r1)
  t39 = jnp.tanh(0.147409e1 + 0.442455 * r0 + 0.442455 * r1)
  res = 0.625039 - 0.130351e1 * t4 - 0.137026e1 * t9 - 0.129598e1 * t14 + 0.104305e1 * t19 - 0.909651 * t24 - 0.991782 * t29 - 0.915745 * t34 - 0.195026e1 * t39
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = jnp.tanh(0.10953e1 + 0.334789e-1 * r0)
  t7 = jnp.tanh(-0.414661 + 0.152399 * r0)
  t11 = jnp.tanh(-0.354691 + 0.390837e-1 * r0)
  t15 = jnp.tanh(0.748531e-1 + 0.136598 * r0)
  t19 = jnp.tanh(-0.141063e1 + 0.496577e-2 * r0)
  t23 = jnp.tanh(0.48315 + 0.402905e1 * r0)
  t27 = jnp.tanh(-0.420166 + 0.104352e-1 * r0)
  t31 = jnp.tanh(0.147409e1 + 0.442455 * r0)
  res = 0.625039 - 0.130351e1 * t3 - 0.137026e1 * t7 - 0.129598e1 * t11 + 0.104305e1 * t15 - 0.909651 * t19 - 0.991782 * t23 - 0.915745 * t27 - 0.195026e1 * t31
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