"""Generated from lda_c_2d_prm.mpl."""

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
  t1 = jnp.sqrt(jnp.pi)
  t3 = jnp.sqrt(r0 + r1)
  t4 = t1 * t3
  t7 = 0.39274e1 * t3 + t1 / 0.2e1
  t9 = t3 / t7
  t11 = 0.39274e1 * t9 - 0.1e1
  t12 = 0.2e1 + params.c
  t13 = jnp.sqrt(t12)
  t22 = t7 ** 2
  t24 = t12 ** (-0.15e1)
  t28 = 0.1e1 + params.c
  t29 = jnp.sqrt(t28)
  res = 0.32416023070084253575e-1 * (0.19637e1 * t4 * t11 / t13 + 0.39274e1 * t9 * t11 / t12 + 0.98185 * t4 / t22 * t24 + 0.39274e1 * t4 * t11 / t29 + 0.39274e1 * t9 / t28) * jnp.pi
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.sqrt(jnp.pi)
  t2 = jnp.sqrt(r0)
  t3 = t1 * t2
  t6 = 0.39274e1 * t2 + t1 / 0.2e1
  t8 = t2 / t6
  t10 = 0.39274e1 * t8 - 0.1e1
  t11 = 0.2e1 + params.c
  t12 = jnp.sqrt(t11)
  t21 = t6 ** 2
  t23 = t11 ** (-0.15e1)
  t27 = 0.1e1 + params.c
  t28 = jnp.sqrt(t27)
  res = 0.32416023070084253575e-1 * (0.19637e1 * t3 * t10 / t12 + 0.39274e1 * t8 * t10 / t11 + 0.98185 * t3 / t21 * t23 + 0.39274e1 * t3 * t10 / t28 + 0.39274e1 * t8 / t27) * jnp.pi
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