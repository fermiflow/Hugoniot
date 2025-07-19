"""Generated from lda_c_1d_loos.mpl."""

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
  t1 = r0 + r1
  t5 = jnp.sqrt(0.1e1 + 0.6166 / t1)
  t6 = t5 - 0.1e1
  t7 = t6 ** 2
  t8 = t1 ** 2
  t10 = jnp.sqrt(0.2e1)
  t11 = jnp.sqrt(jnp.pi)
  t13 = jnp.log(t10 * t11)
  t18 = 0.1e1 - 0.32435939020434641583e1 * t6 * t1
  t19 = t18 ** 2
  t28 = jnp.pi ** 2
  res = 0.10520901401373545762e2 * t7 * t8 * ((-0.3083 * t13 - 0.231225) * t19 * t18 + 0.32435939020434641583e1 * (-0.12332e1 * t13 - 0.86328563835932653793) * t6 * t1 * t19 + 0.10520901401373545762e2 * (-t28 / 0.72e2 + 0.2315926046059033409e-1) * t7 * t8 * t18 + 0.24365629583459979481 * t7 * t6 * t8 * t1)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t4 = jnp.sqrt(0.1e1 + 0.6166 / r0)
  t5 = t4 - 0.1e1
  t6 = t5 ** 2
  t7 = r0 ** 2
  t9 = jnp.sqrt(0.2e1)
  t10 = jnp.sqrt(jnp.pi)
  t12 = jnp.log(t9 * t10)
  t17 = 0.1e1 - 0.32435939020434641583e1 * t5 * r0
  t18 = t17 ** 2
  t27 = jnp.pi ** 2
  res = 0.10520901401373545762e2 * t6 * t7 * ((-0.3083 * t12 - 0.231225) * t18 * t17 + 0.32435939020434641583e1 * (-0.12332e1 * t12 - 0.86328563835932653793) * t5 * r0 * t18 + 0.10520901401373545762e2 * (-t27 / 0.72e2 + 0.2315926046059033409e-1) * t6 * t7 * t17 + 0.24365629583459979481 * t6 * t5 * t7 * r0)
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