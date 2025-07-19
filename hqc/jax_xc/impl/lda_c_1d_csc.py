"""Generated from lda_c_1d_csc.mpl."""

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
  t2 = 0.1e1 / t1
  t3 = t2 / 0.2e1
  t5 = t1 ** 2
  t6 = 0.1e1 / t5
  t15 = t3 ** params.para[9]
  t18 = jnp.log(0.1e1 + params.para[7] * t2 / 0.2e1 + params.para[8] * t15)
  t26 = t3 ** params.para[5]
  t31 = t3 ** params.para[6]
  t36 = (t3 + params.para[4] * t6 / 0.4e1) * t18 / (params.para[1] * t2 + 0.2e1 * params.para[2] * t26 + 0.2e1 * params.para[3] * t31 + 0.2e1 * params.para[0])
  t46 = t3 ** params.ferro[9]
  t49 = jnp.log(0.1e1 + params.ferro[7] * t2 / 0.2e1 + params.ferro[8] * t46)
  t57 = t3 ** params.ferro[5]
  t62 = t3 ** params.ferro[6]
  t70 = (r0 - r1) ** 2
  res = -t36 + (-(t3 + params.ferro[4] * t6 / 0.4e1) * t49 / (params.ferro[1] * t2 + 0.2e1 * params.ferro[2] * t57 + 0.2e1 * params.ferro[3] * t62 + 0.2e1 * params.ferro[0]) + t36) * t70 * t6
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = 0.1e1 / r0
  t2 = t1 / 0.2e1
  t4 = r0 ** 2
  t14 = t2 ** params.para[9]
  t17 = jnp.log(0.1e1 + params.para[7] * t1 / 0.2e1 + params.para[8] * t14)
  t25 = t2 ** params.para[5]
  t30 = t2 ** params.para[6]
  res = -(t2 + params.para[4] / t4 / 0.4e1) * t17 / (params.para[1] * t1 + 0.2e1 * params.para[2] * t25 + 0.2e1 * params.para[3] * t30 + 0.2e1 * params.para[0])
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