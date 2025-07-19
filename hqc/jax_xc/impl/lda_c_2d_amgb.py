"""Generated from lda_c_2d_amgb.mpl."""

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
  t2 = 0.1e1 / t1
  t3 = r0 + r1
  t4 = jnp.sqrt(t3)
  t6 = t2 / t4
  t9 = 0.1e1 / t3
  t10 = 0.1e1 / jnp.pi * t9
  t16 = 0.1e1 / t1 / jnp.pi / t4 / t3
  t20 = t6 ** 0.15e1
  t27 = jnp.log(0.1e1 + 0.1e1 / (0.10022e1 * t6 - 0.2069e-1 * t20 + 0.33997 * t10 + 0.1747e-1 * t16))
  t39 = jnp.log(0.1e1 + 0.1e1 / (0.4133 * t6 + 0.668467e-1 * t10 + 0.7799e-3 * t16))
  t42 = r0 - r1
  t43 = t42 ** 2
  t45 = t3 ** 2
  t46 = 0.1e1 / t45
  t57 = jnp.log(0.1e1 + 0.1e1 / (0.1424301e1 * t6 + 0.1163099e1 * t16))
  t60 = t43 ** 2
  t62 = t45 ** 2
  t63 = 0.1e1 / t62
  t66 = jnp.exp(-0.13386e1 * t6)
  t68 = jnp.sqrt(0.2e1)
  t71 = t42 * t9
  t72 = 0.1e1 + t71
  t74 = jnp.sqrt(p.zeta_threshold)
  t75 = t74 * p.zeta_threshold
  t76 = jnp.sqrt(t72)
  t78 = jnp.where(t72 <= p.zeta_threshold, t75, t76 * t72)
  t80 = 0.1e1 - t71
  t82 = jnp.sqrt(t80)
  t84 = jnp.where(t80 <= p.zeta_threshold, t75, t82 * t80)
  res = -0.1925 + (0.863136e-1 * t6 + 0.572384e-1 * t10 + 0.3362975e-2 * t16) * t27 + (0.117331 + (-0.3394e-1 * t6 - 0.766765e-2 * t10 - 0.915064469e-4 * t16) * t39) * t43 * t46 + (0.234188e-1 + (-0.37093e-1 * t6 + 0.163618e-1 * t10 - 0.272383828612e-1 * t16) * t57) * t60 * t63 - 0.4e1 / 0.3e1 * (t66 - 0.1e1) * t68 * t2 * t4 * (t78 / 0.2e1 + t84 / 0.2e1 - 0.1e1 - 0.3e1 / 0.8e1 * t43 * t46 - 0.3e1 / 0.128e3 * t60 * t63)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.sqrt(jnp.pi)
  t2 = 0.1e1 / t1
  t3 = jnp.sqrt(r0)
  t5 = t2 / t3
  t9 = 0.1e1 / jnp.pi / r0
  t15 = 0.1e1 / t1 / jnp.pi / t3 / r0
  t19 = t5 ** 0.15e1
  t26 = jnp.log(0.1e1 + 0.1e1 / (0.10022e1 * t5 - 0.2069e-1 * t19 + 0.33997 * t9 + 0.1747e-1 * t15))
  t29 = jnp.exp(-0.13386e1 * t5)
  t31 = jnp.sqrt(0.2e1)
  t35 = jnp.sqrt(p.zeta_threshold)
  t37 = jnp.where(0.1e1 <= p.zeta_threshold, t35 * p.zeta_threshold, 1)
  res = -0.1925 + (0.863136e-1 * t5 + 0.572384e-1 * t9 + 0.3362975e-2 * t15) * t26 - 0.4e1 / 0.3e1 * (t29 - 0.1e1) * t31 * t2 * t3 * (t37 - 0.1e1)
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