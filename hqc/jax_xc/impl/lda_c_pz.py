"""Generated from lda_c_pz.mpl."""

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
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = r0 + r1
  t8 = jnp.cbrt(t7)
  t9 = 0.1e1 / t8
  t10 = t6 * t9
  t11 = t1 * t3 * t10
  t12 = t11 / 0.4e1
  t13 = 0.1e1 <= t12
  t16 = jnp.sqrt(t11)
  t22 = t3 * t6 * t9
  t29 = jnp.log(t12)
  t35 = t10 * t29
  t43 = jnp.where(t13, params.gamma[0] / (0.1e1 + params.beta1[0] * t16 / 0.2e1 + params.beta2[0] * t1 * t22 / 0.4e1), params.a[0] * t29 + params.b[0] + params.c[0] * t1 * t3 * t35 / 0.4e1 + params.d[0] * t1 * t22 / 0.4e1)
  t68 = jnp.where(t13, params.gamma[1] / (0.1e1 + params.beta1[1] * t16 / 0.2e1 + params.beta2[1] * t1 * t22 / 0.4e1), params.a[1] * t29 + params.b[1] + params.c[1] * t1 * t3 * t35 / 0.4e1 + params.d[1] * t1 * t22 / 0.4e1)
  t72 = (r0 - r1) / t7
  t73 = 0.1e1 + t72
  t75 = jnp.cbrt(p.zeta_threshold)
  t76 = t75 * p.zeta_threshold
  t77 = jnp.cbrt(t73)
  t79 = jnp.where(t73 <= p.zeta_threshold, t76, t77 * t73)
  t80 = 0.1e1 - t72
  t82 = jnp.cbrt(t80)
  t84 = jnp.where(t80 <= p.zeta_threshold, t76, t82 * t80)
  t87 = jnp.cbrt(2)
  res = t43 + (t68 - t43) * (t79 + t84 - 0.2e1) / (0.2e1 * t87 - 0.2e1)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(3)
  t3 = jnp.cbrt(0.1e1 / jnp.pi)
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = jnp.cbrt(r0)
  t8 = 0.1e1 / t7
  t9 = t6 * t8
  t10 = t1 * t3 * t9
  t11 = t10 / 0.4e1
  t12 = 0.1e1 <= t11
  t15 = jnp.sqrt(t10)
  t21 = t3 * t6 * t8
  t28 = jnp.log(t11)
  t34 = t9 * t28
  t42 = jnp.where(t12, params.gamma[0] / (0.1e1 + params.beta1[0] * t15 / 0.2e1 + params.beta2[0] * t1 * t21 / 0.4e1), params.a[0] * t28 + params.b[0] + params.c[0] * t1 * t3 * t34 / 0.4e1 + params.d[0] * t1 * t21 / 0.4e1)
  t67 = jnp.where(t12, params.gamma[1] / (0.1e1 + params.beta1[1] * t15 / 0.2e1 + params.beta2[1] * t1 * t21 / 0.4e1), params.a[1] * t28 + params.b[1] + params.c[1] * t1 * t3 * t34 / 0.4e1 + params.d[1] * t1 * t21 / 0.4e1)
  t70 = jnp.cbrt(p.zeta_threshold)
  t72 = jnp.where(0.1e1 <= p.zeta_threshold, t70 * p.zeta_threshold, 1)
  t76 = jnp.cbrt(2)
  res = t42 + (t67 - t42) * (0.2e1 * t72 - 0.2e1) / (0.2e1 * t76 - 0.2e1)
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