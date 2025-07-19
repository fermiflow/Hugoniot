"""Generated from lda_c_vwn_rpa.mpl."""

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
  t7 = r0 + r1
  t8 = jnp.cbrt(t7)
  t10 = t6 / t8
  t11 = t4 * t10
  t12 = t11 / 0.4e1
  t13 = jnp.sqrt(t11)
  t16 = 0.1e1 / (t12 + 0.6536e1 * t13 + 0.427198e2)
  t20 = jnp.log(t4 * t10 * t16 / 0.4e1)
  t25 = jnp.arctan(0.44899888641287296627e-1 / (t13 + 0.13072e2))
  t27 = t13 / 0.2e1
  t29 = (t27 + 0.409286) ** 2
  t31 = jnp.log(t29 * t16)
  t36 = (r0 - r1) / t7
  t37 = 0.1e1 + t36
  t39 = jnp.cbrt(p.zeta_threshold)
  t40 = t39 * p.zeta_threshold
  t41 = jnp.cbrt(t37)
  t43 = jnp.where(t37 <= p.zeta_threshold, t40, t41 * t37)
  t44 = 0.1e1 - t36
  t46 = jnp.cbrt(t44)
  t48 = jnp.where(t44 <= p.zeta_threshold, t40, t46 * t44)
  t49 = t43 + t48 - 0.2e1
  t50 = jnp.cbrt(2)
  t53 = 0.1e1 / (0.2e1 * t50 - 0.2e1)
  t59 = 0.1e1 / (t12 + 0.1006155e2 * t13 + 0.101578e3)
  t63 = jnp.log(t4 * t10 * t59 / 0.4e1)
  t68 = jnp.arctan(0.11716852777089929792e1 / (t13 + 0.201231e2))
  t71 = (t27 + 0.743294) ** 2
  t73 = jnp.log(t71 * t59)
  res = (0.310907e-1 * t20 + 0.20521972937837502661e2 * t25 + 0.44313737677495382697e-2 * t31) * (-t49 * t53 + 0.1e1) + (0.1554535e-1 * t63 + 0.61881802979060631482 * t68 + 0.26673100072733151594e-2 * t73) * t49 * t53
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
  t12 = jnp.sqrt(t10)
  t15 = 0.1e1 / (t11 + 0.6536e1 * t12 + 0.427198e2)
  t19 = jnp.log(t4 * t9 * t15 / 0.4e1)
  t24 = jnp.arctan(0.44899888641287296627e-1 / (t12 + 0.13072e2))
  t26 = t12 / 0.2e1
  t28 = (t26 + 0.409286) ** 2
  t30 = jnp.log(t28 * t15)
  t34 = jnp.cbrt(p.zeta_threshold)
  t36 = jnp.where(0.1e1 <= p.zeta_threshold, t34 * p.zeta_threshold, 1)
  t38 = 0.2e1 * t36 - 0.2e1
  t39 = jnp.cbrt(2)
  t42 = 0.1e1 / (0.2e1 * t39 - 0.2e1)
  t48 = 0.1e1 / (t11 + 0.1006155e2 * t12 + 0.101578e3)
  t52 = jnp.log(t4 * t9 * t48 / 0.4e1)
  t57 = jnp.arctan(0.11716852777089929792e1 / (t12 + 0.201231e2))
  t60 = (t26 + 0.743294) ** 2
  t62 = jnp.log(t60 * t48)
  res = (0.310907e-1 * t19 + 0.20521972937837502661e2 * t24 + 0.44313737677495382697e-2 * t30) * (-t38 * t42 + 0.1e1) + (0.1554535e-1 * t52 + 0.61881802979060631482 * t57 + 0.26673100072733151594e-2 * t62) * t38 * t42
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