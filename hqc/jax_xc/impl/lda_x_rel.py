"""Generated from lda_x_rel.mpl."""

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
  t4 = 0.1e1 / t3
  t5 = t2 * t4
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
  t26 = r1 * t7
  t30 = jnp.cbrt(t26)
  t34 = jnp.where(0.2e1 * t26 <= p.zeta_threshold, t12, 0.2e1 * t13 * r1 * t7 * t30)
  t38 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t34 * t20)
  t40 = jnp.cbrt(9)
  t41 = t40 ** 2
  t42 = t3 ** 2
  t45 = jnp.cbrt(0.1e1 / jnp.pi)
  t46 = t45 ** 2
  t49 = t20 ** 2
  t54 = jnp.sqrt(0.1e1 + 0.17750451365686221606e-4 * t41 * t42 * t2 / t46 * t49)
  t63 = t2 ** 2
  t69 = jnp.arcsinh(0.24324508467583486202e-2 * t40 * t3 * t63 / t45 * t20)
  t79 = (0.15226222180972388889e2 * t54 * t41 * t4 * t2 * t45 / t20 - 0.20865405771390201384e4 * t69 * t40 / t42 * t63 * t46 / t49) ** 2
  res = (t24 + t38) * (0.1e1 - 0.15e1 * t79)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = jnp.cbrt(3)
  t4 = jnp.cbrt(jnp.pi)
  t5 = 0.1e1 / t4
  t8 = jnp.cbrt(p.zeta_threshold)
  t10 = jnp.where(0.1e1 <= p.zeta_threshold, t8 * p.zeta_threshold, 1)
  t11 = jnp.cbrt(r0)
  t15 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 * t5 * t10 * t11)
  t16 = jnp.cbrt(9)
  t17 = t16 ** 2
  t18 = t4 ** 2
  t21 = jnp.cbrt(0.1e1 / jnp.pi)
  t22 = t21 ** 2
  t25 = t11 ** 2
  t30 = jnp.sqrt(0.1e1 + 0.17750451365686221606e-4 * t17 * t18 * t3 / t22 * t25)
  t39 = t3 ** 2
  t45 = jnp.arcsinh(0.24324508467583486202e-2 * t16 * t4 * t39 / t21 * t11)
  t55 = (0.15226222180972388889e2 * t30 * t17 * t5 * t3 * t21 / t11 - 0.20865405771390201384e4 * t45 * t16 / t18 * t39 * t22 / t25) ** 2
  res = 0.2e1 * t15 * (0.1e1 - 0.15e1 * t55)
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