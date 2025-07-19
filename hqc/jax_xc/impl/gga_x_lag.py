"""Generated from gga_x_lag.mpl."""

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
  t10 = 0.2e1 * r0 * t7 <= p.zeta_threshold
  t11 = p.zeta_threshold - 0.1e1
  t14 = 0.2e1 * r1 * t7 <= p.zeta_threshold
  t15 = -t11
  t17 = (r0 - r1) * t7
  t18 = jnp.where(t14, t15, t17)
  t19 = jnp.where(t10, t11, t18)
  t20 = 0.1e1 + t19
  t22 = jnp.cbrt(p.zeta_threshold)
  t23 = t22 * p.zeta_threshold
  t24 = jnp.cbrt(t20)
  t26 = jnp.where(t20 <= p.zeta_threshold, t23, t24 * t20)
  t28 = jnp.cbrt(t6)
  t29 = jnp.cbrt(6)
  t30 = t29 ** 2
  t31 = jnp.pi ** 2
  t32 = jnp.cbrt(t31)
  t34 = t30 / t32
  t35 = jnp.sqrt(s0)
  t36 = jnp.cbrt(r0)
  t41 = (t34 * t35 / t36 / r0) ** 0.2626712e1
  t45 = (0.1e1 + 0.13471619689594796103e-3 * t41) ** (-0.657946)
  t49 = jnp.where(r0 <= p.dens_threshold, 0, -0.22554757207579166202e-4 * t5 * t26 * t28 * t41 * t45)
  t51 = jnp.where(t10, t15, -t17)
  t52 = jnp.where(t14, t11, t51)
  t53 = 0.1e1 + t52
  t55 = jnp.cbrt(t53)
  t57 = jnp.where(t53 <= p.zeta_threshold, t23, t55 * t53)
  t59 = jnp.sqrt(s2)
  t60 = jnp.cbrt(r1)
  t65 = (t34 * t59 / t60 / r1) ** 0.2626712e1
  t69 = (0.1e1 + 0.13471619689594796103e-3 * t65) ** (-0.657946)
  t73 = jnp.where(r1 <= p.dens_threshold, 0, -0.22554757207579166202e-4 * t5 * t57 * t28 * t65 * t69)
  res = t49 + t73
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = jnp.cbrt(3)
  t4 = jnp.cbrt(jnp.pi)
  t7 = 0.1e1 <= p.zeta_threshold
  t8 = p.zeta_threshold - 0.1e1
  t10 = jnp.where(t7, -t8, 0)
  t11 = jnp.where(t7, t8, t10)
  t12 = 0.1e1 + t11
  t14 = jnp.cbrt(p.zeta_threshold)
  t16 = jnp.cbrt(t12)
  t18 = jnp.where(t12 <= p.zeta_threshold, t14 * p.zeta_threshold, t16 * t12)
  t20 = jnp.cbrt(r0)
  t21 = jnp.cbrt(6)
  t22 = t21 ** 2
  t23 = jnp.pi ** 2
  t24 = jnp.cbrt(t23)
  t27 = jnp.sqrt(s0)
  t28 = jnp.cbrt(2)
  t34 = (t22 / t24 * t27 * t28 / t20 / r0) ** 0.2626712e1
  t38 = (0.1e1 + 0.13471619689594796103e-3 * t34) ** (-0.657946)
  t42 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.22554757207579166202e-4 * t3 / t4 * t18 * t20 * t34 * t38)
  res = 0.2e1 * t42
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