"""Generated from mgga_x_gdme.mpl."""

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
  t27 = jnp.cbrt(t6)
  t31 = jnp.cbrt(2)
  t34 = jnp.cbrt(0.1e1 / jnp.pi)
  t35 = 0.1e1 / t34
  t36 = jnp.cbrt(4)
  t38 = jnp.pi ** 2
  t39 = jnp.cbrt(t38)
  t40 = t39 ** 2
  t44 = 0.2e1 / 0.9e1 * (params.AA + 0.3e1 / 0.5e1 * params.BB) * t31 * t35 * t36 / t40
  t46 = params.BB * t2 * t35
  t47 = t31 ** 2
  t48 = t36 * t47
  t50 = 0.1e1 / t39 / t38
  t51 = params.a ** 2
  t52 = t51 - params.a + 0.1e1 / 0.2e1
  t54 = jnp.cbrt(r0)
  t55 = t54 ** 2
  t57 = 0.1e1 / t55 / r0
  t70 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (t44 + t46 * t48 * t50 * (t52 * l0 * t57 - 0.2e1 * tau0 * t57) / 0.27e2))
  t72 = jnp.where(t10, t15, -t17)
  t73 = jnp.where(t14, t11, t72)
  t74 = 0.1e1 + t73
  t76 = jnp.cbrt(t74)
  t78 = jnp.where(t74 <= p.zeta_threshold, t23, t76 * t74)
  t81 = jnp.cbrt(r1)
  t82 = t81 ** 2
  t84 = 0.1e1 / t82 / r1
  t97 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t78 * t27 * (t44 + t46 * t48 * t50 * (t52 * l1 * t84 - 0.2e1 * tau1 * t84) / 0.27e2))
  res = t70 + t97
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
  t19 = jnp.cbrt(r0)
  t23 = jnp.cbrt(2)
  t26 = jnp.cbrt(0.1e1 / jnp.pi)
  t27 = 0.1e1 / t26
  t28 = jnp.cbrt(4)
  t30 = jnp.pi ** 2
  t31 = jnp.cbrt(t30)
  t32 = t31 ** 2
  t39 = t23 ** 2
  t43 = params.a ** 2
  t46 = t19 ** 2
  t48 = 0.1e1 / t46 / r0
  t63 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.2e1 / 0.9e1 * (params.AA + 0.3e1 / 0.5e1 * params.BB) * t23 * t27 * t28 / t32 + params.BB * t3 * t27 * t28 * t39 / t31 / t30 * ((t43 - params.a + 0.1e1 / 0.2e1) * l0 * t39 * t48 - 0.2e1 * tau0 * t39 * t48) / 0.27e2))
  res = 0.2e1 * t63
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