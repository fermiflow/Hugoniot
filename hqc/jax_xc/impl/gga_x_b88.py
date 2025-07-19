"""Generated from gga_x_b88.mpl."""

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
  t29 = t2 ** 2
  t32 = jnp.cbrt(0.1e1 / jnp.pi)
  t34 = params.beta * t29 / t32
  t35 = jnp.cbrt(4)
  t37 = r0 ** 2
  t38 = jnp.cbrt(r0)
  t39 = t38 ** 2
  t42 = params.gamma * params.beta
  t43 = jnp.sqrt(s0)
  t46 = t43 / t38 / r0
  t47 = jnp.arcsinh(t46)
  t60 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + 0.2e1 / 0.9e1 * t34 * t35 * s0 / t39 / t37 / (t42 * t46 * t47 + 0.1e1)))
  t62 = jnp.where(t10, t15, -t17)
  t63 = jnp.where(t14, t11, t62)
  t64 = 0.1e1 + t63
  t66 = jnp.cbrt(t64)
  t68 = jnp.where(t64 <= p.zeta_threshold, t23, t66 * t64)
  t71 = r1 ** 2
  t72 = jnp.cbrt(r1)
  t73 = t72 ** 2
  t76 = jnp.sqrt(s2)
  t79 = t76 / t72 / r1
  t80 = jnp.arcsinh(t79)
  t93 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t68 * t27 * (0.1e1 + 0.2e1 / 0.9e1 * t34 * t35 * s2 / t73 / t71 / (t42 * t79 * t80 + 0.1e1)))
  res = t60 + t93
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
  t21 = t3 ** 2
  t24 = jnp.cbrt(0.1e1 / jnp.pi)
  t26 = jnp.cbrt(4)
  t29 = jnp.cbrt(2)
  t30 = t29 ** 2
  t32 = r0 ** 2
  t33 = t19 ** 2
  t37 = jnp.sqrt(s0)
  t40 = 0.1e1 / t19 / r0
  t44 = jnp.arcsinh(t37 * t29 * t40)
  t57 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + 0.2e1 / 0.9e1 * params.beta * t21 / t24 * t26 * s0 * t30 / t33 / t32 / (params.gamma * params.beta * t37 * t29 * t40 * t44 + 0.1e1)))
  res = 0.2e1 * t57
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