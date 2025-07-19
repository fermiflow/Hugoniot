"""Generated from gga_xc_th1.mpl."""

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
  t2 = r0 ** (0.1e1 / 0.6e1)
  t4 = r1 ** (0.1e1 / 0.6e1)
  t6 = t2 * r0 + t4 * r1
  t9 = jnp.cbrt(r0)
  t10 = t9 * r0
  t11 = jnp.cbrt(r1)
  t12 = t11 * r1
  t13 = t10 + t12
  t16 = jnp.sqrt(r0)
  t18 = jnp.sqrt(r1)
  t20 = t16 * r0 + t18 * r1
  t23 = t9 ** 2
  t25 = t11 ** 2
  t27 = t23 * r0 + t25 * r1
  t31 = jnp.sqrt(s0)
  t34 = r0 - r1
  t35 = r0 + r1
  t36 = 0.1e1 / t35
  t37 = t34 * t36
  t38 = 0.1e1 + t37
  t40 = jnp.cbrt(p.zeta_threshold)
  t41 = t40 * p.zeta_threshold
  t42 = jnp.cbrt(t38)
  t44 = jnp.where(t38 <= p.zeta_threshold, t41, t42 * t38)
  t45 = jnp.cbrt(2)
  t46 = t45 ** 2
  t49 = jnp.sqrt(s2)
  t52 = 0.1e1 - t37
  t54 = jnp.cbrt(t52)
  t56 = jnp.where(t52 <= p.zeta_threshold, t41, t54 * t52)
  t60 = t31 / t10 * t44 * t46 / 0.4e1 + t49 / t12 * t56 * t46 / 0.4e1
  t72 = t2 ** 2
  t73 = t72 ** 2
  t76 = t4 ** 2
  t77 = t76 ** 2
  t80 = t73 * t2 * r0 + t77 * t4 * r1
  t86 = r0 ** 2
  t90 = t44 ** 2
  t92 = s0 / t23 / t86 * t90 * t45
  t93 = r1 ** 2
  t97 = t56 ** 2
  t99 = s2 / t25 / t93 * t97 * t45
  t101 = t92 / 0.8e1 + t99 / 0.8e1
  t114 = t86 + t93
  t124 = t35 ** 2
  t125 = jnp.cbrt(t35)
  t126 = t125 ** 2
  t130 = t92 / 0.4e1 + t99 / 0.4e1 - (s0 + 0.2e1 * s1 + s2) / t126 / t124
  t143 = t34 ** 2
  t145 = t143 / t124
  t158 = params.omega[10] * t80 * t101 / 0.2e1 + params.omega[11] * t114 * t101 / 0.2e1 + params.omega[12] * t20 * t130 + params.omega[13] * t27 * t130 + params.omega[14] * t80 * t130 + params.omega[15] * t114 * t130 + params.omega[16] * t6 * t145 + params.omega[17] * t13 * t145 + params.omega[18] * t20 * t145 + params.omega[19] * t27 * t145 + params.omega[20] * t35
  res = (params.omega[0] * t6 + params.omega[1] * t13 + params.omega[2] * t20 + params.omega[3] * t27 + params.omega[4] * t13 * t60 / 0.2e1 + params.omega[5] * t20 * t60 / 0.2e1 + params.omega[6] * t27 * t60 / 0.2e1 + params.omega[7] * t80 * t60 / 0.2e1 + params.omega[8] * t20 * t101 / 0.2e1 + params.omega[9] * t27 * t101 / 0.2e1 + t158) * t36
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t2 = 2 ** (0.1e1 / 0.6e1)
  t3 = t2 ** 2
  t4 = t3 ** 2
  t7 = r0 ** (0.1e1 / 0.6e1)
  t8 = t7 * r0
  t12 = jnp.cbrt(2)
  t13 = t12 ** 2
  t15 = jnp.cbrt(r0)
  t20 = jnp.sqrt(0.2e1)
  t22 = jnp.sqrt(r0)
  t23 = t22 * r0
  t28 = t15 ** 2
  t29 = t28 * r0
  t34 = jnp.sqrt(s0)
  t36 = jnp.cbrt(p.zeta_threshold)
  t38 = jnp.where(0.1e1 <= p.zeta_threshold, t36 * p.zeta_threshold, 1)
  t64 = t38 ** 2
  t70 = 0.1e1 / r0
  t77 = t7 ** 2
  t78 = t77 ** 2
  t79 = t78 * t7
  t93 = r0 ** 2
  t96 = s0 / t28 / t93
  t98 = t96 * t64 - t96
  t119 = params.omega[0] * t4 * t2 * t8 / 0.2e1 + params.omega[1] * t13 * t15 * r0 / 0.2e1 + params.omega[2] * t20 * t23 / 0.2e1 + params.omega[3] * t12 * t29 / 0.2e1 + params.omega[4] * t13 * t34 * t38 / 0.4e1 + params.omega[5] * t20 * t7 * t34 * t38 / 0.4e1 + params.omega[6] * t12 * t15 * t34 * t38 / 0.4e1 + params.omega[7] * t2 * t22 * t34 * t38 / 0.4e1 + params.omega[8] * t20 / t8 * s0 * t64 / 0.8e1 + params.omega[9] * t12 * t70 * s0 * t64 / 0.8e1 + params.omega[10] * t2 / t79 * s0 * t64 / 0.8e1 + params.omega[11] / t28 * s0 * t64 / 0.8e1 + params.omega[12] * t20 * t23 * t98 / 0.2e1 + params.omega[13] * t12 * t29 * t98 / 0.2e1 + params.omega[14] * t2 * t79 * r0 * t98 / 0.2e1 + params.omega[15] * t93 * t98 / 0.2e1 + params.omega[20] * r0
  res = t119 * t70
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