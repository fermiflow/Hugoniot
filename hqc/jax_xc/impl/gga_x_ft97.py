"""Generated from gga_x_ft97.mpl."""

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
  t30 = r0 ** 2
  t31 = jnp.cbrt(r0)
  t32 = t31 ** 2
  t34 = 0.1e1 / t32 / t30
  t35 = jnp.cbrt(2)
  t38 = t20 ** 2
  t39 = t6 ** 2
  t40 = t38 * t39
  t42 = jnp.cbrt(t20 * t6)
  t43 = t42 ** 2
  t44 = s0 * t34
  t55 = params.beta0 + params.beta1 * s0 * t34 * t35 * t40 * t43 / (params.beta2 + t44 * t35 * t40 * t43 / 0.8e1) / 0.8e1
  t58 = t2 ** 2
  t60 = jnp.cbrt(0.1e1 / jnp.pi)
  t62 = t58 / t60
  t63 = jnp.cbrt(4)
  t64 = t55 ** 2
  t65 = jnp.arcsinh(t44)
  t66 = t65 ** 2
  t71 = jnp.sqrt(0.9e1 * t44 * t64 * t66 + 0.1e1)
  t81 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + 0.2e1 / 0.9e1 * t55 * s0 * t34 * t62 * t63 / t71))
  t83 = jnp.where(t10, t15, -t17)
  t84 = jnp.where(t14, t11, t83)
  t85 = 0.1e1 + t84
  t87 = jnp.cbrt(t85)
  t89 = jnp.where(t85 <= p.zeta_threshold, t23, t87 * t85)
  t92 = r1 ** 2
  t93 = jnp.cbrt(r1)
  t94 = t93 ** 2
  t96 = 0.1e1 / t94 / t92
  t99 = t85 ** 2
  t100 = t99 * t39
  t102 = jnp.cbrt(t85 * t6)
  t103 = t102 ** 2
  t104 = s2 * t96
  t115 = params.beta0 + params.beta1 * s2 * t96 * t35 * t100 * t103 / (params.beta2 + t104 * t35 * t100 * t103 / 0.8e1) / 0.8e1
  t118 = t115 ** 2
  t119 = jnp.arcsinh(t104)
  t120 = t119 ** 2
  t125 = jnp.sqrt(0.9e1 * t104 * t118 * t120 + 0.1e1)
  t135 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t89 * t27 * (0.1e1 + 0.2e1 / 0.9e1 * t115 * s2 * t96 * t62 * t63 / t125))
  res = t81 + t135
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
  t22 = t19 ** 2
  t23 = 0.1e1 / t22
  t25 = t12 ** 2
  t27 = jnp.cbrt(t12 * r0)
  t28 = t27 ** 2
  t29 = t25 * t28
  t38 = params.beta0 + params.beta1 * s0 * t23 * t29 / (params.beta2 + s0 * t23 * t29 / 0.4e1) / 0.4e1
  t40 = jnp.cbrt(2)
  t41 = t40 ** 2
  t42 = r0 ** 2
  t44 = 0.1e1 / t22 / t42
  t47 = t3 ** 2
  t49 = jnp.cbrt(0.1e1 / jnp.pi)
  t52 = jnp.cbrt(4)
  t53 = s0 * t41
  t54 = t38 ** 2
  t57 = jnp.arcsinh(t53 * t44)
  t58 = t57 ** 2
  t63 = jnp.sqrt(0.9e1 * t53 * t44 * t54 * t58 + 0.1e1)
  t73 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + 0.2e1 / 0.9e1 * t38 * s0 * t41 * t44 * t47 / t49 * t52 / t63))
  res = 0.2e1 * t73
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