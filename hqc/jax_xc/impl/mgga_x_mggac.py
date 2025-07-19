"""Generated from mgga_x_mggac.mpl."""

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
  t2 = r0 + r1
  t3 = 0.1e1 / t2
  t6 = 0.2e1 * r0 * t3 <= p.zeta_threshold
  t7 = p.zeta_threshold - 0.1e1
  t10 = 0.2e1 * r1 * t3 <= p.zeta_threshold
  t11 = -t7
  t13 = (r0 - r1) * t3
  t14 = jnp.where(t10, t11, t13)
  t15 = jnp.where(t6, t7, t14)
  t16 = 0.1e1 + t15
  t18 = jnp.cbrt(p.zeta_threshold)
  t19 = t18 * p.zeta_threshold
  t20 = jnp.cbrt(t16)
  t22 = jnp.where(t16 <= p.zeta_threshold, t19, t20 * t16)
  t23 = jnp.cbrt(t2)
  t25 = jnp.cbrt(32)
  t27 = jnp.cbrt(0.1e1 / jnp.pi)
  t29 = t25 / t27
  t31 = jnp.cbrt(4)
  t32 = t25 ** 2
  t33 = jnp.cbrt(jnp.pi)
  t34 = t33 ** 2
  t35 = t32 * t34
  t36 = jnp.cbrt(r0)
  t37 = t36 ** 2
  t41 = jnp.cbrt(6)
  t42 = jnp.pi ** 2
  t43 = jnp.cbrt(t42)
  t44 = t43 ** 2
  t46 = t41 / t44
  t51 = s0 / r0 / tau0 / 0.8e1
  t55 = jnp.where(0. < 0.9999999999 - t51, 0.1e1 - t51, 0.1e-9)
  t57 = tau0 / t37 / r0 * t46 * t55
  t59 = tau0 ** 2
  t60 = r0 ** 2
  t65 = t41 ** 2
  t68 = t65 / t43 / t42
  t69 = t55 ** 2
  t71 = t59 / t36 / t60 / r0 * t68 * t69
  t78 = (0.1e1 + 0.55555555555555555556e-1 * t57 - 0.34860833333333333333e1 * t71) / (0.3712e1 + 0.11111111111111111111e1 * t57 + 0.11620277777777777778e1 * t71)
  t79 = jnp.abs(t78)
  t84 = t35 * t78 / 0.6e1
  t86 = jnp.where(0. < t84, 0.5e-12, -0.5e-12)
  t87 = jnp.where(t35 * t79 / 0.6e1 < 0.5e-12, t86, t84)
  t88 = mbrxc_x(t87)
  t90 = jnp.exp(t88 / 0.3e1)
  t92 = jnp.exp(-t88)
  t93 = t88 ** 2
  t101 = jnp.cbrt(0.1e1 + t88)
  t107 = jnp.where(r0 <= p.dens_threshold, 0, -t22 * t23 * t29 * t31 * t90 * (0.8e1 - t92 * (t93 + 0.5e1 * t88 + 0.8e1)) / t88 / t101 / 0.64e2)
  t109 = jnp.where(t6, t11, -t13)
  t110 = jnp.where(t10, t7, t109)
  t111 = 0.1e1 + t110
  t113 = jnp.cbrt(t111)
  t115 = jnp.where(t111 <= p.zeta_threshold, t19, t113 * t111)
  t118 = jnp.cbrt(r1)
  t119 = t118 ** 2
  t127 = s2 / r1 / tau1 / 0.8e1
  t131 = jnp.where(0. < 0.9999999999 - t127, 0.1e1 - t127, 0.1e-9)
  t133 = tau1 / t119 / r1 * t46 * t131
  t135 = tau1 ** 2
  t136 = r1 ** 2
  t141 = t131 ** 2
  t143 = t135 / t118 / t136 / r1 * t68 * t141
  t150 = (0.1e1 + 0.55555555555555555556e-1 * t133 - 0.34860833333333333333e1 * t143) / (0.3712e1 + 0.11111111111111111111e1 * t133 + 0.11620277777777777778e1 * t143)
  t151 = jnp.abs(t150)
  t156 = t35 * t150 / 0.6e1
  t158 = jnp.where(0. < t156, 0.5e-12, -0.5e-12)
  t159 = jnp.where(t35 * t151 / 0.6e1 < 0.5e-12, t158, t156)
  t160 = mbrxc_x(t159)
  t162 = jnp.exp(t160 / 0.3e1)
  t164 = jnp.exp(-t160)
  t165 = t160 ** 2
  t173 = jnp.cbrt(0.1e1 + t160)
  t179 = jnp.where(r1 <= p.dens_threshold, 0, -t115 * t23 * t29 * t31 * t162 * (0.8e1 - t164 * (t165 + 0.5e1 * t160 + 0.8e1)) / t160 / t173 / 0.64e2)
  res = t107 + t179
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = 0.1e1 <= p.zeta_threshold
  t4 = p.zeta_threshold - 0.1e1
  t6 = jnp.where(t3, -t4, 0)
  t7 = jnp.where(t3, t4, t6)
  t8 = 0.1e1 + t7
  t10 = jnp.cbrt(p.zeta_threshold)
  t12 = jnp.cbrt(t8)
  t14 = jnp.where(t8 <= p.zeta_threshold, t10 * p.zeta_threshold, t12 * t8)
  t15 = jnp.cbrt(r0)
  t17 = jnp.cbrt(32)
  t19 = jnp.cbrt(0.1e1 / jnp.pi)
  t23 = jnp.cbrt(4)
  t24 = t17 ** 2
  t25 = jnp.cbrt(jnp.pi)
  t26 = t25 ** 2
  t27 = t24 * t26
  t28 = jnp.cbrt(2)
  t29 = t28 ** 2
  t31 = t15 ** 2
  t35 = jnp.cbrt(6)
  t36 = jnp.pi ** 2
  t37 = jnp.cbrt(t36)
  t38 = t37 ** 2
  t45 = s0 / r0 / tau0 / 0.8e1
  t49 = jnp.where(0. < 0.9999999999 - t45, 0.1e1 - t45, 0.1e-9)
  t51 = tau0 * t29 / t31 / r0 * t35 / t38 * t49
  t53 = tau0 ** 2
  t55 = r0 ** 2
  t60 = t35 ** 2
  t64 = t49 ** 2
  t66 = t53 * t28 / t15 / t55 / r0 * t60 / t37 / t36 * t64
  t73 = (0.1e1 + 0.55555555555555555556e-1 * t51 - 0.69721666666666666666e1 * t66) / (0.3712e1 + 0.11111111111111111111e1 * t51 + 0.23240555555555555556e1 * t66)
  t74 = jnp.abs(t73)
  t79 = t27 * t73 / 0.6e1
  t81 = jnp.where(0. < t79, 0.5e-12, -0.5e-12)
  t82 = jnp.where(t27 * t74 / 0.6e1 < 0.5e-12, t81, t79)
  t83 = mbrxc_x(t82)
  t85 = jnp.exp(t83 / 0.3e1)
  t87 = jnp.exp(-t83)
  t88 = t83 ** 2
  t96 = jnp.cbrt(0.1e1 + t83)
  t102 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -t14 * t15 * t17 / t19 * t23 * t85 * (0.8e1 - t87 * (t88 + 0.5e1 * t83 + 0.8e1)) / t83 / t96 / 0.64e2)
  res = 0.2e1 * t102
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