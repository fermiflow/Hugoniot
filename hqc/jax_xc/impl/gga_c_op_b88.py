"""Generated from gga_c_op_b88.mpl."""

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
  t4 = (r0 - r1) * t3
  t5 = jnp.abs(t4)
  t10 = jnp.logical_and(r0 <= p.dens_threshold, r1 <= p.dens_threshold)
  t11 = jnp.logical_or(0.1e1 - t5 <= p.zeta_threshold, t10)
  t14 = p.zeta_threshold - 0.1e1
  t17 = -t14
  t18 = jnp.where(0.1e1 - t4 <= p.zeta_threshold, t17, t4)
  t19 = jnp.where(0.1e1 + t4 <= p.zeta_threshold, t14, t18)
  t20 = t19 ** 2
  t29 = jnp.where(0.2e1 * r1 * t3 <= p.zeta_threshold, t17, t4)
  t30 = jnp.where(0.2e1 * r0 * t3 <= p.zeta_threshold, t14, t29)
  t31 = 0.1e1 + t30
  t35 = jnp.cbrt(3)
  t36 = t35 ** 2
  t38 = jnp.cbrt(0.1e1 / jnp.pi)
  t41 = jnp.cbrt(4)
  t42 = t36 / t38 * t41
  t43 = jnp.cbrt(2)
  t44 = t31 <= p.zeta_threshold
  t45 = 0.1e1 - t30
  t46 = t45 <= p.zeta_threshold
  t47 = jnp.where(t46, t17, t30)
  t48 = jnp.where(t44, t14, t47)
  t51 = jnp.cbrt((0.1e1 + t48) * t2)
  t54 = r0 ** 2
  t55 = jnp.cbrt(r0)
  t56 = t55 ** 2
  t60 = jnp.sqrt(s0)
  t63 = t60 / t55 / r0
  t64 = jnp.arcsinh(t63)
  t77 = jnp.where(t31 * t2 / 0.2e1 <= p.dens_threshold, 0, t42 * t43 / t51 / (0.1e1 + 0.93333333333333333332e-3 * t42 * s0 / t56 / t54 / (0.1e1 + 0.252e-1 * t63 * t64)) / 0.9e1)
  t82 = jnp.where(t44, t17, -t30)
  t83 = jnp.where(t46, t14, t82)
  t86 = jnp.cbrt((0.1e1 + t83) * t2)
  t89 = r1 ** 2
  t90 = jnp.cbrt(r1)
  t91 = t90 ** 2
  t95 = jnp.sqrt(s2)
  t98 = t95 / t90 / r1
  t99 = jnp.arcsinh(t98)
  t112 = jnp.where(t45 * t2 / 0.2e1 <= p.dens_threshold, 0, t42 * t43 / t86 / (0.1e1 + 0.93333333333333333332e-3 * t42 * s2 / t91 / t89 / (0.1e1 + 0.252e-1 * t98 * t99)) / 0.9e1)
  t113 = t77 + t112
  t115 = jnp.where(t113 == 0., 2.220446049250313e-16, t113)
  t119 = t115 ** 2
  t120 = t119 ** 2
  res = jnp.where(t11, 0, -0.25 * (0.1e1 - t20) * t2 * (0.36011538e1 / t115 + 0.5764) / (0.31390124030721e2 / t120 + 0.149643497914092e2 / t119 / t115 + 0.17833359087e1 / t119))
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = 0.1e1 <= p.zeta_threshold
  t3 = r0 / 0.2e1 <= p.dens_threshold
  t4 = jnp.logical_and(t3, t3)
  t5 = jnp.logical_or(t1, t4)
  t6 = p.zeta_threshold - 0.1e1
  t7 = -t6
  t8 = jnp.where(t1, t7, 0)
  t9 = jnp.where(t1, t6, t8)
  t10 = t9 ** 2
  t13 = 0.1e1 + t9
  t17 = jnp.cbrt(3)
  t18 = t17 ** 2
  t20 = jnp.cbrt(0.1e1 / jnp.pi)
  t23 = jnp.cbrt(4)
  t24 = t18 / t20 * t23
  t25 = jnp.cbrt(2)
  t26 = t13 <= p.zeta_threshold
  t27 = 0.1e1 - t9
  t28 = t27 <= p.zeta_threshold
  t29 = jnp.where(t28, t7, t9)
  t30 = jnp.where(t26, t6, t29)
  t33 = jnp.cbrt((0.1e1 + t30) * r0)
  t36 = t25 ** 2
  t38 = r0 ** 2
  t39 = jnp.cbrt(r0)
  t40 = t39 ** 2
  t43 = jnp.sqrt(s0)
  t44 = t43 * t25
  t46 = 0.1e1 / t39 / r0
  t48 = jnp.arcsinh(t44 * t46)
  t59 = 0.1e1 / (0.1e1 + 0.93333333333333333332e-3 * t24 * s0 * t36 / t40 / t38 / (0.1e1 + 0.252e-1 * t44 * t46 * t48))
  t63 = jnp.where(t13 * r0 / 0.2e1 <= p.dens_threshold, 0, t24 * t25 / t33 * t59 / 0.9e1)
  t68 = jnp.where(t26, t7, -t9)
  t69 = jnp.where(t28, t6, t68)
  t72 = jnp.cbrt((0.1e1 + t69) * r0)
  t78 = jnp.where(t27 * r0 / 0.2e1 <= p.dens_threshold, 0, t24 * t25 / t72 * t59 / 0.9e1)
  t79 = t63 + t78
  t81 = jnp.where(t79 == 0., 2.220446049250313e-16, t79)
  t85 = t81 ** 2
  t86 = t85 ** 2
  res = jnp.where(t5, 0, -0.25 * (0.1e1 - t10) * r0 * (0.36011538e1 / t81 + 0.5764) / (0.31390124030721e2 / t86 + 0.149643497914092e2 / t85 / t81 + 0.17833359087e1 / t85))
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