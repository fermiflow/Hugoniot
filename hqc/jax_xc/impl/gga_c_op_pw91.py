"""Generated from gga_c_op_pw91.mpl."""

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
  t54 = jnp.cbrt(6)
  t55 = jnp.pi ** 2
  t56 = jnp.cbrt(t55)
  t57 = t56 ** 2
  t58 = 0.1e1 / t57
  t59 = t54 * t58
  t60 = r0 ** 2
  t61 = jnp.cbrt(r0)
  t62 = t61 ** 2
  t64 = 0.1e1 / t62 / t60
  t68 = jnp.exp(-0.25e2 / 0.6e1 * t59 * s0 * t64)
  t76 = t54 ** 2
  t79 = t76 / t56 / t55
  t80 = s0 ** 2
  t81 = t60 ** 2
  t87 = 0.69444444444444444444e-5 * t79 * t80 / t61 / t81 / r0
  t90 = t76 / t56
  t91 = jnp.sqrt(s0)
  t94 = t91 / t61 / r0
  t97 = jnp.arcsinh(0.64963333333333333333 * t90 * t94)
  t109 = jnp.where(t31 * t2 / 0.2e1 <= p.dens_threshold, 0, t42 * t43 / t51 / (0.1e1 + ((0.2743 - 0.1508 * t68) * t54 * t58 * s0 * t64 / 0.24e2 - t87) / (0.1e1 + 0.16370833333333333333e-1 * t90 * t94 * t97 + t87)) / 0.9e1)
  t114 = jnp.where(t44, t17, -t30)
  t115 = jnp.where(t46, t14, t114)
  t118 = jnp.cbrt((0.1e1 + t115) * t2)
  t121 = r1 ** 2
  t122 = jnp.cbrt(r1)
  t123 = t122 ** 2
  t125 = 0.1e1 / t123 / t121
  t129 = jnp.exp(-0.25e2 / 0.6e1 * t59 * s2 * t125)
  t137 = s2 ** 2
  t138 = t121 ** 2
  t144 = 0.69444444444444444444e-5 * t79 * t137 / t122 / t138 / r1
  t146 = jnp.sqrt(s2)
  t149 = t146 / t122 / r1
  t152 = jnp.arcsinh(0.64963333333333333333 * t90 * t149)
  t164 = jnp.where(t45 * t2 / 0.2e1 <= p.dens_threshold, 0, t42 * t43 / t118 / (0.1e1 + ((0.2743 - 0.1508 * t129) * t54 * t58 * s2 * t125 / 0.24e2 - t144) / (0.1e1 + 0.16370833333333333333e-1 * t90 * t149 * t152 + t144)) / 0.9e1)
  t165 = t109 + t164
  t167 = jnp.where(t165 == 0., 2.220446049250313e-16, t165)
  t171 = t167 ** 2
  t172 = t171 ** 2
  res = jnp.where(t11, 0, -0.25 * (0.1e1 - t20) * t2 * (0.360663084e1 / t167 + 0.5764) / (0.315815266717518096e2 / t172 + 0.150327320916243744e2 / t171 / t167 + 0.1788764629788e1 / t171))
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
  t36 = jnp.cbrt(6)
  t37 = jnp.pi ** 2
  t38 = jnp.cbrt(t37)
  t39 = t38 ** 2
  t40 = 0.1e1 / t39
  t42 = t25 ** 2
  t44 = r0 ** 2
  t45 = jnp.cbrt(r0)
  t46 = t45 ** 2
  t49 = s0 * t42 / t46 / t44
  t52 = jnp.exp(-0.25e2 / 0.6e1 * t36 * t40 * t49)
  t59 = t36 ** 2
  t63 = s0 ** 2
  t65 = t44 ** 2
  t71 = 0.13888888888888888889e-4 * t59 / t38 / t37 * t63 * t25 / t45 / t65 / r0
  t74 = t59 / t38
  t75 = jnp.sqrt(s0)
  t78 = 0.1e1 / t45 / r0
  t84 = jnp.arcsinh(0.64963333333333333333 * t74 * t75 * t25 * t78)
  t92 = 0.1e1 / (0.1e1 + ((0.2743 - 0.1508 * t52) * t36 * t40 * t49 / 0.24e2 - t71) / (0.1e1 + 0.16370833333333333333e-1 * t74 * t75 * t25 * t78 * t84 + t71))
  t96 = jnp.where(t13 * r0 / 0.2e1 <= p.dens_threshold, 0, t24 * t25 / t33 * t92 / 0.9e1)
  t101 = jnp.where(t26, t7, -t9)
  t102 = jnp.where(t28, t6, t101)
  t105 = jnp.cbrt((0.1e1 + t102) * r0)
  t111 = jnp.where(t27 * r0 / 0.2e1 <= p.dens_threshold, 0, t24 * t25 / t105 * t92 / 0.9e1)
  t112 = t96 + t111
  t114 = jnp.where(t112 == 0., 2.220446049250313e-16, t112)
  t118 = t114 ** 2
  t119 = t118 ** 2
  res = jnp.where(t5, 0, -0.25 * (0.1e1 - t10) * r0 * (0.360663084e1 / t114 + 0.5764) / (0.315815266717518096e2 / t119 + 0.150327320916243744e2 / t118 / t114 + 0.1788764629788e1 / t118))
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