"""Generated from mgga_k_csk_loc.mpl."""

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
  t3 = t2 ** 2
  t4 = jnp.cbrt(jnp.pi)
  t6 = t3 * t4 * jnp.pi
  t7 = r0 + r1
  t8 = 0.1e1 / t7
  t11 = 0.2e1 * r0 * t8 <= p.zeta_threshold
  t12 = p.zeta_threshold - 0.1e1
  t15 = 0.2e1 * r1 * t8 <= p.zeta_threshold
  t16 = -t12
  t18 = (r0 - r1) * t8
  t19 = jnp.where(t15, t16, t18)
  t20 = jnp.where(t11, t12, t19)
  t21 = 0.1e1 + t20
  t23 = jnp.cbrt(p.zeta_threshold)
  t24 = t23 ** 2
  t25 = t24 * p.zeta_threshold
  t26 = jnp.cbrt(t21)
  t27 = t26 ** 2
  t29 = jnp.where(t21 <= p.zeta_threshold, t25, t27 * t21)
  t30 = jnp.cbrt(t7)
  t31 = t30 ** 2
  t33 = jnp.cbrt(6)
  t34 = jnp.pi ** 2
  t35 = jnp.cbrt(t34)
  t36 = t35 ** 2
  t37 = 0.1e1 / t36
  t38 = t33 * t37
  t39 = r0 ** 2
  t40 = jnp.cbrt(r0)
  t41 = t40 ** 2
  t43 = 0.1e1 / t41 / t39
  t46 = 0.5e1 / 0.72e2 * t38 * s0 * t43
  t47 = params.csk_cp * t33
  t52 = params.csk_cq * t33
  t59 = t47 * t37 * s0 * t43 / 0.24e2 + t52 * t37 * l0 / t41 / r0 / 0.24e2 - t46
  t61 = jnp.log(0.1e1 - 2.220446049250313e-16)
  t62 = 0.1e1 / params.csk_a
  t63 = (-t61) ** (-t62)
  t65 = jnp.log(2.220446049250313e-16)
  t66 = (-t65) ** (-t62)
  t67 = -t66 < t59
  t68 = jnp.where(t67, -t66, t59)
  t70 = jnp.where(-t63 < t68, t68, -t63)
  t71 = jnp.abs(t70)
  t72 = t71 ** params.csk_a
  t74 = jnp.exp(-0.1e1 / t72)
  t76 = (0.1e1 - t74) ** t62
  t77 = jnp.where(t67, 1, t76)
  t78 = jnp.where(t59 < -t63, 0, t77)
  t84 = jnp.where(r0 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t29 * t31 * (t59 * t78 + t46 + 0.1e1))
  t86 = jnp.where(t11, t16, -t18)
  t87 = jnp.where(t15, t12, t86)
  t88 = 0.1e1 + t87
  t90 = jnp.cbrt(t88)
  t91 = t90 ** 2
  t93 = jnp.where(t88 <= p.zeta_threshold, t25, t91 * t88)
  t95 = r1 ** 2
  t96 = jnp.cbrt(r1)
  t97 = t96 ** 2
  t99 = 0.1e1 / t97 / t95
  t102 = 0.5e1 / 0.72e2 * t38 * s2 * t99
  t113 = t47 * t37 * s2 * t99 / 0.24e2 + t52 * t37 * l1 / t97 / r1 / 0.24e2 - t102
  t115 = -t66 < t113
  t116 = jnp.where(t115, -t66, t113)
  t118 = jnp.where(-t63 < t116, t116, -t63)
  t119 = jnp.abs(t118)
  t120 = t119 ** params.csk_a
  t122 = jnp.exp(-0.1e1 / t120)
  t124 = (0.1e1 - t122) ** t62
  t125 = jnp.where(t115, 1, t124)
  t126 = jnp.where(t113 < -t63, 0, t125)
  t132 = jnp.where(r1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t93 * t31 * (t113 * t126 + t102 + 0.1e1))
  res = t84 + t132
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = jnp.cbrt(3)
  t4 = t3 ** 2
  t5 = jnp.cbrt(jnp.pi)
  t8 = 0.1e1 <= p.zeta_threshold
  t9 = p.zeta_threshold - 0.1e1
  t11 = jnp.where(t8, -t9, 0)
  t12 = jnp.where(t8, t9, t11)
  t13 = 0.1e1 + t12
  t15 = jnp.cbrt(p.zeta_threshold)
  t16 = t15 ** 2
  t18 = jnp.cbrt(t13)
  t19 = t18 ** 2
  t21 = jnp.where(t13 <= p.zeta_threshold, t16 * p.zeta_threshold, t19 * t13)
  t22 = jnp.cbrt(r0)
  t23 = t22 ** 2
  t25 = jnp.cbrt(6)
  t26 = jnp.pi ** 2
  t27 = jnp.cbrt(t26)
  t28 = t27 ** 2
  t29 = 0.1e1 / t28
  t31 = jnp.cbrt(2)
  t32 = t31 ** 2
  t34 = r0 ** 2
  t37 = s0 * t32 / t23 / t34
  t39 = 0.5e1 / 0.72e2 * t25 * t29 * t37
  t52 = params.csk_cp * t25 * t29 * t37 / 0.24e2 + params.csk_cq * t25 * t29 * l0 * t32 / t23 / r0 / 0.24e2 - t39
  t54 = jnp.log(0.1e1 - 2.220446049250313e-16)
  t55 = 0.1e1 / params.csk_a
  t56 = (-t54) ** (-t55)
  t58 = jnp.log(2.220446049250313e-16)
  t59 = (-t58) ** (-t55)
  t60 = -t59 < t52
  t61 = jnp.where(t60, -t59, t52)
  t63 = jnp.where(-t56 < t61, t61, -t56)
  t64 = jnp.abs(t63)
  t65 = t64 ** params.csk_a
  t67 = jnp.exp(-0.1e1 / t65)
  t69 = (0.1e1 - t67) ** t55
  t70 = jnp.where(t60, 1, t69)
  t71 = jnp.where(t52 < -t56, 0, t70)
  t77 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t4 * t5 * jnp.pi * t21 * t23 * (t52 * t71 + t39 + 0.1e1))
  res = 0.2e1 * t77
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