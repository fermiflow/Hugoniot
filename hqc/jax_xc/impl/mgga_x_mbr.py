"""Generated from mgga_x_mbr.mpl."""

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
  t26 = jnp.cbrt(0.1e1 / jnp.pi)
  t27 = 0.1e1 / t26
  t29 = jnp.cbrt(4)
  t30 = params.lambda_ ** 2
  t31 = t30 - params.lambda_ + 0.1e1 / 0.2e1
  t32 = jnp.cbrt(r0)
  t33 = t32 ** 2
  t37 = 0.2e1 * tau0 / t33 / r0
  t38 = jnp.cbrt(6)
  t39 = t38 ** 2
  t40 = jnp.pi ** 2
  t41 = jnp.cbrt(t40)
  t42 = t41 ** 2
  t43 = t39 * t42
  t44 = 0.3e1 / 0.5e1 * t43
  t45 = r0 ** 2
  t47 = 0.1e1 / t33 / t45
  t54 = (0.2e1 * params.lambda_ - 0.1e1) ** 2
  t55 = t54 * t38
  t56 = 0.1e1 / t42
  t61 = t54 ** 2
  t63 = params.beta * t61 * t39
  t65 = 0.1e1 / t41 / t40
  t66 = s0 ** 2
  t68 = t45 ** 2
  t76 = (0.1e1 + 0.175e3 / 0.162e3 * t55 * t56 * s0 * t47 + t63 * t65 * t66 / t32 / t68 / r0 / 0.576e3) ** (0.1e1 / 0.5e1)
  t86 = t31 * (t37 - t44 - s0 * t47 / 0.36e2) + t43 * (t76 - 0.1e1) / 0.5e1 - params.gamma * (t37 - t54 * s0 * t47 / 0.4e1) / 0.3e1
  t87 = jnp.abs(t86)
  t90 = jnp.where(0. < t86, 0.5e-12, -0.5e-12)
  t91 = jnp.where(t87 < 0.5e-12, t90, t86)
  t92 = br89_x(t91)
  t94 = jnp.exp(t92 / 0.3e1)
  t96 = jnp.exp(-t92)
  t106 = jnp.where(r0 <= p.dens_threshold, 0, -t22 * t23 * t27 * t29 * t94 * (0.1e1 - t96 * (0.1e1 + t92 / 0.2e1)) / t92 / 0.4e1)
  t108 = jnp.where(t6, t11, -t13)
  t109 = jnp.where(t10, t7, t108)
  t110 = 0.1e1 + t109
  t112 = jnp.cbrt(t110)
  t114 = jnp.where(t110 <= p.zeta_threshold, t19, t112 * t110)
  t117 = jnp.cbrt(r1)
  t118 = t117 ** 2
  t122 = 0.2e1 * tau1 / t118 / r1
  t123 = r1 ** 2
  t125 = 0.1e1 / t118 / t123
  t134 = s2 ** 2
  t136 = t123 ** 2
  t144 = (0.1e1 + 0.175e3 / 0.162e3 * t55 * t56 * s2 * t125 + t63 * t65 * t134 / t117 / t136 / r1 / 0.576e3) ** (0.1e1 / 0.5e1)
  t154 = t31 * (t122 - t44 - s2 * t125 / 0.36e2) + t43 * (t144 - 0.1e1) / 0.5e1 - params.gamma * (t122 - t54 * s2 * t125 / 0.4e1) / 0.3e1
  t155 = jnp.abs(t154)
  t158 = jnp.where(0. < t154, 0.5e-12, -0.5e-12)
  t159 = jnp.where(t155 < 0.5e-12, t158, t154)
  t160 = br89_x(t159)
  t162 = jnp.exp(t160 / 0.3e1)
  t164 = jnp.exp(-t160)
  t174 = jnp.where(r1 <= p.dens_threshold, 0, -t114 * t23 * t27 * t29 * t162 * (0.1e1 - t164 * (0.1e1 + t160 / 0.2e1)) / t160 / 0.4e1)
  res = t106 + t174
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
  t18 = jnp.cbrt(0.1e1 / jnp.pi)
  t21 = jnp.cbrt(4)
  t22 = params.lambda_ ** 2
  t24 = jnp.cbrt(2)
  t25 = t24 ** 2
  t27 = t15 ** 2
  t31 = 0.2e1 * tau0 * t25 / t27 / r0
  t32 = jnp.cbrt(6)
  t33 = t32 ** 2
  t34 = jnp.pi ** 2
  t35 = jnp.cbrt(t34)
  t36 = t35 ** 2
  t37 = t33 * t36
  t40 = r0 ** 2
  t42 = 0.1e1 / t27 / t40
  t43 = s0 * t25 * t42
  t49 = (0.2e1 * params.lambda_ - 0.1e1) ** 2
  t55 = t49 ** 2
  t60 = s0 ** 2
  t62 = t40 ** 2
  t71 = (0.1e1 + 0.175e3 / 0.162e3 * t49 * t32 / t36 * t43 + params.beta * t55 * t33 / t35 / t34 * t60 * t24 / t15 / t62 / r0 / 0.288e3) ** (0.1e1 / 0.5e1)
  t82 = (t22 - params.lambda_ + 0.1e1 / 0.2e1) * (t31 - 0.3e1 / 0.5e1 * t37 - t43 / 0.36e2) + t37 * (t71 - 0.1e1) / 0.5e1 - params.gamma * (t31 - t49 * s0 * t25 * t42 / 0.4e1) / 0.3e1
  t83 = jnp.abs(t82)
  t86 = jnp.where(0. < t82, 0.5e-12, -0.5e-12)
  t87 = jnp.where(t83 < 0.5e-12, t86, t82)
  t88 = br89_x(t87)
  t90 = jnp.exp(t88 / 0.3e1)
  t92 = jnp.exp(-t88)
  t102 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -t14 * t15 / t18 * t21 * t90 * (0.1e1 - t92 * (0.1e1 + t88 / 0.2e1)) / t88 / 0.4e1)
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