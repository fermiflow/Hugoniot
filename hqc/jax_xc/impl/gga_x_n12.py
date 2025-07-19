"""Generated from gga_x_n12.mpl."""

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
  t29 = params.CC[0][0]
  t30 = params.CC[0][1]
  t32 = r0 ** 2
  t33 = jnp.cbrt(r0)
  t34 = t33 ** 2
  t36 = 0.1e1 / t34 / t32
  t39 = 0.1e1 + 0.4e-2 * s0 * t36
  t41 = t36 / t39
  t44 = params.CC[0][2]
  t45 = s0 ** 2
  t47 = t32 ** 2
  t51 = t39 ** 2
  t53 = 0.1e1 / t33 / t47 / r0 / t51
  t56 = params.CC[0][3]
  t57 = t45 * s0
  t59 = t47 ** 2
  t63 = 0.1e1 / t59 / t51 / t39
  t66 = params.CC[1][0]
  t67 = params.CC[1][1]
  t71 = params.CC[1][2]
  t75 = params.CC[1][3]
  t81 = jnp.cbrt(2)
  t82 = 0.1e1 / t27 * t81
  t84 = 0.1e1 + t17 <= p.zeta_threshold
  t86 = 0.1e1 - t17 <= p.zeta_threshold
  t87 = jnp.where(t86, t15, t17)
  t88 = jnp.where(t84, t11, t87)
  t89 = 0.1e1 + t88
  t91 = 0.1e1 / t22
  t92 = jnp.cbrt(t89)
  t94 = jnp.where(t89 <= p.zeta_threshold, t91, 0.1e1 / t92)
  t97 = 0.1e1 + 0.39999999999999999998 * t82 * t94
  t100 = params.CC[2][0]
  t101 = params.CC[2][1]
  t105 = params.CC[2][2]
  t109 = params.CC[2][3]
  t114 = t97 ** 2
  t117 = params.CC[3][0]
  t118 = params.CC[3][1]
  t122 = params.CC[3][2]
  t126 = params.CC[3][3]
  t138 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (t29 + 0.4e-2 * t30 * s0 * t41 + 0.16e-4 * t44 * t45 * t53 + 0.64e-7 * t56 * t57 * t63 + (t66 + 0.4e-2 * t67 * s0 * t41 + 0.16e-4 * t71 * t45 * t53 + 0.64e-7 * t75 * t57 * t63) / t97 + (t100 + 0.4e-2 * t101 * s0 * t41 + 0.16e-4 * t105 * t45 * t53 + 0.64e-7 * t109 * t57 * t63) / t114 + (t117 + 0.4e-2 * t118 * s0 * t41 + 0.16e-4 * t122 * t45 * t53 + 0.64e-7 * t126 * t57 * t63) / t114 / t97))
  t140 = jnp.where(t10, t15, -t17)
  t141 = jnp.where(t14, t11, t140)
  t142 = 0.1e1 + t141
  t144 = jnp.cbrt(t142)
  t146 = jnp.where(t142 <= p.zeta_threshold, t23, t144 * t142)
  t149 = r1 ** 2
  t150 = jnp.cbrt(r1)
  t151 = t150 ** 2
  t153 = 0.1e1 / t151 / t149
  t156 = 0.1e1 + 0.4e-2 * s2 * t153
  t158 = t153 / t156
  t161 = s2 ** 2
  t163 = t149 ** 2
  t167 = t156 ** 2
  t169 = 0.1e1 / t150 / t163 / r1 / t167
  t172 = t161 * s2
  t174 = t163 ** 2
  t178 = 0.1e1 / t174 / t167 / t156
  t191 = jnp.where(t84, t15, -t17)
  t192 = jnp.where(t86, t11, t191)
  t193 = 0.1e1 + t192
  t195 = jnp.cbrt(t193)
  t197 = jnp.where(t193 <= p.zeta_threshold, t91, 0.1e1 / t195)
  t200 = 0.1e1 + 0.39999999999999999998 * t82 * t197
  t213 = t200 ** 2
  t233 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t146 * t27 * (t29 + 0.4e-2 * t30 * s2 * t158 + 0.16e-4 * t44 * t161 * t169 + 0.64e-7 * t56 * t172 * t178 + (t66 + 0.4e-2 * t67 * s2 * t158 + 0.16e-4 * t71 * t161 * t169 + 0.64e-7 * t75 * t172 * t178) / t200 + (t100 + 0.4e-2 * t101 * s2 * t158 + 0.16e-4 * t105 * t161 * t169 + 0.64e-7 * t109 * t172 * t178) / t213 + (t117 + 0.4e-2 * t118 * s2 * t158 + 0.16e-4 * t122 * t161 * t169 + 0.64e-7 * t126 * t172 * t178) / t213 / t200))
  res = t138 + t233
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
  t13 = t12 <= p.zeta_threshold
  t14 = jnp.cbrt(p.zeta_threshold)
  t16 = jnp.cbrt(t12)
  t18 = jnp.where(t13, t14 * p.zeta_threshold, t16 * t12)
  t19 = jnp.cbrt(r0)
  t24 = jnp.cbrt(2)
  t25 = t24 ** 2
  t26 = r0 ** 2
  t27 = t19 ** 2
  t29 = 0.1e1 / t27 / t26
  t34 = 0.1e1 + 0.4e-2 * s0 * t25 * t29
  t36 = t25 * t29 / t34
  t40 = s0 ** 2
  t42 = t26 ** 2
  t47 = t34 ** 2
  t49 = t24 / t19 / t42 / r0 / t47
  t53 = t40 * s0
  t55 = t42 ** 2
  t59 = 0.1e1 / t55 / t47 / t34
  t80 = jnp.where(t13, 0.1e1 / t14, 0.1e1 / t16)
  t83 = 0.1e1 + 0.39999999999999999998 / t19 * t24 * t80
  t100 = t83 ** 2
  t124 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (params.CC[0][0] + 0.4e-2 * params.CC[0][1] * s0 * t36 + 0.32e-4 * params.CC[0][2] * t40 * t49 + 0.256e-6 * params.CC[0][3] * t53 * t59 + (params.CC[1][0] + 0.4e-2 * params.CC[1][1] * s0 * t36 + 0.32e-4 * params.CC[1][2] * t40 * t49 + 0.256e-6 * params.CC[1][3] * t53 * t59) / t83 + (params.CC[2][0] + 0.4e-2 * params.CC[2][1] * s0 * t36 + 0.32e-4 * params.CC[2][2] * t40 * t49 + 0.256e-6 * params.CC[2][3] * t53 * t59) / t100 + (params.CC[3][0] + 0.4e-2 * params.CC[3][1] * s0 * t36 + 0.32e-4 * params.CC[3][2] * t40 * t49 + 0.256e-6 * params.CC[3][3] * t53 * t59) / t100 / t83))
  res = 0.2e1 * t124
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