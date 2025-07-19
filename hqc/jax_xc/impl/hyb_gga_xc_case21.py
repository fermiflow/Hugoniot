"""Generated from hyb_gga_xc_case21.mpl."""

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
  t3 = jnp.cbrt(3)
  t4 = jnp.cbrt(jnp.pi)
  t6 = t3 / t4
  t7 = r0 + r1
  t8 = 0.1e1 / t7
  t11 = 0.2e1 * r0 * t8 <= p.zeta_threshold
  t12 = p.zeta_threshold - 0.1e1
  t15 = 0.2e1 * r1 * t8 <= p.zeta_threshold
  t16 = -t12
  t17 = r0 - r1
  t18 = t17 * t8
  t19 = jnp.where(t15, t16, t18)
  t20 = jnp.where(t11, t12, t19)
  t21 = 0.1e1 + t20
  t23 = jnp.cbrt(p.zeta_threshold)
  t24 = t23 * p.zeta_threshold
  t25 = jnp.cbrt(t21)
  t27 = jnp.where(t21 <= p.zeta_threshold, t24, t25 * t21)
  t28 = jnp.cbrt(t7)
  t30 = jnp.cbrt(6)
  t31 = params.gammax * t30
  t32 = jnp.pi ** 2
  t33 = jnp.cbrt(t32)
  t34 = t33 ** 2
  t35 = 0.1e1 / t34
  t36 = t31 * t35
  t37 = r0 ** 2
  t38 = jnp.cbrt(r0)
  t39 = t38 ** 2
  t41 = 0.1e1 / t39 / t37
  t52 = xbspline(t36 * s0 * t41 / (0.1e1 + t31 * t35 * s0 * t41 / 0.24e2) / 0.24e2, 0, params)
  t56 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t6 * t27 * t28 * t52)
  t58 = jnp.where(t11, t16, -t18)
  t59 = jnp.where(t15, t12, t58)
  t60 = 0.1e1 + t59
  t62 = jnp.cbrt(t60)
  t64 = jnp.where(t60 <= p.zeta_threshold, t24, t62 * t60)
  t66 = r1 ** 2
  t67 = jnp.cbrt(r1)
  t68 = t67 ** 2
  t70 = 0.1e1 / t68 / t66
  t81 = xbspline(t36 * s2 * t70 / (0.1e1 + t31 * t35 * s2 * t70 / 0.24e2) / 0.24e2, 0, params)
  t85 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t6 * t64 * t28 * t81)
  t88 = t18 + 0.1e1
  t89 = t88 <= p.zeta_threshold
  t90 = t23 ** 2
  t91 = jnp.cbrt(t88)
  t92 = t91 ** 2
  t93 = jnp.where(t89, t90, t92)
  t94 = 0.1e1 - t18
  t95 = t94 <= p.zeta_threshold
  t96 = jnp.cbrt(t94)
  t97 = t96 ** 2
  t98 = jnp.where(t95, t90, t97)
  t101 = t3 ** 2
  t102 = (t93 / 0.2e1 + t98 / 0.2e1) * t101
  t104 = jnp.sqrt(s0)
  t105 = jnp.sqrt(s2)
  t107 = (t104 + t105) ** 2
  t108 = t7 ** 2
  t110 = 0.1e1 / t28 / t108
  t117 = jnp.cbrt(0.1e1 / jnp.pi)
  t119 = jnp.cbrt(4)
  t120 = t119 ** 2
  t123 = t3 * t117 * t120 / t28
  t126 = jnp.sqrt(t123)
  t129 = t123 ** 0.15e1
  t131 = t117 ** 2
  t133 = t28 ** 2
  t136 = t101 * t131 * t119 / t133
  t142 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t126 + 0.8969 * t123 + 0.204775 * t129 + 0.123235 * t136))
  t144 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t123) * t142
  t145 = t17 ** 2
  t146 = t145 ** 2
  t147 = t108 ** 2
  t151 = jnp.where(t89, t24, t91 * t88)
  t153 = jnp.where(t95, t24, t96 * t94)
  t155 = jnp.cbrt(2)
  t159 = (t151 + t153 - 0.2e1) / (0.2e1 * t155 - 0.2e1)
  t170 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t126 + 0.1549425e1 * t123 + 0.420775 * t129 + 0.1562925 * t136))
  t183 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t126 + 0.905775 * t123 + 0.1100325 * t129 + 0.1241775 * t136))
  t184 = (0.1e1 + 0.278125e-1 * t123) * t183
  t191 = -t144 + t146 / t147 * t159 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t123) * t170 + t144 - 0.19751673498613801407e-1 * t184) + 0.19751673498613801407e-1 * t159 * t184
  t198 = cbspline(-t102 * t4 * t107 * t110 / (-t102 * t4 * t107 * t110 / 0.48e2 + params.gammac * t191) / 0.48e2, 0, params)
  res = (0.1e1 - params.ax) * (t56 + t85) + t198 * t191
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t4 = jnp.cbrt(3)
  t5 = jnp.cbrt(jnp.pi)
  t8 = 0.1e1 <= p.zeta_threshold
  t9 = p.zeta_threshold - 0.1e1
  t11 = jnp.where(t8, -t9, 0)
  t12 = jnp.where(t8, t9, t11)
  t13 = 0.1e1 + t12
  t15 = jnp.cbrt(p.zeta_threshold)
  t16 = t15 * p.zeta_threshold
  t17 = jnp.cbrt(t13)
  t19 = jnp.where(t13 <= p.zeta_threshold, t16, t17 * t13)
  t20 = jnp.cbrt(r0)
  t22 = jnp.cbrt(6)
  t24 = jnp.pi ** 2
  t25 = jnp.cbrt(t24)
  t26 = t25 ** 2
  t28 = params.gammax * t22 / t26
  t29 = jnp.cbrt(2)
  t30 = t29 ** 2
  t31 = s0 * t30
  t32 = r0 ** 2
  t33 = t20 ** 2
  t35 = 0.1e1 / t33 / t32
  t45 = xbspline(t28 * t31 * t35 / (0.1e1 + t28 * t31 * t35 / 0.24e2) / 0.24e2, 0, params)
  t49 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t4 / t5 * t19 * t20 * t45)
  t52 = t15 ** 2
  t53 = jnp.where(t8, t52, 1)
  t54 = t4 ** 2
  t55 = t53 * t54
  t58 = 0.1e1 / t20 / t32
  t65 = jnp.cbrt(0.1e1 / jnp.pi)
  t67 = jnp.cbrt(4)
  t68 = t67 ** 2
  t71 = t4 * t65 * t68 / t20
  t74 = jnp.sqrt(t71)
  t77 = t71 ** 0.15e1
  t79 = t65 ** 2
  t83 = t54 * t79 * t67 / t33
  t89 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t74 + 0.8969 * t71 + 0.204775 * t77 + 0.123235 * t83))
  t92 = jnp.where(t8, t16, 1)
  t109 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t74 + 0.905775 * t71 + 0.1100325 * t77 + 0.1241775 * t83))
  t113 = -0.621814e-1 * (0.1e1 + 0.53425e-1 * t71) * t89 + 0.19751673498613801407e-1 * (0.2e1 * t92 - 0.2e1) / (0.2e1 * t29 - 0.2e1) * (0.1e1 + 0.278125e-1 * t71) * t109
  t120 = cbspline(-t55 * t5 * s0 * t58 / (-t55 * t5 * s0 * t58 / 0.48e2 + params.gammac * t113) / 0.48e2, 0, params)
  res = 0.2e1 * (0.1e1 - params.ax) * t49 + t120 * t113
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