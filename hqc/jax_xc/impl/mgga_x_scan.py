"""Generated from mgga_x_scan.mpl."""

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
  t28 = jnp.cbrt(t6)
  t29 = jnp.cbrt(6)
  t30 = jnp.pi ** 2
  t31 = jnp.cbrt(t30)
  t32 = t31 ** 2
  t33 = 0.1e1 / t32
  t34 = t29 * t33
  t35 = r0 ** 2
  t36 = jnp.cbrt(r0)
  t37 = t36 ** 2
  t39 = 0.1e1 / t37 / t35
  t40 = s0 * t39
  t45 = 0.1e3 / 0.6561e4 / params.k1 - 0.73e2 / 0.648e3
  t46 = t29 ** 2
  t50 = t45 * t46 / t31 / t30
  t51 = s0 ** 2
  t52 = t35 ** 2
  t57 = t45 * t29
  t59 = t33 * s0 * t39
  t62 = jnp.exp(-0.27e2 / 0.8e2 * t57 * t59)
  t66 = jnp.sqrt(0.146e3)
  t67 = t66 * t29
  t77 = 0.5e1 / 0.9e1 * (tau0 / t37 / r0 - t40 / 0.8e1) * t29 * t33
  t78 = 0.1e1 - t77
  t80 = t78 ** 2
  t82 = jnp.exp(-t80 / 0.2e1)
  t86 = (0.7e1 / 0.1296e5 * t67 * t59 + t66 * t78 * t82 / 0.1e3) ** 2
  t94 = jnp.log(2.220446049250313e-16)
  t97 = t94 / (-t94 + params.c1)
  t100 = jnp.where(t77 < -t97, t77, -t97)
  t105 = jnp.exp(-params.c1 * t100 / (0.1e1 - t100))
  t106 = jnp.where(-t97 < t77, 0, t105)
  t107 = jnp.abs(params.d)
  t110 = jnp.log(2.220446049250313e-16 / t107)
  t113 = (-t110 + params.c2) / t110
  t114 = t77 < -t113
  t115 = jnp.where(t114, -t113, t77)
  t119 = jnp.exp(params.c2 / (0.1e1 - t115))
  t121 = jnp.where(t114, 0, -params.d * t119)
  t122 = jnp.where(t77 <= 0.1e1, t106, t121)
  t128 = jnp.sqrt(0.3e1)
  t130 = t46 / t31
  t131 = jnp.sqrt(s0)
  t136 = jnp.sqrt(t130 * t131 / t36 / r0)
  t140 = jnp.exp(-0.98958e1 * t128 / t136)
  t145 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * ((0.1e1 + params.k1 * (0.1e1 - params.k1 / (params.k1 + 0.5e1 / 0.972e3 * t34 * t40 + t50 * t51 / t36 / t52 / r0 * t62 / 0.576e3 + t86))) * (0.1e1 - t122) + 0.1174e1 * t122) * (0.1e1 - t140))
  t147 = jnp.where(t10, t15, -t17)
  t148 = jnp.where(t14, t11, t147)
  t149 = 0.1e1 + t148
  t151 = jnp.cbrt(t149)
  t153 = jnp.where(t149 <= p.zeta_threshold, t23, t151 * t149)
  t155 = r1 ** 2
  t156 = jnp.cbrt(r1)
  t157 = t156 ** 2
  t159 = 0.1e1 / t157 / t155
  t160 = s2 * t159
  t163 = s2 ** 2
  t164 = t155 ** 2
  t170 = t33 * s2 * t159
  t173 = jnp.exp(-0.27e2 / 0.8e2 * t57 * t170)
  t186 = 0.5e1 / 0.9e1 * (tau1 / t157 / r1 - t160 / 0.8e1) * t29 * t33
  t187 = 0.1e1 - t186
  t189 = t187 ** 2
  t191 = jnp.exp(-t189 / 0.2e1)
  t195 = (0.7e1 / 0.1296e5 * t67 * t170 + t66 * t187 * t191 / 0.1e3) ** 2
  t205 = jnp.where(t186 < -t97, t186, -t97)
  t210 = jnp.exp(-params.c1 * t205 / (0.1e1 - t205))
  t211 = jnp.where(-t97 < t186, 0, t210)
  t212 = t186 < -t113
  t213 = jnp.where(t212, -t113, t186)
  t217 = jnp.exp(params.c2 / (0.1e1 - t213))
  t219 = jnp.where(t212, 0, -params.d * t217)
  t220 = jnp.where(t186 <= 0.1e1, t211, t219)
  t226 = jnp.sqrt(s2)
  t231 = jnp.sqrt(t130 * t226 / t156 / r1)
  t235 = jnp.exp(-0.98958e1 * t128 / t231)
  t240 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t153 * t28 * ((0.1e1 + params.k1 * (0.1e1 - params.k1 / (params.k1 + 0.5e1 / 0.972e3 * t34 * t160 + t50 * t163 / t156 / t164 / r1 * t173 / 0.576e3 + t195))) * (0.1e1 - t220) + 0.1174e1 * t220) * (0.1e1 - t235))
  res = t145 + t240
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
  t20 = jnp.cbrt(r0)
  t21 = jnp.cbrt(6)
  t22 = jnp.pi ** 2
  t23 = jnp.cbrt(t22)
  t24 = t23 ** 2
  t25 = 0.1e1 / t24
  t27 = jnp.cbrt(2)
  t28 = t27 ** 2
  t30 = r0 ** 2
  t31 = t20 ** 2
  t34 = s0 * t28 / t31 / t30
  t39 = 0.1e3 / 0.6561e4 / params.k1 - 0.73e2 / 0.648e3
  t40 = t21 ** 2
  t45 = s0 ** 2
  t47 = t30 ** 2
  t55 = jnp.exp(-0.27e2 / 0.8e2 * t39 * t21 * t25 * t34)
  t60 = jnp.sqrt(0.146e3)
  t73 = 0.5e1 / 0.9e1 * (tau0 * t28 / t31 / r0 - t34 / 0.8e1) * t21 * t25
  t74 = 0.1e1 - t73
  t76 = t74 ** 2
  t78 = jnp.exp(-t76 / 0.2e1)
  t82 = (0.7e1 / 0.1296e5 * t60 * t21 * t25 * t34 + t60 * t74 * t78 / 0.1e3) ** 2
  t90 = jnp.log(2.220446049250313e-16)
  t93 = t90 / (-t90 + params.c1)
  t96 = jnp.where(t73 < -t93, t73, -t93)
  t101 = jnp.exp(-params.c1 * t96 / (0.1e1 - t96))
  t102 = jnp.where(-t93 < t73, 0, t101)
  t103 = jnp.abs(params.d)
  t106 = jnp.log(2.220446049250313e-16 / t103)
  t109 = (-t106 + params.c2) / t106
  t110 = t73 < -t109
  t111 = jnp.where(t110, -t109, t73)
  t115 = jnp.exp(params.c2 / (0.1e1 - t111))
  t117 = jnp.where(t110, 0, -params.d * t115)
  t118 = jnp.where(t73 <= 0.1e1, t102, t117)
  t124 = jnp.sqrt(0.3e1)
  t127 = jnp.sqrt(s0)
  t133 = jnp.sqrt(t40 / t23 * t127 * t27 / t20 / r0)
  t137 = jnp.exp(-0.98958e1 * t124 / t133)
  t142 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * ((0.1e1 + params.k1 * (0.1e1 - params.k1 / (params.k1 + 0.5e1 / 0.972e3 * t21 * t25 * t34 + t39 * t40 / t23 / t22 * t45 * t27 / t20 / t47 / r0 * t55 / 0.288e3 + t82))) * (0.1e1 - t118) + 0.1174e1 * t118) * (0.1e1 - t137))
  res = 0.2e1 * t142
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