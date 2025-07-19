"""Generated from mgga_x_ft98.mpl."""

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
  t37 = jnp.sqrt(params.a1 * s0 * t34 + 0.1e1)
  t42 = (params.b1 * s0 * t34 + 0.1e1) ** (0.1e1 / 0.4e1)
  t43 = t42 ** 2
  t49 = s0 * t34
  t54 = (t49 - l0 / t32 / r0) ** 2
  t57 = (0.1e1 + t49) ** 2
  t62 = params.b2 ** 2
  t64 = jnp.sqrt(t62 + 0.1e1)
  t65 = t64 - params.b2
  t66 = s0 ** 2
  t67 = t30 ** 2
  t71 = t66 / t31 / t67 / r0
  t72 = l0 ** 2
  t76 = t72 / t31 / t30 / r0
  t77 = t71 - t76 - params.b2
  t78 = 2.220446049250313e-16 ** (0.1e1 / 0.4e1)
  t79 = 0.1e1 / t78
  t83 = 0.2e1 * params.b2
  t89 = jnp.where(0. < t77, t77, -t77)
  t91 = t77 ** 2
  t93 = t91 ** 2
  t97 = jnp.where(-t79 < t77, t77, -t79)
  t98 = t97 ** 2
  t100 = jnp.sqrt(0.1e1 + t98)
  t103 = jnp.where(t89 < t78, 0.1e1 - t71 + t76 + params.b2 + t91 / 0.2e1 - t93 / 0.8e1, 0.1e1 / (t97 + t100))
  t104 = jnp.where(t77 < -t79, -0.2e1 * t71 + 0.2e1 * t76 + t83 - 0.1e1 / t77 / 0.2e1, t103)
  t107 = jnp.cbrt(2)
  t109 = (t107 - 0.1e1) * t65
  t111 = t109 * t104 + 0.1e1
  t112 = t111 ** 2
  t119 = t2 ** 2
  t121 = jnp.cbrt(0.1e1 / jnp.pi)
  t122 = t121 ** 2
  t124 = jnp.cbrt(4)
  t125 = t119 * t122 * t124
  t133 = jnp.sqrt((0.1e1 + params.a * t37 / t43 / t42 * s0 * t34 + params.b * (0.1e1 + params.a2 * t54 / t57) * (t65 * t104 + 0.1e1) / t112 / t111 * t54) / (0.1e1 + 0.81e2 / 0.4e1 * t125 * params.b * s0 * t34))
  t137 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * t133)
  t139 = jnp.where(t10, t15, -t17)
  t140 = jnp.where(t14, t11, t139)
  t141 = 0.1e1 + t140
  t143 = jnp.cbrt(t141)
  t145 = jnp.where(t141 <= p.zeta_threshold, t23, t143 * t141)
  t148 = r1 ** 2
  t149 = jnp.cbrt(r1)
  t150 = t149 ** 2
  t152 = 0.1e1 / t150 / t148
  t155 = jnp.sqrt(params.a1 * s2 * t152 + 0.1e1)
  t160 = (params.b1 * s2 * t152 + 0.1e1) ** (0.1e1 / 0.4e1)
  t161 = t160 ** 2
  t167 = s2 * t152
  t172 = (t167 - l1 / t150 / r1) ** 2
  t175 = (0.1e1 + t167) ** 2
  t180 = s2 ** 2
  t181 = t148 ** 2
  t185 = t180 / t149 / t181 / r1
  t186 = l1 ** 2
  t190 = t186 / t149 / t148 / r1
  t191 = t185 - t190 - params.b2
  t200 = jnp.where(0. < t191, t191, -t191)
  t202 = t191 ** 2
  t204 = t202 ** 2
  t208 = jnp.where(-t79 < t191, t191, -t79)
  t209 = t208 ** 2
  t211 = jnp.sqrt(0.1e1 + t209)
  t214 = jnp.where(t200 < t78, 0.1e1 - t185 + t190 + params.b2 + t202 / 0.2e1 - t204 / 0.8e1, 0.1e1 / (t208 + t211))
  t215 = jnp.where(t191 < -t79, -0.2e1 * t185 + 0.2e1 * t190 + t83 - 0.1e1 / t191 / 0.2e1, t214)
  t219 = t109 * t215 + 0.1e1
  t220 = t219 ** 2
  t234 = jnp.sqrt((0.1e1 + params.a * t155 / t161 / t160 * s2 * t152 + params.b * (0.1e1 + params.a2 * t172 / t175) * (t65 * t215 + 0.1e1) / t220 / t219 * t172) / (0.1e1 + 0.81e2 / 0.4e1 * t125 * params.b * s2 * t152))
  t238 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t145 * t27 * t234)
  res = t137 + t238
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
  t22 = jnp.cbrt(2)
  t23 = t22 ** 2
  t24 = r0 ** 2
  t25 = t19 ** 2
  t27 = 0.1e1 / t25 / t24
  t28 = t23 * t27
  t31 = jnp.sqrt(params.a1 * s0 * t28 + 0.1e1)
  t36 = (params.b1 * s0 * t28 + 0.1e1) ** (0.1e1 / 0.4e1)
  t37 = t36 ** 2
  t42 = s0 * t23 * t27
  t49 = (t42 - l0 * t23 / t25 / r0) ** 2
  t52 = (0.1e1 + t42) ** 2
  t57 = params.b2 ** 2
  t59 = jnp.sqrt(t57 + 0.1e1)
  t60 = t59 - params.b2
  t61 = s0 ** 2
  t63 = t24 ** 2
  t67 = t61 * t22 / t19 / t63 / r0
  t68 = 0.2e1 * t67
  t69 = l0 ** 2
  t74 = t69 * t22 / t19 / t24 / r0
  t75 = 0.2e1 * t74
  t76 = t68 - t75 - params.b2
  t77 = 2.220446049250313e-16 ** (0.1e1 / 0.4e1)
  t78 = 0.1e1 / t77
  t88 = jnp.where(0. < t76, t76, -t76)
  t90 = t76 ** 2
  t92 = t90 ** 2
  t96 = jnp.where(-t78 < t76, t76, -t78)
  t97 = t96 ** 2
  t99 = jnp.sqrt(0.1e1 + t97)
  t102 = jnp.where(t88 < t77, 0.1e1 - t68 + t75 + params.b2 + t90 / 0.2e1 - t92 / 0.8e1, 0.1e1 / (t96 + t99))
  t103 = jnp.where(t76 < -t78, -0.4e1 * t67 + 0.4e1 * t74 + 0.2e1 * params.b2 - 0.1e1 / t76 / 0.2e1, t102)
  t109 = 0.1e1 + (t22 - 0.1e1) * t60 * t103
  t110 = t109 ** 2
  t117 = t3 ** 2
  t119 = jnp.cbrt(0.1e1 / jnp.pi)
  t120 = t119 ** 2
  t122 = jnp.cbrt(4)
  t131 = jnp.sqrt((0.1e1 + params.a * t31 / t37 / t36 * t42 + params.b * (0.1e1 + params.a2 * t49 / t52) * (t60 * t103 + 0.1e1) / t110 / t109 * t49) / (0.1e1 + 0.81e2 / 0.4e1 * t117 * t120 * t122 * params.b * s0 * t28))
  t135 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * t131)
  res = 0.2e1 * t135
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