"""Generated from mgga_x_rppscan.mpl."""

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
  t76 = 0.3e1 / 0.1e2 * t46 * t32
  t82 = (tau0 / t37 / r0 - t40 / 0.8e1) / (t76 + params.eta * s0 * t39 / 0.8e1)
  t83 = 0.1e1 - t82
  t85 = t83 ** 2
  t87 = jnp.exp(-t85 / 0.2e1)
  t91 = (0.7e1 / 0.1296e5 * t67 * t59 + t66 * t83 * t87 / 0.1e3) ** 2
  t99 = 0.25e1 < t82
  t100 = jnp.where(t99, 0.25e1, t82)
  t102 = t100 ** 2
  t104 = t102 * t100
  t106 = t102 ** 2
  t115 = jnp.where(t99, t82, 0.25e1)
  t119 = jnp.exp(params.c2 / (0.1e1 - t115))
  t121 = jnp.where(t82 <= 0.25e1, 0.1e1 - 0.667 * t100 - 0.4445555 * t102 - 0.663086601049 * t104 + 0.145129704449e1 * t106 - 0.887998041597 * t106 * t100 + 0.234528941479 * t106 * t102 - 0.23185843322e-1 * t106 * t104, -params.d * t119)
  t127 = jnp.sqrt(0.3e1)
  t129 = t46 / t31
  t130 = jnp.sqrt(s0)
  t135 = jnp.sqrt(t129 * t130 / t36 / r0)
  t139 = jnp.exp(-0.98958e1 * t127 / t135)
  t144 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * ((0.1e1 + params.k1 * (0.1e1 - params.k1 / (params.k1 + 0.5e1 / 0.972e3 * t34 * t40 + t50 * t51 / t36 / t52 / r0 * t62 / 0.576e3 + t91))) * (0.1e1 - t121) + 0.1174e1 * t121) * (0.1e1 - t139))
  t146 = jnp.where(t10, t15, -t17)
  t147 = jnp.where(t14, t11, t146)
  t148 = 0.1e1 + t147
  t150 = jnp.cbrt(t148)
  t152 = jnp.where(t148 <= p.zeta_threshold, t23, t150 * t148)
  t154 = r1 ** 2
  t155 = jnp.cbrt(r1)
  t156 = t155 ** 2
  t158 = 0.1e1 / t156 / t154
  t159 = s2 * t158
  t162 = s2 ** 2
  t163 = t154 ** 2
  t169 = t33 * s2 * t158
  t172 = jnp.exp(-0.27e2 / 0.8e2 * t57 * t169)
  t188 = (tau1 / t156 / r1 - t159 / 0.8e1) / (t76 + params.eta * s2 * t158 / 0.8e1)
  t189 = 0.1e1 - t188
  t191 = t189 ** 2
  t193 = jnp.exp(-t191 / 0.2e1)
  t197 = (0.7e1 / 0.1296e5 * t67 * t169 + t66 * t189 * t193 / 0.1e3) ** 2
  t205 = 0.25e1 < t188
  t206 = jnp.where(t205, 0.25e1, t188)
  t208 = t206 ** 2
  t210 = t208 * t206
  t212 = t208 ** 2
  t221 = jnp.where(t205, t188, 0.25e1)
  t225 = jnp.exp(params.c2 / (0.1e1 - t221))
  t227 = jnp.where(t188 <= 0.25e1, 0.1e1 - 0.667 * t206 - 0.4445555 * t208 - 0.663086601049 * t210 + 0.145129704449e1 * t212 - 0.887998041597 * t212 * t206 + 0.234528941479 * t212 * t208 - 0.23185843322e-1 * t212 * t210, -params.d * t225)
  t233 = jnp.sqrt(s2)
  t238 = jnp.sqrt(t129 * t233 / t155 / r1)
  t242 = jnp.exp(-0.98958e1 * t127 / t238)
  t247 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t152 * t28 * ((0.1e1 + params.k1 * (0.1e1 - params.k1 / (params.k1 + 0.5e1 / 0.972e3 * t34 * t159 + t50 * t162 / t155 / t163 / r1 * t172 / 0.576e3 + t197))) * (0.1e1 - t227) + 0.1174e1 * t227) * (0.1e1 - t242))
  res = t144 + t247
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
  t33 = 0.1e1 / t31 / t30
  t34 = s0 * t28 * t33
  t39 = 0.1e3 / 0.6561e4 / params.k1 - 0.73e2 / 0.648e3
  t40 = t21 ** 2
  t45 = s0 ** 2
  t47 = t30 ** 2
  t55 = jnp.exp(-0.27e2 / 0.8e2 * t39 * t21 * t25 * t34)
  t60 = jnp.sqrt(0.146e3)
  t79 = (tau0 * t28 / t31 / r0 - t34 / 0.8e1) / (0.3e1 / 0.1e2 * t40 * t24 + params.eta * s0 * t28 * t33 / 0.8e1)
  t80 = 0.1e1 - t79
  t82 = t80 ** 2
  t84 = jnp.exp(-t82 / 0.2e1)
  t88 = (0.7e1 / 0.1296e5 * t60 * t21 * t25 * t34 + t60 * t80 * t84 / 0.1e3) ** 2
  t96 = 0.25e1 < t79
  t97 = jnp.where(t96, 0.25e1, t79)
  t99 = t97 ** 2
  t101 = t99 * t97
  t103 = t99 ** 2
  t112 = jnp.where(t96, t79, 0.25e1)
  t116 = jnp.exp(params.c2 / (0.1e1 - t112))
  t118 = jnp.where(t79 <= 0.25e1, 0.1e1 - 0.667 * t97 - 0.4445555 * t99 - 0.663086601049 * t101 + 0.145129704449e1 * t103 - 0.887998041597 * t103 * t97 + 0.234528941479 * t103 * t99 - 0.23185843322e-1 * t103 * t101, -params.d * t116)
  t124 = jnp.sqrt(0.3e1)
  t127 = jnp.sqrt(s0)
  t133 = jnp.sqrt(t40 / t23 * t127 * t27 / t20 / r0)
  t137 = jnp.exp(-0.98958e1 * t124 / t133)
  t142 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * ((0.1e1 + params.k1 * (0.1e1 - params.k1 / (params.k1 + 0.5e1 / 0.972e3 * t21 * t25 * t34 + t39 * t40 / t23 / t22 * t45 * t27 / t20 / t47 / r0 * t55 / 0.288e3 + t88))) * (0.1e1 - t118) + 0.1174e1 * t118) * (0.1e1 - t137))
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