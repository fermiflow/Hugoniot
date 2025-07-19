"""Generated from mgga_x_rscan.mpl."""

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
  t70 = t20 ** 2
  t71 = t70 ** 2
  t73 = t6 ** 2
  t74 = t73 ** 2
  t75 = t74 * t6
  t81 = tau0 / t37 / r0 - t40 / 0.8e1
  t83 = jnp.where(0. < t81, t81, 0)
  t84 = t83 ** 2
  t86 = jnp.cbrt(2)
  t87 = t20 * t6
  t88 = jnp.cbrt(t87)
  t89 = t88 ** 2
  t92 = t46 * t32
  t95 = params.taur / 0.2e1
  t96 = 0.3e1 / 0.4e2 * t86 * t89 * t87 * t92 + t95
  t97 = t96 ** 2
  t101 = t86 ** 2
  t103 = t73 * t6
  t115 = t71 * t20 * t75 * t84 * t83 / t97 / t96 / (t101 * t88 * t70 * t20 * t103 * t84 / t97 / 0.16e2 + params.alphar) / 0.32e2
  t116 = 0.1e1 - t115
  t118 = t116 ** 2
  t120 = jnp.exp(-t118 / 0.2e1)
  t124 = (0.7e1 / 0.1296e5 * t67 * t59 + t66 * t116 * t120 / 0.1e3) ** 2
  t132 = 0.25e1 < t115
  t133 = jnp.where(t132, 0.25e1, t115)
  t135 = t133 ** 2
  t137 = t135 * t133
  t139 = t135 ** 2
  t148 = jnp.where(t132, t115, 0.25e1)
  t152 = jnp.exp(params.c2 / (0.1e1 - t148))
  t154 = jnp.where(t115 <= 0.25e1, 0.1e1 - 0.667 * t133 - 0.4445555 * t135 - 0.663086601049 * t137 + 0.145129704449e1 * t139 - 0.887998041597 * t139 * t133 + 0.234528941479 * t139 * t135 - 0.23185843322e-1 * t139 * t137, -params.d * t152)
  t160 = jnp.sqrt(0.3e1)
  t162 = t46 / t31
  t163 = jnp.sqrt(s0)
  t168 = jnp.sqrt(t162 * t163 / t36 / r0)
  t172 = jnp.exp(-0.98958e1 * t160 / t168)
  t177 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * ((0.1e1 + params.k1 * (0.1e1 - params.k1 / (params.k1 + 0.5e1 / 0.972e3 * t34 * t40 + t50 * t51 / t36 / t52 / r0 * t62 / 0.576e3 + t124))) * (0.1e1 - t154) + 0.1174e1 * t154) * (0.1e1 - t172))
  t179 = jnp.where(t10, t15, -t17)
  t180 = jnp.where(t14, t11, t179)
  t181 = 0.1e1 + t180
  t183 = jnp.cbrt(t181)
  t185 = jnp.where(t181 <= p.zeta_threshold, t23, t183 * t181)
  t187 = r1 ** 2
  t188 = jnp.cbrt(r1)
  t189 = t188 ** 2
  t191 = 0.1e1 / t189 / t187
  t192 = s2 * t191
  t195 = s2 ** 2
  t196 = t187 ** 2
  t202 = t33 * s2 * t191
  t205 = jnp.exp(-0.27e2 / 0.8e2 * t57 * t202)
  t211 = t181 ** 2
  t212 = t211 ** 2
  t219 = tau1 / t189 / r1 - t192 / 0.8e1
  t221 = jnp.where(0. < t219, t219, 0)
  t222 = t221 ** 2
  t224 = t181 * t6
  t225 = jnp.cbrt(t224)
  t226 = t225 ** 2
  t231 = 0.3e1 / 0.4e2 * t86 * t226 * t224 * t92 + t95
  t232 = t231 ** 2
  t248 = t212 * t181 * t75 * t222 * t221 / t232 / t231 / (t101 * t225 * t211 * t181 * t103 * t222 / t232 / 0.16e2 + params.alphar) / 0.32e2
  t249 = 0.1e1 - t248
  t251 = t249 ** 2
  t253 = jnp.exp(-t251 / 0.2e1)
  t257 = (0.7e1 / 0.1296e5 * t67 * t202 + t66 * t249 * t253 / 0.1e3) ** 2
  t265 = 0.25e1 < t248
  t266 = jnp.where(t265, 0.25e1, t248)
  t268 = t266 ** 2
  t270 = t268 * t266
  t272 = t268 ** 2
  t281 = jnp.where(t265, t248, 0.25e1)
  t285 = jnp.exp(params.c2 / (0.1e1 - t281))
  t287 = jnp.where(t248 <= 0.25e1, 0.1e1 - 0.667 * t266 - 0.4445555 * t268 - 0.663086601049 * t270 + 0.145129704449e1 * t272 - 0.887998041597 * t272 * t266 + 0.234528941479 * t272 * t268 - 0.23185843322e-1 * t272 * t270, -params.d * t285)
  t293 = jnp.sqrt(s2)
  t298 = jnp.sqrt(t162 * t293 / t188 / r1)
  t302 = jnp.exp(-0.98958e1 * t160 / t298)
  t307 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t185 * t28 * ((0.1e1 + params.k1 * (0.1e1 - params.k1 / (params.k1 + 0.5e1 / 0.972e3 * t34 * t192 + t50 * t195 / t188 / t196 / r1 * t205 / 0.576e3 + t257))) * (0.1e1 - t287) + 0.1174e1 * t287) * (0.1e1 - t302))
  res = t177 + t307
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
  t48 = t47 * r0
  t55 = jnp.exp(-0.27e2 / 0.8e2 * t39 * t21 * t25 * t34)
  t60 = jnp.sqrt(0.146e3)
  t65 = t12 ** 2
  t66 = t65 ** 2
  t74 = tau0 * t28 / t31 / r0 - t34 / 0.8e1
  t76 = jnp.where(0. < t74, t74, 0)
  t77 = t76 ** 2
  t79 = t12 * r0
  t80 = jnp.cbrt(t79)
  t81 = t80 ** 2
  t88 = 0.3e1 / 0.4e2 * t27 * t81 * t79 * t40 * t24 + params.taur / 0.2e1
  t89 = t88 ** 2
  t106 = t66 * t12 * t48 * t77 * t76 / t89 / t88 / (t28 * t80 * t65 * t12 * t30 * r0 * t77 / t89 / 0.16e2 + params.alphar) / 0.32e2
  t107 = 0.1e1 - t106
  t109 = t107 ** 2
  t111 = jnp.exp(-t109 / 0.2e1)
  t115 = (0.7e1 / 0.1296e5 * t60 * t21 * t25 * t34 + t60 * t107 * t111 / 0.1e3) ** 2
  t123 = 0.25e1 < t106
  t124 = jnp.where(t123, 0.25e1, t106)
  t126 = t124 ** 2
  t128 = t126 * t124
  t130 = t126 ** 2
  t139 = jnp.where(t123, t106, 0.25e1)
  t143 = jnp.exp(params.c2 / (0.1e1 - t139))
  t145 = jnp.where(t106 <= 0.25e1, 0.1e1 - 0.667 * t124 - 0.4445555 * t126 - 0.663086601049 * t128 + 0.145129704449e1 * t130 - 0.887998041597 * t130 * t124 + 0.234528941479 * t130 * t126 - 0.23185843322e-1 * t130 * t128, -params.d * t143)
  t151 = jnp.sqrt(0.3e1)
  t154 = jnp.sqrt(s0)
  t160 = jnp.sqrt(t40 / t23 * t154 * t27 / t20 / r0)
  t164 = jnp.exp(-0.98958e1 * t151 / t160)
  t169 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * ((0.1e1 + params.k1 * (0.1e1 - params.k1 / (params.k1 + 0.5e1 / 0.972e3 * t21 * t25 * t34 + t39 * t40 / t23 / t22 * t45 * t27 / t20 / t48 * t55 / 0.288e3 + t115))) * (0.1e1 - t145) + 0.1174e1 * t145) * (0.1e1 - t164))
  res = 0.2e1 * t169
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