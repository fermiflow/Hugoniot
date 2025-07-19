"""Generated from mgga_c_m08.mpl."""

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
  t3 = jnp.cbrt(6)
  t4 = t3 ** 2
  t5 = jnp.pi ** 2
  t6 = jnp.cbrt(t5)
  t7 = t6 ** 2
  t9 = 0.3e1 / 0.1e2 * t4 * t7
  t10 = jnp.cbrt(2)
  t11 = t10 ** 2
  t12 = jnp.cbrt(r0)
  t13 = t12 ** 2
  t17 = r0 - r1
  t18 = r0 + r1
  t20 = t17 / t18
  t21 = 0.1e1 + t20
  t22 = t21 / 0.2e1
  t23 = jnp.cbrt(t22)
  t24 = t23 ** 2
  t27 = jnp.cbrt(r1)
  t28 = t27 ** 2
  t32 = 0.1e1 - t20
  t33 = t32 / 0.2e1
  t34 = jnp.cbrt(t33)
  t35 = t34 ** 2
  t39 = t11 * (tau0 / t13 / r0 * t24 * t22 + tau1 / t28 / r1 * t35 * t33)
  t40 = t9 - t39
  t42 = t9 + t39
  t43 = 0.1e1 / t42
  t46 = t40 ** 2
  t48 = t42 ** 2
  t49 = 0.1e1 / t48
  t52 = t46 * t40
  t54 = t48 * t42
  t55 = 0.1e1 / t54
  t58 = t46 ** 2
  t60 = t48 ** 2
  t61 = 0.1e1 / t60
  t64 = t58 * t40
  t67 = 0.1e1 / t60 / t42
  t70 = t58 * t46
  t73 = 0.1e1 / t60 / t48
  t76 = t58 * t52
  t79 = 0.1e1 / t60 / t54
  t82 = t58 ** 2
  t84 = t60 ** 2
  t85 = 0.1e1 / t84
  t88 = t82 * t40
  t91 = 0.1e1 / t84 / t42
  t94 = t82 * t46
  t97 = 0.1e1 / t84 / t48
  t100 = t82 * t52
  t103 = 0.1e1 / t84 / t54
  t105 = params.m08_a[0] + params.m08_a[1] * t40 * t43 + params.m08_a[2] * t46 * t49 + params.m08_a[3] * t52 * t55 + params.m08_a[4] * t58 * t61 + params.m08_a[5] * t64 * t67 + params.m08_a[6] * t70 * t73 + params.m08_a[7] * t76 * t79 + params.m08_a[8] * t82 * t85 + params.m08_a[9] * t88 * t91 + params.m08_a[10] * t94 * t97 + params.m08_a[11] * t100 * t103
  t106 = jnp.cbrt(3)
  t108 = jnp.cbrt(0.1e1 / jnp.pi)
  t110 = jnp.cbrt(4)
  t111 = t110 ** 2
  t112 = jnp.cbrt(t18)
  t115 = t106 * t108 * t111 / t112
  t118 = jnp.sqrt(t115)
  t121 = t115 ** 0.15e1
  t123 = t106 ** 2
  t124 = t108 ** 2
  t126 = t112 ** 2
  t129 = t123 * t124 * t110 / t126
  t135 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t118 + 0.8969 * t115 + 0.204775 * t121 + 0.123235 * t129))
  t137 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t115) * t135
  t138 = t17 ** 2
  t139 = t138 ** 2
  t140 = t18 ** 2
  t141 = t140 ** 2
  t144 = t21 <= p.zeta_threshold
  t145 = jnp.cbrt(p.zeta_threshold)
  t146 = t145 * p.zeta_threshold
  t147 = jnp.cbrt(t21)
  t149 = jnp.where(t144, t146, t147 * t21)
  t150 = t32 <= p.zeta_threshold
  t151 = jnp.cbrt(t32)
  t153 = jnp.where(t150, t146, t151 * t32)
  t158 = (t149 + t153 - 0.2e1) / (0.2e1 * t10 - 0.2e1)
  t169 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t118 + 0.1549425e1 * t115 + 0.420775 * t121 + 0.1562925 * t129))
  t182 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t118 + 0.905775 * t115 + 0.1100325 * t121 + 0.1241775 * t129))
  t183 = (0.1e1 + 0.278125e-1 * t115) * t182
  t190 = -t137 + t139 / t141 * t158 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t115) * t169 + t137 - 0.19751673498613801407e-1 * t183) + 0.19751673498613801407e-1 * t158 * t183
  t226 = params.m08_b[0] + params.m08_b[1] * t40 * t43 + params.m08_b[2] * t46 * t49 + params.m08_b[3] * t52 * t55 + params.m08_b[4] * t58 * t61 + params.m08_b[5] * t64 * t67 + params.m08_b[6] * t70 * t73 + params.m08_b[7] * t76 * t79 + params.m08_b[8] * t82 * t85 + params.m08_b[9] * t88 * t91 + params.m08_b[10] * t94 * t97 + params.m08_b[11] * t100 * t103
  t227 = jnp.log(0.2e1)
  t228 = 0.1e1 - t227
  t231 = t145 ** 2
  t232 = t147 ** 2
  t233 = jnp.where(t144, t231, t232)
  t234 = t151 ** 2
  t235 = jnp.where(t150, t231, t234)
  t237 = t233 / 0.2e1 + t235 / 0.2e1
  t238 = t237 ** 2
  t239 = t238 * t237
  t242 = s0 + 0.2e1 * s1 + s2
  t254 = 0.1e1 / t228
  t255 = t254 * t5
  t260 = jnp.exp(-t190 * t254 * t5 / t239)
  t262 = 0.1e1 / (t260 - 0.1e1)
  t263 = t242 ** 2
  t269 = t238 ** 2
  t278 = t242 / t112 / t140 * t10 / t238 * t123 / t108 * t110 / 0.96e2 + 0.21720231316129303386e-4 * t255 * t262 * t263 / t126 / t141 * t11 / t269 * t106 / t124 * t111
  t289 = jnp.log(0.1e1 + 0.6672455060314922e-1 * t278 * t254 * t5 / (0.1e1 + 0.6672455060314922e-1 * t255 * t262 * t278))
  res = t105 * t190 + t226 * t228 / t5 * t239 * t289
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = jnp.cbrt(6)
  t4 = t3 ** 2
  t5 = jnp.pi ** 2
  t6 = jnp.cbrt(t5)
  t7 = t6 ** 2
  t9 = 0.3e1 / 0.1e2 * t4 * t7
  t10 = jnp.cbrt(2)
  t11 = t10 ** 2
  t13 = jnp.cbrt(r0)
  t14 = t13 ** 2
  t17 = tau0 * t11 / t14 / r0
  t18 = t9 - t17
  t20 = t9 + t17
  t21 = 0.1e1 / t20
  t24 = t18 ** 2
  t26 = t20 ** 2
  t27 = 0.1e1 / t26
  t30 = t24 * t18
  t32 = t26 * t20
  t33 = 0.1e1 / t32
  t36 = t24 ** 2
  t38 = t26 ** 2
  t39 = 0.1e1 / t38
  t42 = t36 * t18
  t45 = 0.1e1 / t38 / t20
  t48 = t36 * t24
  t51 = 0.1e1 / t38 / t26
  t54 = t36 * t30
  t57 = 0.1e1 / t38 / t32
  t60 = t36 ** 2
  t62 = t38 ** 2
  t63 = 0.1e1 / t62
  t66 = t60 * t18
  t69 = 0.1e1 / t62 / t20
  t72 = t60 * t24
  t75 = 0.1e1 / t62 / t26
  t78 = t60 * t30
  t81 = 0.1e1 / t62 / t32
  t83 = params.m08_a[0] + params.m08_a[1] * t18 * t21 + params.m08_a[2] * t24 * t27 + params.m08_a[3] * t30 * t33 + params.m08_a[4] * t36 * t39 + params.m08_a[5] * t42 * t45 + params.m08_a[6] * t48 * t51 + params.m08_a[7] * t54 * t57 + params.m08_a[8] * t60 * t63 + params.m08_a[9] * t66 * t69 + params.m08_a[10] * t72 * t75 + params.m08_a[11] * t78 * t81
  t84 = jnp.cbrt(3)
  t86 = jnp.cbrt(0.1e1 / jnp.pi)
  t88 = jnp.cbrt(4)
  t89 = t88 ** 2
  t92 = t84 * t86 * t89 / t13
  t95 = jnp.sqrt(t92)
  t98 = t92 ** 0.15e1
  t100 = t84 ** 2
  t101 = t86 ** 2
  t105 = t100 * t101 * t88 / t14
  t111 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t95 + 0.8969 * t92 + 0.204775 * t98 + 0.123235 * t105))
  t114 = 0.1e1 <= p.zeta_threshold
  t115 = jnp.cbrt(p.zeta_threshold)
  t117 = jnp.where(t114, t115 * p.zeta_threshold, 1)
  t134 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t95 + 0.905775 * t92 + 0.1100325 * t98 + 0.1241775 * t105))
  t138 = -0.621814e-1 * (0.1e1 + 0.53425e-1 * t92) * t111 + 0.19751673498613801407e-1 * (0.2e1 * t117 - 0.2e1) / (0.2e1 * t10 - 0.2e1) * (0.1e1 + 0.278125e-1 * t92) * t134
  t174 = params.m08_b[0] + params.m08_b[1] * t18 * t21 + params.m08_b[2] * t24 * t27 + params.m08_b[3] * t30 * t33 + params.m08_b[4] * t36 * t39 + params.m08_b[5] * t42 * t45 + params.m08_b[6] * t48 * t51 + params.m08_b[7] * t54 * t57 + params.m08_b[8] * t60 * t63 + params.m08_b[9] * t66 * t69 + params.m08_b[10] * t72 * t75 + params.m08_b[11] * t78 * t81
  t175 = jnp.log(0.2e1)
  t176 = 0.1e1 - t175
  t179 = t115 ** 2
  t180 = jnp.where(t114, t179, 1)
  t181 = t180 ** 2
  t182 = t181 * t180
  t184 = r0 ** 2
  t196 = 0.1e1 / t176
  t197 = t196 * t5
  t202 = jnp.exp(-t138 * t196 * t5 / t182)
  t204 = 0.1e1 / (t202 - 0.1e1)
  t205 = s0 ** 2
  t207 = t184 ** 2
  t212 = t181 ** 2
  t221 = s0 / t13 / t184 * t10 / t181 * t100 / t86 * t88 / 0.96e2 + 0.21720231316129303386e-4 * t197 * t204 * t205 / t14 / t207 * t11 / t212 * t84 / t101 * t89
  t232 = jnp.log(0.1e1 + 0.6672455060314922e-1 * t221 * t196 * t5 / (0.1e1 + 0.6672455060314922e-1 * t197 * t204 * t221))
  res = t83 * t138 + t174 * t176 / t5 * t182 * t232
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