"""Generated from mgga_x_m06l.mpl."""

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
  t29 = jnp.cbrt(6)
  t30 = jnp.pi ** 2
  t31 = jnp.cbrt(t30)
  t32 = t31 ** 2
  t34 = t29 / t32
  t35 = r0 ** 2
  t36 = jnp.cbrt(r0)
  t37 = t36 ** 2
  t39 = 0.1e1 / t37 / t35
  t40 = s0 * t39
  t47 = params.a[0]
  t48 = params.a[1]
  t49 = t29 ** 2
  t50 = t49 * t32
  t51 = 0.3e1 / 0.1e2 * t50
  t54 = tau0 / t37 / r0
  t55 = t51 - t54
  t57 = t51 + t54
  t60 = params.a[2]
  t61 = t55 ** 2
  t63 = t57 ** 2
  t66 = params.a[3]
  t67 = t61 * t55
  t69 = t63 * t57
  t72 = params.a[4]
  t73 = t61 ** 2
  t75 = t63 ** 2
  t78 = params.a[5]
  t84 = params.a[6]
  t90 = params.a[7]
  t96 = params.a[8]
  t97 = t73 ** 2
  t99 = t75 ** 2
  t102 = params.a[9]
  t108 = params.a[10]
  t114 = params.a[11]
  t120 = t47 + t48 * t55 / t57 + t60 * t61 / t63 + t66 * t67 / t69 + t72 * t73 / t75 + t78 * t73 * t55 / t75 / t57 + t84 * t73 * t61 / t75 / t63 + t90 * t73 * t67 / t75 / t69 + t96 * t97 / t99 + t102 * t97 * t55 / t99 / t57 + t108 * t97 * t61 / t99 / t63 + t114 * t97 * t67 / t99 / t69
  t122 = params.d[0]
  t125 = 0.1120356e-2 * t50
  t126 = 0.1e1 + 0.186726e-2 * t40 + 0.373452e-2 * t54 - t125
  t129 = params.d[1]
  t132 = params.d[2]
  t134 = 0.3e1 / 0.5e1 * t50
  t135 = 0.2e1 * t54 - t134
  t138 = t126 ** 2
  t141 = params.d[3]
  t142 = s0 ** 2
  t144 = t35 ** 2
  t149 = params.d[4]
  t153 = params.d[5]
  t154 = t135 ** 2
  t164 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * ((0.1804e1 - 0.646416 / (0.804 + 0.91464571985215458336e-2 * t34 * t40)) * t120 + t122 / t126 + (t129 * s0 * t39 + t132 * t135) / t138 + (t141 * t142 / t36 / t144 / r0 + t149 * s0 * t39 * t135 + t153 * t154) / t138 / t126))
  t166 = jnp.where(t10, t15, -t17)
  t167 = jnp.where(t14, t11, t166)
  t168 = 0.1e1 + t167
  t170 = jnp.cbrt(t168)
  t172 = jnp.where(t168 <= p.zeta_threshold, t23, t170 * t168)
  t174 = r1 ** 2
  t175 = jnp.cbrt(r1)
  t176 = t175 ** 2
  t178 = 0.1e1 / t176 / t174
  t179 = s2 * t178
  t188 = tau1 / t176 / r1
  t189 = t51 - t188
  t191 = t51 + t188
  t194 = t189 ** 2
  t196 = t191 ** 2
  t199 = t194 * t189
  t201 = t196 * t191
  t204 = t194 ** 2
  t206 = t196 ** 2
  t224 = t204 ** 2
  t226 = t206 ** 2
  t244 = t47 + t48 * t189 / t191 + t60 * t194 / t196 + t66 * t199 / t201 + t72 * t204 / t206 + t78 * t204 * t189 / t206 / t191 + t84 * t204 * t194 / t206 / t196 + t90 * t204 * t199 / t206 / t201 + t96 * t224 / t226 + t102 * t224 * t189 / t226 / t191 + t108 * t224 * t194 / t226 / t196 + t114 * t224 * t199 / t226 / t201
  t248 = 0.1e1 + 0.186726e-2 * t179 + 0.373452e-2 * t188 - t125
  t254 = 0.2e1 * t188 - t134
  t257 = t248 ** 2
  t260 = s2 ** 2
  t262 = t174 ** 2
  t270 = t254 ** 2
  t280 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t172 * t27 * ((0.1804e1 - 0.646416 / (0.804 + 0.91464571985215458336e-2 * t34 * t179)) * t244 + t122 / t248 + (t129 * s2 * t178 + t132 * t254) / t257 + (t141 * t260 / t175 / t262 / r1 + t149 * s2 * t178 * t254 + t153 * t270) / t257 / t248))
  res = t164 + t280
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
  t21 = jnp.cbrt(6)
  t22 = jnp.pi ** 2
  t23 = jnp.cbrt(t22)
  t24 = t23 ** 2
  t27 = jnp.cbrt(2)
  t28 = t27 ** 2
  t30 = r0 ** 2
  t31 = t19 ** 2
  t33 = 0.1e1 / t31 / t30
  t34 = s0 * t28 * t33
  t43 = t21 ** 2
  t44 = t43 * t24
  t45 = 0.3e1 / 0.1e2 * t44
  t49 = tau0 * t28 / t31 / r0
  t50 = t45 - t49
  t52 = t45 + t49
  t56 = t50 ** 2
  t58 = t52 ** 2
  t62 = t56 * t50
  t64 = t58 * t52
  t68 = t56 ** 2
  t70 = t58 ** 2
  t92 = t68 ** 2
  t94 = t70 ** 2
  t115 = params.a[0] + params.a[1] * t50 / t52 + params.a[2] * t56 / t58 + params.a[3] * t62 / t64 + params.a[4] * t68 / t70 + params.a[5] * t68 * t50 / t70 / t52 + params.a[6] * t68 * t56 / t70 / t58 + params.a[7] * t68 * t62 / t70 / t64 + params.a[8] * t92 / t94 + params.a[9] * t92 * t50 / t94 / t52 + params.a[10] * t92 * t56 / t94 / t58 + params.a[11] * t92 * t62 / t94 / t64
  t121 = 0.1e1 + 0.186726e-2 * t34 + 0.373452e-2 * t49 - 0.1120356e-2 * t44
  t126 = t28 * t33
  t131 = 0.2e1 * t49 - 0.3e1 / 0.5e1 * t44
  t134 = t121 ** 2
  t138 = s0 ** 2
  t140 = t30 ** 2
  t152 = t131 ** 2
  t162 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * ((0.1804e1 - 0.646416 / (0.804 + 0.91464571985215458336e-2 * t21 / t24 * t34)) * t115 + params.d[0] / t121 + (params.d[1] * s0 * t126 + params.d[2] * t131) / t134 + (0.2e1 * params.d[3] * t138 * t27 / t19 / t140 / r0 + params.d[4] * s0 * t126 * t131 + params.d[5] * t152) / t134 / t121))
  res = 0.2e1 * t162
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