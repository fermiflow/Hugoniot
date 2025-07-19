"""Generated from lda_xc_ksdt.mpl."""

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
  t1 = 0.1e1 / jnp.pi
  t2 = jnp.cbrt(4)
  t3 = t2 ** 2
  t5 = jnp.cbrt(9)
  t7 = jnp.cbrt(t1)
  t8 = 0.1e1 / t7
  t9 = t5 ** 2
  t10 = t7 * t1
  t11 = 0.1e1 / t10
  t13 = 0.1e1 / params.T
  t14 = t9 * t11 * t13
  t15 = jnp.cbrt(3)
  t16 = r0 + r1
  t17 = jnp.cbrt(t16)
  t18 = t17 ** 2
  t19 = t15 * t18
  t20 = r0 - r1
  t22 = 0.1e1 / t16
  t24 = params.thetaParam * t20 * t22 + 0.1e1
  t25 = jnp.cbrt(t24)
  t26 = t25 ** 2
  t27 = 0.1e1 / t26
  t31 = jnp.tanh(t14 * t19 * t27 / 0.6e1)
  t33 = jnp.pi ** 2
  t35 = t7 ** 2
  t36 = t35 / t33
  t38 = params.T ** 2
  t39 = t9 * t36 * t38
  t41 = 0.1e1 / t17 / t16
  t42 = t15 * t41
  t43 = t25 * t24
  t45 = t39 * t42 * t43
  t47 = t33 ** 2
  t51 = t16 ** 2
  t53 = t24 ** 2
  t55 = 0.1e1 / t47 * t38 * params.T / t51 * t53
  t59 = t7 / t47 / jnp.pi
  t61 = t38 ** 2
  t62 = t5 * t59 * t61
  t63 = t15 ** 2
  t65 = 0.1e1 / t18 / t51
  t66 = t63 * t65
  t67 = t26 * t53
  t69 = t62 * t66 * t67
  t80 = jnp.sqrt(0.2e1)
  t82 = t5 * t10 * params.T
  t84 = t63 / t18
  t87 = jnp.sqrt(t82 * t84 * t26)
  t91 = jnp.tanh(0.3e1 / 0.2e1 * t80 / t87)
  t98 = t38 * t15 * t41 * t43
  t106 = t61 * t63 * t65 * t67
  t126 = t15 * t7 * t3 / t17
  t127 = jnp.sqrt(t126)
  t141 = jnp.exp(-params.c[0][2] * t9 * t11 * t13 * t15 * t18 * t27 / 0.6e1)
  t156 = params.e[0][0] + 0.4e1 / 0.27e2 * params.e[0][1] * t9 * t36 * t98 + 0.16e2 / 0.81e2 * params.e[0][2] * t5 * t59 * t106
  t168 = 0.1e1 / (0.1e1 + 0.4e1 / 0.27e2 * params.e[0][3] * t9 * t36 * t98 + 0.16e2 / 0.81e2 * params.e[0][4] * t5 * t59 * t106)
  t211 = t20 * t22
  t212 = 0.1e1 + t211
  t229 = jnp.exp(-0.2e1 / 0.9e1 * t82 * t84 * t26 * (0.1064009e1 + 0.63618333333333333335e-1 * t82 * t84 * t26 * t127))
  t231 = 0.2e1 - (0.2e1 / 0.3e1 - 0.3481525e-2 * t126) / (0.1e1 + 0.45802e-1 * t126) * t229
  t232 = p.zeta_threshold ** t231
  t233 = t212 ** t231
  t234 = jnp.where(t212 <= p.zeta_threshold, t232, t233)
  t235 = 0.1e1 - t211
  t237 = t235 ** t231
  t238 = jnp.where(t235 <= p.zeta_threshold, t232, t237)
  t240 = 2 ** t231
  t243 = (t234 + t238 - 0.2e1) / (t240 - 0.2e1)
  t248 = jnp.cbrt(2)
  t252 = t248 ** 2
  t254 = t19 * t27 * t252
  t257 = jnp.tanh(t14 * t254 / 0.6e1)
  t260 = t42 * t43 * t252
  t261 = t39 * t260
  t265 = t66 * t67 * t248
  t266 = t62 * t265
  t280 = jnp.sqrt(t82 * t84 * t26 * t248)
  t283 = jnp.tanh(0.3e1 / t280)
  t287 = t36 * t38
  t293 = t59 * t61
  t322 = jnp.exp(-params.c[1][2] * t9 * t11 * t13 * t254 / 0.6e1)
  t337 = params.e[1][0] + params.e[1][1] * t9 * t287 * t260 / 0.27e2 + 0.2e1 / 0.81e2 * params.e[1][2] * t5 * t293 * t265
  t349 = 0.1e1 / (0.1e1 + params.e[1][3] * t9 * t287 * t260 / 0.27e2 + 0.2e1 / 0.81e2 * params.e[1][4] * t5 * t293 * t265)
  res = -(t1 * t3 * t5 * t8 * t31 * (0.75 + 0.45090814814814814815 * t45 - 0.82017777777777777776e-1 * t55 + 0.33649382716049382717 * t69) / (0.1e1 + 0.12311866666666666667e1 * t45 + 0.10094814814814814815e1 * t69) / 0.4e1 + t91 * (params.b[0][0] + 0.4e1 / 0.27e2 * params.b[0][1] * t9 * t36 * t98 + 0.16e2 / 0.81e2 * params.b[0][2] * t5 * t59 * t106) / (0.1e1 + 0.4e1 / 0.27e2 * params.b[0][3] * t9 * t36 * t98 + 0.16e2 / 0.81e2 * params.b[0][4] * t5 * t59 * t106) * t127 / 0.2e1 + (params.c[0][1] * t141 + params.c[0][0]) * t31 * t156 * t168 * t126 / 0.4e1) * t63 * t8 * t2 * t17 / (0.1e1 + t91 * (params.d[0][0] + 0.4e1 / 0.27e2 * params.d[0][1] * t9 * t36 * t98 + 0.16e2 / 0.81e2 * params.d[0][2] * t5 * t59 * t106) / (0.1e1 + 0.4e1 / 0.27e2 * params.d[0][3] * t9 * t36 * t98 + 0.16e2 / 0.81e2 * params.d[0][4] * t5 * t59 * t106) * t127 / 0.2e1 + t31 * t156 * t168 * t126 / 0.4e1) * (0.1e1 - t243) / 0.3e1 - (t248 * t1 * t3 * t5 * t8 * t257 * (0.75 + 0.11272703703703703704 * t261 - 0.20504444444444444444e-1 * t55 + 0.42061728395061728396e-1 * t266) / (0.1e1 + 0.30779666666666666667 * t261 + 0.12618518518518518519 * t266) / 0.4e1 + t283 * (params.b[1][0] + params.b[1][1] * t9 * t287 * t260 / 0.27e2 + 0.2e1 / 0.81e2 * params.b[1][2] * t5 * t293 * t265) / (0.1e1 + params.b[1][3] * t9 * t287 * t260 / 0.27e2 + 0.2e1 / 0.81e2 * params.b[1][4] * t5 * t293 * t265) * t127 / 0.2e1 + (params.c[1][1] * t322 + params.c[1][0]) * t257 * t337 * t349 * t126 / 0.4e1) * t63 * t8 * t2 * t17 / (0.1e1 + t283 * (params.d[1][0] + params.d[1][1] * t9 * t287 * t260 / 0.27e2 + 0.2e1 / 0.81e2 * params.d[1][2] * t5 * t293 * t265) / (0.1e1 + params.d[1][3] * t9 * t287 * t260 / 0.27e2 + 0.2e1 / 0.81e2 * params.d[1][4] * t5 * t293 * t265) * t127 / 0.2e1 + t257 * t337 * t349 * t126 / 0.4e1) * t243 / 0.3e1
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = 0.1e1 / jnp.pi
  t2 = jnp.cbrt(4)
  t3 = t2 ** 2
  t5 = jnp.cbrt(9)
  t7 = jnp.cbrt(t1)
  t8 = 0.1e1 / t7
  t9 = t5 ** 2
  t10 = t7 * t1
  t11 = 0.1e1 / t10
  t12 = t9 * t11
  t13 = 0.1e1 / params.T
  t14 = jnp.cbrt(3)
  t15 = t13 * t14
  t16 = jnp.cbrt(r0)
  t17 = t16 ** 2
  t18 = t15 * t17
  t21 = jnp.tanh(t12 * t18 / 0.6e1)
  t23 = jnp.pi ** 2
  t25 = t7 ** 2
  t26 = t25 / t23
  t27 = t9 * t26
  t28 = params.T ** 2
  t29 = t28 * t14
  t31 = 0.1e1 / t16 / r0
  t32 = t29 * t31
  t33 = t27 * t32
  t35 = t23 ** 2
  t39 = r0 ** 2
  t41 = 0.1e1 / t35 * t28 * params.T / t39
  t45 = t7 / t35 / jnp.pi
  t46 = t5 * t45
  t47 = t28 ** 2
  t48 = t14 ** 2
  t49 = t47 * t48
  t51 = 0.1e1 / t17 / t39
  t52 = t49 * t51
  t53 = t46 * t52
  t64 = jnp.sqrt(0.2e1)
  t65 = t5 * t10
  t67 = 0.1e1 / t17
  t70 = jnp.sqrt(t65 * params.T * t48 * t67)
  t74 = jnp.tanh(0.3e1 / 0.2e1 * t64 / t70)
  t103 = t14 * t7 * t3 / t16
  t104 = jnp.sqrt(t103)
  t115 = jnp.exp(-params.c[0][2] * t9 * t11 * t18 / 0.6e1)
  t130 = params.e[0][0] + 0.4e1 / 0.27e2 * params.e[0][1] * t9 * t26 * t32 + 0.16e2 / 0.81e2 * params.e[0][2] * t5 * t45 * t52
  t142 = 0.1e1 / (0.1e1 + 0.4e1 / 0.27e2 * params.e[0][3] * t9 * t26 * t32 + 0.16e2 / 0.81e2 * params.e[0][4] * t5 * t45 * t52)
  t192 = t65 * params.T
  t193 = t48 * t67
  t201 = jnp.exp(-0.2e1 / 0.9e1 * t192 * t193 * (0.1064009e1 + 0.63618333333333333335e-1 * t192 * t193 * t104))
  t203 = 0.2e1 - (0.2e1 / 0.3e1 - 0.3481525e-2 * t103) / (0.1e1 + 0.45802e-1 * t103) * t201
  t204 = p.zeta_threshold ** t203
  t205 = jnp.where(0.1e1 <= p.zeta_threshold, t204, 1)
  t208 = 2 ** t203
  t211 = (0.2e1 * t205 - 0.2e1) / (t208 - 0.2e1)
  t216 = jnp.cbrt(2)
  t222 = t216 ** 2
  t226 = jnp.tanh(t12 * t13 * t14 * t17 * t222 / 0.6e1)
  t231 = t27 * t28 * t14 * t31 * t222
  t237 = t46 * t47 * t48 * t51 * t216
  t250 = jnp.sqrt(t192 * t193 * t216)
  t253 = jnp.tanh(0.3e1 / t250)
  t259 = t29 * t31 * t222
  t266 = t49 * t51 * t216
  t295 = jnp.exp(-params.c[1][2] * t9 * t11 * t15 * t17 * t222 / 0.6e1)
  t310 = params.e[1][0] + params.e[1][1] * t9 * t26 * t259 / 0.27e2 + 0.2e1 / 0.81e2 * params.e[1][2] * t5 * t45 * t266
  t322 = 0.1e1 / (0.1e1 + params.e[1][3] * t9 * t26 * t259 / 0.27e2 + 0.2e1 / 0.81e2 * params.e[1][4] * t5 * t45 * t266)
  res = -(t1 * t3 * t5 * t8 * t21 * (0.75 + 0.45090814814814814815 * t33 - 0.82017777777777777776e-1 * t41 + 0.33649382716049382717 * t53) / (0.1e1 + 0.12311866666666666667e1 * t33 + 0.10094814814814814815e1 * t53) / 0.4e1 + t74 * (params.b[0][0] + 0.4e1 / 0.27e2 * params.b[0][1] * t9 * t26 * t32 + 0.16e2 / 0.81e2 * params.b[0][2] * t5 * t45 * t52) / (0.1e1 + 0.4e1 / 0.27e2 * params.b[0][3] * t9 * t26 * t32 + 0.16e2 / 0.81e2 * params.b[0][4] * t5 * t45 * t52) * t104 / 0.2e1 + (params.c[0][1] * t115 + params.c[0][0]) * t21 * t130 * t142 * t103 / 0.4e1) * t48 * t8 * t2 * t16 / (0.1e1 + t74 * (params.d[0][0] + 0.4e1 / 0.27e2 * params.d[0][1] * t9 * t26 * t32 + 0.16e2 / 0.81e2 * params.d[0][2] * t5 * t45 * t52) / (0.1e1 + 0.4e1 / 0.27e2 * params.d[0][3] * t9 * t26 * t32 + 0.16e2 / 0.81e2 * params.d[0][4] * t5 * t45 * t52) * t104 / 0.2e1 + t21 * t130 * t142 * t103 / 0.4e1) * (0.1e1 - t211) / 0.3e1 - (t216 * t1 * t3 * t5 * t8 * t226 * (0.75 + 0.11272703703703703704 * t231 - 0.20504444444444444444e-1 * t41 + 0.42061728395061728396e-1 * t237) / (0.1e1 + 0.30779666666666666667 * t231 + 0.12618518518518518519 * t237) / 0.4e1 + t253 * (params.b[1][0] + params.b[1][1] * t9 * t26 * t259 / 0.27e2 + 0.2e1 / 0.81e2 * params.b[1][2] * t5 * t45 * t266) / (0.1e1 + params.b[1][3] * t9 * t26 * t259 / 0.27e2 + 0.2e1 / 0.81e2 * params.b[1][4] * t5 * t45 * t266) * t104 / 0.2e1 + (params.c[1][1] * t295 + params.c[1][0]) * t226 * t310 * t322 * t103 / 0.4e1) * t48 * t8 * t2 * t16 / (0.1e1 + t253 * (params.d[1][0] + params.d[1][1] * t9 * t26 * t259 / 0.27e2 + 0.2e1 / 0.81e2 * params.d[1][2] * t5 * t45 * t266) / (0.1e1 + params.d[1][3] * t9 * t26 * t259 / 0.27e2 + 0.2e1 / 0.81e2 * params.d[1][4] * t5 * t45 * t266) * t104 / 0.2e1 + t226 * t310 * t322 * t103 / 0.4e1) * t211 / 0.3e1
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