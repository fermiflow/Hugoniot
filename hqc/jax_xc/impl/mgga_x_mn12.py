"""Generated from mgga_x_mn12.mpl."""

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
  t29 = params.c[0]
  t30 = params.c[1]
  t31 = jnp.cbrt(6)
  t32 = t31 ** 2
  t33 = jnp.pi ** 2
  t34 = jnp.cbrt(t33)
  t35 = t34 ** 2
  t37 = 0.3e1 / 0.1e2 * t32 * t35
  t38 = jnp.cbrt(r0)
  t39 = t38 ** 2
  t42 = tau0 / t39 / r0
  t43 = t37 - t42
  t45 = t37 + t42
  t46 = 0.1e1 / t45
  t48 = params.c[2]
  t49 = t43 ** 2
  t51 = t45 ** 2
  t52 = 0.1e1 / t51
  t54 = params.c[3]
  t55 = t49 * t43
  t58 = 0.1e1 / t51 / t45
  t60 = params.c[4]
  t61 = t49 ** 2
  t63 = t51 ** 2
  t64 = 0.1e1 / t63
  t66 = params.c[5]
  t72 = params.c[6]
  t73 = params.c[7]
  t76 = params.c[8]
  t79 = params.c[9]
  t82 = params.c[10]
  t87 = r0 ** 2
  t89 = 0.1e1 / t39 / t87
  t92 = 0.1e1 + 0.4e-2 * s0 * t89
  t94 = t89 / t92
  t97 = params.c[11]
  t98 = params.c[12]
  t101 = params.c[13]
  t104 = params.c[14]
  t108 = s0 ** 2
  t110 = t87 ** 2
  t114 = t92 ** 2
  t116 = 0.1e1 / t38 / t110 / r0 / t114
  t119 = params.c[15]
  t120 = params.c[16]
  t123 = params.c[17]
  t129 = t110 ** 2
  t136 = params.c[18]
  t137 = params.c[19]
  t140 = params.c[20]
  t143 = params.c[21]
  t146 = params.c[22]
  t151 = jnp.cbrt(2)
  t152 = 0.1e1 / t27 * t151
  t154 = 0.1e1 + t17 <= p.zeta_threshold
  t156 = 0.1e1 - t17 <= p.zeta_threshold
  t157 = jnp.where(t156, t15, t17)
  t158 = jnp.where(t154, t11, t157)
  t161 = jnp.cbrt(0.1e1 / (0.1e1 + t158))
  t164 = 0.1e1 + 0.39999999999999999998 * t152 * t161
  t165 = 0.1e1 / t164
  t167 = params.c[23]
  t168 = params.c[24]
  t171 = params.c[25]
  t174 = params.c[26]
  t182 = params.c[27]
  t183 = params.c[28]
  t186 = params.c[29]
  t194 = params.c[30]
  t195 = params.c[31]
  t198 = params.c[32]
  t201 = params.c[33]
  t205 = t164 ** 2
  t206 = 0.1e1 / t205
  t208 = params.c[34]
  t209 = params.c[35]
  t212 = params.c[36]
  t220 = params.c[37]
  t221 = params.c[38]
  t224 = params.c[39]
  t231 = t29 + t30 * t43 * t46 + t48 * t49 * t52 + t54 * t55 * t58 + t60 * t61 * t64 + t66 * t61 * t43 / t63 / t45 + 0.4e-2 * (t73 * t43 * t46 + t76 * t49 * t52 + t79 * t55 * t58 + t82 * t61 * t64 + t72) * s0 * t94 + 0.16e-4 * (t101 * t49 * t52 + t104 * t55 * t58 + t98 * t43 * t46 + t97) * t108 * t116 + 0.64e-7 * (t120 * t43 * t46 + t123 * t49 * t52 + t119) * t108 * s0 / t129 / t114 / t92 + (t137 * t43 * t46 + t140 * t49 * t52 + t143 * t55 * t58 + t146 * t61 * t64 + t136) * t165 + 0.4e-2 * (t168 * t43 * t46 + t171 * t49 * t52 + t174 * t55 * t58 + t167) * s0 * t94 * t165 + 0.16e-4 * (t183 * t43 * t46 + t186 * t49 * t52 + t182) * t108 * t116 * t165 + (t195 * t43 * t46 + t198 * t49 * t52 + t201 * t55 * t58 + t194) * t206 + 0.4e-2 * (t209 * t43 * t46 + t212 * t49 * t52 + t208) * s0 * t94 * t206 + (t221 * t43 * t46 + t224 * t49 * t52 + t220) / t205 / t164
  t235 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * t231)
  t237 = jnp.where(t10, t15, -t17)
  t238 = jnp.where(t14, t11, t237)
  t239 = 0.1e1 + t238
  t241 = jnp.cbrt(t239)
  t243 = jnp.where(t239 <= p.zeta_threshold, t23, t241 * t239)
  t245 = jnp.cbrt(r1)
  t246 = t245 ** 2
  t249 = tau1 / t246 / r1
  t250 = t37 - t249
  t252 = t37 + t249
  t253 = 0.1e1 / t252
  t255 = t250 ** 2
  t257 = t252 ** 2
  t258 = 0.1e1 / t257
  t260 = t255 * t250
  t263 = 0.1e1 / t257 / t252
  t265 = t255 ** 2
  t267 = t257 ** 2
  t268 = 0.1e1 / t267
  t285 = r1 ** 2
  t287 = 0.1e1 / t246 / t285
  t290 = 0.1e1 + 0.4e-2 * s2 * t287
  t292 = t287 / t290
  t302 = s2 ** 2
  t304 = t285 ** 2
  t308 = t290 ** 2
  t310 = 0.1e1 / t245 / t304 / r1 / t308
  t320 = t304 ** 2
  t336 = jnp.where(t154, t15, -t17)
  t337 = jnp.where(t156, t11, t336)
  t340 = jnp.cbrt(0.1e1 / (0.1e1 + t337))
  t343 = 0.1e1 + 0.39999999999999999998 * t152 * t340
  t344 = 0.1e1 / t343
  t373 = t343 ** 2
  t374 = 0.1e1 / t373
  t393 = t29 + t30 * t250 * t253 + t48 * t255 * t258 + t54 * t260 * t263 + t60 * t265 * t268 + t66 * t265 * t250 / t267 / t252 + 0.4e-2 * (t73 * t250 * t253 + t76 * t255 * t258 + t79 * t260 * t263 + t82 * t265 * t268 + t72) * s2 * t292 + 0.16e-4 * (t101 * t255 * t258 + t104 * t260 * t263 + t98 * t250 * t253 + t97) * t302 * t310 + 0.64e-7 * (t120 * t250 * t253 + t123 * t255 * t258 + t119) * t302 * s2 / t320 / t308 / t290 + (t137 * t250 * t253 + t140 * t255 * t258 + t143 * t260 * t263 + t146 * t265 * t268 + t136) * t344 + 0.4e-2 * (t168 * t250 * t253 + t171 * t255 * t258 + t174 * t260 * t263 + t167) * s2 * t292 * t344 + 0.16e-4 * (t183 * t250 * t253 + t186 * t255 * t258 + t182) * t302 * t310 * t344 + (t195 * t250 * t253 + t198 * t255 * t258 + t201 * t260 * t263 + t194) * t374 + 0.4e-2 * (t209 * t250 * t253 + t212 * t255 * t258 + t208) * s2 * t292 * t374 + (t221 * t250 * t253 + t224 * t255 * t258 + t220) / t373 / t343
  t397 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t243 * t27 * t393)
  res = t235 + t397
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
  t23 = jnp.cbrt(6)
  t24 = t23 ** 2
  t25 = jnp.pi ** 2
  t26 = jnp.cbrt(t25)
  t27 = t26 ** 2
  t29 = 0.3e1 / 0.1e2 * t24 * t27
  t30 = jnp.cbrt(2)
  t31 = t30 ** 2
  t33 = t19 ** 2
  t36 = tau0 * t31 / t33 / r0
  t37 = t29 - t36
  t39 = t29 + t36
  t40 = 0.1e1 / t39
  t43 = t37 ** 2
  t45 = t39 ** 2
  t46 = 0.1e1 / t45
  t49 = t43 * t37
  t52 = 0.1e1 / t45 / t39
  t55 = t43 ** 2
  t57 = t45 ** 2
  t58 = 0.1e1 / t57
  t81 = r0 ** 2
  t83 = 0.1e1 / t33 / t81
  t88 = 0.1e1 + 0.4e-2 * s0 * t31 * t83
  t89 = 0.1e1 / t88
  t104 = s0 ** 2
  t106 = t81 ** 2
  t109 = 0.1e1 / t19 / t106 / r0
  t111 = t88 ** 2
  t112 = 0.1e1 / t111
  t126 = t106 ** 2
  t150 = jnp.cbrt(0.1e1 / t12)
  t153 = 0.1e1 + 0.39999999999999999998 / t19 * t30 * t150
  t154 = 0.1e1 / t153
  t169 = t83 * t89
  t198 = t153 ** 2
  t199 = 0.1e1 / t198
  t225 = params.c[0] + params.c[1] * t37 * t40 + params.c[2] * t43 * t46 + params.c[3] * t49 * t52 + params.c[4] * t55 * t58 + params.c[5] * t55 * t37 / t57 / t39 + 0.4e-2 * (params.c[7] * t37 * t40 + params.c[8] * t43 * t46 + params.c[9] * t49 * t52 + params.c[10] * t55 * t58 + params.c[6]) * s0 * t31 * t83 * t89 + 0.32e-4 * (params.c[12] * t37 * t40 + params.c[13] * t43 * t46 + params.c[14] * t49 * t52 + params.c[11]) * t104 * t30 * t109 * t112 + 0.256e-6 * (params.c[16] * t37 * t40 + params.c[17] * t43 * t46 + params.c[15]) * t104 * s0 / t126 / t111 / t88 + (params.c[19] * t37 * t40 + params.c[20] * t43 * t46 + params.c[21] * t49 * t52 + params.c[22] * t55 * t58 + params.c[18]) * t154 + 0.4e-2 * (params.c[24] * t37 * t40 + params.c[25] * t43 * t46 + params.c[26] * t49 * t52 + params.c[23]) * s0 * t31 * t169 * t154 + 0.32e-4 * (params.c[28] * t37 * t40 + params.c[29] * t43 * t46 + params.c[27]) * t104 * t30 * t109 * t112 * t154 + (params.c[31] * t37 * t40 + params.c[32] * t43 * t46 + params.c[33] * t49 * t52 + params.c[30]) * t199 + 0.4e-2 * (params.c[35] * t37 * t40 + params.c[36] * t43 * t46 + params.c[34]) * s0 * t31 * t169 * t199 + (params.c[38] * t37 * t40 + params.c[39] * t43 * t46 + params.c[37]) / t198 / t153
  t229 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * t225)
  res = 0.2e1 * t229
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