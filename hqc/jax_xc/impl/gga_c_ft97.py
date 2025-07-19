"""Generated from gga_c_ft97.mpl."""

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
  t1 = jnp.log(0.2e1)
  t2 = 0.1e1 - t1
  t3 = jnp.pi ** 2
  t5 = t2 / t3
  t6 = jnp.cbrt(3)
  t7 = 0.1e1 / jnp.pi
  t8 = jnp.cbrt(t7)
  t9 = t6 * t8
  t10 = t5 * t9
  t11 = jnp.cbrt(4)
  t12 = t11 ** 2
  t13 = r0 + r1
  t14 = jnp.cbrt(t13)
  t15 = 0.1e1 / t14
  t16 = t12 * t15
  t17 = jnp.cbrt(2)
  t20 = (r0 - r1) / t13
  t21 = 0.1e1 + t20
  t23 = jnp.cbrt(0.1e1 / t21)
  t24 = t17 * t23
  t25 = 4 ** (0.1e1 / 0.5e1)
  t26 = t9 * t12
  t27 = t15 * t17
  t28 = t27 * t23
  t29 = t26 * t28
  t30 = t29 ** (0.1e1 / 0.5e1)
  t31 = t30 ** 2
  t32 = t31 ** 2
  t35 = jnp.exp(-0.2081897e-1 * t25 * t32)
  t38 = (0.942486901 + 0.349064173 * t35) ** 2
  t39 = t6 ** 2
  t40 = t8 ** 2
  t41 = t39 * t40
  t43 = r0 ** 2
  t44 = jnp.cbrt(r0)
  t45 = t44 ** 2
  t47 = 0.1e1 / t45 / t43
  t52 = t6 * t8 * t7
  t53 = s0 ** 2
  t55 = t43 ** 2
  t60 = t52 * t12 * t53 / t44 / t55 / r0
  t61 = 0.55569193573523559258e-3 * t60
  t63 = (0.1e1 + 0.45058854638888888889e-1 * t41 * t11 * s0 * t47 + t61) ** 2
  t65 = jnp.exp(-t61)
  t66 = t65 ** 2
  t70 = t17 ** 2
  t74 = t9 * t12 * s0 * t47 * t14 * t70 / t23
  t79 = t38 * t63 * t66 / (0.1e1 + 0.19153082513888888889e-1 * t74)
  t81 = jnp.where(0.1e-59 < t79, t79, 0.1e-59)
  t82 = 0.1e1 / t81
  t85 = t10 * t16 * t24 * t82
  t86 = t85 / 0.6e1
  t88 = xc_E1_scaled(t86)
  t89 = t5 * t26
  t90 = jnp.sqrt(0.6e1)
  t91 = t90 * t7
  t94 = t2 * t6 * t8 * t12
  t98 = jnp.sqrt(t94 * t27 * t23 * t82)
  t99 = t91 * t98
  t102 = 0.3e1 + t99 / 0.3e1 + t85 / 0.3e1
  t105 = 0.1e1 / (0.3e1 + t99 + t85)
  t115 = 0.1e1 - t20
  t119 = jnp.where(0.1e8 <= t86, 0, t5 * (-t88 * (0.1e1 + t89 * t28 * t82 * t102 * t105 / 0.3e1) + 0.2e1 * t102 * t105) * t115 / 0.4e1)
  t120 = jnp.sqrt(t29)
  t122 = jnp.exp(-0.544669424 * t120)
  t124 = t25 ** 2
  t125 = t124 * t25
  t128 = jnp.exp(-0.16390970575 * t125 * t31)
  t131 = (0.1247511874e1 - 0.859614445 * t122 + 0.812904345 * t128) ** 2
  t132 = 0.56633563016285904186e-1 * t60
  t134 = (0.1e1 + t132) ** 2
  t136 = jnp.exp(-t132)
  t137 = t136 ** 2
  t142 = t131 * t134 * t137 / (0.1e1 + 0.50008500819444444447e-1 * t74)
  t144 = jnp.where(0.1e-59 < t142, t142, 0.1e-59)
  t145 = 0.1e1 / t144
  t148 = t10 * t16 * t24 * t145
  t149 = t148 / 0.6e1
  t151 = xc_E1_scaled(t149)
  t155 = jnp.sqrt(t94 * t27 * t23 * t145)
  t156 = t91 * t155
  t159 = 0.3e1 + t156 / 0.3e1 + t148 / 0.3e1
  t162 = 0.1e1 / (0.3e1 + t156 + t148)
  t172 = t41 * t11
  t173 = t14 ** 2
  t175 = 0.1e1 / t173 * t70
  t176 = t23 ** 2
  t180 = (0.469508 * t120 + 0.4332925 * t29) ** 2
  t186 = jnp.exp(-t172 * t175 * t176 / t180 / 0.4e1)
  t191 = jnp.where(0.1e8 <= t149, 0, t5 * (-t151 * (0.1e1 + t89 * t28 * t145 * t159 * t162 / 0.3e1) + 0.2e1 * t159 * t162) * t186 * t21 / 0.4e1)
  t193 = jnp.cbrt(0.1e1 / t115)
  t194 = t17 * t193
  t195 = t27 * t193
  t196 = t26 * t195
  t197 = t196 ** (0.1e1 / 0.5e1)
  t198 = t197 ** 2
  t199 = t198 ** 2
  t202 = jnp.exp(-0.2081897e-1 * t25 * t199)
  t205 = (0.942486901 + 0.349064173 * t202) ** 2
  t207 = r1 ** 2
  t208 = jnp.cbrt(r1)
  t209 = t208 ** 2
  t211 = 0.1e1 / t209 / t207
  t215 = s2 ** 2
  t217 = t207 ** 2
  t222 = t52 * t12 * t215 / t208 / t217 / r1
  t223 = 0.55569193573523559258e-3 * t222
  t225 = (0.1e1 + 0.45058854638888888889e-1 * t41 * t11 * s2 * t211 + t223) ** 2
  t227 = jnp.exp(-t223)
  t228 = t227 ** 2
  t235 = t9 * t12 * s2 * t211 * t14 * t70 / t193
  t240 = t205 * t225 * t228 / (0.1e1 + 0.19153082513888888889e-1 * t235)
  t242 = jnp.where(0.1e-59 < t240, t240, 0.1e-59)
  t243 = 0.1e1 / t242
  t246 = t10 * t16 * t194 * t243
  t247 = t246 / 0.6e1
  t249 = xc_E1_scaled(t247)
  t253 = jnp.sqrt(t94 * t27 * t193 * t243)
  t254 = t91 * t253
  t257 = 0.3e1 + t254 / 0.3e1 + t246 / 0.3e1
  t260 = 0.1e1 / (0.3e1 + t254 + t246)
  t273 = jnp.where(0.1e8 <= t247, 0, t5 * (-t249 * (0.1e1 + t89 * t195 * t243 * t257 * t260 / 0.3e1) + 0.2e1 * t257 * t260) * t21 / 0.4e1)
  t274 = jnp.sqrt(t196)
  t276 = jnp.exp(-0.544669424 * t274)
  t280 = jnp.exp(-0.16390970575 * t125 * t198)
  t283 = (0.1247511874e1 - 0.859614445 * t276 + 0.812904345 * t280) ** 2
  t284 = 0.56633563016285904186e-1 * t222
  t286 = (0.1e1 + t284) ** 2
  t288 = jnp.exp(-t284)
  t289 = t288 ** 2
  t294 = t283 * t286 * t289 / (0.1e1 + 0.50008500819444444447e-1 * t235)
  t296 = jnp.where(0.1e-59 < t294, t294, 0.1e-59)
  t297 = 0.1e1 / t296
  t300 = t10 * t16 * t194 * t297
  t301 = t300 / 0.6e1
  t303 = xc_E1_scaled(t301)
  t307 = jnp.sqrt(t94 * t27 * t193 * t297)
  t308 = t91 * t307
  t311 = 0.3e1 + t308 / 0.3e1 + t300 / 0.3e1
  t314 = 0.1e1 / (0.3e1 + t308 + t300)
  t324 = t193 ** 2
  t328 = (0.469508 * t274 + 0.4332925 * t196) ** 2
  t334 = jnp.exp(-t172 * t175 * t324 / t328 / 0.4e1)
  t339 = jnp.where(0.1e8 <= t301, 0, t5 * (-t303 * (0.1e1 + t89 * t195 * t297 * t311 * t314 / 0.3e1) + 0.2e1 * t311 * t314) * t334 * t115 / 0.4e1)
  res = t119 + t191 + t273 + t339
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.log(0.2e1)
  t2 = 0.1e1 - t1
  t3 = jnp.pi ** 2
  t5 = t2 / t3
  t6 = jnp.cbrt(3)
  t7 = 0.1e1 / jnp.pi
  t8 = jnp.cbrt(t7)
  t9 = t6 * t8
  t10 = t5 * t9
  t11 = jnp.cbrt(4)
  t12 = t11 ** 2
  t13 = jnp.cbrt(r0)
  t14 = 0.1e1 / t13
  t15 = t12 * t14
  t16 = jnp.cbrt(2)
  t17 = 4 ** (0.1e1 / 0.5e1)
  t19 = t9 * t15 * t16
  t20 = t19 ** (0.1e1 / 0.5e1)
  t21 = t20 ** 2
  t22 = t21 ** 2
  t25 = jnp.exp(-0.2081897e-1 * t17 * t22)
  t28 = (0.942486901 + 0.349064173 * t25) ** 2
  t29 = t6 ** 2
  t30 = t8 ** 2
  t32 = t29 * t30 * t11
  t33 = t16 ** 2
  t35 = r0 ** 2
  t36 = t13 ** 2
  t45 = s0 ** 2
  t47 = t35 ** 2
  t52 = t6 * t8 * t7 * t12 * t45 * t16 / t13 / t47 / r0
  t53 = 0.11113838714704711852e-2 * t52
  t55 = (0.1e1 + 0.45058854638888888889e-1 * t32 * s0 * t33 / t36 / t35 + t53) ** 2
  t57 = jnp.exp(-t53)
  t58 = t57 ** 2
  t59 = t9 * t12
  t64 = t59 * s0 * t16 / t13 / t35
  t69 = t28 * t55 * t58 / (0.1e1 + 0.38306165027777777778e-1 * t64)
  t71 = jnp.where(0.1e-59 < t69, t69, 0.1e-59)
  t72 = 0.1e1 / t71
  t74 = t15 * t16 * t72
  t75 = t10 * t74
  t76 = t75 / 0.6e1
  t78 = xc_E1_scaled(t76)
  t79 = t5 * t59
  t80 = t14 * t16
  t81 = jnp.sqrt(0.6e1)
  t82 = t81 * t7
  t84 = t2 * t6 * t8
  t86 = jnp.sqrt(t84 * t74)
  t87 = t82 * t86
  t90 = 0.3e1 + t87 / 0.3e1 + t75 / 0.3e1
  t93 = 0.1e1 / (0.3e1 + t87 + t75)
  t105 = jnp.where(0.1e8 <= t76, 0, t5 * (-t78 * (0.1e1 + t79 * t80 * t72 * t90 * t93 / 0.3e1) + 0.2e1 * t90 * t93) / 0.4e1)
  t106 = jnp.sqrt(t19)
  t108 = jnp.exp(-0.544669424 * t106)
  t110 = t17 ** 2
  t114 = jnp.exp(-0.16390970575 * t110 * t17 * t21)
  t117 = (0.1247511874e1 - 0.859614445 * t108 + 0.812904345 * t114) ** 2
  t118 = 0.11326712603257180837 * t52
  t120 = (0.1e1 + t118) ** 2
  t122 = jnp.exp(-t118)
  t123 = t122 ** 2
  t128 = t117 * t120 * t123 / (0.1e1 + 0.10001700163888888889 * t64)
  t130 = jnp.where(0.1e-59 < t128, t128, 0.1e-59)
  t131 = 0.1e1 / t130
  t133 = t15 * t16 * t131
  t134 = t10 * t133
  t135 = t134 / 0.6e1
  t137 = xc_E1_scaled(t135)
  t139 = jnp.sqrt(t84 * t133)
  t140 = t82 * t139
  t143 = 0.3e1 + t140 / 0.3e1 + t134 / 0.3e1
  t146 = 0.1e1 / (0.3e1 + t140 + t134)
  t161 = (0.469508 * t106 + 0.4332925 * t19) ** 2
  t166 = jnp.exp(-t32 / t36 * t33 / t161 / 0.4e1)
  t170 = jnp.where(0.1e8 <= t135, 0, t5 * (-t137 * (0.1e1 + t79 * t80 * t131 * t143 * t146 / 0.3e1) + 0.2e1 * t143 * t146) * t166 / 0.4e1)
  res = 0.2e1 * t105 + 0.2e1 * t170
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