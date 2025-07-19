"""Generated from mgga_xc_b98.mpl."""

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
  t1 = jnp.cbrt(3)
  t2 = jnp.cbrt(jnp.pi)
  t4 = t1 / t2
  t5 = r0 + r1
  t6 = 0.1e1 / t5
  t7 = r0 * t6
  t10 = jnp.cbrt(p.zeta_threshold)
  t11 = t10 * p.zeta_threshold
  t12 = jnp.cbrt(2)
  t14 = jnp.cbrt(t7)
  t18 = jnp.where(0.2e1 * t7 <= p.zeta_threshold, t11, 0.2e1 * t12 * r0 * t6 * t14)
  t19 = jnp.cbrt(t5)
  t21 = jnp.cbrt(r0)
  t22 = t21 ** 2
  t24 = 0.1e1 / t22 / r0
  t26 = r0 ** 2
  t34 = jnp.cbrt(6)
  t36 = jnp.pi ** 2
  t37 = jnp.cbrt(t36)
  t38 = t37 ** 2
  t39 = 0.1e1 / t38
  t40 = (tau0 * t24 - s0 / t22 / t26 / 0.8e1 - l0 * t24 / 0.4e1) * t34 * t39
  t42 = 0.1e1 - 0.5e1 / 0.9e1 * t40
  t43 = t42 ** 2
  t45 = 0.1e1 + 0.121e-1 * t43
  t46 = jnp.sqrt(t45)
  t57 = r1 * t6
  t61 = jnp.cbrt(t57)
  t65 = jnp.where(0.2e1 * t57 <= p.zeta_threshold, t11, 0.2e1 * t12 * r1 * t6 * t61)
  t67 = jnp.cbrt(r1)
  t68 = t67 ** 2
  t70 = 0.1e1 / t68 / r1
  t72 = r1 ** 2
  t81 = (tau1 * t70 - s2 / t68 / t72 / 0.8e1 - l1 * t70 / 0.4e1) * t34 * t39
  t83 = 0.1e1 - 0.5e1 / 0.9e1 * t81
  t84 = t83 ** 2
  t86 = 0.1e1 + 0.121e-1 * t84
  t87 = jnp.sqrt(t86)
  t99 = r0 - r1
  t100 = t99 * t6
  t101 = 0.1e1 + t100
  t102 = t101 <= p.zeta_threshold
  t103 = jnp.logical_or(r0 <= p.dens_threshold, t102)
  t104 = jnp.where(t102, p.zeta_threshold, t101)
  t106 = jnp.cbrt(0.1e1 / jnp.pi)
  t107 = t1 * t106
  t108 = jnp.cbrt(4)
  t109 = t108 ** 2
  t110 = t107 * t109
  t111 = 0.1e1 / t19
  t112 = t111 * t12
  t113 = 0.1e1 / t10
  t114 = jnp.cbrt(t101)
  t116 = jnp.where(t102, t113, 0.1e1 / t114)
  t118 = t110 * t112 * t116
  t121 = jnp.sqrt(t118)
  t124 = t118 ** 0.15e1
  t126 = t1 ** 2
  t127 = t106 ** 2
  t128 = t126 * t127
  t129 = t128 * t108
  t130 = t19 ** 2
  t131 = 0.1e1 / t130
  t132 = t12 ** 2
  t133 = t131 * t132
  t134 = t116 ** 2
  t136 = t129 * t133 * t134
  t142 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t121 + 0.8969 * t118 + 0.204775 * t124 + 0.123235 * t136))
  t144 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t118) * t142
  t147 = jnp.where(0.2e1 <= p.zeta_threshold, t11, 0.2e1 * t12)
  t149 = jnp.where(0. <= p.zeta_threshold, t11, 0)
  t153 = 0.1e1 / (0.2e1 * t12 - 0.2e1)
  t154 = (t147 + t149 - 0.2e1) * t153
  t165 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t121 + 0.1549425e1 * t118 + 0.420775 * t124 + 0.1562925 * t136))
  t178 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t121 + 0.905775 * t118 + 0.1100325 * t124 + 0.1241775 * t136))
  t179 = (0.1e1 + 0.278125e-1 * t118) * t178
  t188 = jnp.where(t103, 0, t104 * (-t144 + t154 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t118) * t165 + t144 - 0.19751789702565206229e-1 * t179) + 0.19751789702565206229e-1 * t154 * t179) / 0.2e1)
  t190 = 0.1e1 + 0.256e1 * t43
  t191 = jnp.sqrt(t190)
  t208 = 0.1e1 - t100
  t209 = t208 <= p.zeta_threshold
  t210 = jnp.logical_or(r1 <= p.dens_threshold, t209)
  t211 = jnp.where(t209, p.zeta_threshold, t208)
  t212 = jnp.cbrt(t208)
  t214 = jnp.where(t209, t113, 0.1e1 / t212)
  t216 = t110 * t112 * t214
  t219 = jnp.sqrt(t216)
  t222 = t216 ** 0.15e1
  t224 = t214 ** 2
  t226 = t129 * t133 * t224
  t232 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t219 + 0.8969 * t216 + 0.204775 * t222 + 0.123235 * t226))
  t234 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t216) * t232
  t245 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t219 + 0.1549425e1 * t216 + 0.420775 * t222 + 0.1562925 * t226))
  t258 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t219 + 0.905775 * t216 + 0.1100325 * t222 + 0.1241775 * t226))
  t259 = (0.1e1 + 0.278125e-1 * t216) * t258
  t268 = jnp.where(t210, 0, t211 * (-t234 + t154 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t216) * t245 + t234 - 0.19751789702565206229e-1 * t259) + 0.19751789702565206229e-1 * t154 * t259) / 0.2e1)
  t270 = 0.1e1 + 0.256e1 * t84
  t271 = jnp.sqrt(t270)
  t288 = t107 * t109 * t111
  t291 = jnp.sqrt(t288)
  t294 = t288 ** 0.15e1
  t297 = t128 * t108 * t131
  t303 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t291 + 0.8969 * t288 + 0.204775 * t294 + 0.123235 * t297))
  t305 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t288) * t303
  t306 = t99 ** 2
  t307 = t306 ** 2
  t308 = t5 ** 2
  t309 = t308 ** 2
  t313 = jnp.where(t102, t11, t114 * t101)
  t315 = jnp.where(t209, t11, t212 * t208)
  t317 = (t313 + t315 - 0.2e1) * t153
  t328 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t291 + 0.1549425e1 * t288 + 0.420775 * t294 + 0.1562925 * t297))
  t341 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t291 + 0.905775 * t288 + 0.1100325 * t294 + 0.1241775 * t297))
  t342 = (0.1e1 + 0.278125e-1 * t288) * t341
  t352 = 0.1e1 - 0.5e1 / 0.18e2 * t40 - 0.5e1 / 0.18e2 * t81
  t353 = t352 ** 2
  t355 = 0.1e1 + 0.196e-1 * t353
  t356 = jnp.sqrt(t355)
  res = -0.3e1 / 0.8e1 * t4 * t18 * t19 * (0.8085 + 0.73502e-1 * t42 / t46 + 0.17182e-2 * t43 / t45) - 0.3e1 / 0.8e1 * t4 * t65 * t19 * (0.8085 + 0.73502e-1 * t83 / t87 + 0.17182e-2 * t84 / t86) + t188 * (0.2606 - 0.153728e1 * t42 / t191 + 0.2309888e1 * t43 / t190) * (0.1e1 - s0 / r0 / tau0 / 0.8e1) + t268 * (0.2606 - 0.153728e1 * t83 / t271 + 0.2309888e1 * t84 / t270) * (0.1e1 - s2 / r1 / tau1 / 0.8e1) + (-t305 + t307 / t309 * t317 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t288) * t328 + t305 - 0.19751789702565206229e-1 * t342) + 0.19751789702565206229e-1 * t317 * t342 - t188 - t268) * (0.12033e1 - 0.318038 * t352 / t356 + 0.1880816e-1 * t353 / t355)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(3)
  t2 = jnp.cbrt(jnp.pi)
  t5 = 0.1e1 <= p.zeta_threshold
  t6 = jnp.cbrt(p.zeta_threshold)
  t7 = t6 * p.zeta_threshold
  t8 = jnp.where(t5, t7, 1)
  t9 = jnp.cbrt(r0)
  t11 = jnp.cbrt(2)
  t12 = t11 ** 2
  t14 = t9 ** 2
  t16 = 0.1e1 / t14 / r0
  t19 = r0 ** 2
  t28 = jnp.cbrt(6)
  t30 = jnp.pi ** 2
  t31 = jnp.cbrt(t30)
  t32 = t31 ** 2
  t36 = 0.1e1 - 0.5e1 / 0.9e1 * (tau0 * t12 * t16 - s0 * t12 / t14 / t19 / 0.8e1 - l0 * t12 * t16 / 0.4e1) * t28 / t32
  t37 = t36 ** 2
  t39 = 0.1e1 + 0.121e-1 * t37
  t40 = jnp.sqrt(t39)
  t53 = jnp.logical_or(r0 / 0.2e1 <= p.dens_threshold, t5)
  t54 = jnp.where(t5, p.zeta_threshold, 1)
  t56 = jnp.cbrt(0.1e1 / jnp.pi)
  t57 = t1 * t56
  t58 = jnp.cbrt(4)
  t59 = t58 ** 2
  t61 = 0.1e1 / t9
  t64 = jnp.where(t5, 0.1e1 / t6, 1)
  t66 = t57 * t59 * t61 * t11 * t64
  t69 = jnp.sqrt(t66)
  t72 = t66 ** 0.15e1
  t74 = t1 ** 2
  t75 = t56 ** 2
  t76 = t74 * t75
  t78 = 0.1e1 / t14
  t80 = t64 ** 2
  t82 = t76 * t58 * t78 * t12 * t80
  t88 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t69 + 0.8969 * t66 + 0.204775 * t72 + 0.123235 * t82))
  t90 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t66) * t88
  t93 = jnp.where(0.2e1 <= p.zeta_threshold, t7, 0.2e1 * t11)
  t95 = jnp.where(0. <= p.zeta_threshold, t7, 0)
  t99 = 0.1e1 / (0.2e1 * t11 - 0.2e1)
  t100 = (t93 + t95 - 0.2e1) * t99
  t111 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t69 + 0.1549425e1 * t66 + 0.420775 * t72 + 0.1562925 * t82))
  t124 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t69 + 0.905775 * t66 + 0.1100325 * t72 + 0.1241775 * t82))
  t125 = (0.1e1 + 0.278125e-1 * t66) * t124
  t134 = jnp.where(t53, 0, t54 * (-t90 + t100 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t66) * t111 + t90 - 0.19751789702565206229e-1 * t125) + 0.19751789702565206229e-1 * t100 * t125) / 0.2e1)
  t136 = 0.1e1 + 0.256e1 * t37
  t137 = jnp.sqrt(t136)
  t155 = t57 * t59 * t61
  t158 = jnp.sqrt(t155)
  t161 = t155 ** 0.15e1
  t164 = t76 * t58 * t78
  t170 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t158 + 0.8969 * t155 + 0.204775 * t161 + 0.123235 * t164))
  t186 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t158 + 0.905775 * t155 + 0.1100325 * t161 + 0.1241775 * t164))
  t193 = 0.1e1 + 0.196e-1 * t37
  t194 = jnp.sqrt(t193)
  res = -0.3e1 / 0.4e1 * t1 / t2 * t8 * t9 * (0.8085 + 0.73502e-1 * t36 / t40 + 0.17182e-2 * t37 / t39) + 0.2e1 * t134 * (0.2606 - 0.153728e1 * t36 / t137 + 0.2309888e1 * t37 / t136) * (0.1e1 - s0 / r0 / tau0 / 0.8e1) + (-0.62182e-1 * (0.1e1 + 0.53425e-1 * t155) * t170 + 0.19751789702565206229e-1 * (0.2e1 * t8 - 0.2e1) * t99 * (0.1e1 + 0.278125e-1 * t155) * t186 - 0.2e1 * t134) * (0.12033e1 - 0.318038 * t36 / t194 + 0.1880816e-1 * t37 / t193)
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