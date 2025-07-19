"""Generated from mgga_x_regtpss.mpl."""

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
  t33 = (s0 / r0 / tau0) ** 0.3e1
  t34 = s0 ** 2
  t35 = r0 ** 2
  t38 = tau0 ** 2
  t40 = t34 / t35 / t38
  t43 = (0.1e1 + t40 / 0.64e2) ** 2
  t48 = jnp.cbrt(6)
  t50 = jnp.pi ** 2
  t51 = jnp.cbrt(t50)
  t52 = t51 ** 2
  t53 = 0.1e1 / t52
  t55 = jnp.cbrt(r0)
  t56 = t55 ** 2
  t58 = 0.1e1 / t56 / t35
  t65 = s0 * t58
  t67 = tau0 / t56 / r0 - t65 / 0.8e1
  t68 = t67 * t48
  t71 = 0.5e1 / 0.9e1 * t68 * t53 - 0.1e1
  t76 = jnp.sqrt(0.1e1 + 0.22222222222222222222 * t68 * t53 * t71)
  t80 = t48 * t53
  t81 = t80 * t65
  t82 = t81 / 0.36e2
  t83 = 0.9e1 / 0.2e2 * t71 / t76 + t82
  t84 = t83 ** 2
  t87 = t48 ** 2
  t89 = 0.1e1 / t51 / t50
  t90 = t87 * t89
  t91 = t35 ** 2
  t96 = t90 * t34 / t55 / t91 / r0
  t97 = 0.5e2 * t96
  t99 = jnp.sqrt(0.162e3 * t40 + t97)
  t102 = 0.32911784453572541028e-4 * t96
  t104 = t50 ** 2
  t105 = 0.1e1 / t104
  t108 = t91 ** 2
  t111 = 0.13171780538194444444e-3 * t105 * t34 * s0 / t108
  t115 = (0.1e1 + 0.61346278355378295562e-1 * t81) ** 2
  t116 = 0.1e1 / t115
  t120 = 0.646416 / (0.804 + ((0.1e2 / 0.81e2 + 0.45938270703125e-2 * t33 / t43) * t48 * t53 * s0 * t58 / 0.24e2 + 0.146e3 / 0.2025e4 * t84 - 0.73e2 / 0.972e5 * t83 * t99 + t102 + 0.20448759451792765188e-2 * t40 + t111) * t116)
  t121 = -t71
  t122 = t121 ** 2
  t124 = t67 ** 2
  t128 = 0.1e1 + 0.67148919753086419753 * t124 * t87 * t89
  t129 = jnp.sqrt(t128)
  t134 = jnp.exp(-t81 / 0.8e1)
  t136 = -0.45 + t82
  t137 = t136 ** 2
  t140 = jnp.sqrt(0.10368e5 + t97)
  t155 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1804e1 - t120 + t122 * t121 / t129 / t128 * t134 * (-0.646416 / (0.804 + (0.29644443963477366255e-1 * t81 + 0.146e3 / 0.2025e4 * t137 - 0.73e2 / 0.972e5 * t136 * t140 + t102 + 0.1308720604914736972 + t111) * t116) + t120)))
  t157 = jnp.where(t10, t15, -t17)
  t158 = jnp.where(t14, t11, t157)
  t159 = 0.1e1 + t158
  t161 = jnp.cbrt(t159)
  t163 = jnp.where(t159 <= p.zeta_threshold, t23, t161 * t159)
  t169 = (s2 / r1 / tau1) ** 0.3e1
  t170 = s2 ** 2
  t171 = r1 ** 2
  t174 = tau1 ** 2
  t176 = t170 / t171 / t174
  t179 = (0.1e1 + t176 / 0.64e2) ** 2
  t186 = jnp.cbrt(r1)
  t187 = t186 ** 2
  t189 = 0.1e1 / t187 / t171
  t196 = s2 * t189
  t198 = tau1 / t187 / r1 - t196 / 0.8e1
  t199 = t198 * t48
  t202 = 0.5e1 / 0.9e1 * t199 * t53 - 0.1e1
  t207 = jnp.sqrt(0.1e1 + 0.22222222222222222222 * t199 * t53 * t202)
  t211 = t80 * t196
  t212 = t211 / 0.36e2
  t213 = 0.9e1 / 0.2e2 * t202 / t207 + t212
  t214 = t213 ** 2
  t217 = t171 ** 2
  t222 = t90 * t170 / t186 / t217 / r1
  t223 = 0.5e2 * t222
  t225 = jnp.sqrt(0.162e3 * t176 + t223)
  t228 = 0.32911784453572541028e-4 * t222
  t232 = t217 ** 2
  t235 = 0.13171780538194444444e-3 * t105 * t170 * s2 / t232
  t239 = (0.1e1 + 0.61346278355378295562e-1 * t211) ** 2
  t240 = 0.1e1 / t239
  t244 = 0.646416 / (0.804 + ((0.1e2 / 0.81e2 + 0.45938270703125e-2 * t169 / t179) * t48 * t53 * s2 * t189 / 0.24e2 + 0.146e3 / 0.2025e4 * t214 - 0.73e2 / 0.972e5 * t213 * t225 + t228 + 0.20448759451792765188e-2 * t176 + t235) * t240)
  t245 = -t202
  t246 = t245 ** 2
  t248 = t198 ** 2
  t252 = 0.1e1 + 0.67148919753086419753 * t248 * t87 * t89
  t253 = jnp.sqrt(t252)
  t258 = jnp.exp(-t211 / 0.8e1)
  t260 = -0.45 + t212
  t261 = t260 ** 2
  t264 = jnp.sqrt(0.10368e5 + t223)
  t279 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t163 * t27 * (0.1804e1 - t244 + t246 * t245 / t253 / t252 * t258 * (-0.646416 / (0.804 + (0.29644443963477366255e-1 * t211 + 0.146e3 / 0.2025e4 * t261 - 0.73e2 / 0.972e5 * t260 * t264 + t228 + 0.1308720604914736972 + t235) * t240) + t244)))
  res = t155 + t279
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
  t25 = (s0 / r0 / tau0) ** 0.3e1
  t26 = s0 ** 2
  t27 = r0 ** 2
  t30 = tau0 ** 2
  t32 = t26 / t27 / t30
  t35 = (0.1e1 + t32 / 0.64e2) ** 2
  t40 = jnp.cbrt(6)
  t42 = jnp.pi ** 2
  t43 = jnp.cbrt(t42)
  t44 = t43 ** 2
  t45 = 0.1e1 / t44
  t47 = jnp.cbrt(2)
  t48 = t47 ** 2
  t50 = t19 ** 2
  t53 = s0 * t48 / t50 / t27
  t61 = tau0 * t48 / t50 / r0 - t53 / 0.8e1
  t62 = t61 * t40
  t65 = 0.5e1 / 0.9e1 * t62 * t45 - 0.1e1
  t70 = jnp.sqrt(0.1e1 + 0.22222222222222222222 * t62 * t45 * t65)
  t75 = t40 * t45 * t53
  t76 = t75 / 0.36e2
  t77 = 0.9e1 / 0.2e2 * t65 / t70 + t76
  t78 = t77 ** 2
  t81 = t40 ** 2
  t83 = 0.1e1 / t43 / t42
  t86 = t27 ** 2
  t91 = t81 * t83 * t26 * t47 / t19 / t86 / r0
  t94 = jnp.sqrt(0.162e3 * t32 + 0.1e3 * t91)
  t97 = 0.65823568907145082056e-4 * t91
  t99 = t42 ** 2
  t103 = t86 ** 2
  t106 = 0.52687122152777777776e-3 / t99 * t26 * s0 / t103
  t110 = (0.1e1 + 0.61346278355378295562e-1 * t75) ** 2
  t111 = 0.1e1 / t110
  t115 = 0.646416 / (0.804 + ((0.1e2 / 0.81e2 + 0.45938270703125e-2 * t25 / t35) * t40 * t45 * t53 / 0.24e2 + 0.146e3 / 0.2025e4 * t78 - 0.73e2 / 0.972e5 * t77 * t94 + t97 + 0.20448759451792765188e-2 * t32 + t106) * t111)
  t116 = -t65
  t117 = t116 ** 2
  t119 = t61 ** 2
  t123 = 0.1e1 + 0.67148919753086419753 * t119 * t81 * t83
  t124 = jnp.sqrt(t123)
  t129 = jnp.exp(-t75 / 0.8e1)
  t131 = -0.45 + t76
  t132 = t131 ** 2
  t136 = jnp.sqrt(0.2592e4 + 0.25e2 * t91)
  t151 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1804e1 - t115 + t117 * t116 / t124 / t123 * t129 * (-0.646416 / (0.804 + (0.29644443963477366255e-1 * t75 + 0.146e3 / 0.2025e4 * t132 - 0.73e2 / 0.486e5 * t131 * t136 + t97 + 0.1308720604914736972 + t106) * t111) + t115)))
  res = 0.2e1 * t151
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