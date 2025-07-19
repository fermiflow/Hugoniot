"""Generated from mgga_c_rmggac.mpl."""

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
  t3 = jnp.cbrt(0.1e1 / jnp.pi)
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = r0 + r1
  t8 = jnp.cbrt(t7)
  t11 = t1 * t3 * t6 / t8
  t12 = jnp.sqrt(t11)
  t16 = 0.1e1 / (0.1e1 + 0.4445e-1 * t12 + 0.3138525e-1 * t11)
  t19 = jnp.exp(0.1e1 * t16)
  t21 = jnp.cbrt(6)
  t22 = jnp.pi ** 2
  t23 = jnp.cbrt(t22)
  t24 = t23 ** 2
  t26 = t21 / t24
  t27 = jnp.cbrt(2)
  t28 = t27 ** 2
  t30 = s0 + 0.2e1 * s1 + s2
  t32 = t7 ** 2
  t33 = t8 ** 2
  t35 = 0.1e1 / t33 / t32
  t37 = t26 * t28 * t30 * t35
  t40 = (0.1e1 + 0.21337642104376358333e-1 * t37) ** (0.1e1 / 0.4e1)
  t45 = jnp.log(0.1e1 + (t19 - 0.1e1) * (0.1e1 - 0.1e1 / t40))
  t48 = t27 - 0.1e1
  t49 = r0 - r1
  t51 = t49 / t7
  t52 = 0.1e1 + t51
  t53 = t52 <= p.zeta_threshold
  t54 = jnp.cbrt(p.zeta_threshold)
  t55 = t54 * p.zeta_threshold
  t56 = jnp.cbrt(t52)
  t58 = jnp.where(t53, t55, t56 * t52)
  t59 = 0.1e1 - t51
  t60 = t59 <= p.zeta_threshold
  t61 = jnp.cbrt(t59)
  t63 = jnp.where(t60, t55, t61 * t59)
  t64 = t58 + t63 - 0.2e1
  t67 = 0.1e1 / t48 / 0.2e1
  t72 = t49 ** 2
  t73 = t72 ** 2
  t74 = t73 ** 2
  t76 = t32 ** 2
  t77 = t76 ** 2
  t82 = jnp.cbrt(r0)
  t83 = t82 ** 2
  t87 = t52 / 0.2e1
  t88 = jnp.cbrt(t87)
  t89 = t88 ** 2
  t93 = jnp.cbrt(r1)
  t94 = t93 ** 2
  t98 = t59 / 0.2e1
  t99 = jnp.cbrt(t98)
  t100 = t99 ** 2
  t106 = 0.2e1 * tau0 / t83 / r0 * t89 * t87 + 0.2e1 * tau1 / t94 / r1 * t100 * t98 - t30 * t35 / 0.4e1
  t107 = t106 ** 2
  t108 = t107 * t106
  t109 = t22 ** 2
  t110 = 0.1e1 / t109
  t111 = t108 * t110
  t116 = 0.8e-1 + 0.5e1 / 0.18e2 * t106 * t28 * t26 + 0.125e-1 * t37
  t117 = t116 ** 2
  t119 = 0.1e1 / t117 / t116
  t122 = t107 ** 2
  t124 = t109 ** 2
  t127 = t117 ** 2
  t133 = 0.1e1 / (0.1e1 + 0.648 * t111 * t119 + 0.419904 * t122 * t107 / t124 / t127 / t117)
  t144 = t11 ** 0.15e1
  t146 = t1 ** 2
  t147 = t3 ** 2
  t151 = t146 * t147 * t5 / t33
  t157 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t12 + 0.8969 * t11 + 0.204775 * t144 + 0.123235 * t151))
  t159 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t11) * t157
  t162 = t64 * t67
  t173 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t12 + 0.1549425e1 * t11 + 0.420775 * t144 + 0.1562925 * t151))
  t186 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t12 + 0.905775 * t11 + 0.1100325 * t144 + 0.1241775 * t151))
  t187 = (0.1e1 + 0.278125e-1 * t11) * t186
  t191 = t73 / t76 * t162 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t11) * t173 + t159 - 0.19751673498613801407e-1 * t187)
  t193 = 0.19751673498613801407e-1 * t162 * t187
  t194 = t54 ** 2
  t195 = t56 ** 2
  t196 = jnp.where(t53, t194, t195)
  t197 = t61 ** 2
  t198 = jnp.where(t60, t194, t197)
  t200 = t196 / 0.2e1 + t198 / 0.2e1
  t201 = t200 ** 2
  t202 = t201 * t200
  t203 = -t159 + t191 + t193
  t204 = 0.1e1 / t202
  t207 = jnp.exp(-0.32163648644302209643e2 * t203 * t204)
  t209 = jnp.log(0.2e1)
  t211 = 0.1e1 / (0.1e1 - t209)
  t216 = jnp.exp(-t203 * t211 * t22 * t204)
  t233 = (0.1e1 + 0.27802083333333333333e-2 * t211 * t22 / (t216 - 0.1e1) * t30 / t8 / t32 * t27 / t201 * t146 / t3 * t5) ** (0.1e1 / 0.4e1)
  t238 = jnp.log(0.1e1 + (t207 - 0.1e1) * (0.1e1 - 0.1e1 / t233))
  res = (-0.285764e-1 * t16 + 0.285764e-1 * t45) * (0.1e1 - 0.2363e1 * t48 * t64 * t67) * (0.1e1 - t74 * t73 / t77 / t76) * (0.1e1 - 0.1944e1 * t111 * t119 * t133) + 0.1944e1 * (-t159 + t191 + t193 + 0.31091e-1 * t202 * t238) * t108 * t110 * t119 * t133
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(3)
  t3 = jnp.cbrt(0.1e1 / jnp.pi)
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = jnp.cbrt(r0)
  t10 = t1 * t3 * t6 / t7
  t11 = jnp.sqrt(t10)
  t15 = 0.1e1 / (0.1e1 + 0.4445e-1 * t11 + 0.3138525e-1 * t10)
  t18 = jnp.exp(0.1e1 * t15)
  t20 = jnp.cbrt(6)
  t21 = jnp.pi ** 2
  t22 = jnp.cbrt(t21)
  t23 = t22 ** 2
  t25 = t20 / t23
  t26 = jnp.cbrt(2)
  t27 = t26 ** 2
  t29 = r0 ** 2
  t30 = t7 ** 2
  t32 = 0.1e1 / t30 / t29
  t34 = t25 * t27 * s0 * t32
  t37 = (0.1e1 + 0.21337642104376358333e-1 * t34) ** (0.1e1 / 0.4e1)
  t42 = jnp.log(0.1e1 + (t18 - 0.1e1) * (0.1e1 - 0.1e1 / t37))
  t45 = t26 - 0.1e1
  t46 = 0.1e1 <= p.zeta_threshold
  t47 = jnp.cbrt(p.zeta_threshold)
  t49 = jnp.where(t46, t47 * p.zeta_threshold, 1)
  t51 = 0.2e1 * t49 - 0.2e1
  t54 = 0.1e1 / t45 / 0.2e1
  t65 = 0.2e1 * tau0 / t30 / r0 - s0 * t32 / 0.4e1
  t66 = t65 ** 2
  t67 = t66 * t65
  t68 = t21 ** 2
  t69 = 0.1e1 / t68
  t70 = t67 * t69
  t75 = 0.8e-1 + 0.5e1 / 0.18e2 * t65 * t27 * t25 + 0.125e-1 * t34
  t76 = t75 ** 2
  t78 = 0.1e1 / t76 / t75
  t81 = t66 ** 2
  t83 = t68 ** 2
  t86 = t76 ** 2
  t92 = 0.1e1 / (0.1e1 + 0.648 * t70 * t78 + 0.419904 * t81 * t66 / t83 / t86 / t76)
  t102 = t10 ** 0.15e1
  t104 = t1 ** 2
  t105 = t3 ** 2
  t109 = t104 * t105 * t5 / t30
  t115 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t11 + 0.8969 * t10 + 0.204775 * t102 + 0.123235 * t109))
  t117 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t10) * t115
  t129 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t11 + 0.905775 * t10 + 0.1100325 * t102 + 0.1241775 * t109))
  t132 = 0.19751673498613801407e-1 * t51 * t54 * (0.1e1 + 0.278125e-1 * t10) * t129
  t133 = t47 ** 2
  t134 = jnp.where(t46, t133, 1)
  t135 = t134 ** 2
  t136 = t135 * t134
  t137 = -t117 + t132
  t138 = 0.1e1 / t136
  t141 = jnp.exp(-0.32163648644302209643e2 * t137 * t138)
  t143 = jnp.log(0.2e1)
  t145 = 0.1e1 / (0.1e1 - t143)
  t150 = jnp.exp(-t137 * t145 * t21 * t138)
  t167 = (0.1e1 + 0.27802083333333333333e-2 * t145 * t21 / (t150 - 0.1e1) * s0 / t7 / t29 * t26 / t135 * t104 / t3 * t5) ** (0.1e1 / 0.4e1)
  t172 = jnp.log(0.1e1 + (t141 - 0.1e1) * (0.1e1 - 0.1e1 / t167))
  res = (-0.285764e-1 * t15 + 0.285764e-1 * t42) * (0.1e1 - 0.2363e1 * t45 * t51 * t54) * (0.1e1 - 0.1944e1 * t70 * t78 * t92) + 0.1944e1 * (-t117 + t132 + 0.31091e-1 * t136 * t172) * t67 * t69 * t78 * t92
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