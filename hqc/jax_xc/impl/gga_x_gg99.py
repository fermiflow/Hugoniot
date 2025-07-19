"""Generated from gga_x_gg99.mpl."""

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
  t3 = t2 ** 2
  t4 = jnp.cbrt(jnp.pi)
  t7 = t3 / t4 / jnp.pi
  t8 = r0 + r1
  t9 = 0.1e1 / t8
  t12 = 0.2e1 * r0 * t9 <= p.zeta_threshold
  t13 = p.zeta_threshold - 0.1e1
  t16 = 0.2e1 * r1 * t9 <= p.zeta_threshold
  t17 = -t13
  t19 = (r0 - r1) * t9
  t20 = jnp.where(t16, t17, t19)
  t21 = jnp.where(t12, t13, t20)
  t22 = 0.1e1 + t21
  t24 = jnp.cbrt(p.zeta_threshold)
  t25 = t24 * p.zeta_threshold
  t26 = jnp.cbrt(t22)
  t28 = jnp.where(t22 <= p.zeta_threshold, t25, t26 * t22)
  t29 = jnp.cbrt(t8)
  t32 = jnp.pi ** 2
  t33 = jnp.sqrt(s0)
  t34 = jnp.cbrt(r0)
  t37 = t33 / t34 / r0
  t38 = jnp.cbrt(4)
  t39 = jnp.sqrt(0.3e1)
  t41 = t39 * t32 * jnp.pi
  t42 = jnp.cbrt(t41)
  t43 = t38 * t42
  t45 = 3 ** (0.1e1 / 0.4e1)
  t46 = jnp.sqrt(0.2e1)
  t48 = jnp.sqrt(jnp.pi)
  t51 = t45 * t46 / t48 / jnp.pi
  t52 = t43 - 0.1e-9
  t54 = jnp.where(t52 < t37, t52, t37)
  t55 = t54 ** 2
  t56 = 0.4e1 * t41
  t57 = t32 ** 2
  t58 = t57 * t32
  t59 = 0.48e2 * t58
  t60 = t55 ** 2
  t63 = jnp.sqrt(-t60 * t55 + t59)
  t64 = t56 + t63
  t65 = jnp.cbrt(t64)
  t66 = t65 ** 2
  t68 = jnp.sqrt(t55 + t66)
  t70 = t64 ** (0.1e1 / 0.6e1)
  t75 = jnp.arcsinh(t51 * t54 * t68 / t70 / 0.4e1)
  t76 = 0.1e1 / jnp.pi
  t77 = t43 + 0.1e-9
  t79 = jnp.where(t77 < t37, t37, t77)
  t80 = t79 ** 2
  t83 = 0.1e1 / t58
  t84 = t80 ** 2
  t89 = jnp.sqrt(0.3e1 * t83 * t84 * t80 - 0.144e3)
  t91 = jnp.arctan(t89 / 0.12e2)
  t93 = jnp.cos(t91 / 0.3e1)
  t96 = jnp.sqrt(t80 * t79 * t39 * t76 * t93)
  t99 = jnp.arcsinh(t76 * t96 / 0.2e1)
  t100 = jnp.where(t37 < t43, t75, t99)
  t102 = jnp.exp(-0.2e1 * t100)
  t104 = jnp.log(0.1e1 + t102)
  t107 = my_dilog(-t102)
  t112 = 0.1e1 / jnp.cosh(t100)
  t113 = jnp.cbrt(t112)
  t114 = t113 ** 2
  t116 = jnp.cbrt(t76)
  t117 = 0.1e1 / t116
  t123 = jnp.where(r0 <= p.dens_threshold, 0, -t7 * t28 * t29 * (-0.12e2 * t100 * t104 + 0.12e2 * t107 + t32) / t100 / t114 * t117 * t38 / 0.24e2)
  t125 = jnp.where(t12, t17, -t19)
  t126 = jnp.where(t16, t13, t125)
  t127 = 0.1e1 + t126
  t129 = jnp.cbrt(t127)
  t131 = jnp.where(t127 <= p.zeta_threshold, t25, t129 * t127)
  t134 = jnp.sqrt(s2)
  t135 = jnp.cbrt(r1)
  t138 = t134 / t135 / r1
  t141 = jnp.where(t52 < t138, t52, t138)
  t142 = t141 ** 2
  t143 = t142 ** 2
  t146 = jnp.sqrt(-t143 * t142 + t59)
  t147 = t56 + t146
  t148 = jnp.cbrt(t147)
  t149 = t148 ** 2
  t151 = jnp.sqrt(t142 + t149)
  t153 = t147 ** (0.1e1 / 0.6e1)
  t158 = jnp.arcsinh(t51 * t141 * t151 / t153 / 0.4e1)
  t160 = jnp.where(t77 < t138, t138, t77)
  t161 = t160 ** 2
  t164 = t161 ** 2
  t169 = jnp.sqrt(0.3e1 * t83 * t164 * t161 - 0.144e3)
  t171 = jnp.arctan(t169 / 0.12e2)
  t173 = jnp.cos(t171 / 0.3e1)
  t176 = jnp.sqrt(t161 * t160 * t39 * t76 * t173)
  t179 = jnp.arcsinh(t76 * t176 / 0.2e1)
  t180 = jnp.where(t138 < t43, t158, t179)
  t182 = jnp.exp(-0.2e1 * t180)
  t184 = jnp.log(0.1e1 + t182)
  t187 = my_dilog(-t182)
  t192 = 0.1e1 / jnp.cosh(t180)
  t193 = jnp.cbrt(t192)
  t194 = t193 ** 2
  t201 = jnp.where(r1 <= p.dens_threshold, 0, -t7 * t131 * t29 * (-0.12e2 * t180 * t184 + 0.12e2 * t187 + t32) / t180 / t194 * t117 * t38 / 0.24e2)
  res = t123 + t201
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = jnp.cbrt(3)
  t4 = t3 ** 2
  t5 = jnp.cbrt(jnp.pi)
  t9 = 0.1e1 <= p.zeta_threshold
  t10 = p.zeta_threshold - 0.1e1
  t12 = jnp.where(t9, -t10, 0)
  t13 = jnp.where(t9, t10, t12)
  t14 = 0.1e1 + t13
  t16 = jnp.cbrt(p.zeta_threshold)
  t18 = jnp.cbrt(t14)
  t20 = jnp.where(t14 <= p.zeta_threshold, t16 * p.zeta_threshold, t18 * t14)
  t21 = jnp.cbrt(r0)
  t24 = jnp.pi ** 2
  t25 = jnp.sqrt(s0)
  t26 = jnp.cbrt(2)
  t30 = t25 * t26 / t21 / r0
  t31 = jnp.cbrt(4)
  t32 = jnp.sqrt(0.3e1)
  t34 = t32 * t24 * jnp.pi
  t35 = jnp.cbrt(t34)
  t36 = t31 * t35
  t38 = 3 ** (0.1e1 / 0.4e1)
  t39 = jnp.sqrt(0.2e1)
  t41 = jnp.sqrt(jnp.pi)
  t45 = t36 - 0.1e-9
  t47 = jnp.where(t45 < t30, t45, t30)
  t48 = t47 ** 2
  t50 = t24 ** 2
  t51 = t50 * t24
  t53 = t48 ** 2
  t56 = jnp.sqrt(-t53 * t48 + 0.48e2 * t51)
  t57 = 0.4e1 * t34 + t56
  t58 = jnp.cbrt(t57)
  t59 = t58 ** 2
  t61 = jnp.sqrt(t48 + t59)
  t63 = t57 ** (0.1e1 / 0.6e1)
  t68 = jnp.arcsinh(t38 * t39 / t41 / jnp.pi * t47 * t61 / t63 / 0.4e1)
  t69 = 0.1e1 / jnp.pi
  t70 = t36 + 0.1e-9
  t72 = jnp.where(t70 < t30, t30, t70)
  t73 = t72 ** 2
  t77 = t73 ** 2
  t82 = jnp.sqrt(0.3e1 / t51 * t77 * t73 - 0.144e3)
  t84 = jnp.arctan(t82 / 0.12e2)
  t86 = jnp.cos(t84 / 0.3e1)
  t89 = jnp.sqrt(t73 * t72 * t32 * t69 * t86)
  t92 = jnp.arcsinh(t69 * t89 / 0.2e1)
  t93 = jnp.where(t30 < t36, t68, t92)
  t95 = jnp.exp(-0.2e1 * t93)
  t97 = jnp.log(0.1e1 + t95)
  t100 = my_dilog(-t95)
  t105 = 0.1e1 / jnp.cosh(t93)
  t106 = jnp.cbrt(t105)
  t107 = t106 ** 2
  t109 = jnp.cbrt(t69)
  t116 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -t4 / t5 / jnp.pi * t20 * t21 * (-0.12e2 * t93 * t97 + 0.12e2 * t100 + t24) / t93 / t107 / t109 * t31 / 0.24e2)
  res = 0.2e1 * t116
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