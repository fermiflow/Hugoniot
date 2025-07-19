"""Generated from gga_x_am05.mpl."""

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
  t30 = params.alpha * t29
  t31 = jnp.pi ** 2
  t32 = jnp.cbrt(t31)
  t33 = t32 ** 2
  t34 = 0.1e1 / t33
  t35 = t30 * t34
  t36 = r0 ** 2
  t37 = jnp.cbrt(r0)
  t38 = t37 ** 2
  t40 = 0.1e1 / t38 / t36
  t42 = t34 * s0
  t43 = t42 * t40
  t47 = 0.1e1 / (0.1e1 + t30 * t43 / 0.24e2)
  t53 = params.c * t29
  t57 = t29 ** 2
  t58 = params.c * t57
  t59 = 0.1e1 / t32
  t60 = jnp.sqrt(s0)
  t63 = 0.1e1 / t37 / r0
  t67 = t2 ** 2
  t68 = 0.1e1 / jnp.pi * t67
  t69 = jnp.cbrt(2)
  t70 = jnp.sqrt(0.12e2)
  t71 = t57 * t59
  t73 = t71 * t60 * t63
  t74 = jnp.sqrt(t73)
  t77 = jnp.sqrt(0.6e1)
  t80 = lambertw(t70 * t74 * t73 * t77 / 0.1728e4)
  t81 = jnp.cbrt(t80)
  t82 = t81 ** 2
  t84 = t69 ** 2
  t85 = t2 * t84
  t90 = (0.2823705740248932030511071641312341561894e2 + 0.3e1 / 0.4e1 * t85 * t81 * t80) ** (0.1e1 / 0.4e1)
  t105 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 - t35 * s0 * t40 * t47 / 0.24e2 + t30 * t42 * t40 * t47 * (0.1e1 + t53 * t43 / 0.24e2) / (0.1e1 + t58 * t59 * t60 * t63 * t68 * t69 * t82 * t90 / 0.8e1) / 0.24e2))
  t107 = jnp.where(t10, t15, -t17)
  t108 = jnp.where(t14, t11, t107)
  t109 = 0.1e1 + t108
  t111 = jnp.cbrt(t109)
  t113 = jnp.where(t109 <= p.zeta_threshold, t23, t111 * t109)
  t115 = r1 ** 2
  t116 = jnp.cbrt(r1)
  t117 = t116 ** 2
  t119 = 0.1e1 / t117 / t115
  t121 = t34 * s2
  t122 = t121 * t119
  t126 = 0.1e1 / (0.1e1 + t30 * t122 / 0.24e2)
  t135 = jnp.sqrt(s2)
  t138 = 0.1e1 / t116 / r1
  t142 = t71 * t135 * t138
  t143 = jnp.sqrt(t142)
  t148 = lambertw(t70 * t143 * t142 * t77 / 0.1728e4)
  t149 = jnp.cbrt(t148)
  t150 = t149 ** 2
  t156 = (0.2823705740248932030511071641312341561894e2 + 0.3e1 / 0.4e1 * t85 * t149 * t148) ** (0.1e1 / 0.4e1)
  t171 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t113 * t27 * (0.1e1 - t35 * s2 * t119 * t126 / 0.24e2 + t30 * t121 * t119 * t126 * (0.1e1 + t53 * t122 / 0.24e2) / (0.1e1 + t58 * t59 * t135 * t138 * t68 * t69 * t150 * t156 / 0.8e1) / 0.24e2))
  res = t105 + t171
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
  t22 = params.alpha * t21
  t23 = jnp.pi ** 2
  t24 = jnp.cbrt(t23)
  t25 = t24 ** 2
  t26 = 0.1e1 / t25
  t27 = t22 * t26
  t28 = jnp.cbrt(2)
  t29 = t28 ** 2
  t30 = s0 * t29
  t31 = r0 ** 2
  t32 = t19 ** 2
  t34 = 0.1e1 / t32 / t31
  t35 = t30 * t34
  t39 = 0.1e1 / (0.1e1 + t27 * t35 / 0.24e2)
  t53 = t21 ** 2
  t55 = 0.1e1 / t24
  t56 = jnp.sqrt(s0)
  t61 = 0.1e1 / t19 / r0
  t64 = t3 ** 2
  t65 = jnp.sqrt(0.12e2)
  t69 = t53 * t55 * t56 * t28 * t61
  t70 = jnp.sqrt(t69)
  t73 = jnp.sqrt(0.6e1)
  t76 = lambertw(t65 * t70 * t69 * t73 / 0.1728e4)
  t77 = jnp.cbrt(t76)
  t78 = t77 ** 2
  t85 = (0.2823705740248932030511071641312341561894e2 + 0.3e1 / 0.4e1 * t3 * t29 * t77 * t76) ** (0.1e1 / 0.4e1)
  t100 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 - t27 * t30 * t34 * t39 / 0.24e2 + t22 * t26 * s0 * t29 * t34 * t39 * (0.1e1 + params.c * t21 * t26 * t35 / 0.24e2) / (0.1e1 + params.c * t53 * t55 * t56 * t29 * t61 / jnp.pi * t64 * t78 * t85 / 0.8e1) / 0.24e2))
  res = 0.2e1 * t100
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