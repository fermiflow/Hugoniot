"""Generated from gga_c_zvpbeint.mpl."""

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
  t14 = jnp.sqrt(t11)
  t17 = t11 ** 0.15e1
  t19 = t1 ** 2
  t20 = t3 ** 2
  t22 = t8 ** 2
  t25 = t19 * t20 * t5 / t22
  t31 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t14 + 0.8969 * t11 + 0.204775 * t17 + 0.123235 * t25))
  t33 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t11) * t31
  t34 = r0 - r1
  t35 = t34 ** 2
  t36 = t35 ** 2
  t37 = t7 ** 2
  t38 = t37 ** 2
  t39 = 0.1e1 / t38
  t42 = t34 / t7
  t43 = 0.1e1 + t42
  t44 = t43 <= p.zeta_threshold
  t45 = jnp.cbrt(p.zeta_threshold)
  t46 = t45 * p.zeta_threshold
  t47 = jnp.cbrt(t43)
  t49 = jnp.where(t44, t46, t47 * t43)
  t50 = 0.1e1 - t42
  t51 = t50 <= p.zeta_threshold
  t52 = jnp.cbrt(t50)
  t54 = jnp.where(t51, t46, t52 * t50)
  t56 = jnp.cbrt(2)
  t60 = (t49 + t54 - 0.2e1) / (0.2e1 * t56 - 0.2e1)
  t71 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t14 + 0.1549425e1 * t11 + 0.420775 * t17 + 0.1562925 * t25))
  t84 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t14 + 0.905775 * t11 + 0.1100325 * t17 + 0.1241775 * t25))
  t85 = (0.1e1 + 0.278125e-1 * t11) * t84
  t89 = t36 * t39 * t60 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t11) * t71 + t33 - 0.19751673498613801407e-1 * t85)
  t91 = 0.19751673498613801407e-1 * t60 * t85
  t93 = s0 + 0.2e1 * s1 + s2
  t94 = jnp.sqrt(t93)
  t100 = 0.1e1 / t3
  t104 = jnp.sqrt(t19 * t100 * t5 * t8)
  t107 = t35 / t37
  t109 = jnp.where(0.1e-19 < t107, t107, 0.1e-19)
  t111 = t109 ** (params.omega / 0.2e1)
  t115 = jnp.exp(-params.alpha * t94 * t93 * t39 / t14 / t11 * t104 * t111 / 0.16e2)
  t116 = jnp.log(0.2e1)
  t117 = 0.1e1 - t116
  t119 = jnp.pi ** 2
  t121 = t45 ** 2
  t122 = t47 ** 2
  t123 = jnp.where(t44, t121, t122)
  t124 = t52 ** 2
  t125 = jnp.where(t51, t121, t124)
  t127 = t123 / 0.2e1 + t125 / 0.2e1
  t128 = t127 ** 2
  t129 = t128 * t127
  t141 = 0.1e1 / t117
  t142 = params.beta * t141
  t148 = jnp.exp(-(-t33 + t89 + t91) * t141 * t119 / t129)
  t151 = t119 / (t148 - 0.1e1)
  t152 = t93 ** 2
  t157 = t56 ** 2
  t159 = t128 ** 2
  t168 = t93 / t8 / t37 * t56 / t128 * t19 * t100 * t5 / 0.96e2 + t142 * t151 * t152 / t22 / t38 * t157 / t159 * t1 / t20 * t6 / 0.3072e4
  t178 = jnp.log(0.1e1 + params.beta * t168 * t141 * t119 / (t142 * t151 * t168 + 0.1e1))
  res = -t33 + t89 + t91 + t115 * t117 / t119 * t129 * t178
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
  t13 = jnp.sqrt(t10)
  t16 = t10 ** 0.15e1
  t18 = t1 ** 2
  t19 = t3 ** 2
  t21 = t7 ** 2
  t24 = t18 * t19 * t5 / t21
  t30 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t13 + 0.8969 * t10 + 0.204775 * t16 + 0.123235 * t24))
  t32 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t10) * t30
  t33 = 0.1e1 <= p.zeta_threshold
  t34 = jnp.cbrt(p.zeta_threshold)
  t36 = jnp.where(t33, t34 * p.zeta_threshold, 1)
  t39 = jnp.cbrt(2)
  t54 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t13 + 0.905775 * t10 + 0.1100325 * t16 + 0.1241775 * t24))
  t57 = 0.19751673498613801407e-1 * (0.2e1 * t36 - 0.2e1) / (0.2e1 * t39 - 0.2e1) * (0.1e1 + 0.278125e-1 * t10) * t54
  t58 = jnp.sqrt(s0)
  t61 = r0 ** 2
  t62 = t61 ** 2
  t67 = 0.1e1 / t3
  t71 = jnp.sqrt(t18 * t67 * t5 * t7)
  t74 = jnp.where(0.1e-19 < 0., 0, 0.1e-19)
  t76 = t74 ** (params.omega / 0.2e1)
  t80 = jnp.exp(-params.alpha * t58 * s0 / t62 / t13 / t10 * t71 * t76 / 0.16e2)
  t81 = jnp.log(0.2e1)
  t82 = 0.1e1 - t81
  t84 = jnp.pi ** 2
  t86 = t34 ** 2
  t87 = jnp.where(t33, t86, 1)
  t88 = t87 ** 2
  t89 = t88 * t87
  t101 = 0.1e1 / t82
  t102 = params.beta * t101
  t108 = jnp.exp(-(-t32 + t57) * t101 * t84 / t89)
  t111 = t84 / (t108 - 0.1e1)
  t112 = s0 ** 2
  t117 = t39 ** 2
  t119 = t88 ** 2
  t128 = s0 / t7 / t61 * t39 / t88 * t18 * t67 * t5 / 0.96e2 + t102 * t111 * t112 / t21 / t62 * t117 / t119 * t1 / t19 * t6 / 0.3072e4
  t138 = jnp.log(0.1e1 + params.beta * t128 * t101 * t84 / (t102 * t111 * t128 + 0.1e1))
  res = -t32 + t57 + t80 * t82 / t84 * t89 * t138
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