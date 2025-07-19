"""Generated from gga_c_sogga11.mpl."""

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
  t92 = -t33 + t36 / t38 * t60 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t11) * t71 + t33 - 0.19751673498613801407e-1 * t85) + 0.19751673498613801407e-1 * t60 * t85
  t95 = t45 ** 2
  t96 = t47 ** 2
  t97 = jnp.where(t44, t95, t96)
  t98 = t52 ** 2
  t99 = jnp.where(t51, t95, t98)
  t115 = 0.69506584583333333332e-3 * t56 * (t97 / 0.2e1 + t99 / 0.2e1) * (s0 + 0.2e1 * s1 + s2) / t8 / t37 * t19 / t3 * t5 / t92
  t118 = 0.1e1 - 0.1e1 / (0.1e1 - t115)
  t121 = t118 ** 2
  t127 = t121 ** 2
  t134 = jnp.exp(t115)
  t135 = 0.1e1 - t134
  t138 = t135 ** 2
  t144 = t138 ** 2
  t149 = params.sogga11_a[3] * t121 * t118 + params.sogga11_a[5] * t127 * t118 + params.sogga11_b[3] * t138 * t135 + params.sogga11_b[5] * t144 * t135 + params.sogga11_a[1] * t118 + params.sogga11_a[2] * t121 + params.sogga11_a[4] * t127 + params.sogga11_b[1] * t135 + params.sogga11_b[2] * t138 + params.sogga11_b[4] * t144 + params.sogga11_a[0] + params.sogga11_b[0]
  res = t92 * t149
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
  t33 = 0.1e1 <= p.zeta_threshold
  t34 = jnp.cbrt(p.zeta_threshold)
  t36 = jnp.where(t33, t34 * p.zeta_threshold, 1)
  t39 = jnp.cbrt(2)
  t54 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t13 + 0.905775 * t10 + 0.1100325 * t16 + 0.1241775 * t24))
  t58 = -0.621814e-1 * (0.1e1 + 0.53425e-1 * t10) * t30 + 0.19751673498613801407e-1 * (0.2e1 * t36 - 0.2e1) / (0.2e1 * t39 - 0.2e1) * (0.1e1 + 0.278125e-1 * t10) * t54
  t61 = t34 ** 2
  t62 = jnp.where(t33, t61, 1)
  t64 = r0 ** 2
  t75 = 0.69506584583333333332e-3 * t39 * t62 * s0 / t7 / t64 * t18 / t3 * t5 / t58
  t78 = 0.1e1 - 0.1e1 / (0.1e1 - t75)
  t81 = t78 ** 2
  t87 = t81 ** 2
  t94 = jnp.exp(t75)
  t95 = 0.1e1 - t94
  t98 = t95 ** 2
  t104 = t98 ** 2
  t109 = params.sogga11_b[5] * t104 * t95 + params.sogga11_a[3] * t81 * t78 + params.sogga11_a[5] * t87 * t78 + params.sogga11_b[3] * t98 * t95 + params.sogga11_b[4] * t104 + params.sogga11_a[1] * t78 + params.sogga11_a[2] * t81 + params.sogga11_a[4] * t87 + params.sogga11_b[1] * t95 + params.sogga11_b[2] * t98 + params.sogga11_a[0] + params.sogga11_b[0]
  res = t58 * t109
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