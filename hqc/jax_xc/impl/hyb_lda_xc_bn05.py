"""Generated from hyb_lda_xc_bn05.mpl."""

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
  t4 = t1 * t3
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = t4 * t6
  t8 = jnp.cbrt(2)
  t9 = t8 ** 2
  t10 = r0 - r1
  t11 = r0 + r1
  t13 = t10 / t11
  t14 = 0.1e1 + t13
  t15 = t14 <= p.zeta_threshold
  t16 = jnp.cbrt(p.zeta_threshold)
  t17 = t16 * p.zeta_threshold
  t18 = jnp.cbrt(t14)
  t20 = jnp.where(t15, t17, t18 * t14)
  t22 = jnp.cbrt(t11)
  t23 = jnp.cbrt(9)
  t24 = t23 ** 2
  t25 = t3 ** 2
  t27 = t24 * t25 * p.cam_omega
  t28 = 0.1e1 / t22
  t29 = t1 * t28
  t30 = jnp.where(t15, t16, t18)
  t34 = t27 * t29 / t30 / 0.18e2
  t36 = 0.192e1 < t34
  t37 = jnp.where(t36, t34, 0.192e1)
  t38 = t37 ** 2
  t41 = t38 ** 2
  t44 = t41 * t38
  t47 = t41 ** 2
  t50 = t47 * t38
  t53 = t47 * t41
  t56 = t47 * t44
  t59 = t47 ** 2
  t83 = t59 ** 2
  t92 = 0.1e1 / t38 / 0.9e1 - 0.1e1 / t41 / 0.3e2 + 0.1e1 / t44 / 0.7e2 - 0.1e1 / t47 / 0.135e3 + 0.1e1 / t50 / 0.231e3 - 0.1e1 / t53 / 0.364e3 + 0.1e1 / t56 / 0.54e3 - 0.1e1 / t59 / 0.765e3 + 0.1e1 / t59 / t38 / 0.1045e4 - 0.1e1 / t59 / t41 / 0.1386e4 + 0.1e1 / t59 / t44 / 0.1794e4 - 0.1e1 / t59 / t47 / 0.2275e4 + 0.1e1 / t59 / t50 / 0.2835e4 - 0.1e1 / t59 / t53 / 0.348e4 + 0.1e1 / t59 / t56 / 0.4216e4 - 0.1e1 / t83 / 0.5049e4 + 0.1e1 / t83 / t38 / 0.5985e4 - 0.1e1 / t83 / t41 / 0.703e4
  t93 = jnp.where(t36, 0.192e1, t34)
  t94 = jnp.arctan2(0.1e1, t93)
  t95 = t93 ** 2
  t99 = jnp.log(0.1e1 + 0.1e1 / t95)
  t108 = jnp.where(0.192e1 <= t34, t92, 0.1e1 - 0.8e1 / 0.3e1 * t93 * (t94 + t93 * (0.1e1 - (t95 + 0.3e1) * t99) / 0.4e1))
  t113 = 0.1e1 - t13
  t114 = t113 <= p.zeta_threshold
  t115 = jnp.cbrt(t113)
  t117 = jnp.where(t114, t17, t115 * t113)
  t119 = jnp.where(t114, t16, t115)
  t123 = t27 * t29 / t119 / 0.18e2
  t125 = 0.192e1 < t123
  t126 = jnp.where(t125, t123, 0.192e1)
  t127 = t126 ** 2
  t130 = t127 ** 2
  t133 = t130 * t127
  t136 = t130 ** 2
  t139 = t136 * t127
  t142 = t136 * t130
  t145 = t136 * t133
  t148 = t136 ** 2
  t172 = t148 ** 2
  t181 = 0.1e1 / t127 / 0.9e1 - 0.1e1 / t130 / 0.3e2 + 0.1e1 / t133 / 0.7e2 - 0.1e1 / t136 / 0.135e3 + 0.1e1 / t139 / 0.231e3 - 0.1e1 / t142 / 0.364e3 + 0.1e1 / t145 / 0.54e3 - 0.1e1 / t148 / 0.765e3 + 0.1e1 / t148 / t127 / 0.1045e4 - 0.1e1 / t148 / t130 / 0.1386e4 + 0.1e1 / t148 / t133 / 0.1794e4 - 0.1e1 / t148 / t136 / 0.2275e4 + 0.1e1 / t148 / t139 / 0.2835e4 - 0.1e1 / t148 / t142 / 0.348e4 + 0.1e1 / t148 / t145 / 0.4216e4 - 0.1e1 / t172 / 0.5049e4 + 0.1e1 / t172 / t127 / 0.5985e4 - 0.1e1 / t172 / t130 / 0.703e4
  t182 = jnp.where(t125, 0.192e1, t123)
  t183 = jnp.arctan2(0.1e1, t182)
  t184 = t182 ** 2
  t188 = jnp.log(0.1e1 + 0.1e1 / t184)
  t197 = jnp.where(0.192e1 <= t123, t181, 0.1e1 - 0.8e1 / 0.3e1 * t182 * (t183 + t182 * (0.1e1 - (t184 + 0.3e1) * t188) / 0.4e1))
  t203 = t4 * t6 * t28
  t206 = jnp.sqrt(t203)
  t209 = t203 ** 0.15e1
  t211 = t1 ** 2
  t213 = t22 ** 2
  t216 = t211 * t25 * t5 / t213
  t222 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t206 + 0.8969 * t203 + 0.204775 * t209 + 0.123235 * t216))
  t224 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t203) * t222
  t225 = t10 ** 2
  t226 = t225 ** 2
  t227 = t11 ** 2
  t228 = t227 ** 2
  t235 = (t20 + t117 - 0.2e1) / (0.2e1 * t8 - 0.2e1)
  t246 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t206 + 0.1549425e1 * t203 + 0.420775 * t209 + 0.1562925 * t216))
  t259 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t206 + 0.905775 * t203 + 0.1100325 * t209 + 0.1241775 * t216))
  t260 = (0.1e1 + 0.278125e-1 * t203) * t259
  res = -0.3e1 / 0.32e2 * t7 * t9 * t20 * t22 * t108 - 0.3e1 / 0.32e2 * t7 * t9 * t117 * t22 * t197 + 0.34602e1 * (-t224 + t226 / t228 * t235 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t203) * t246 + t224 - 0.19751673498613801407e-1 * t260) + 0.19751673498613801407e-1 * t235 * t260) / (0.32e1 - 0.225 * t203 + t216 / 0.4e1)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(3)
  t3 = jnp.cbrt(0.1e1 / jnp.pi)
  t4 = t1 * t3
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t8 = jnp.cbrt(2)
  t9 = t8 ** 2
  t10 = 0.1e1 <= p.zeta_threshold
  t11 = jnp.cbrt(p.zeta_threshold)
  t13 = jnp.where(t10, t11 * p.zeta_threshold, 1)
  t15 = jnp.cbrt(r0)
  t16 = jnp.cbrt(9)
  t17 = t16 ** 2
  t18 = t3 ** 2
  t21 = 0.1e1 / t15
  t23 = jnp.where(t10, t11, 1)
  t27 = t17 * t18 * p.cam_omega * t1 * t21 / t23 / 0.18e2
  t29 = 0.192e1 < t27
  t30 = jnp.where(t29, t27, 0.192e1)
  t31 = t30 ** 2
  t34 = t31 ** 2
  t37 = t34 * t31
  t40 = t34 ** 2
  t43 = t40 * t31
  t46 = t40 * t34
  t49 = t40 * t37
  t52 = t40 ** 2
  t76 = t52 ** 2
  t85 = 0.1e1 / t31 / 0.9e1 - 0.1e1 / t34 / 0.3e2 + 0.1e1 / t37 / 0.7e2 - 0.1e1 / t40 / 0.135e3 + 0.1e1 / t43 / 0.231e3 - 0.1e1 / t46 / 0.364e3 + 0.1e1 / t49 / 0.54e3 - 0.1e1 / t52 / 0.765e3 + 0.1e1 / t52 / t31 / 0.1045e4 - 0.1e1 / t52 / t34 / 0.1386e4 + 0.1e1 / t52 / t37 / 0.1794e4 - 0.1e1 / t52 / t40 / 0.2275e4 + 0.1e1 / t52 / t43 / 0.2835e4 - 0.1e1 / t52 / t46 / 0.348e4 + 0.1e1 / t52 / t49 / 0.4216e4 - 0.1e1 / t76 / 0.5049e4 + 0.1e1 / t76 / t31 / 0.5985e4 - 0.1e1 / t76 / t34 / 0.703e4
  t86 = jnp.where(t29, 0.192e1, t27)
  t87 = jnp.arctan2(0.1e1, t86)
  t88 = t86 ** 2
  t92 = jnp.log(0.1e1 + 0.1e1 / t88)
  t101 = jnp.where(0.192e1 <= t27, t85, 0.1e1 - 0.8e1 / 0.3e1 * t86 * (t87 + t86 * (0.1e1 - (t88 + 0.3e1) * t92) / 0.4e1))
  t107 = t4 * t6 * t21
  t110 = jnp.sqrt(t107)
  t113 = t107 ** 0.15e1
  t115 = t1 ** 2
  t117 = t15 ** 2
  t120 = t115 * t18 * t5 / t117
  t126 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t110 + 0.8969 * t107 + 0.204775 * t113 + 0.123235 * t120))
  t145 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t110 + 0.905775 * t107 + 0.1100325 * t113 + 0.1241775 * t120))
  res = -0.3e1 / 0.16e2 * t4 * t6 * t9 * t13 * t15 * t101 + 0.34602e1 * (-0.621814e-1 * (0.1e1 + 0.53425e-1 * t107) * t126 + 0.19751673498613801407e-1 * (0.2e1 * t13 - 0.2e1) / (0.2e1 * t8 - 0.2e1) * (0.1e1 + 0.278125e-1 * t107) * t145) / (0.32e1 - 0.225 * t107 + t120 / 0.4e1)
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