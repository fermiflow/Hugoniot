"""Generated from mgga_c_bc95.mpl."""

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
  t2 = r0 - r1
  t3 = r0 + r1
  t5 = t2 / t3
  t6 = 0.1e1 + t5
  t7 = t6 <= p.zeta_threshold
  t8 = jnp.logical_or(r0 <= p.dens_threshold, t7)
  t9 = jnp.where(t7, p.zeta_threshold, t6)
  t10 = jnp.cbrt(3)
  t12 = jnp.cbrt(0.1e1 / jnp.pi)
  t13 = t10 * t12
  t14 = jnp.cbrt(4)
  t15 = t14 ** 2
  t16 = t13 * t15
  t17 = jnp.cbrt(t3)
  t18 = 0.1e1 / t17
  t19 = jnp.cbrt(2)
  t20 = t18 * t19
  t21 = jnp.cbrt(p.zeta_threshold)
  t22 = 0.1e1 / t21
  t23 = jnp.cbrt(t6)
  t25 = jnp.where(t7, t22, 0.1e1 / t23)
  t27 = t16 * t20 * t25
  t30 = jnp.sqrt(t27)
  t33 = t27 ** 0.15e1
  t35 = t10 ** 2
  t36 = t12 ** 2
  t37 = t35 * t36
  t38 = t37 * t14
  t39 = t17 ** 2
  t40 = 0.1e1 / t39
  t41 = t19 ** 2
  t42 = t40 * t41
  t43 = t25 ** 2
  t45 = t38 * t42 * t43
  t51 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t30 + 0.8969 * t27 + 0.204775 * t33 + 0.123235 * t45))
  t53 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t27) * t51
  t55 = t21 * p.zeta_threshold
  t57 = jnp.where(0.2e1 <= p.zeta_threshold, t55, 0.2e1 * t19)
  t59 = jnp.where(0. <= p.zeta_threshold, t55, 0)
  t63 = 0.1e1 / (0.2e1 * t19 - 0.2e1)
  t64 = (t57 + t59 - 0.2e1) * t63
  t75 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t30 + 0.1549425e1 * t27 + 0.420775 * t33 + 0.1562925 * t45))
  t88 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t30 + 0.905775 * t27 + 0.1100325 * t33 + 0.1241775 * t45))
  t89 = (0.1e1 + 0.278125e-1 * t27) * t88
  t98 = jnp.where(t8, 0, t9 * (-t53 + t64 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t27) * t75 + t53 - 0.19751673498613801407e-1 * t89) + 0.19751673498613801407e-1 * t64 * t89) / 0.2e1)
  t100 = jnp.cbrt(r0)
  t101 = t100 ** 2
  t111 = jnp.cbrt(6)
  t113 = jnp.pi ** 2
  t114 = jnp.cbrt(t113)
  t115 = t114 ** 2
  t116 = 0.1e1 / t115
  t118 = r0 ** 2
  t120 = 0.1e1 / t101 / t118
  t123 = (params.css * s0 * t120 + 0.1e1) ** 2
  t130 = 0.1e1 - t5
  t131 = t130 <= p.zeta_threshold
  t132 = jnp.logical_or(r1 <= p.dens_threshold, t131)
  t133 = jnp.where(t131, p.zeta_threshold, t130)
  t134 = jnp.cbrt(t130)
  t136 = jnp.where(t131, t22, 0.1e1 / t134)
  t138 = t16 * t20 * t136
  t141 = jnp.sqrt(t138)
  t144 = t138 ** 0.15e1
  t146 = t136 ** 2
  t148 = t38 * t42 * t146
  t154 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t141 + 0.8969 * t138 + 0.204775 * t144 + 0.123235 * t148))
  t156 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t138) * t154
  t167 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t141 + 0.1549425e1 * t138 + 0.420775 * t144 + 0.1562925 * t148))
  t180 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t141 + 0.905775 * t138 + 0.1100325 * t144 + 0.1241775 * t148))
  t181 = (0.1e1 + 0.278125e-1 * t138) * t180
  t190 = jnp.where(t132, 0, t133 * (-t156 + t64 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t138) * t167 + t156 - 0.19751673498613801407e-1 * t181) + 0.19751673498613801407e-1 * t64 * t181) / 0.2e1)
  t192 = jnp.cbrt(r1)
  t193 = t192 ** 2
  t205 = r1 ** 2
  t207 = 0.1e1 / t193 / t205
  t210 = (params.css * s2 * t207 + 0.1e1) ** 2
  t217 = t13 * t15 * t18
  t220 = jnp.sqrt(t217)
  t223 = t217 ** 0.15e1
  t226 = t37 * t14 * t40
  t232 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t220 + 0.8969 * t217 + 0.204775 * t223 + 0.123235 * t226))
  t234 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t217) * t232
  t235 = t2 ** 2
  t236 = t235 ** 2
  t237 = t3 ** 2
  t238 = t237 ** 2
  t242 = jnp.where(t7, t55, t23 * t6)
  t244 = jnp.where(t131, t55, t134 * t130)
  t246 = (t242 + t244 - 0.2e1) * t63
  t257 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t220 + 0.1549425e1 * t217 + 0.420775 * t223 + 0.1562925 * t226))
  t270 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t220 + 0.905775 * t217 + 0.1100325 * t223 + 0.1241775 * t226))
  t271 = (0.1e1 + 0.278125e-1 * t217) * t270
  res = 0.5e1 / 0.9e1 * t98 * tau0 / t101 / r0 * (0.1e1 - s0 / r0 / tau0 / 0.8e1) * t111 * t116 / t123 + 0.5e1 / 0.9e1 * t190 * tau1 / t193 / r1 * (0.1e1 - s2 / r1 / tau1 / 0.8e1) * t111 * t116 / t210 + (-t234 + t236 / t238 * t246 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t217) * t257 + t234 - 0.19751673498613801407e-1 * t271) + 0.19751673498613801407e-1 * t246 * t271 - t98 - t190) / (0.1e1 + params.copp * (s0 * t120 + s2 * t207))
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = 0.1e1 <= p.zeta_threshold
  t4 = jnp.logical_or(r0 / 0.2e1 <= p.dens_threshold, t3)
  t5 = jnp.where(t3, p.zeta_threshold, 1)
  t6 = jnp.cbrt(3)
  t8 = jnp.cbrt(0.1e1 / jnp.pi)
  t9 = t6 * t8
  t10 = jnp.cbrt(4)
  t11 = t10 ** 2
  t13 = jnp.cbrt(r0)
  t14 = 0.1e1 / t13
  t15 = jnp.cbrt(2)
  t17 = jnp.cbrt(p.zeta_threshold)
  t19 = jnp.where(t3, 0.1e1 / t17, 1)
  t21 = t9 * t11 * t14 * t15 * t19
  t24 = jnp.sqrt(t21)
  t27 = t21 ** 0.15e1
  t29 = t6 ** 2
  t30 = t8 ** 2
  t31 = t29 * t30
  t33 = t13 ** 2
  t34 = 0.1e1 / t33
  t35 = t15 ** 2
  t37 = t19 ** 2
  t39 = t31 * t10 * t34 * t35 * t37
  t45 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t24 + 0.8969 * t21 + 0.204775 * t27 + 0.123235 * t39))
  t47 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t21) * t45
  t49 = t17 * p.zeta_threshold
  t51 = jnp.where(0.2e1 <= p.zeta_threshold, t49, 0.2e1 * t15)
  t53 = jnp.where(0. <= p.zeta_threshold, t49, 0)
  t57 = 0.1e1 / (0.2e1 * t15 - 0.2e1)
  t58 = (t51 + t53 - 0.2e1) * t57
  t69 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t24 + 0.1549425e1 * t21 + 0.420775 * t27 + 0.1562925 * t39))
  t82 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t24 + 0.905775 * t21 + 0.1100325 * t27 + 0.1241775 * t39))
  t83 = (0.1e1 + 0.278125e-1 * t21) * t82
  t92 = jnp.where(t4, 0, t5 * (-t47 + t58 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t21) * t69 + t47 - 0.19751673498613801407e-1 * t83) + 0.19751673498613801407e-1 * t58 * t83) / 0.2e1)
  t104 = jnp.cbrt(6)
  t106 = jnp.pi ** 2
  t107 = jnp.cbrt(t106)
  t108 = t107 ** 2
  t111 = r0 ** 2
  t114 = t35 / t33 / t111
  t117 = (params.css * s0 * t114 + 0.1e1) ** 2
  t124 = t9 * t11 * t14
  t127 = jnp.sqrt(t124)
  t130 = t124 ** 0.15e1
  t133 = t31 * t10 * t34
  t139 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t127 + 0.8969 * t124 + 0.204775 * t130 + 0.123235 * t133))
  t142 = jnp.where(t3, t49, 1)
  t156 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t127 + 0.905775 * t124 + 0.1100325 * t130 + 0.1241775 * t133))
  res = 0.1e2 / 0.9e1 * t92 * tau0 * t35 / t33 / r0 * (0.1e1 - s0 / r0 / tau0 / 0.8e1) * t104 / t108 / t117 + (-0.621814e-1 * (0.1e1 + 0.53425e-1 * t124) * t139 + 0.19751673498613801407e-1 * (0.2e1 * t142 - 0.2e1) * t57 * (0.1e1 + 0.278125e-1 * t124) * t156 - 0.2e1 * t92) / (0.2e1 * params.copp * s0 * t114 + 0.1e1)
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