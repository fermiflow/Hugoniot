"""Generated from gga_c_bmk.mpl."""

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
  t51 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t30 + 0.8969 * t27 + 0.204775 * t33 + 0.123235 * t45))
  t53 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t27) * t51
  t55 = t21 * p.zeta_threshold
  t57 = jnp.where(0.2e1 <= p.zeta_threshold, t55, 0.2e1 * t19)
  t59 = jnp.where(0. <= p.zeta_threshold, t55, 0)
  t63 = 0.1e1 / (0.2e1 * t19 - 0.2e1)
  t64 = (t57 + t59 - 0.2e1) * t63
  t75 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t30 + 0.1549425e1 * t27 + 0.420775 * t33 + 0.1562925 * t45))
  t88 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t30 + 0.905775 * t27 + 0.1100325 * t33 + 0.1241775 * t45))
  t89 = (0.1e1 + 0.278125e-1 * t27) * t88
  t98 = jnp.where(t8, 0, t9 * (-t53 + t64 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t27) * t75 + t53 - 0.19751789702565206229e-1 * t89) + 0.19751789702565206229e-1 * t64 * t89) / 0.2e1)
  t99 = params.c_ss[0]
  t100 = params.c_ss[1]
  t102 = r0 ** 2
  t103 = jnp.cbrt(r0)
  t104 = t103 ** 2
  t106 = 0.1e1 / t104 / t102
  t107 = s0 * t106
  t109 = 0.1e1 + 0.2 * t107
  t114 = params.c_ss[2]
  t115 = s0 ** 2
  t117 = t102 ** 2
  t121 = t109 ** 2
  t126 = params.c_ss[3]
  t129 = t117 ** 2
  t136 = params.c_ss[4]
  t137 = t115 ** 2
  t142 = t121 ** 2
  t150 = 0.1e1 - t5
  t151 = t150 <= p.zeta_threshold
  t152 = jnp.logical_or(r1 <= p.dens_threshold, t151)
  t153 = jnp.where(t151, p.zeta_threshold, t150)
  t154 = jnp.cbrt(t150)
  t156 = jnp.where(t151, t22, 0.1e1 / t154)
  t158 = t16 * t20 * t156
  t161 = jnp.sqrt(t158)
  t164 = t158 ** 0.15e1
  t166 = t156 ** 2
  t168 = t38 * t42 * t166
  t174 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t161 + 0.8969 * t158 + 0.204775 * t164 + 0.123235 * t168))
  t176 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t158) * t174
  t187 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t161 + 0.1549425e1 * t158 + 0.420775 * t164 + 0.1562925 * t168))
  t200 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t161 + 0.905775 * t158 + 0.1100325 * t164 + 0.1241775 * t168))
  t201 = (0.1e1 + 0.278125e-1 * t158) * t200
  t210 = jnp.where(t152, 0, t153 * (-t176 + t64 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t158) * t187 + t176 - 0.19751789702565206229e-1 * t201) + 0.19751789702565206229e-1 * t64 * t201) / 0.2e1)
  t212 = r1 ** 2
  t213 = jnp.cbrt(r1)
  t214 = t213 ** 2
  t216 = 0.1e1 / t214 / t212
  t217 = s2 * t216
  t219 = 0.1e1 + 0.2 * t217
  t224 = s2 ** 2
  t226 = t212 ** 2
  t230 = t219 ** 2
  t237 = t226 ** 2
  t244 = t224 ** 2
  t249 = t230 ** 2
  t257 = t13 * t15 * t18
  t260 = jnp.sqrt(t257)
  t263 = t257 ** 0.15e1
  t266 = t37 * t14 * t40
  t272 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t260 + 0.8969 * t257 + 0.204775 * t263 + 0.123235 * t266))
  t274 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t257) * t272
  t275 = t2 ** 2
  t276 = t275 ** 2
  t277 = t3 ** 2
  t278 = t277 ** 2
  t282 = jnp.where(t7, t55, t23 * t6)
  t284 = jnp.where(t151, t55, t154 * t150)
  t286 = (t282 + t284 - 0.2e1) * t63
  t297 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t260 + 0.1549425e1 * t257 + 0.420775 * t263 + 0.1562925 * t266))
  t310 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t260 + 0.905775 * t257 + 0.1100325 * t263 + 0.1241775 * t266))
  t311 = (0.1e1 + 0.278125e-1 * t257) * t310
  t321 = t107 + t217
  t325 = 0.1e1 + 0.3e-2 * t107 + 0.3e-2 * t217
  t330 = t321 ** 2
  t332 = t325 ** 2
  t344 = t330 ** 2
  t346 = t332 ** 2
  res = t98 * (t99 + 0.2 * t100 * s0 * t106 / t109 + 0.4e-1 * t114 * t115 / t103 / t117 / r0 / t121 + 0.8e-2 * t126 * t115 * s0 / t129 / t121 / t109 + 0.16e-2 * t136 * t137 / t104 / t129 / t102 / t142) + t210 * (t99 + 0.2 * t100 * s2 * t216 / t219 + 0.4e-1 * t114 * t224 / t213 / t226 / r1 / t230 + 0.8e-2 * t126 * t224 * s2 / t237 / t230 / t219 + 0.16e-2 * t136 * t244 / t214 / t237 / t212 / t249) + (-t274 + t276 / t278 * t286 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t257) * t297 + t274 - 0.19751789702565206229e-1 * t311) + 0.19751789702565206229e-1 * t286 * t311 - t98 - t210) * (params.c_ab[0] + 0.3e-2 * params.c_ab[1] * t321 / t325 + 0.9e-5 * params.c_ab[2] * t330 / t332 + 0.27e-7 * params.c_ab[3] * t330 * t321 / t332 / t325 + 0.81e-10 * params.c_ab[4] * t344 / t346)
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
  t45 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t24 + 0.8969 * t21 + 0.204775 * t27 + 0.123235 * t39))
  t47 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t21) * t45
  t49 = t17 * p.zeta_threshold
  t51 = jnp.where(0.2e1 <= p.zeta_threshold, t49, 0.2e1 * t15)
  t53 = jnp.where(0. <= p.zeta_threshold, t49, 0)
  t57 = 0.1e1 / (0.2e1 * t15 - 0.2e1)
  t58 = (t51 + t53 - 0.2e1) * t57
  t69 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t24 + 0.1549425e1 * t21 + 0.420775 * t27 + 0.1562925 * t39))
  t82 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t24 + 0.905775 * t21 + 0.1100325 * t27 + 0.1241775 * t39))
  t83 = (0.1e1 + 0.278125e-1 * t21) * t82
  t92 = jnp.where(t4, 0, t5 * (-t47 + t58 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t21) * t69 + t47 - 0.19751789702565206229e-1 * t83) + 0.19751789702565206229e-1 * t58 * t83) / 0.2e1)
  t96 = r0 ** 2
  t98 = 0.1e1 / t33 / t96
  t99 = t35 * t98
  t101 = s0 * t35 * t98
  t103 = 0.1e1 + 0.2 * t101
  t109 = s0 ** 2
  t111 = t96 ** 2
  t115 = t15 / t13 / t111 / r0
  t116 = t103 ** 2
  t122 = t109 * s0
  t124 = t111 ** 2
  t125 = 0.1e1 / t124
  t132 = t109 ** 2
  t137 = t35 / t33 / t124 / t96
  t138 = t116 ** 2
  t147 = t9 * t11 * t14
  t150 = jnp.sqrt(t147)
  t153 = t147 ** 0.15e1
  t156 = t31 * t10 * t34
  t162 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t150 + 0.8969 * t147 + 0.204775 * t153 + 0.123235 * t156))
  t165 = jnp.where(t3, t49, 1)
  t179 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t150 + 0.905775 * t147 + 0.1100325 * t153 + 0.1241775 * t156))
  t189 = 0.1e1 + 0.6e-2 * t101
  t196 = t189 ** 2
  t210 = t196 ** 2
  res = 0.2e1 * t92 * (params.c_ss[0] + 0.2 * params.c_ss[1] * s0 * t99 / t103 + 0.8e-1 * params.c_ss[2] * t109 * t115 / t116 + 0.32e-1 * params.c_ss[3] * t122 * t125 / t116 / t103 + 0.64e-2 * params.c_ss[4] * t132 * t137 / t138) + (-0.62182e-1 * (0.1e1 + 0.53425e-1 * t147) * t162 + 0.19751789702565206229e-1 * (0.2e1 * t165 - 0.2e1) * t57 * (0.1e1 + 0.278125e-1 * t147) * t179 - 0.2e1 * t92) * (params.c_ab[0] + 0.6e-2 * params.c_ab[1] * s0 * t99 / t189 + 0.72e-4 * params.c_ab[2] * t109 * t115 / t196 + 0.864e-6 * params.c_ab[3] * t122 * t125 / t196 / t189 + 0.5184e-8 * params.c_ab[4] * t132 * t137 / t210)
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