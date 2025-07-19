"""Generated from lda_c_pmgb06.mpl."""

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
  t1 = r0 - r1
  t2 = r0 + r1
  t3 = 0.1e1 / t2
  t4 = t1 * t3
  t5 = 0.1e1 + t4
  t6 = t5 <= p.zeta_threshold
  t7 = jnp.cbrt(p.zeta_threshold)
  t8 = t7 ** 2
  t9 = jnp.cbrt(t5)
  t10 = t9 ** 2
  t11 = jnp.where(t6, t8, t10)
  t12 = 0.1e1 - t4
  t13 = t12 <= p.zeta_threshold
  t14 = jnp.cbrt(t12)
  t15 = t14 ** 2
  t16 = jnp.where(t13, t8, t15)
  t18 = t11 / 0.2e1 + t16 / 0.2e1
  t19 = t18 ** 2
  t20 = t19 * t18
  t21 = jnp.log(0.2e1)
  t22 = t21 - 0.1e1
  t25 = jnp.pi ** 2
  t26 = 0.1e1 / t25
  t27 = jnp.cbrt(3)
  t28 = 0.1e1 / jnp.pi
  t29 = jnp.cbrt(t28)
  t30 = t27 * t29
  t31 = jnp.cbrt(4)
  t32 = t31 ** 2
  t33 = jnp.cbrt(t2)
  t34 = 0.1e1 / t33
  t35 = t32 * t34
  t36 = t30 * t35
  t37 = jnp.sqrt(t36)
  t41 = 0.2923025e1 * p.cam_omega * t37 / t18
  t43 = jnp.cbrt(9)
  t44 = t43 ** 2
  t52 = p.cam_omega ** 2
  t55 = t29 * t32
  t56 = 0.1e1 / t19
  t61 = t52 * p.cam_omega
  t76 = jnp.log((0.1e1 + t41 + (0.344851e1 - jnp.pi * t31 * t44 * t29 / t22 / 0.12e2) * t52 * t27 * t55 * t34 * t56 / 0.4e1 + 0.48968 * t61 * t37 * t36 / t20) / (0.1e1 + t41 + 0.8621275 * t52 * t27 * t29 * t35 * t56))
  t79 = jnp.sqrt(jnp.pi)
  t83 = t1 ** 2
  t84 = t2 ** 2
  t85 = 0.1e1 / t84
  t86 = t83 * t85
  t87 = 0.1e1 - t86
  t101 = t27 ** 2
  t102 = t29 ** 2
  t103 = t101 * t102
  t104 = t33 ** 2
  t105 = 0.1e1 / t104
  t106 = t31 * t105
  t107 = t103 * t106
  t109 = t28 * t3
  t111 = t29 * t28
  t112 = t27 * t111
  t115 = t32 / t33 / t2
  t120 = jnp.exp(-0.1881 * t36)
  t121 = (0.1e1 - (0.2e1 / 0.45e2 * t31 * t44 * t29 * (t25 + 0.6e1 * t21 - 0.3e1) * t28 - 0.7524) * t27 * t55 * t34 / 0.4e1 + 0.204825e-1 * t107 - 0.95775e-2 * t109 + 0.3485625e-3 * t112 * t115) * t120
  t122 = jnp.sqrt(0.2e1)
  t127 = t101 * t102 * t26
  t128 = t127 * t31
  t130 = 0.1e1 / t104 / t2
  t131 = p.zeta_threshold ** 2
  t132 = t5 ** 2
  t133 = jnp.where(t6, t131, t132)
  t136 = 0.1e1 / t111 * t27
  t139 = jnp.cbrt(0.1e1 / t5)
  t140 = t139 ** 2
  t143 = t30 * t32
  t144 = jnp.cbrt(2)
  t145 = t34 * t144
  t147 = t143 * t145 * t139
  t151 = t103 * t31
  t152 = t144 ** 2
  t153 = t105 * t152
  t162 = t133 * t44 * t136 * t104 / t140 * (0.1e1 - 0.56675e-2 * t147) / (0.1e1 + 0.107975 * t147 + 0.1e-1 * t151 * t153 * t140) / 0.3e2
  t163 = t12 ** 2
  t164 = jnp.where(t13, t131, t163)
  t168 = jnp.cbrt(0.1e1 / t12)
  t169 = t168 ** 2
  t173 = t143 * t145 * t168
  t185 = t164 * t44 * t136 * t104 / t169 * (0.1e1 - 0.56675e-2 * t173) / (0.1e1 + 0.107975 * t173 + 0.1e-1 * t151 * t153 * t169) / 0.3e2
  t191 = jnp.exp(-0.775e-1 * t36)
  t198 = t122 * t79
  t205 = t121 / 0.2e1 - 0.1e1 / 0.2e1 + t86 / 0.2e1
  t214 = jnp.exp(-0.13675 * t36)
  t222 = t8 * t131
  t224 = jnp.where(t6, t222, t10 * t132)
  t226 = jnp.where(t13, t222, t15 * t163)
  t242 = t36 ** 0.15e1
  t249 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t37 + 0.8969 * t36 + 0.204775 * t242 + 0.123235 * t107))
  t251 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t36) * t249
  t252 = t83 ** 2
  t253 = t84 ** 2
  t256 = t7 * p.zeta_threshold
  t258 = jnp.where(t6, t256, t9 * t5)
  t260 = jnp.where(t13, t256, t14 * t12)
  t265 = (t258 + t260 - 0.2e1) / (0.2e1 * t144 - 0.2e1)
  t276 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t37 + 0.1549425e1 * t36 + 0.420775 * t242 + 0.1562925 * t107))
  t289 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t37 + 0.905775 * t36 + 0.1100325 * t242 + 0.1241775 * t107))
  t290 = (0.1e1 + 0.278125e-1 * t36) * t289
  t297 = -t251 + t252 / t253 * t265 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t36) * t276 + t251 - 0.19751673498613801407e-1 * t290) + 0.19751673498613801407e-1 * t265 * t290
  t302 = t52 ** 2
  t326 = t302 ** 2
  t335 = (0.1e1 + 0.15403623315025 * t103 * t106 * t52) ** 2
  t336 = t335 ** 2
  res = (0.2e1 * t20 * t22 * t26 * t76 + (-0.17543244109220059985 / t79 / jnp.pi * t3 * t87 * t121 * t122 - 0.3040082144797017415e-2 * t128 * t130 * (t162 + t185 + 0.4e1 / 0.3e1 * t87 * (-0.12375e1 * t36 + t107 / 0.4e1) * t191 * jnp.pi * t2) * t198) * t61 + (-0.26314866163830089978 * t109 * t87 * t205 - 0.38001026809962717688e-2 * t128 * t130 * (t162 + t185 + t87 * (-0.97e-1 * t36 + 0.169 * t107) * t214 * t27 / t102 * t32 * t104 / 0.3e1 - (t224 / 0.2e1 + t226 / 0.2e1) * t44 * t136 * t104 / 0.15e2) * jnp.pi + 0.42708890021612718669 * t112 * t115 * t297) * t302 - 0.6755738099548927589e-2 * t127 * t31 * t130 * t87 * t121 * t198 * t302 * p.cam_omega + (-0.10133607149323391384e-1 * t128 * t130 * t87 * t205 * jnp.pi + 0.52629732327660179956 * t26 * t85 * t297) * t302 * t52 + 0.20267214298646782767e-1 * t128 / t104 / t84 * t297 * t326) / t336
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = 0.1e1 <= p.zeta_threshold
  t2 = jnp.cbrt(p.zeta_threshold)
  t3 = t2 ** 2
  t4 = jnp.where(t1, t3, 1)
  t5 = t4 ** 2
  t6 = t5 * t4
  t7 = jnp.log(0.2e1)
  t8 = t7 - 0.1e1
  t11 = jnp.pi ** 2
  t12 = 0.1e1 / t11
  t13 = jnp.cbrt(3)
  t14 = 0.1e1 / jnp.pi
  t15 = jnp.cbrt(t14)
  t16 = t13 * t15
  t17 = jnp.cbrt(4)
  t18 = t17 ** 2
  t19 = jnp.cbrt(r0)
  t20 = 0.1e1 / t19
  t21 = t18 * t20
  t22 = t16 * t21
  t23 = jnp.sqrt(t22)
  t27 = 0.2923025e1 * p.cam_omega * t23 / t4
  t29 = jnp.cbrt(9)
  t30 = t29 ** 2
  t38 = p.cam_omega ** 2
  t41 = t15 * t18
  t42 = 0.1e1 / t5
  t47 = t38 * p.cam_omega
  t62 = jnp.log((0.1e1 + t27 + (0.344851e1 - jnp.pi * t17 * t30 * t15 / t8 / 0.12e2) * t38 * t13 * t41 * t20 * t42 / 0.4e1 + 0.48968 * t47 * t23 * t22 / t6) / (0.1e1 + t27 + 0.8621275 * t38 * t13 * t15 * t21 * t42))
  t65 = jnp.sqrt(jnp.pi)
  t68 = 0.1e1 / r0
  t82 = t13 ** 2
  t83 = t15 ** 2
  t84 = t82 * t83
  t85 = t19 ** 2
  t87 = t17 / t85
  t88 = t84 * t87
  t90 = t14 * t68
  t92 = t15 * t14
  t93 = t13 * t92
  t96 = t18 / t19 / r0
  t101 = jnp.exp(-0.1881 * t22)
  t102 = (0.1e1 - (0.2e1 / 0.45e2 * t17 * t30 * t15 * (t11 + 0.6e1 * t7 - 0.3e1) * t14 - 0.7524) * t13 * t41 * t20 / 0.4e1 + 0.204825e-1 * t88 - 0.95775e-2 * t90 + 0.3485625e-3 * t93 * t96) * t101
  t103 = jnp.sqrt(0.2e1)
  t108 = t82 * t83 * t12
  t109 = t108 * t17
  t111 = 0.1e1 / t85 / r0
  t112 = p.zeta_threshold ** 2
  t113 = jnp.where(t1, t112, 1)
  t115 = 0.1e1 / t92
  t118 = jnp.cbrt(2)
  t120 = t16 * t21 * t118
  t124 = t118 ** 2
  t133 = t113 * t30 * t115 * t13 * t85 * (0.1e1 - 0.56675e-2 * t120) / (0.1e1 + 0.107975 * t120 + 0.1e-1 * t84 * t87 * t124) / 0.15e2
  t138 = jnp.exp(-0.775e-1 * t22)
  t145 = t103 * t65
  t152 = t102 / 0.2e1 - 0.1e1 / 0.2e1
  t159 = jnp.exp(-0.13675 * t22)
  t168 = jnp.where(t1, t3 * t112, 1)
  t183 = t22 ** 0.15e1
  t190 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t23 + 0.8969 * t22 + 0.204775 * t183 + 0.123235 * t88))
  t194 = jnp.where(t1, t2 * p.zeta_threshold, 1)
  t211 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t23 + 0.905775 * t22 + 0.1100325 * t183 + 0.1241775 * t88))
  t215 = -0.621814e-1 * (0.1e1 + 0.53425e-1 * t22) * t190 + 0.19751673498613801407e-1 * (0.2e1 * t194 - 0.2e1) / (0.2e1 * t118 - 0.2e1) * (0.1e1 + 0.278125e-1 * t22) * t211
  t220 = t38 ** 2
  t233 = r0 ** 2
  t244 = t220 ** 2
  t253 = (0.1e1 + 0.15403623315025 * t84 * t87 * t38) ** 2
  t254 = t253 ** 2
  res = (0.2e1 * t6 * t8 * t12 * t62 + (-0.17543244109220059985 / t65 / jnp.pi * t68 * t102 * t103 - 0.3040082144797017415e-2 * t109 * t111 * (t133 + 0.4e1 / 0.3e1 * (-0.12375e1 * t22 + t88 / 0.4e1) * t138 * jnp.pi * r0) * t145) * t47 + (-0.26314866163830089978 * t90 * t152 - 0.38001026809962717688e-2 * t109 * t111 * (t133 + (-0.97e-1 * t22 + 0.169 * t88) * t159 * t13 / t83 * t18 * t85 / 0.3e1 - t168 * t30 * t115 * t13 * t85 / 0.15e2) * jnp.pi + 0.42708890021612718669 * t93 * t96 * t215) * t220 - 0.6755738099548927589e-2 * t108 * t17 * t111 * t102 * t145 * t220 * p.cam_omega + (-0.10133607149323391384e-1 * t109 * t111 * t152 * jnp.pi + 0.52629732327660179956 * t12 / t233 * t215) * t220 * t38 + 0.20267214298646782767e-1 * t109 / t85 / t233 * t215 * t244) / t254
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