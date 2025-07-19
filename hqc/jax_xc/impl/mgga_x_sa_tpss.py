"""Generated from mgga_x_sa_tpss.mpl."""

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
  t29 = jnp.sqrt(0.5e1)
  t30 = jnp.pi * t29
  t31 = jnp.cbrt(r0)
  t32 = t31 ** 2
  t36 = r0 ** 2
  t38 = 0.1e1 / t32 / t36
  t39 = s0 * t38
  t42 = jnp.cbrt(6)
  t43 = (tau0 / t32 / r0 - t39 / 0.8e1) * t42
  t44 = jnp.pi ** 2
  t45 = jnp.cbrt(t44)
  t46 = t45 ** 2
  t47 = 0.1e1 / t46
  t48 = t43 * t47
  t51 = jnp.sqrt(0.5e1 * t48 + 0.9e1)
  t52 = 0.5e1 / 0.9e1 * t48
  t54 = jnp.log(t52 + 0.348)
  t56 = jnp.sqrt(0.2413e1 + t54)
  t58 = t51 / t56
  t61 = s0 ** 2
  t63 = t61 / t36
  t64 = tau0 ** 2
  t65 = 0.1e1 / t64
  t66 = t63 * t65
  t69 = (0.1e1 + t66 / 0.64e2) ** 2
  t80 = t52 - 0.1e1
  t85 = jnp.sqrt(0.1e1 + 0.22222222222222222222 * t43 * t47 * t80)
  t89 = t42 * t47
  t90 = t89 * t39
  t92 = 0.9e1 / 0.2e2 * t80 / t85 + t90 / 0.36e2
  t93 = t92 ** 2
  t96 = t42 ** 2
  t99 = t96 / t45 / t44
  t100 = t36 ** 2
  t105 = t99 * t61 / t31 / t100 / r0
  t108 = jnp.sqrt(0.162e3 * t66 + 0.5e2 * t105)
  t112 = 0.1e1 / jnp.pi * t29
  t119 = t44 ** 2
  t120 = 0.1e1 / t119
  t123 = t100 ** 2
  t130 = (0.1e1 + 0.51656585037899841583e-1 * t90) ** 2
  t146 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + 0.2e1 / 0.45e2 * t30 * t58 * (0.1e1 - 0.2e1 / 0.45e2 * t30 * t58 / (0.2e1 / 0.45e2 * t30 * t58 + ((0.1e2 / 0.81e2 + 0.2485875e-1 * t63 * t65 / t69) * t42 * t47 * s0 * t38 / 0.24e2 + 0.146e3 / 0.2025e4 * t93 - 0.73e2 / 0.972e5 * t92 * t108 + 0.25e2 / 0.209952e6 * t112 / t51 * t56 * t105 + 0.17218861679299947194e-2 * t66 + 0.1464352734375e-3 * t120 * t61 * s0 / t123) / t130))))
  t148 = jnp.where(t10, t15, -t17)
  t149 = jnp.where(t14, t11, t148)
  t150 = 0.1e1 + t149
  t152 = jnp.cbrt(t150)
  t154 = jnp.where(t150 <= p.zeta_threshold, t23, t152 * t150)
  t156 = jnp.cbrt(r1)
  t157 = t156 ** 2
  t161 = r1 ** 2
  t163 = 0.1e1 / t157 / t161
  t164 = s2 * t163
  t167 = (tau1 / t157 / r1 - t164 / 0.8e1) * t42
  t168 = t167 * t47
  t171 = jnp.sqrt(0.5e1 * t168 + 0.9e1)
  t172 = 0.5e1 / 0.9e1 * t168
  t174 = jnp.log(t172 + 0.348)
  t176 = jnp.sqrt(0.2413e1 + t174)
  t178 = t171 / t176
  t181 = s2 ** 2
  t183 = t181 / t161
  t184 = tau1 ** 2
  t185 = 0.1e1 / t184
  t186 = t183 * t185
  t189 = (0.1e1 + t186 / 0.64e2) ** 2
  t200 = t172 - 0.1e1
  t205 = jnp.sqrt(0.1e1 + 0.22222222222222222222 * t167 * t47 * t200)
  t209 = t89 * t164
  t211 = 0.9e1 / 0.2e2 * t200 / t205 + t209 / 0.36e2
  t212 = t211 ** 2
  t215 = t161 ** 2
  t220 = t99 * t181 / t156 / t215 / r1
  t223 = jnp.sqrt(0.162e3 * t186 + 0.5e2 * t220)
  t234 = t215 ** 2
  t241 = (0.1e1 + 0.51656585037899841583e-1 * t209) ** 2
  t257 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t154 * t27 * (0.1e1 + 0.2e1 / 0.45e2 * t30 * t178 * (0.1e1 - 0.2e1 / 0.45e2 * t30 * t178 / (0.2e1 / 0.45e2 * t30 * t178 + ((0.1e2 / 0.81e2 + 0.2485875e-1 * t183 * t185 / t189) * t42 * t47 * s2 * t163 / 0.24e2 + 0.146e3 / 0.2025e4 * t212 - 0.73e2 / 0.972e5 * t211 * t223 + 0.25e2 / 0.209952e6 * t112 / t171 * t176 * t220 + 0.17218861679299947194e-2 * t186 + 0.1464352734375e-3 * t120 * t181 * s2 / t234) / t241))))
  res = t146 + t257
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
  t21 = jnp.sqrt(0.5e1)
  t22 = jnp.pi * t21
  t23 = jnp.cbrt(2)
  t24 = t23 ** 2
  t26 = t19 ** 2
  t31 = r0 ** 2
  t34 = s0 * t24 / t26 / t31
  t37 = jnp.cbrt(6)
  t38 = (tau0 * t24 / t26 / r0 - t34 / 0.8e1) * t37
  t39 = jnp.pi ** 2
  t40 = jnp.cbrt(t39)
  t41 = t40 ** 2
  t42 = 0.1e1 / t41
  t43 = t38 * t42
  t46 = jnp.sqrt(0.5e1 * t43 + 0.9e1)
  t47 = 0.5e1 / 0.9e1 * t43
  t49 = jnp.log(t47 + 0.348)
  t51 = jnp.sqrt(0.2413e1 + t49)
  t53 = t46 / t51
  t56 = s0 ** 2
  t58 = t56 / t31
  t59 = tau0 ** 2
  t60 = 0.1e1 / t59
  t61 = t58 * t60
  t64 = (0.1e1 + t61 / 0.64e2) ** 2
  t74 = t47 - 0.1e1
  t79 = jnp.sqrt(0.1e1 + 0.22222222222222222222 * t38 * t42 * t74)
  t84 = t37 * t42 * t34
  t86 = 0.9e1 / 0.2e2 * t74 / t79 + t84 / 0.36e2
  t87 = t86 ** 2
  t90 = t37 ** 2
  t95 = t31 ** 2
  t100 = t90 / t40 / t39 * t56 * t23 / t19 / t95 / r0
  t103 = jnp.sqrt(0.162e3 * t61 + 0.1e3 * t100)
  t114 = t39 ** 2
  t118 = t95 ** 2
  t125 = (0.1e1 + 0.51656585037899841583e-1 * t84) ** 2
  t141 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + 0.2e1 / 0.45e2 * t22 * t53 * (0.1e1 - 0.2e1 / 0.45e2 * t22 * t53 / (0.2e1 / 0.45e2 * t22 * t53 + ((0.1e2 / 0.81e2 + 0.2485875e-1 * t58 * t60 / t64) * t37 * t42 * t34 / 0.24e2 + 0.146e3 / 0.2025e4 * t87 - 0.73e2 / 0.972e5 * t86 * t103 + 0.25e2 / 0.104976e6 / jnp.pi * t21 / t46 * t51 * t100 + 0.17218861679299947194e-2 * t61 + 0.58574109375e-3 / t114 * t56 * s0 / t118) / t125))))
  res = 0.2e1 * t141
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