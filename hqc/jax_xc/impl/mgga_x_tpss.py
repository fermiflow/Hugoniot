"""Generated from mgga_x_tpss.mpl."""

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
  t29 = 0.1e1 / r0
  t31 = 0.1e1 / tau0
  t39 = (s0 * t29 * t31 / 0.8e1) ** (params.BLOC_a + params.BLOC_b * s0 * t29 * t31 / 0.8e1)
  t41 = s0 ** 2
  t42 = r0 ** 2
  t43 = 0.1e1 / t42
  t45 = tau0 ** 2
  t46 = 0.1e1 / t45
  t47 = t41 * t43 * t46
  t50 = (0.1e1 + t47 / 0.64e2) ** 2
  t54 = jnp.cbrt(6)
  t56 = jnp.pi ** 2
  t57 = jnp.cbrt(t56)
  t58 = t57 ** 2
  t59 = 0.1e1 / t58
  t61 = jnp.cbrt(r0)
  t62 = t61 ** 2
  t64 = 0.1e1 / t62 / t42
  t65 = t59 * s0 * t64
  t71 = s0 * t64
  t73 = tau0 / t62 / r0 - t71 / 0.8e1
  t77 = 0.5e1 / 0.9e1 * t73 * t54 * t59 - 0.1e1
  t79 = t54 * t59
  t84 = jnp.sqrt(0.5e1 * params.b * t73 * t79 * t77 + 0.9e1)
  t90 = 0.27e2 / 0.2e2 * t77 / t84 + t79 * t71 / 0.36e2
  t91 = t90 ** 2
  t94 = t54 ** 2
  t96 = 0.1e1 / t57 / t56
  t97 = t94 * t96
  t98 = t42 ** 2
  t101 = 0.1e1 / t61 / t98 / r0
  t106 = jnp.sqrt(0.5e2 * t97 * t41 * t101 + 0.162e3 * t47)
  t110 = 0.1e1 / params.kappa * t94
  t115 = jnp.sqrt(params.e)
  t120 = params.e * params.mu
  t121 = t56 ** 2
  t122 = 0.1e1 / t121
  t125 = t98 ** 2
  t131 = t115 * t54
  t135 = (0.1e1 + t131 * t65 / 0.24e2) ** 2
  t147 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + params.kappa * (0.1e1 - params.kappa / (params.kappa + ((0.1e2 / 0.81e2 + params.c * t39 / t50) * t54 * t65 / 0.24e2 + 0.146e3 / 0.2025e4 * t91 - 0.73e2 / 0.972e5 * t90 * t106 + 0.25e2 / 0.944784e6 * t110 * t96 * t41 * t101 + t115 * t41 * t43 * t46 / 0.72e3 + t120 * t122 * t41 * s0 / t125 / 0.2304e4) / t135))))
  t149 = jnp.where(t10, t15, -t17)
  t150 = jnp.where(t14, t11, t149)
  t151 = 0.1e1 + t150
  t153 = jnp.cbrt(t151)
  t155 = jnp.where(t151 <= p.zeta_threshold, t23, t153 * t151)
  t157 = 0.1e1 / r1
  t159 = 0.1e1 / tau1
  t167 = (s2 * t157 * t159 / 0.8e1) ** (params.BLOC_a + params.BLOC_b * s2 * t157 * t159 / 0.8e1)
  t169 = s2 ** 2
  t170 = r1 ** 2
  t171 = 0.1e1 / t170
  t173 = tau1 ** 2
  t174 = 0.1e1 / t173
  t175 = t169 * t171 * t174
  t178 = (0.1e1 + t175 / 0.64e2) ** 2
  t184 = jnp.cbrt(r1)
  t185 = t184 ** 2
  t187 = 0.1e1 / t185 / t170
  t188 = t59 * s2 * t187
  t194 = s2 * t187
  t196 = tau1 / t185 / r1 - t194 / 0.8e1
  t200 = 0.5e1 / 0.9e1 * t196 * t54 * t59 - 0.1e1
  t206 = jnp.sqrt(0.5e1 * params.b * t196 * t79 * t200 + 0.9e1)
  t212 = 0.27e2 / 0.2e2 * t200 / t206 + t79 * t194 / 0.36e2
  t213 = t212 ** 2
  t216 = t170 ** 2
  t219 = 0.1e1 / t184 / t216 / r1
  t224 = jnp.sqrt(0.5e2 * t97 * t169 * t219 + 0.162e3 * t175)
  t237 = t216 ** 2
  t246 = (0.1e1 + t131 * t188 / 0.24e2) ** 2
  t258 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t155 * t27 * (0.1e1 + params.kappa * (0.1e1 - params.kappa / (params.kappa + ((0.1e2 / 0.81e2 + params.c * t167 / t178) * t54 * t188 / 0.24e2 + 0.146e3 / 0.2025e4 * t213 - 0.73e2 / 0.972e5 * t212 * t224 + 0.25e2 / 0.944784e6 * t110 * t96 * t169 * t219 + t115 * t169 * t171 * t174 / 0.72e3 + t120 * t122 * t169 * s2 / t237 / 0.2304e4) / t246))))
  res = t147 + t258
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
  t21 = 0.1e1 / r0
  t23 = 0.1e1 / tau0
  t31 = (s0 * t21 * t23 / 0.8e1) ** (params.BLOC_a + params.BLOC_b * s0 * t21 * t23 / 0.8e1)
  t33 = s0 ** 2
  t34 = r0 ** 2
  t35 = 0.1e1 / t34
  t37 = tau0 ** 2
  t38 = 0.1e1 / t37
  t39 = t33 * t35 * t38
  t42 = (0.1e1 + t39 / 0.64e2) ** 2
  t46 = jnp.cbrt(6)
  t48 = jnp.pi ** 2
  t49 = jnp.cbrt(t48)
  t50 = t49 ** 2
  t51 = 0.1e1 / t50
  t53 = jnp.cbrt(2)
  t54 = t53 ** 2
  t56 = t19 ** 2
  t59 = s0 * t54 / t56 / t34
  t67 = tau0 * t54 / t56 / r0 - t59 / 0.8e1
  t71 = 0.5e1 / 0.9e1 * t67 * t46 * t51 - 0.1e1
  t73 = t46 * t51
  t78 = jnp.sqrt(0.5e1 * params.b * t67 * t73 * t71 + 0.9e1)
  t84 = 0.27e2 / 0.2e2 * t71 / t78 + t73 * t59 / 0.36e2
  t85 = t84 ** 2
  t88 = t46 ** 2
  t90 = 0.1e1 / t49 / t48
  t93 = t34 ** 2
  t97 = t33 * t53 / t19 / t93 / r0
  t101 = jnp.sqrt(0.1e3 * t88 * t90 * t97 + 0.162e3 * t39)
  t109 = jnp.sqrt(params.e)
  t115 = t48 ** 2
  t119 = t93 ** 2
  t130 = (0.1e1 + t109 * t46 * t51 * t59 / 0.24e2) ** 2
  t142 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + params.kappa * (0.1e1 - params.kappa / (params.kappa + ((0.1e2 / 0.81e2 + params.c * t31 / t42) * t46 * t51 * t59 / 0.24e2 + 0.146e3 / 0.2025e4 * t85 - 0.73e2 / 0.972e5 * t84 * t101 + 0.25e2 / 0.472392e6 / params.kappa * t88 * t90 * t97 + t109 * t33 * t35 * t38 / 0.72e3 + params.e * params.mu / t115 * t33 * s0 / t119 / 0.576e3) / t130))))
  res = 0.2e1 * t142
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