"""Generated from gga_x_ityh_optx.mpl."""

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
  t28 = jnp.cbrt(t6)
  t29 = t2 ** 2
  t30 = jnp.pi * t29
  t32 = jnp.cbrt(0.1e1 / jnp.pi)
  t34 = jnp.cbrt(4)
  t35 = 0.1e1 / t32 * t34
  t36 = s0 ** 2
  t38 = r0 ** 2
  t39 = t38 ** 2
  t41 = jnp.cbrt(r0)
  t44 = t41 ** 2
  t50 = (0.1e1 + 0.6e1 * s0 / t44 / t38) ** 2
  t55 = params.a + 0.36e2 * params.b * t36 / t41 / t39 / r0 / t50
  t59 = jnp.sqrt(t30 * t35 / t55)
  t62 = jnp.cbrt(2)
  t64 = jnp.cbrt(t20 * t6)
  t68 = p.cam_omega / t59 * t62 / t64 / 0.2e1
  t70 = 0.135e1 < t68
  t71 = jnp.where(t70, t68, 0.135e1)
  t72 = t71 ** 2
  t75 = t72 ** 2
  t78 = t75 * t72
  t81 = t75 ** 2
  t93 = t81 ** 2
  t97 = jnp.where(t70, 0.135e1, t68)
  t98 = jnp.sqrt(jnp.pi)
  t101 = jax.lax.erf(0.1e1 / t97 / 0.2e1)
  t103 = t97 ** 2
  t106 = jnp.exp(-0.1e1 / t103 / 0.4e1)
  t117 = jnp.where(0.135e1 <= t68, 0.1e1 / t72 / 0.36e2 - 0.1e1 / t75 / 0.96e3 + 0.1e1 / t78 / 0.2688e5 - 0.1e1 / t81 / 0.82944e6 + 0.1e1 / t81 / t72 / 0.2838528e8 - 0.1e1 / t81 / t75 / 0.107347968e10 + 0.1e1 / t81 / t78 / 0.445906944e11 - 0.1e1 / t93 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t97 * (t98 * t101 + 0.2e1 * t97 * (t106 - 0.3e1 / 0.2e1 - 0.2e1 * t103 * (t106 - 0.1e1))))
  t122 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * t117 * t55)
  t124 = jnp.where(t10, t15, -t17)
  t125 = jnp.where(t14, t11, t124)
  t126 = 0.1e1 + t125
  t128 = jnp.cbrt(t126)
  t130 = jnp.where(t126 <= p.zeta_threshold, t23, t128 * t126)
  t132 = s2 ** 2
  t134 = r1 ** 2
  t135 = t134 ** 2
  t137 = jnp.cbrt(r1)
  t140 = t137 ** 2
  t146 = (0.1e1 + 0.6e1 * s2 / t140 / t134) ** 2
  t151 = params.a + 0.36e2 * params.b * t132 / t137 / t135 / r1 / t146
  t155 = jnp.sqrt(t30 * t35 / t151)
  t159 = jnp.cbrt(t126 * t6)
  t163 = p.cam_omega / t155 * t62 / t159 / 0.2e1
  t165 = 0.135e1 < t163
  t166 = jnp.where(t165, t163, 0.135e1)
  t167 = t166 ** 2
  t170 = t167 ** 2
  t173 = t170 * t167
  t176 = t170 ** 2
  t188 = t176 ** 2
  t192 = jnp.where(t165, 0.135e1, t163)
  t195 = jax.lax.erf(0.1e1 / t192 / 0.2e1)
  t197 = t192 ** 2
  t200 = jnp.exp(-0.1e1 / t197 / 0.4e1)
  t211 = jnp.where(0.135e1 <= t163, 0.1e1 / t167 / 0.36e2 - 0.1e1 / t170 / 0.96e3 + 0.1e1 / t173 / 0.2688e5 - 0.1e1 / t176 / 0.82944e6 + 0.1e1 / t176 / t167 / 0.2838528e8 - 0.1e1 / t176 / t170 / 0.107347968e10 + 0.1e1 / t176 / t173 / 0.445906944e11 - 0.1e1 / t188 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t192 * (t98 * t195 + 0.2e1 * t192 * (t200 - 0.3e1 / 0.2e1 - 0.2e1 * t197 * (t200 - 0.1e1))))
  t216 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t130 * t28 * t211 * t151)
  res = t122 + t216
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
  t20 = jnp.cbrt(r0)
  t21 = t3 ** 2
  t24 = jnp.cbrt(0.1e1 / jnp.pi)
  t26 = jnp.cbrt(4)
  t28 = s0 ** 2
  t30 = jnp.cbrt(2)
  t31 = r0 ** 2
  t32 = t31 ** 2
  t37 = t30 ** 2
  t39 = t20 ** 2
  t45 = (0.1e1 + 0.6e1 * s0 * t37 / t39 / t31) ** 2
  t50 = params.a + 0.72e2 * params.b * t28 * t30 / t20 / t32 / r0 / t45
  t54 = jnp.sqrt(jnp.pi * t21 / t24 * t26 / t50)
  t58 = jnp.cbrt(t12 * r0)
  t62 = p.cam_omega / t54 * t30 / t58 / 0.2e1
  t64 = 0.135e1 < t62
  t65 = jnp.where(t64, t62, 0.135e1)
  t66 = t65 ** 2
  t69 = t66 ** 2
  t72 = t69 * t66
  t75 = t69 ** 2
  t87 = t75 ** 2
  t91 = jnp.where(t64, 0.135e1, t62)
  t92 = jnp.sqrt(jnp.pi)
  t95 = jax.lax.erf(0.1e1 / t91 / 0.2e1)
  t97 = t91 ** 2
  t100 = jnp.exp(-0.1e1 / t97 / 0.4e1)
  t111 = jnp.where(0.135e1 <= t62, 0.1e1 / t66 / 0.36e2 - 0.1e1 / t69 / 0.96e3 + 0.1e1 / t72 / 0.2688e5 - 0.1e1 / t75 / 0.82944e6 + 0.1e1 / t75 / t66 / 0.2838528e8 - 0.1e1 / t75 / t69 / 0.107347968e10 + 0.1e1 / t75 / t72 / 0.445906944e11 - 0.1e1 / t87 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t91 * (t92 * t95 + 0.2e1 * t91 * (t100 - 0.3e1 / 0.2e1 - 0.2e1 * t97 * (t100 - 0.1e1))))
  t116 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * t111 * t50)
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