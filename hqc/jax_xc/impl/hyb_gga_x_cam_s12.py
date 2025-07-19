"""Generated from hyb_gga_x_cam_s12.mpl."""

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
  t30 = r0 ** 2
  t31 = jnp.cbrt(r0)
  t32 = t31 ** 2
  t34 = 0.1e1 / t32 / t30
  t36 = s0 ** 2
  t38 = t30 ** 2
  t53 = params.A + params.B * (0.1e1 - 0.1e1 / (0.1e1 + params.C * s0 * t34 + params.D * t36 / t31 / t38 / r0)) * (0.1e1 - 0.1e1 / (params.E * s0 * t34 + 0.1e1))
  t55 = t2 ** 2
  t56 = jnp.pi * t55
  t58 = jnp.cbrt(0.1e1 / jnp.pi)
  t60 = jnp.cbrt(4)
  t61 = 0.1e1 / t58 * t60
  t65 = jnp.sqrt(t56 * t61 / t53)
  t68 = jnp.cbrt(2)
  t70 = jnp.cbrt(t20 * t6)
  t74 = p.cam_omega / t65 * t68 / t70 / 0.2e1
  t76 = 0.135e1 < t74
  t77 = jnp.where(t76, t74, 0.135e1)
  t78 = t77 ** 2
  t81 = t78 ** 2
  t84 = t81 * t78
  t87 = t81 ** 2
  t99 = t87 ** 2
  t103 = jnp.where(t76, 0.135e1, t74)
  t104 = jnp.sqrt(jnp.pi)
  t107 = jax.lax.erf(0.1e1 / t103 / 0.2e1)
  t109 = t103 ** 2
  t112 = jnp.exp(-0.1e1 / t109 / 0.4e1)
  t123 = jnp.where(0.135e1 <= t74, 0.1e1 / t78 / 0.36e2 - 0.1e1 / t81 / 0.96e3 + 0.1e1 / t84 / 0.2688e5 - 0.1e1 / t87 / 0.82944e6 + 0.1e1 / t87 / t78 / 0.2838528e8 - 0.1e1 / t87 / t81 / 0.107347968e10 + 0.1e1 / t87 / t84 / 0.445906944e11 - 0.1e1 / t99 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t103 * (t104 * t107 + 0.2e1 * t103 * (t112 - 0.3e1 / 0.2e1 - 0.2e1 * t109 * (t112 - 0.1e1))))
  t129 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * t53 * (-p.cam_beta * t123 - p.cam_alpha + 0.1e1))
  t131 = jnp.where(t10, t15, -t17)
  t132 = jnp.where(t14, t11, t131)
  t133 = 0.1e1 + t132
  t135 = jnp.cbrt(t133)
  t137 = jnp.where(t133 <= p.zeta_threshold, t23, t135 * t133)
  t140 = r1 ** 2
  t141 = jnp.cbrt(r1)
  t142 = t141 ** 2
  t144 = 0.1e1 / t142 / t140
  t146 = s2 ** 2
  t148 = t140 ** 2
  t163 = params.A + params.B * (0.1e1 - 0.1e1 / (0.1e1 + params.C * s2 * t144 + params.D * t146 / t141 / t148 / r1)) * (0.1e1 - 0.1e1 / (params.E * s2 * t144 + 0.1e1))
  t168 = jnp.sqrt(t56 * t61 / t163)
  t172 = jnp.cbrt(t133 * t6)
  t176 = p.cam_omega / t168 * t68 / t172 / 0.2e1
  t178 = 0.135e1 < t176
  t179 = jnp.where(t178, t176, 0.135e1)
  t180 = t179 ** 2
  t183 = t180 ** 2
  t186 = t183 * t180
  t189 = t183 ** 2
  t201 = t189 ** 2
  t205 = jnp.where(t178, 0.135e1, t176)
  t208 = jax.lax.erf(0.1e1 / t205 / 0.2e1)
  t210 = t205 ** 2
  t213 = jnp.exp(-0.1e1 / t210 / 0.4e1)
  t224 = jnp.where(0.135e1 <= t176, 0.1e1 / t180 / 0.36e2 - 0.1e1 / t183 / 0.96e3 + 0.1e1 / t186 / 0.2688e5 - 0.1e1 / t189 / 0.82944e6 + 0.1e1 / t189 / t180 / 0.2838528e8 - 0.1e1 / t189 / t183 / 0.107347968e10 + 0.1e1 / t189 / t186 / 0.445906944e11 - 0.1e1 / t201 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t205 * (t104 * t208 + 0.2e1 * t205 * (t213 - 0.3e1 / 0.2e1 - 0.2e1 * t210 * (t213 - 0.1e1))))
  t230 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t137 * t28 * t163 * (-p.cam_beta * t224 - p.cam_alpha + 0.1e1))
  res = t129 + t230
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
  t22 = jnp.cbrt(2)
  t23 = t22 ** 2
  t24 = r0 ** 2
  t25 = t20 ** 2
  t28 = t23 / t25 / t24
  t30 = s0 ** 2
  t32 = t24 ** 2
  t49 = params.A + params.B * (0.1e1 - 0.1e1 / (0.1e1 + params.C * s0 * t28 + 0.2e1 * params.D * t30 * t22 / t20 / t32 / r0)) * (0.1e1 - 0.1e1 / (params.E * s0 * t28 + 0.1e1))
  t51 = t3 ** 2
  t54 = jnp.cbrt(0.1e1 / jnp.pi)
  t56 = jnp.cbrt(4)
  t61 = jnp.sqrt(jnp.pi * t51 / t54 * t56 / t49)
  t65 = jnp.cbrt(t12 * r0)
  t69 = p.cam_omega / t61 * t22 / t65 / 0.2e1
  t71 = 0.135e1 < t69
  t72 = jnp.where(t71, t69, 0.135e1)
  t73 = t72 ** 2
  t76 = t73 ** 2
  t79 = t76 * t73
  t82 = t76 ** 2
  t94 = t82 ** 2
  t98 = jnp.where(t71, 0.135e1, t69)
  t99 = jnp.sqrt(jnp.pi)
  t102 = jax.lax.erf(0.1e1 / t98 / 0.2e1)
  t104 = t98 ** 2
  t107 = jnp.exp(-0.1e1 / t104 / 0.4e1)
  t118 = jnp.where(0.135e1 <= t69, 0.1e1 / t73 / 0.36e2 - 0.1e1 / t76 / 0.96e3 + 0.1e1 / t79 / 0.2688e5 - 0.1e1 / t82 / 0.82944e6 + 0.1e1 / t82 / t73 / 0.2838528e8 - 0.1e1 / t82 / t76 / 0.107347968e10 + 0.1e1 / t82 / t79 / 0.445906944e11 - 0.1e1 / t94 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t98 * (t99 * t102 + 0.2e1 * t98 * (t107 - 0.3e1 / 0.2e1 - 0.2e1 * t104 * (t107 - 0.1e1))))
  t124 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * t49 * (-p.cam_beta * t118 - p.cam_alpha + 0.1e1))
  res = 0.2e1 * t124
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