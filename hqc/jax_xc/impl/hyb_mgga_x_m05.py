"""Generated from hyb_mgga_x_m05.mpl."""

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
  t29 = t28 * params.csi_HF
  t30 = jnp.cbrt(6)
  t31 = jnp.pi ** 2
  t32 = jnp.cbrt(t31)
  t33 = t32 ** 2
  t35 = t30 / t33
  t36 = r0 ** 2
  t37 = jnp.cbrt(r0)
  t38 = t37 ** 2
  t48 = params.a[0]
  t49 = params.a[1]
  t50 = t30 ** 2
  t52 = 0.3e1 / 0.1e2 * t50 * t33
  t55 = tau0 / t38 / r0
  t56 = t52 - t55
  t58 = t52 + t55
  t61 = params.a[2]
  t62 = t56 ** 2
  t64 = t58 ** 2
  t67 = params.a[3]
  t68 = t62 * t56
  t70 = t64 * t58
  t73 = params.a[4]
  t74 = t62 ** 2
  t76 = t64 ** 2
  t79 = params.a[5]
  t85 = params.a[6]
  t91 = params.a[7]
  t97 = params.a[8]
  t98 = t74 ** 2
  t100 = t76 ** 2
  t103 = params.a[9]
  t109 = params.a[10]
  t115 = params.a[11]
  t121 = t48 + t49 * t56 / t58 + t61 * t62 / t64 + t67 * t68 / t70 + t73 * t74 / t76 + t79 * t74 * t56 / t76 / t58 + t85 * t74 * t62 / t76 / t64 + t91 * t74 * t68 / t76 / t70 + t97 * t98 / t100 + t103 * t98 * t56 / t100 / t58 + t109 * t98 * t62 / t100 / t64 + t115 * t98 * t68 / t100 / t70
  t126 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t29 * (0.1804e1 - 0.646416 / (0.804 + 0.91464571985215458336e-2 * t35 * s0 / t38 / t36)) * t121)
  t128 = jnp.where(t10, t15, -t17)
  t129 = jnp.where(t14, t11, t128)
  t130 = 0.1e1 + t129
  t132 = jnp.cbrt(t130)
  t134 = jnp.where(t130 <= p.zeta_threshold, t23, t132 * t130)
  t136 = r1 ** 2
  t137 = jnp.cbrt(r1)
  t138 = t137 ** 2
  t150 = tau1 / t138 / r1
  t151 = t52 - t150
  t153 = t52 + t150
  t156 = t151 ** 2
  t158 = t153 ** 2
  t161 = t156 * t151
  t163 = t158 * t153
  t166 = t156 ** 2
  t168 = t158 ** 2
  t186 = t166 ** 2
  t188 = t168 ** 2
  t206 = t48 + t49 * t151 / t153 + t61 * t156 / t158 + t67 * t161 / t163 + t73 * t166 / t168 + t79 * t166 * t151 / t168 / t153 + t85 * t166 * t156 / t168 / t158 + t91 * t166 * t161 / t168 / t163 + t97 * t186 / t188 + t103 * t186 * t151 / t188 / t153 + t109 * t186 * t156 / t188 / t158 + t115 * t186 * t161 / t188 / t163
  t211 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t134 * t29 * (0.1804e1 - 0.646416 / (0.804 + 0.91464571985215458336e-2 * t35 * s2 / t138 / t136)) * t206)
  res = t126 + t211
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
  t22 = jnp.cbrt(6)
  t23 = jnp.pi ** 2
  t24 = jnp.cbrt(t23)
  t25 = t24 ** 2
  t28 = jnp.cbrt(2)
  t29 = t28 ** 2
  t31 = r0 ** 2
  t32 = t20 ** 2
  t44 = t22 ** 2
  t46 = 0.3e1 / 0.1e2 * t44 * t25
  t50 = tau0 * t29 / t32 / r0
  t51 = t46 - t50
  t53 = t46 + t50
  t57 = t51 ** 2
  t59 = t53 ** 2
  t63 = t57 * t51
  t65 = t59 * t53
  t69 = t57 ** 2
  t71 = t59 ** 2
  t93 = t69 ** 2
  t95 = t71 ** 2
  t116 = params.a[0] + params.a[1] * t51 / t53 + params.a[2] * t57 / t59 + params.a[3] * t63 / t65 + params.a[4] * t69 / t71 + params.a[5] * t69 * t51 / t71 / t53 + params.a[6] * t69 * t57 / t71 / t59 + params.a[7] * t69 * t63 / t71 / t65 + params.a[8] * t93 / t95 + params.a[9] * t93 * t51 / t95 / t53 + params.a[10] * t93 * t57 / t95 / t59 + params.a[11] * t93 * t63 / t95 / t65
  t121 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * params.csi_HF * (0.1804e1 - 0.646416 / (0.804 + 0.91464571985215458336e-2 * t22 / t25 * s0 * t29 / t32 / t31)) * t116)
  res = 0.2e1 * t121
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