"""Generated from mgga_x_gvt4.mpl."""

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
  t2 = jnp.cbrt(jnp.pi)
  t3 = 0.1e1 / t2
  t4 = r0 + r1
  t5 = 0.1e1 / t4
  t8 = 0.2e1 * r0 * t5 <= p.zeta_threshold
  t9 = p.zeta_threshold - 0.1e1
  t12 = 0.2e1 * r1 * t5 <= p.zeta_threshold
  t13 = -t9
  t15 = (r0 - r1) * t5
  t16 = jnp.where(t12, t13, t15)
  t17 = jnp.where(t8, t9, t16)
  t18 = 0.1e1 + t17
  t20 = jnp.cbrt(p.zeta_threshold)
  t21 = t20 * p.zeta_threshold
  t22 = jnp.cbrt(t18)
  t24 = jnp.where(t18 <= p.zeta_threshold, t21, t22 * t18)
  t26 = jnp.cbrt(t4)
  t28 = r0 ** 2
  t29 = jnp.cbrt(r0)
  t30 = t29 ** 2
  t33 = s0 / t30 / t28
  t37 = tau0 / t30 / r0
  t39 = jnp.cbrt(6)
  t40 = t39 ** 2
  t41 = jnp.pi ** 2
  t42 = jnp.cbrt(t41)
  t43 = t42 ** 2
  t44 = t40 * t43
  t45 = 0.1120356e-2 * t44
  t46 = 0.1e1 + 0.186726e-2 * t33 + 0.373452e-2 * t37 - t45
  t51 = 0.37501956e-2 * t44
  t53 = t46 ** 2
  t56 = s0 ** 2
  t57 = t28 ** 2
  t64 = 0.3e1 / 0.5e1 * t44
  t65 = 0.2e1 * t37 - t64
  t68 = t65 ** 2
  t76 = jnp.cbrt(0.1e1 / jnp.pi)
  t77 = 0.1e1 / t76
  t79 = jnp.cbrt(4)
  t83 = jnp.where(r0 <= p.dens_threshold, 0, t3 * t24 * t26 * (-0.9800683 / t46 + (-0.3556788e-2 * t33 + 0.12500652e-1 * t37 - t51) / t53 + (-0.2354518e-4 * t56 / t29 / t57 / r0 - 0.1282732e-3 * t33 * t65 + 0.3574822e-3 * t68) / t53 / t46) * t77 * t79 / 0.4e1)
  t85 = jnp.where(t8, t13, -t15)
  t86 = jnp.where(t12, t9, t85)
  t87 = 0.1e1 + t86
  t89 = jnp.cbrt(t87)
  t91 = jnp.where(t87 <= p.zeta_threshold, t21, t89 * t87)
  t94 = r1 ** 2
  t95 = jnp.cbrt(r1)
  t96 = t95 ** 2
  t99 = s2 / t96 / t94
  t103 = tau1 / t96 / r1
  t105 = 0.1e1 + 0.186726e-2 * t99 + 0.373452e-2 * t103 - t45
  t111 = t105 ** 2
  t114 = s2 ** 2
  t115 = t94 ** 2
  t122 = 0.2e1 * t103 - t64
  t125 = t122 ** 2
  t136 = jnp.where(r1 <= p.dens_threshold, 0, t3 * t91 * t26 * (-0.9800683 / t105 + (-0.3556788e-2 * t99 + 0.12500652e-1 * t103 - t51) / t111 + (-0.2354518e-4 * t114 / t95 / t115 / r1 - 0.1282732e-3 * t99 * t122 + 0.3574822e-3 * t125) / t111 / t105) * t77 * t79 / 0.4e1)
  res = t83 + t136
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = jnp.cbrt(jnp.pi)
  t5 = 0.1e1 <= p.zeta_threshold
  t6 = p.zeta_threshold - 0.1e1
  t8 = jnp.where(t5, -t6, 0)
  t9 = jnp.where(t5, t6, t8)
  t10 = 0.1e1 + t9
  t12 = jnp.cbrt(p.zeta_threshold)
  t14 = jnp.cbrt(t10)
  t16 = jnp.where(t10 <= p.zeta_threshold, t12 * p.zeta_threshold, t14 * t10)
  t18 = jnp.cbrt(r0)
  t20 = jnp.cbrt(2)
  t21 = t20 ** 2
  t22 = s0 * t21
  t23 = r0 ** 2
  t24 = t18 ** 2
  t26 = 0.1e1 / t24 / t23
  t27 = t22 * t26
  t32 = tau0 * t21 / t24 / r0
  t34 = jnp.cbrt(6)
  t35 = t34 ** 2
  t36 = jnp.pi ** 2
  t37 = jnp.cbrt(t36)
  t38 = t37 ** 2
  t39 = t35 * t38
  t41 = 0.1e1 + 0.186726e-2 * t27 + 0.373452e-2 * t32 - 0.1120356e-2 * t39
  t48 = t41 ** 2
  t51 = s0 ** 2
  t53 = t23 ** 2
  t61 = 0.2e1 * t32 - 0.3e1 / 0.5e1 * t39
  t65 = t61 ** 2
  t73 = jnp.cbrt(0.1e1 / jnp.pi)
  t76 = jnp.cbrt(4)
  t80 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, 0.1e1 / t3 * t16 * t18 * (-0.9800683 / t41 + (-0.3556788e-2 * t27 + 0.12500652e-1 * t32 - 0.37501956e-2 * t39) / t48 + (-0.4709036e-4 * t51 * t20 / t18 / t53 / r0 - 0.1282732e-3 * t22 * t26 * t61 + 0.3574822e-3 * t65) / t48 / t41) / t73 * t76 / 0.4e1)
  res = 0.2e1 * t80
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