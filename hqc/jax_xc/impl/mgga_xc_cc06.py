"""Generated from mgga_xc_cc06.mpl."""

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
  t8 = r0 * t7
  t11 = jnp.cbrt(p.zeta_threshold)
  t12 = t11 * p.zeta_threshold
  t13 = jnp.cbrt(2)
  t15 = jnp.cbrt(t8)
  t19 = jnp.where(0.2e1 * t8 <= p.zeta_threshold, t12, 0.2e1 * t13 * r0 * t7 * t15)
  t20 = jnp.cbrt(t6)
  t24 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t19 * t20)
  t26 = r1 * t7
  t30 = jnp.cbrt(t26)
  t34 = jnp.where(0.2e1 * t26 <= p.zeta_threshold, t12, 0.2e1 * t13 * r1 * t7 * t30)
  t38 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t34 * t20)
  t40 = jnp.cbrt(0.1e1 / jnp.pi)
  t42 = jnp.cbrt(4)
  t43 = t42 ** 2
  t46 = t2 * t40 * t43 / t20
  t49 = jnp.sqrt(t46)
  t52 = t46 ** 0.15e1
  t54 = t2 ** 2
  t55 = t40 ** 2
  t57 = t20 ** 2
  t60 = t54 * t55 * t42 / t57
  t66 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t49 + 0.8969 * t46 + 0.204775 * t52 + 0.123235 * t60))
  t68 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t46) * t66
  t69 = r0 - r1
  t70 = t69 ** 2
  t71 = t70 ** 2
  t72 = t6 ** 2
  t73 = t72 ** 2
  t76 = t69 * t7
  t77 = 0.1e1 + t76
  t79 = jnp.cbrt(t77)
  t81 = jnp.where(t77 <= p.zeta_threshold, t12, t79 * t77)
  t82 = 0.1e1 - t76
  t84 = jnp.cbrt(t82)
  t86 = jnp.where(t82 <= p.zeta_threshold, t12, t84 * t82)
  t91 = (t81 + t86 - 0.2e1) / (0.2e1 * t13 - 0.2e1)
  t102 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t49 + 0.1549425e1 * t46 + 0.420775 * t52 + 0.1562925 * t60))
  t115 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t49 + 0.905775 * t46 + 0.1100325 * t52 + 0.1241775 * t60))
  t116 = (0.1e1 + 0.278125e-1 * t46) * t115
  t125 = jnp.cbrt(r0)
  t126 = t125 ** 2
  t130 = t77 / 0.2e1
  t131 = jnp.cbrt(t130)
  t132 = t131 ** 2
  t135 = jnp.cbrt(r1)
  t136 = t135 ** 2
  t140 = t82 / 0.2e1
  t141 = jnp.cbrt(t140)
  t142 = t141 ** 2
  t147 = t54 * t42 * t55 * (l0 / t126 / r0 * t132 * t130 + l1 / t136 / r1 * t142 * t140)
  res = (t24 + t38 - t68 + t71 / t73 * t91 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t46) * t102 + t68 - 0.19751789702565206229e-1 * t116) + 0.19751789702565206229e-1 * t91 * t116) * (0.1e1 + (-0.7e-3 + 0.2e-2 * t147) / (0.1e1 + 0.65e-2 * t147))
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = jnp.cbrt(3)
  t4 = jnp.cbrt(jnp.pi)
  t8 = jnp.cbrt(p.zeta_threshold)
  t10 = jnp.where(0.1e1 <= p.zeta_threshold, t8 * p.zeta_threshold, 1)
  t11 = jnp.cbrt(r0)
  t15 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t10 * t11)
  t18 = jnp.cbrt(0.1e1 / jnp.pi)
  t20 = jnp.cbrt(4)
  t21 = t20 ** 2
  t24 = t3 * t18 * t21 / t11
  t27 = jnp.sqrt(t24)
  t30 = t24 ** 0.15e1
  t32 = t3 ** 2
  t33 = t18 ** 2
  t35 = t11 ** 2
  t38 = t32 * t33 * t20 / t35
  t44 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t27 + 0.8969 * t24 + 0.204775 * t30 + 0.123235 * t38))
  t49 = jnp.cbrt(2)
  t64 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t27 + 0.905775 * t24 + 0.1100325 * t30 + 0.1241775 * t38))
  t74 = t32 * t20 * t33 * l0 / t35 / r0
  res = (0.2e1 * t15 - 0.62182e-1 * (0.1e1 + 0.53425e-1 * t24) * t44 + 0.19751789702565206229e-1 * (0.2e1 * t10 - 0.2e1) / (0.2e1 * t49 - 0.2e1) * (0.1e1 + 0.278125e-1 * t24) * t64) * (0.1e1 + (-0.7e-3 + 0.2e-2 * t74) / (0.1e1 + 0.65e-2 * t74))
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