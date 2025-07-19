"""Generated from mgga_x_tm.mpl."""

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
  t33 = s0 / r0 / tau0 / 0.8e1
  t35 = jnp.where(t33 < 0.1e1, t33, 0.1e1)
  t36 = t35 ** 2
  t37 = t36 * t35
  t41 = (0.1e1 + t37) ** 2
  t43 = (t36 + 0.3e1 * t37) / t41
  t44 = jnp.cbrt(6)
  t45 = jnp.pi ** 2
  t46 = jnp.cbrt(t45)
  t47 = t46 ** 2
  t48 = 0.1e1 / t47
  t49 = t44 * t48
  t50 = r0 ** 2
  t51 = jnp.cbrt(r0)
  t52 = t51 ** 2
  t54 = 0.1e1 / t52 / t50
  t55 = s0 * t54
  t56 = t49 * t55
  t58 = t44 ** 2
  t61 = t58 / t46 / t45
  t62 = s0 ** 2
  t63 = t50 ** 2
  t71 = (0.1e1 + 0.15045488888888888889 * t56 + 0.26899490462262948e-2 * t61 * t62 / t51 / t63 / r0) ** (0.1e1 / 0.5e1)
  t76 = tau0 / t52 / r0
  t79 = 0.256337604 * t58 * t47
  t86 = t71 ** 2
  t106 = (t76 - t55 / 0.8e1) * t44 * t48 / 0.4e1 - 0.9e1 / 0.2e2 + t56 / 0.36e2
  t107 = t106 ** 2
  t114 = (0.1e1 + 0.5e1 / 0.12e2 * (0.1e2 / 0.81e2 + 0.25e2 / 0.8748e4 * t56) * t44 * t48 * s0 * t54 + 0.292e3 / 0.405e3 * t107 - 0.146e3 / 0.135e3 * t106 * t35 * (0.1e1 - t35)) ** (0.1e1 / 0.1e2)
  t120 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (t43 * (0.1e1 / t71 + 0.7e1 / 0.9e1 * (0.1e1 + 0.63943327777777777778e-1 * t56 - 0.5e1 / 0.9e1 * (0.14554132 * t76 + t79 + 0.11867481666666666667e-1 * t55) * t44 * t48) / t86) + (0.1e1 - t43) * t114))
  t122 = jnp.where(t10, t15, -t17)
  t123 = jnp.where(t14, t11, t122)
  t124 = 0.1e1 + t123
  t126 = jnp.cbrt(t124)
  t128 = jnp.where(t124 <= p.zeta_threshold, t23, t126 * t124)
  t134 = s2 / r1 / tau1 / 0.8e1
  t136 = jnp.where(t134 < 0.1e1, t134, 0.1e1)
  t137 = t136 ** 2
  t138 = t137 * t136
  t142 = (0.1e1 + t138) ** 2
  t144 = (t137 + 0.3e1 * t138) / t142
  t145 = r1 ** 2
  t146 = jnp.cbrt(r1)
  t147 = t146 ** 2
  t149 = 0.1e1 / t147 / t145
  t150 = s2 * t149
  t151 = t49 * t150
  t153 = s2 ** 2
  t154 = t145 ** 2
  t162 = (0.1e1 + 0.15045488888888888889 * t151 + 0.26899490462262948e-2 * t61 * t153 / t146 / t154 / r1) ** (0.1e1 / 0.5e1)
  t167 = tau1 / t147 / r1
  t175 = t162 ** 2
  t195 = (t167 - t150 / 0.8e1) * t44 * t48 / 0.4e1 - 0.9e1 / 0.2e2 + t151 / 0.36e2
  t196 = t195 ** 2
  t203 = (0.1e1 + 0.5e1 / 0.12e2 * (0.1e2 / 0.81e2 + 0.25e2 / 0.8748e4 * t151) * t44 * t48 * s2 * t149 + 0.292e3 / 0.405e3 * t196 - 0.146e3 / 0.135e3 * t195 * t136 * (0.1e1 - t136)) ** (0.1e1 / 0.1e2)
  t209 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t128 * t27 * (t144 * (0.1e1 / t162 + 0.7e1 / 0.9e1 * (0.1e1 + 0.63943327777777777778e-1 * t151 - 0.5e1 / 0.9e1 * (0.14554132 * t167 + t79 + 0.11867481666666666667e-1 * t150) * t44 * t48) / t175) + (0.1e1 - t144) * t203))
  res = t120 + t209
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
  t25 = s0 / r0 / tau0 / 0.8e1
  t27 = jnp.where(t25 < 0.1e1, t25, 0.1e1)
  t28 = t27 ** 2
  t29 = t28 * t27
  t33 = (0.1e1 + t29) ** 2
  t35 = (t28 + 0.3e1 * t29) / t33
  t36 = jnp.cbrt(6)
  t37 = jnp.pi ** 2
  t38 = jnp.cbrt(t37)
  t39 = t38 ** 2
  t40 = 0.1e1 / t39
  t42 = jnp.cbrt(2)
  t43 = t42 ** 2
  t45 = r0 ** 2
  t46 = t19 ** 2
  t49 = s0 * t43 / t46 / t45
  t50 = t36 * t40 * t49
  t52 = t36 ** 2
  t56 = s0 ** 2
  t58 = t45 ** 2
  t66 = (0.1e1 + 0.15045488888888888889 * t50 + 0.53798980924525896e-2 * t52 / t38 / t37 * t56 * t42 / t19 / t58 / r0) ** (0.1e1 / 0.5e1)
  t72 = tau0 * t43 / t46 / r0
  t82 = t66 ** 2
  t101 = (t72 - t49 / 0.8e1) * t36 * t40 / 0.4e1 - 0.9e1 / 0.2e2 + t50 / 0.36e2
  t102 = t101 ** 2
  t109 = (0.1e1 + 0.5e1 / 0.12e2 * (0.1e2 / 0.81e2 + 0.25e2 / 0.8748e4 * t50) * t36 * t40 * t49 + 0.292e3 / 0.405e3 * t102 - 0.146e3 / 0.135e3 * t101 * t27 * (0.1e1 - t27)) ** (0.1e1 / 0.1e2)
  t115 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (t35 * (0.1e1 / t66 + 0.7e1 / 0.9e1 * (0.1e1 + 0.63943327777777777778e-1 * t50 - 0.5e1 / 0.9e1 * (0.14554132 * t72 + 0.256337604 * t52 * t39 + 0.11867481666666666667e-1 * t49) * t36 * t40) / t82) + (0.1e1 - t35) * t109))
  res = 0.2e1 * t115
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