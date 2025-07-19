"""Generated from mgga_c_ccalda.mpl."""

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
  t2 = jnp.cbrt(r0)
  t3 = t2 ** 2
  t7 = r0 - r1
  t8 = r0 + r1
  t10 = t7 / t8
  t11 = 0.1e1 + t10
  t12 = t11 / 0.2e1
  t13 = jnp.cbrt(t12)
  t14 = t13 ** 2
  t16 = tau0 / t3 / r0 * t14 * t12
  t17 = jnp.cbrt(r1)
  t18 = t17 ** 2
  t22 = 0.1e1 - t10
  t23 = t22 / 0.2e1
  t24 = jnp.cbrt(t23)
  t25 = t24 ** 2
  t27 = tau1 / t18 / r1 * t25 * t23
  t29 = s0 + 0.2e1 * s1 + s2
  t30 = t8 ** 2
  t31 = jnp.cbrt(t8)
  t32 = t31 ** 2
  t37 = t16 + t27 - t29 / t32 / t30 / 0.8e1
  t38 = (0.1e1 + params.c) * t37
  t39 = jnp.cbrt(6)
  t40 = jnp.pi ** 2
  t41 = jnp.cbrt(t40)
  t42 = t41 ** 2
  t43 = 0.1e1 / t42
  t44 = t39 * t43
  t46 = jnp.cbrt(2)
  t47 = t46 ** 2
  t53 = 0.1e1 / (0.1e1 + 0.5e1 / 0.9e1 * params.c * t37 * t44 * t47)
  t55 = t30 ** 2
  t61 = t7 ** 2
  t66 = jnp.cbrt(3)
  t68 = jnp.cbrt(0.1e1 / jnp.pi)
  t70 = jnp.cbrt(4)
  t71 = t70 ** 2
  t74 = t66 * t68 * t71 / t31
  t77 = jnp.sqrt(t74)
  t80 = t74 ** 0.15e1
  t82 = t66 ** 2
  t83 = t68 ** 2
  t87 = t82 * t83 * t70 / t32
  t93 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t77 + 0.8969 * t74 + 0.204775 * t80 + 0.123235 * t87))
  t95 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t74) * t93
  t96 = t61 ** 2
  t100 = jnp.cbrt(p.zeta_threshold)
  t101 = t100 * p.zeta_threshold
  t102 = jnp.cbrt(t11)
  t104 = jnp.where(t11 <= p.zeta_threshold, t101, t102 * t11)
  t106 = jnp.cbrt(t22)
  t108 = jnp.where(t22 <= p.zeta_threshold, t101, t106 * t22)
  t113 = (t104 + t108 - 0.2e1) / (0.2e1 * t46 - 0.2e1)
  t124 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t77 + 0.1549425e1 * t74 + 0.420775 * t80 + 0.1562925 * t87))
  t137 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t77 + 0.905775 * t74 + 0.1100325 * t80 + 0.1241775 * t87))
  t138 = (0.1e1 + 0.278125e-1 * t74) * t137
  t145 = -t95 + t96 / t55 * t113 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t74) * t124 + t95 - 0.19751673498613801407e-1 * t138) + 0.19751673498613801407e-1 * t113 * t138
  res = 0.5e1 / 0.9e1 * t38 * t44 * t47 * t53 * (0.1e1 - t29 / t32 / t55 / (t16 + t27) * t61 / 0.8e1) * t145 + (0.1e1 - 0.5e1 / 0.9e1 * t38 * t39 * t43 * t47 * t53) * t145
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t2 = jnp.cbrt(r0)
  t3 = t2 ** 2
  t7 = r0 ** 2
  t12 = tau0 / t3 / r0 - s0 / t3 / t7 / 0.8e1
  t14 = jnp.cbrt(6)
  t15 = (0.1e1 + params.c) * t12 * t14
  t16 = jnp.pi ** 2
  t17 = jnp.cbrt(t16)
  t18 = t17 ** 2
  t19 = 0.1e1 / t18
  t20 = jnp.cbrt(2)
  t21 = t20 ** 2
  t22 = t19 * t21
  t29 = 0.1e1 / (0.1e1 + 0.5e1 / 0.9e1 * params.c * t12 * t14 * t19 * t21)
  t30 = jnp.cbrt(3)
  t32 = jnp.cbrt(0.1e1 / jnp.pi)
  t34 = jnp.cbrt(4)
  t35 = t34 ** 2
  t38 = t30 * t32 * t35 / t2
  t41 = jnp.sqrt(t38)
  t44 = t38 ** 0.15e1
  t46 = t30 ** 2
  t47 = t32 ** 2
  t51 = t46 * t47 * t34 / t3
  t57 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t41 + 0.8969 * t38 + 0.204775 * t44 + 0.123235 * t51))
  t61 = jnp.cbrt(p.zeta_threshold)
  t63 = jnp.where(0.1e1 <= p.zeta_threshold, t61 * p.zeta_threshold, 1)
  t80 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t41 + 0.905775 * t38 + 0.1100325 * t44 + 0.1241775 * t51))
  t84 = -0.621814e-1 * (0.1e1 + 0.53425e-1 * t38) * t57 + 0.19751673498613801407e-1 * (0.2e1 * t63 - 0.2e1) / (0.2e1 * t20 - 0.2e1) * (0.1e1 + 0.278125e-1 * t38) * t80
  res = 0.5e1 / 0.9e1 * t15 * t22 * t29 * t84 + (0.1e1 - 0.5e1 / 0.9e1 * t15 * t22 * t29) * t84
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