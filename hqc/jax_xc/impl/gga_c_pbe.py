"""Generated from gga_c_pbe.mpl."""

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
  t1 = jnp.cbrt(3)
  t3 = jnp.cbrt(0.1e1 / jnp.pi)
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = r0 + r1
  t8 = jnp.cbrt(t7)
  t11 = t1 * t3 * t6 / t8
  t14 = jnp.sqrt(t11)
  t17 = t11 ** 0.15e1
  t19 = t1 ** 2
  t20 = t3 ** 2
  t22 = t8 ** 2
  t25 = t19 * t20 * t5 / t22
  t31 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t14 + 0.8969 * t11 + 0.204775 * t17 + 0.123235 * t25))
  t33 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t11) * t31
  t34 = r0 - r1
  t35 = t34 ** 2
  t36 = t35 ** 2
  t37 = t7 ** 2
  t38 = t37 ** 2
  t42 = t34 / t7
  t43 = 0.1e1 + t42
  t44 = t43 <= p.zeta_threshold
  t45 = jnp.cbrt(p.zeta_threshold)
  t46 = t45 * p.zeta_threshold
  t47 = jnp.cbrt(t43)
  t49 = jnp.where(t44, t46, t47 * t43)
  t50 = 0.1e1 - t42
  t51 = t50 <= p.zeta_threshold
  t52 = jnp.cbrt(t50)
  t54 = jnp.where(t51, t46, t52 * t50)
  t56 = jnp.cbrt(2)
  t60 = (t49 + t54 - 0.2e1) / (0.2e1 * t56 - 0.2e1)
  t71 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t14 + 0.1549425e1 * t11 + 0.420775 * t17 + 0.1562925 * t25))
  t84 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t14 + 0.905775 * t11 + 0.1100325 * t17 + 0.1241775 * t25))
  t85 = (0.1e1 + 0.278125e-1 * t11) * t84
  t89 = t36 / t38 * t60 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t11) * t71 + t33 - 0.19751673498613801407e-1 * t85)
  t91 = 0.19751673498613801407e-1 * t60 * t85
  t92 = t45 ** 2
  t93 = t47 ** 2
  t94 = jnp.where(t44, t92, t93)
  t95 = t52 ** 2
  t96 = jnp.where(t51, t92, t95)
  t98 = t94 / 0.2e1 + t96 / 0.2e1
  t99 = t98 ** 2
  t100 = t99 * t98
  t103 = s0 + 0.2e1 * s1 + s2
  t116 = 0.1e1 / params.gamma
  t121 = jnp.exp(-(-t33 + t89 + t91) * t116 / t100)
  t123 = 0.1e1 / (t121 - 0.1e1)
  t125 = t103 ** 2
  t130 = t56 ** 2
  t132 = t99 ** 2
  t141 = t103 / t8 / t37 * t56 / t99 * t19 / t3 * t5 / 0.96e2 + params.BB * params.beta * t116 * t123 * t125 / t22 / t38 * t130 / t132 * t1 / t20 * t6 / 0.3072e4
  t151 = jnp.log(0.1e1 + params.beta * t141 * t116 / (params.beta * t116 * t123 * t141 + 0.1e1))
  res = params.gamma * t100 * t151 - t33 + t89 + t91
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(3)
  t3 = jnp.cbrt(0.1e1 / jnp.pi)
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = jnp.cbrt(r0)
  t10 = t1 * t3 * t6 / t7
  t13 = jnp.sqrt(t10)
  t16 = t10 ** 0.15e1
  t18 = t1 ** 2
  t19 = t3 ** 2
  t21 = t7 ** 2
  t24 = t18 * t19 * t5 / t21
  t30 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t13 + 0.8969 * t10 + 0.204775 * t16 + 0.123235 * t24))
  t32 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t10) * t30
  t33 = 0.1e1 <= p.zeta_threshold
  t34 = jnp.cbrt(p.zeta_threshold)
  t36 = jnp.where(t33, t34 * p.zeta_threshold, 1)
  t39 = jnp.cbrt(2)
  t54 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t13 + 0.905775 * t10 + 0.1100325 * t16 + 0.1241775 * t24))
  t57 = 0.19751673498613801407e-1 * (0.2e1 * t36 - 0.2e1) / (0.2e1 * t39 - 0.2e1) * (0.1e1 + 0.278125e-1 * t10) * t54
  t58 = t34 ** 2
  t59 = jnp.where(t33, t58, 1)
  t60 = t59 ** 2
  t61 = t60 * t59
  t63 = r0 ** 2
  t76 = 0.1e1 / params.gamma
  t81 = jnp.exp(-(-t32 + t57) * t76 / t61)
  t83 = 0.1e1 / (t81 - 0.1e1)
  t85 = s0 ** 2
  t88 = t63 ** 2
  t91 = t39 ** 2
  t93 = t60 ** 2
  t102 = s0 / t7 / t63 * t39 / t60 * t18 / t3 * t5 / 0.96e2 + params.BB * params.beta * t76 * t83 * t85 / t21 / t88 * t91 / t93 * t1 / t19 * t6 / 0.3072e4
  t112 = jnp.log(0.1e1 + params.beta * t102 * t76 / (params.beta * t76 * t83 * t102 + 0.1e1))
  res = params.gamma * t61 * t112 - t32 + t57
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