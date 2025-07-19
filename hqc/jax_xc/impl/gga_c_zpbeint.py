"""Generated from gga_c_zpbeint.mpl."""

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
  t39 = 0.1e1 / t38
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
  t89 = t36 * t39 * t60 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t11) * t71 + t33 - 0.19751673498613801407e-1 * t85)
  t91 = 0.19751673498613801407e-1 * t60 * t85
  t92 = t45 ** 2
  t93 = t47 ** 2
  t94 = jnp.where(t44, t92, t93)
  t95 = t52 ** 2
  t96 = jnp.where(t51, t92, t95)
  t98 = t94 / 0.2e1 + t96 / 0.2e1
  t100 = s0 + 0.2e1 * s1 + s2
  t101 = jnp.sqrt(t100)
  t104 = t98 ** 2
  t105 = t104 * t98
  t106 = 0.1e1 / t105
  t113 = t98 ** (params.alpha * t101 * t100 * t39 * t106 / t14 / t11 / 0.16e2)
  t114 = jnp.log(0.2e1)
  t115 = 0.1e1 - t114
  t117 = jnp.pi ** 2
  t131 = 0.1e1 / t115
  t132 = params.beta * t131
  t137 = jnp.exp(-(-t33 + t89 + t91) * t131 * t117 * t106)
  t140 = t117 / (t137 - 0.1e1)
  t141 = t100 ** 2
  t146 = t56 ** 2
  t148 = t104 ** 2
  t157 = t100 / t8 / t37 * t56 / t104 * t19 / t3 * t5 / 0.96e2 + t132 * t140 * t141 / t22 / t38 * t146 / t148 * t1 / t20 * t6 / 0.3072e4
  t167 = jnp.log(0.1e1 + params.beta * t157 * t131 * t117 / (t132 * t140 * t157 + 0.1e1))
  res = -t33 + t89 + t91 + t113 * t115 / t117 * t105 * t167
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
  t60 = jnp.sqrt(s0)
  t63 = r0 ** 2
  t64 = t63 ** 2
  t66 = t59 ** 2
  t67 = t66 * t59
  t68 = 0.1e1 / t67
  t75 = t59 ** (params.alpha * t60 * s0 / t64 * t68 / t13 / t10 / 0.16e2)
  t76 = jnp.log(0.2e1)
  t77 = 0.1e1 - t76
  t79 = jnp.pi ** 2
  t93 = 0.1e1 / t77
  t94 = params.beta * t93
  t99 = jnp.exp(-(-t32 + t57) * t93 * t79 * t68)
  t102 = t79 / (t99 - 0.1e1)
  t103 = s0 ** 2
  t108 = t39 ** 2
  t110 = t66 ** 2
  t119 = s0 / t7 / t63 * t39 / t66 * t18 / t3 * t5 / 0.96e2 + t94 * t102 * t103 / t21 / t64 * t108 / t110 * t1 / t19 * t6 / 0.3072e4
  t129 = jnp.log(0.1e1 + params.beta * t119 * t93 * t79 / (t94 * t102 * t119 + 0.1e1))
  res = -t32 + t57 + t75 * t77 / t79 * t67 * t129
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