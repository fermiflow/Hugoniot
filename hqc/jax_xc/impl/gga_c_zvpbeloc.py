"""Generated from gga_c_zvpbeloc.mpl."""

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
  t1 = 4 ** (0.1e1 / 0.6e1)
  t2 = t1 ** 2
  t3 = t2 ** 2
  t5 = 3 ** (0.1e1 / 0.6e1)
  t7 = jnp.pi ** 2
  t8 = 0.1e1 / t7
  t9 = t8 ** (0.1e1 / 0.6e1)
  t12 = jnp.cbrt(0.1e1 / jnp.pi)
  t13 = r0 + r1
  t14 = jnp.cbrt(t13)
  t15 = 0.1e1 / t14
  t17 = r0 - r1
  t18 = t17 ** 2
  t19 = t13 ** 2
  t21 = t18 / t19
  t23 = jnp.where(0.1e-19 < t21, t21, 0.1e-19)
  t27 = jnp.exp(-0.99999999999999999999 * t3 * t1 * t5 * t9 * t12 * t15 * t23)
  t28 = jnp.cbrt(3)
  t30 = jnp.cbrt(4)
  t31 = t30 ** 2
  t33 = t28 * t12 * t31 * t15
  t36 = jnp.sqrt(t33)
  t39 = t33 ** 0.15e1
  t41 = t28 ** 2
  t42 = t12 ** 2
  t44 = t14 ** 2
  t47 = t41 * t42 * t30 / t44
  t53 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t36 + 0.8969 * t33 + 0.204775 * t39 + 0.123235 * t47))
  t55 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t33) * t53
  t56 = t18 ** 2
  t57 = t19 ** 2
  t61 = t17 / t13
  t62 = 0.1e1 + t61
  t63 = t62 <= p.zeta_threshold
  t64 = jnp.cbrt(p.zeta_threshold)
  t65 = t64 * p.zeta_threshold
  t66 = jnp.cbrt(t62)
  t68 = jnp.where(t63, t65, t66 * t62)
  t69 = 0.1e1 - t61
  t70 = t69 <= p.zeta_threshold
  t71 = jnp.cbrt(t69)
  t73 = jnp.where(t70, t65, t71 * t69)
  t75 = jnp.cbrt(2)
  t79 = (t68 + t73 - 0.2e1) / (0.2e1 * t75 - 0.2e1)
  t90 = jnp.log(0.1e1 + 0.32163958997385070134e2 / (0.705945e1 * t36 + 0.1549425e1 * t33 + 0.420775 * t39 + 0.1562925 * t47))
  t103 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t36 + 0.905775 * t33 + 0.1100325 * t39 + 0.1241775 * t47))
  t104 = (0.1e1 + 0.278125e-1 * t33) * t103
  t108 = t56 / t57 * t79 * (-0.310907e-1 * (0.1e1 + 0.5137e-1 * t33) * t90 + t55 - 0.19751673498613801407e-1 * t104)
  t110 = 0.19751673498613801407e-1 * t79 * t104
  t111 = jnp.log(0.2e1)
  t112 = 0.1e1 - t111
  t114 = t64 ** 2
  t115 = t66 ** 2
  t116 = jnp.where(t63, t114, t115)
  t117 = t71 ** 2
  t118 = jnp.where(t70, t114, t117)
  t120 = t116 / 0.2e1 + t118 / 0.2e1
  t121 = t120 ** 2
  t122 = t121 * t120
  t124 = s0 + 0.2e1 * s1 + s2
  t127 = t124 / t14 / t19
  t128 = 0.1e1 / t121
  t131 = 0.1e1 / t12
  t134 = jnp.exp(-t47 / 0.4e1)
  t140 = 0.375e-1 + 0.83333333333333333332e-3 * t127 * t75 * t128 * t41 * t131 * t30 * (0.1e1 - t134)
  t147 = 0.1e1 / t112
  t148 = t140 * t147
  t154 = jnp.exp(-(-t55 + t108 + t110) * t147 * t7 / t122)
  t157 = t7 / (t154 - 0.1e1)
  t158 = t124 ** 2
  t163 = t75 ** 2
  t165 = t121 ** 2
  t174 = t127 * t75 * t128 * t41 * t131 * t30 / 0.96e2 + t148 * t157 * t158 / t44 / t57 * t163 / t165 * t28 / t42 * t31 / 0.3072e4
  t184 = jnp.log(0.1e1 + t140 * t174 * t147 * t7 / (t148 * t157 * t174 + 0.1e1))
  res = t27 * (t112 * t8 * t122 * t184 + t108 + t110 - t55)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = 4 ** (0.1e1 / 0.6e1)
  t2 = t1 ** 2
  t3 = t2 ** 2
  t5 = 3 ** (0.1e1 / 0.6e1)
  t7 = jnp.pi ** 2
  t8 = 0.1e1 / t7
  t9 = t8 ** (0.1e1 / 0.6e1)
  t12 = jnp.cbrt(0.1e1 / jnp.pi)
  t13 = jnp.cbrt(r0)
  t14 = 0.1e1 / t13
  t17 = jnp.where(0.1e-19 < 0., 0, 0.1e-19)
  t21 = jnp.exp(-0.99999999999999999999 * t3 * t1 * t5 * t9 * t12 * t14 * t17)
  t22 = jnp.cbrt(3)
  t24 = jnp.cbrt(4)
  t25 = t24 ** 2
  t27 = t22 * t12 * t25 * t14
  t30 = jnp.sqrt(t27)
  t33 = t27 ** 0.15e1
  t35 = t22 ** 2
  t36 = t12 ** 2
  t38 = t13 ** 2
  t41 = t35 * t36 * t24 / t38
  t47 = jnp.log(0.1e1 + 0.16081979498692535067e2 / (0.379785e1 * t30 + 0.8969 * t27 + 0.204775 * t33 + 0.123235 * t41))
  t49 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t27) * t47
  t50 = 0.1e1 <= p.zeta_threshold
  t51 = jnp.cbrt(p.zeta_threshold)
  t53 = jnp.where(t50, t51 * p.zeta_threshold, 1)
  t56 = jnp.cbrt(2)
  t71 = jnp.log(0.1e1 + 0.29608749977793437516e2 / (0.51785e1 * t30 + 0.905775 * t27 + 0.1100325 * t33 + 0.1241775 * t41))
  t74 = 0.19751673498613801407e-1 * (0.2e1 * t53 - 0.2e1) / (0.2e1 * t56 - 0.2e1) * (0.1e1 + 0.278125e-1 * t27) * t71
  t75 = jnp.log(0.2e1)
  t76 = 0.1e1 - t75
  t78 = t51 ** 2
  t79 = jnp.where(t50, t78, 1)
  t80 = t79 ** 2
  t81 = t80 * t79
  t82 = r0 ** 2
  t85 = s0 / t13 / t82
  t86 = 0.1e1 / t80
  t89 = 0.1e1 / t12
  t92 = jnp.exp(-t41 / 0.4e1)
  t98 = 0.375e-1 + 0.83333333333333333332e-3 * t85 * t56 * t86 * t35 * t89 * t24 * (0.1e1 - t92)
  t105 = 0.1e1 / t76
  t106 = t98 * t105
  t112 = jnp.exp(-(-t49 + t74) * t105 * t7 / t81)
  t115 = t7 / (t112 - 0.1e1)
  t116 = s0 ** 2
  t119 = t82 ** 2
  t122 = t56 ** 2
  t124 = t80 ** 2
  t133 = t85 * t56 * t86 * t35 * t89 * t24 / 0.96e2 + t106 * t115 * t116 / t38 / t119 * t122 / t124 * t22 / t36 * t25 / 0.3072e4
  t143 = jnp.log(0.1e1 + t98 * t133 * t105 * t7 / (t106 * t115 * t133 + 0.1e1))
  res = t21 * (t76 * t8 * t81 * t143 - t49 + t74)
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