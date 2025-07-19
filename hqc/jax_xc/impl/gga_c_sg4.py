"""Generated from gga_c_sg4.mpl."""

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
  t112 = t98 ** (0.5e-1 * t101 * t100 * t39 * t106 / t14 / t11)
  t113 = jnp.log(0.2e1)
  t114 = 0.1e1 - t113
  t116 = jnp.pi ** 2
  t117 = 0.1e1 / t116
  t123 = t56 ** 2
  t129 = jnp.exp(-t25 / 0.4e1)
  t134 = 0.786 * t117 + 0.175e-1 * t101 / t8 / t7 * t123 / t98 / t14 * (0.1e1 - t129)
  t146 = 0.1e1 / t114
  t147 = t134 * t146
  t152 = jnp.exp(-(-t33 + t89 + t91) * t146 * t116 * t106)
  t155 = t116 / (t152 - 0.1e1)
  t156 = t100 ** 2
  t162 = t104 ** 2
  t171 = t100 / t8 / t37 * t56 / t104 * t19 / t3 * t5 / 0.96e2 + t147 * t155 * t156 / t22 / t38 * t123 / t162 * t1 / t20 * t6 / 0.3072e4
  t181 = jnp.log(0.1e1 + t134 * t171 * t146 * t116 / (t147 * t155 * t171 + 0.1e1))
  res = t112 * t114 * t117 * t105 * t181 - t33 + t89 + t91
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
  t62 = r0 ** 2
  t63 = t62 ** 2
  t66 = t59 ** 2
  t67 = t66 * t59
  t68 = 0.1e1 / t67
  t74 = t59 ** (0.5e-1 * t60 * s0 / t63 * t68 / t13 / t10)
  t75 = jnp.log(0.2e1)
  t76 = 0.1e1 - t75
  t78 = jnp.pi ** 2
  t79 = 0.1e1 / t78
  t85 = t39 ** 2
  t91 = jnp.exp(-t24 / 0.4e1)
  t96 = 0.786 * t79 + 0.175e-1 * t60 / t7 / r0 * t85 / t59 / t13 * (0.1e1 - t91)
  t108 = 0.1e1 / t76
  t109 = t96 * t108
  t114 = jnp.exp(-(-t32 + t57) * t108 * t78 * t68)
  t117 = t78 / (t114 - 0.1e1)
  t118 = s0 ** 2
  t124 = t66 ** 2
  t133 = s0 / t7 / t62 * t39 / t66 * t18 / t3 * t5 / 0.96e2 + t109 * t117 * t118 / t21 / t63 * t85 / t124 * t1 / t19 * t6 / 0.3072e4
  t143 = jnp.log(0.1e1 + t96 * t133 * t108 * t78 / (t109 * t117 * t133 + 0.1e1))
  res = t74 * t76 * t79 * t67 * t143 - t32 + t57
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