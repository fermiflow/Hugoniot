"""Generated from gga_c_pw91.mpl."""

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
  t2 = 0.1e1 / jnp.pi
  t3 = jnp.cbrt(t2)
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
  t31 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t14 + 0.8969 * t11 + 0.204775 * t17 + 0.123235 * t25))
  t33 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t11) * t31
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
  t71 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t14 + 0.1549425e1 * t11 + 0.420775 * t17 + 0.1562925 * t25))
  t84 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t14 + 0.905775 * t11 + 0.1100325 * t17 + 0.1241775 * t25))
  t85 = (0.1e1 + 0.278125e-1 * t11) * t84
  t89 = t36 / t38 * t60 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t11) * t71 + t33 - 0.19751789702565206229e-1 * t85)
  t91 = 0.19751789702565206229e-1 * t60 * t85
  t92 = jnp.pi ** 2
  t95 = jnp.cbrt(t92)
  t96 = t95 ** 2
  t97 = t45 ** 2
  t98 = t47 ** 2
  t99 = jnp.where(t44, t97, t98)
  t100 = t52 ** 2
  t101 = jnp.where(t51, t97, t100)
  t103 = t99 / 0.2e1 + t101 / 0.2e1
  t104 = t103 ** 2
  t105 = t104 * t103
  t108 = 0.1e1 / t95
  t110 = s0 + 0.2e1 * s1 + s2
  t112 = 0.1e1 / t8 / t37
  t115 = 0.1e1 / t104
  t117 = 0.1e1 / t3
  t118 = t117 * t5
  t127 = 0.1e1 / t96
  t131 = jnp.exp(-0.13067859477648036197e2 * (-t33 + t89 + t91) / t105 * t92 * t1 * t127)
  t132 = t131 - 0.1e1
  t133 = 0.1e1 / t132
  t134 = t110 ** 2
  t139 = t56 ** 2
  t141 = t104 ** 2
  t146 = 0.1e1 / t22 / t38 * t139 / t141 / t20 * t6
  t155 = t112 * t56
  t162 = t132 ** 2
  t175 = jnp.log(0.1e1 + 0.88547815820543093274 * jnp.pi * t19 * t108 * (t110 * t112 * t56 * t115 * t19 * t118 / 0.96e2 + 0.86472476387249114526e-3 * jnp.pi * t108 * t133 * t134 * t146) / (0.1e1 + 0.27671192443919716648e-1 * jnp.pi * t1 * t108 * t133 * t110 * t155 * t115 * t117 * t5 + 0.76569489126843962094e-3 * t92 * t19 * t127 / t162 * t134 * t146))
  t193 = jnp.cbrt(9)
  t194 = t193 ** 2
  t204 = jnp.exp(-0.25e2 / 0.18e2 * t2 * t5 * t194 * t3 / t22 / t37 * t104 * t110 * t56)
  res = -t33 + t89 + t91 + 0.25507875555555555556e-1 / t92 * t19 * t96 * t105 * t175 + t2 * t95 * ((0.2568e1 + 0.58165e1 * t11 + 0.184725e-2 * t25) / (0.1e4 + 0.218075e4 * t11 + 0.118e3 * t25) - 0.18535714285714285714e-2) * t103 * t110 * t155 * t118 * t204 / 0.2e1
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(3)
  t2 = 0.1e1 / jnp.pi
  t3 = jnp.cbrt(t2)
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
  t30 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t13 + 0.8969 * t10 + 0.204775 * t16 + 0.123235 * t24))
  t32 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t10) * t30
  t33 = 0.1e1 <= p.zeta_threshold
  t34 = jnp.cbrt(p.zeta_threshold)
  t36 = jnp.where(t33, t34 * p.zeta_threshold, 1)
  t39 = jnp.cbrt(2)
  t54 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t13 + 0.905775 * t10 + 0.1100325 * t16 + 0.1241775 * t24))
  t57 = 0.19751789702565206229e-1 * (0.2e1 * t36 - 0.2e1) / (0.2e1 * t39 - 0.2e1) * (0.1e1 + 0.278125e-1 * t10) * t54
  t58 = jnp.pi ** 2
  t61 = jnp.cbrt(t58)
  t62 = t61 ** 2
  t63 = t34 ** 2
  t64 = jnp.where(t33, t63, 1)
  t65 = t64 ** 2
  t66 = t65 * t64
  t69 = 0.1e1 / t61
  t70 = r0 ** 2
  t72 = 0.1e1 / t7 / t70
  t75 = 0.1e1 / t65
  t77 = 0.1e1 / t3
  t78 = t77 * t5
  t87 = 0.1e1 / t62
  t91 = jnp.exp(-0.13067859477648036197e2 * (-t32 + t57) / t66 * t58 * t1 * t87)
  t92 = t91 - 0.1e1
  t93 = 0.1e1 / t92
  t94 = s0 ** 2
  t97 = t70 ** 2
  t100 = t39 ** 2
  t102 = t65 ** 2
  t107 = 0.1e1 / t21 / t97 * t100 / t102 / t19 * t6
  t116 = t72 * t39
  t123 = t92 ** 2
  t136 = jnp.log(0.1e1 + 0.88547815820543093274 * jnp.pi * t18 * t69 * (s0 * t72 * t39 * t75 * t18 * t78 / 0.96e2 + 0.86472476387249114526e-3 * jnp.pi * t69 * t93 * t94 * t107) / (0.1e1 + 0.27671192443919716648e-1 * jnp.pi * t1 * t69 * t93 * s0 * t116 * t75 * t77 * t5 + 0.76569489126843962094e-3 * t58 * t18 * t87 / t123 * t94 * t107))
  t154 = jnp.cbrt(9)
  t155 = t154 ** 2
  t165 = jnp.exp(-0.25e2 / 0.18e2 * t2 * t5 * t155 * t3 / t21 / t70 * t65 * s0 * t39)
  res = -t32 + t57 + 0.25507875555555555556e-1 / t58 * t18 * t62 * t66 * t136 + t2 * t61 * ((0.2568e1 + 0.58165e1 * t10 + 0.184725e-2 * t24) / (0.1e4 + 0.218075e4 * t10 + 0.118e3 * t24) - 0.18535714285714285714e-2) * t64 * s0 * t116 * t78 * t165 / 0.2e1
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