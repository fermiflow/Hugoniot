"""Generated from gga_xc_th2.mpl."""

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
  t1 = r0 ** (0.1e1 / 0.12e2)
  t4 = r1 ** (0.1e1 / 0.12e2)
  t7 = r0 ** (0.1e1 / 0.6e1)
  t8 = t7 * r0
  t10 = r1 ** (0.1e1 / 0.6e1)
  t11 = t10 * r1
  t13 = jnp.cbrt(r0)
  t14 = t13 * r0
  t16 = jnp.cbrt(r1)
  t17 = t16 * r1
  t19 = jnp.sqrt(r0)
  t20 = t19 * r0
  t22 = jnp.sqrt(r1)
  t23 = t22 * r1
  t25 = t13 ** 2
  t26 = t25 * r0
  t28 = t16 ** 2
  t29 = t28 * r1
  t31 = t1 ** 2
  t32 = t31 ** 2
  t35 = t4 ** 2
  t36 = t35 ** 2
  t40 = jnp.sqrt(s0)
  t43 = r0 - r1
  t44 = r0 + r1
  t45 = 0.1e1 / t44
  t46 = t43 * t45
  t47 = 0.1e1 + t46
  t49 = jnp.cbrt(p.zeta_threshold)
  t50 = t49 * p.zeta_threshold
  t51 = jnp.cbrt(t47)
  t53 = jnp.where(t47 <= p.zeta_threshold, t50, t51 * t47)
  t54 = jnp.cbrt(2)
  t55 = t54 ** 2
  t58 = jnp.sqrt(s2)
  t61 = 0.1e1 - t46
  t63 = jnp.cbrt(t61)
  t65 = jnp.where(t61 <= p.zeta_threshold, t50, t63 * t61)
  t69 = t40 / t14 * t53 * t55 / 0.4e1 + t58 / t17 * t65 * t55 / 0.4e1
  t72 = t20 + t23
  t75 = 0.678831 * t1 * r0 + 0.678831 * t4 * r1 - 0.175821e1 * t8 - 0.175821e1 * t11 + 0.127676e1 * t14 + 0.127676e1 * t17 - 0.160789e1 * t20 - 0.160789e1 * t23 + 0.36561 * t26 + 0.36561 * t29 - 0.906635e-1 * (t32 * t1 * r0 + t36 * t4 * r1) * t69 + 0.734865e-1 * t72 * t69
  t76 = t26 + t29
  t79 = t7 ** 2
  t80 = t79 ** 2
  t83 = t10 ** 2
  t84 = t83 ** 2
  t87 = t80 * t7 * r0 + t84 * t10 * r1
  t90 = r0 ** 2
  t94 = t53 ** 2
  t96 = s0 / t25 / t90 * t94 * t54
  t97 = r1 ** 2
  t101 = t65 ** 2
  t103 = s2 / t28 / t97 * t101 * t54
  t105 = t96 / 0.8e1 + t103 / 0.8e1
  t110 = t90 + t97
  t117 = t44 ** 2
  t118 = jnp.cbrt(t44)
  t119 = t118 ** 2
  t123 = t96 / 0.4e1 + t103 / 0.4e1 - (s0 + 0.2e1 * s1 + s2) / t119 / t117
  t131 = t43 ** 2
  t133 = 0.1e1 / t117
  t143 = r0 ** 0.16666666666666666667e1
  t144 = r1 ** 0.16666666666666666667e1
  t149 = 0.735705e-1 * t76 * t69 - 0.3584585e-1 * t87 * t69 - 0.2035835e-1 * t76 * t105 + 0.1073125e-1 * t87 * t105 - 0.384078e-3 * t110 * t105 + 0.310377e-1 * t76 * t123 - 0.720326e-1 * t87 * t123 + 0.446562e-1 * t110 * t123 - 0.266802 * (t8 + t11) * t131 * t133 + 0.150822e1 * (t14 + t17) * t131 * t133 - 0.194515e1 * t72 * t131 * t133 + 0.679078 * (t143 + t144) * t131 * t133
  res = (t75 + t149) * t45
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = 2 ** (0.1e1 / 0.12e2)
  t2 = t1 ** 2
  t3 = t2 * t1
  t4 = t2 ** 2
  t5 = t4 ** 2
  t7 = r0 ** (0.1e1 / 0.12e2)
  t11 = 2 ** (0.1e1 / 0.6e1)
  t12 = t11 ** 2
  t13 = t12 ** 2
  t15 = r0 ** (0.1e1 / 0.6e1)
  t19 = jnp.cbrt(2)
  t20 = t19 ** 2
  t21 = jnp.cbrt(r0)
  t25 = jnp.sqrt(0.2e1)
  t26 = jnp.sqrt(r0)
  t30 = t21 ** 2
  t32 = t19 * t30 * r0
  t36 = jnp.sqrt(s0)
  t38 = jnp.cbrt(p.zeta_threshold)
  t40 = jnp.where(0.1e1 <= p.zeta_threshold, t38 * p.zeta_threshold, 1)
  t41 = t36 * t40
  t53 = 0.1e1 / r0
  t55 = t40 ** 2
  t56 = s0 * t55
  t59 = t15 ** 2
  t60 = t59 ** 2
  t61 = t60 * t15
  t70 = r0 ** 2
  t73 = s0 / t30 / t70
  t75 = t73 * t55 - t73
  t84 = 0.3394155 * t5 * t3 * t7 * r0 - 0.879105 * t13 * t11 * t15 * r0 + 0.63838 * t20 * t21 * r0 - 0.803945 * t25 * t26 * r0 + 0.182805 * t32 - 0.4533175e-1 * t4 * t3 * t7 * t41 + 0.3674325e-1 * t25 * t15 * t41 + 0.3678525e-1 * t19 * t21 * t41 - 0.17922925e-1 * t11 * t26 * t41 - 0.50895875e-2 * t19 * t53 * t56 + 0.26828125e-2 * t11 / t61 * t56 - 0.960195e-4 / t30 * s0 * t55 + 0.1551885e-1 * t32 * t75 - 0.360163e-1 * t11 * t61 * r0 * t75 + 0.223281e-1 * t70 * t75
  res = t84 * t53
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