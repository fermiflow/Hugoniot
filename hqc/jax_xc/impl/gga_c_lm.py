"""Generated from gga_c_lm.mpl."""

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
  t1 = 0.1e1 / jnp.pi
  t2 = r0 + r1
  t3 = 0.1e1 / t2
  t4 = t1 * t3
  t7 = jnp.cbrt(3)
  t8 = t7 ** 2
  t9 = jnp.cbrt(t1)
  t12 = jnp.cbrt(4)
  t13 = jnp.cbrt(t2)
  t15 = t8 / t9 * t12 * t13
  t18 = jnp.log(0.1e1 + 0.1e2 * t15)
  t20 = 0.252e-1 * (0.1e1 + t4 / 0.36e5) * t18
  t21 = t9 ** 2
  t23 = t13 ** 2
  t26 = t8 * t21 * t12 / t23
  t29 = t12 ** 2
  t32 = t7 * t9 * t29 / t13
  t35 = (r0 - r1) * t3
  t36 = 0.1e1 + t35
  t37 = t36 <= p.zeta_threshold
  t38 = jnp.cbrt(p.zeta_threshold)
  t39 = t38 * p.zeta_threshold
  t40 = jnp.cbrt(t36)
  t42 = jnp.where(t37, t39, t40 * t36)
  t43 = 0.1e1 - t35
  t44 = t43 <= p.zeta_threshold
  t45 = jnp.cbrt(t43)
  t47 = jnp.where(t44, t39, t45 * t43)
  t49 = jnp.cbrt(2)
  t58 = jnp.log(0.1e1 + 0.25e2 * t15)
  t66 = jnp.pi ** 2
  t67 = jnp.cbrt(t66)
  t70 = r0 ** 2
  t71 = jnp.cbrt(r0)
  t72 = t71 ** 2
  t77 = r1 ** 2
  t78 = jnp.cbrt(r1)
  t79 = t78 ** 2
  t87 = t38 ** 2
  t88 = t87 * p.zeta_threshold
  t89 = t40 ** 2
  t91 = jnp.where(t37, t88, t89 * t36)
  t92 = t45 ** 2
  t94 = jnp.where(t44, t88, t92 * t43)
  t96 = jnp.sqrt(t91 + t94)
  t98 = jnp.sqrt(0.2e1)
  t101 = t1 ** (0.1e1 / 0.6e1)
  t104 = s0 + 0.2e1 * s1 + s2
  t105 = jnp.sqrt(t104)
  t107 = t2 ** (0.1e1 / 0.6e1)
  t112 = jnp.exp(-t7 * params.lm_f / t101 * t105 / t107 / t2)
  t114 = t2 ** 2
  res = -t20 + 0.7e-5 * t26 - 0.105e-3 * t32 + 0.84e-2 + (t42 + t47 - 0.2e1) / (0.2e1 * t49 - 0.2e1) * (-0.127e-1 * (0.1e1 + 0.17777777777777777778e-5 * t4) * t58 - 0.64355555555555555556e-5 * t26 + 0.83833333333333333334e-4 * t32 - 0.41666666666666666667e-2 + t20) + jnp.pi * t8 / t67 / t66 * (-0.7e1 / 0.36e2 * t49 * (s0 / t72 / t70 * t42 + s2 / t79 / t77 * t47) + 0.2e1 / t96 * t98 * t112 * t104 / t23 / t114) * t13 / 0.144e3
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = 0.1e1 / jnp.pi
  t3 = t1 / r0
  t6 = jnp.cbrt(3)
  t7 = t6 ** 2
  t8 = jnp.cbrt(t1)
  t11 = jnp.cbrt(4)
  t12 = jnp.cbrt(r0)
  t14 = t7 / t8 * t11 * t12
  t17 = jnp.log(0.1e1 + 0.1e2 * t14)
  t19 = 0.252e-1 * (0.1e1 + t3 / 0.36e5) * t17
  t20 = t8 ** 2
  t22 = t12 ** 2
  t25 = t7 * t20 * t11 / t22
  t28 = t11 ** 2
  t31 = t6 * t8 * t28 / t12
  t33 = 0.1e1 <= p.zeta_threshold
  t34 = jnp.cbrt(p.zeta_threshold)
  t36 = jnp.where(t33, t34 * p.zeta_threshold, 1)
  t39 = jnp.cbrt(2)
  t48 = jnp.log(0.1e1 + 0.25e2 * t14)
  t56 = jnp.pi ** 2
  t57 = jnp.cbrt(t56)
  t60 = r0 ** 2
  t63 = s0 / t22 / t60
  t66 = t34 ** 2
  t68 = jnp.where(t33, t66 * p.zeta_threshold, 1)
  t69 = jnp.sqrt(t68)
  t72 = t1 ** (0.1e1 / 0.6e1)
  t74 = jnp.sqrt(s0)
  t76 = r0 ** (0.1e1 / 0.6e1)
  t81 = jnp.exp(-t6 * params.lm_f / t72 * t74 / t76 / r0)
  res = -t19 + 0.7e-5 * t25 - 0.105e-3 * t31 + 0.84e-2 + (0.2e1 * t36 - 0.2e1) / (0.2e1 * t39 - 0.2e1) * (-0.127e-1 * (0.1e1 + 0.17777777777777777778e-5 * t3) * t48 - 0.64355555555555555556e-5 * t25 + 0.83833333333333333334e-4 * t31 - 0.41666666666666666667e-2 + t19) + jnp.pi * t7 / t57 / t56 * (-0.7e1 / 0.9e1 * t63 * t36 + 0.2e1 / t69 * t81 * t63) * t12 / 0.144e3
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