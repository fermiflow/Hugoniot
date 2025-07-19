"""Generated from gga_c_revtca.mpl."""

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
  t1 = r0 - r1
  t2 = r0 + r1
  t3 = 0.1e1 / t2
  t4 = t1 * t3
  t5 = 0.1e1 + t4
  t7 = jnp.cbrt(p.zeta_threshold)
  t8 = t7 ** 2
  t9 = jnp.cbrt(t5)
  t10 = t9 ** 2
  t11 = jnp.where(t5 <= p.zeta_threshold, t8, t10)
  t12 = 0.1e1 - t4
  t14 = jnp.cbrt(t12)
  t15 = t14 ** 2
  t16 = jnp.where(t12 <= p.zeta_threshold, t8, t15)
  t18 = t11 / 0.2e1 + t16 / 0.2e1
  t19 = t18 ** 2
  t21 = jnp.cbrt(3)
  t22 = 0.1e1 / jnp.pi
  t23 = jnp.cbrt(t22)
  t25 = jnp.cbrt(4)
  t26 = t25 ** 2
  t27 = jnp.cbrt(t2)
  t33 = jnp.arctan(0.488827e1 + 0.79425925 * t21 * t23 * t26 / t27)
  t37 = t21 ** 2
  t38 = 0.1e1 / t23
  t42 = jnp.cbrt(6)
  t43 = t42 ** 2
  t44 = jnp.pi ** 2
  t45 = jnp.cbrt(t44)
  t47 = t43 / t45
  t48 = jnp.cbrt(2)
  t50 = s0 + 0.2e1 * s1 + s2
  t51 = jnp.sqrt(t50)
  t52 = t48 * t51
  t57 = (t47 * t52 / t27 / t2) ** 0.23e1
  t61 = t1 ** 2
  t62 = t61 ** 2
  t63 = t2 ** 2
  t64 = t63 ** 2
  t65 = 0.1e1 / t64
  t67 = jnp.cbrt(jnp.pi)
  t69 = jnp.cbrt(9)
  t76 = t67 * jnp.pi * t69 * t47 * t52 * t3 * t37 * t38 / 0.36e2
  t77 = 2.220446049250313e-16 ** (0.1e1 / 0.4e1)
  t79 = t67 ** 2
  t81 = t69 ** 2
  t83 = t45 ** 2
  t87 = t48 ** 2
  t91 = t23 ** 2
  t97 = t44 ** 2
  t105 = t50 ** 2
  t116 = jnp.where(t77 < t76, t76, t77)
  t117 = jnp.sin(t116)
  t120 = jnp.where(t76 <= t77, 0.1e1 - t79 * t44 * t81 * t42 / t83 * t87 * t50 / t63 * t21 / t91 / 0.432e3 + t67 * t97 * jnp.pi * t69 * t43 / t45 / t44 * t48 * t105 * t65 * t37 / t23 / t22 / 0.3456e5, t117 / t116)
  t121 = t120 ** 2
  res = t19 * t18 * (-0.655868 * t33 + 0.897889) * t37 * t38 * t25 * t27 / (0.1e1 + 0.47121507034422759993e-2 * t57) * (0.1e1 - t62 * t65 * (0.1e1 - t121)) / 0.3e1
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t2 = jnp.cbrt(p.zeta_threshold)
  t3 = t2 ** 2
  t4 = jnp.where(0.1e1 <= p.zeta_threshold, t3, 1)
  t5 = t4 ** 2
  t7 = jnp.cbrt(3)
  t9 = jnp.cbrt(0.1e1 / jnp.pi)
  t11 = jnp.cbrt(4)
  t12 = t11 ** 2
  t13 = jnp.cbrt(r0)
  t19 = jnp.arctan(0.488827e1 + 0.79425925 * t7 * t9 * t12 / t13)
  t23 = t7 ** 2
  t27 = jnp.cbrt(6)
  t28 = t27 ** 2
  t29 = jnp.pi ** 2
  t30 = jnp.cbrt(t29)
  t33 = jnp.cbrt(2)
  t34 = jnp.sqrt(s0)
  t40 = (t28 / t30 * t33 * t34 / t13 / r0) ** 0.23e1
  res = t5 * t4 * (-0.655868 * t19 + 0.897889) * t23 / t9 * t11 * t13 / (0.1e1 + 0.47121507034422759993e-2 * t40) / 0.3e1
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