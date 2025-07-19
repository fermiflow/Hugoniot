"""Generated from gga_x_chachiyo.mpl."""

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
  t20 = t19 + 0.1e1
  t22 = jnp.cbrt(p.zeta_threshold)
  t23 = t22 * p.zeta_threshold
  t24 = jnp.cbrt(t20)
  t26 = jnp.where(t20 <= p.zeta_threshold, t23, t24 * t20)
  t28 = jnp.cbrt(t6)
  t29 = t3 ** 2
  t30 = t2 * t29
  t31 = jnp.cbrt(2)
  t33 = r0 ** 2
  t34 = jnp.cbrt(r0)
  t35 = t34 ** 2
  t41 = jnp.pi ** 2
  t42 = t2 ** 2
  t43 = t42 * t3
  t44 = t31 ** 2
  t45 = jnp.sqrt(s0)
  t50 = t43 * t44 * t45 / t34 / r0
  t53 = jnp.log(t50 / 0.27e2 + 0.1e1)
  t65 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * (0.2e1 / 0.81e2 * t30 * t31 * s0 / t35 / t33 + t41 * t53) / (t50 / 0.9e1 + t41) / t53)
  t67 = jnp.where(t10, t15, -t17)
  t68 = jnp.where(t14, t11, t67)
  t69 = t68 + 0.1e1
  t71 = jnp.cbrt(t69)
  t73 = jnp.where(t69 <= p.zeta_threshold, t23, t71 * t69)
  t76 = r1 ** 2
  t77 = jnp.cbrt(r1)
  t78 = t77 ** 2
  t84 = jnp.sqrt(s2)
  t89 = t43 * t44 * t84 / t77 / r1
  t92 = jnp.log(t89 / 0.27e2 + 0.1e1)
  t104 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t73 * t28 * (0.2e1 / 0.81e2 * t30 * t31 * s2 / t78 / t76 + t41 * t92) / (t89 / 0.9e1 + t41) / t92)
  res = t65 + t104
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
  t12 = t11 + 0.1e1
  t14 = jnp.cbrt(p.zeta_threshold)
  t16 = jnp.cbrt(t12)
  t18 = jnp.where(t12 <= p.zeta_threshold, t14 * p.zeta_threshold, t16 * t12)
  t20 = jnp.cbrt(r0)
  t21 = t4 ** 2
  t23 = r0 ** 2
  t24 = t20 ** 2
  t30 = jnp.pi ** 2
  t31 = t3 ** 2
  t33 = jnp.sqrt(s0)
  t37 = t31 * t4 * t33 / t20 / r0
  t40 = jnp.log(0.2e1 / 0.27e2 * t37 + 0.1e1)
  t52 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * (0.4e1 / 0.81e2 * t3 * t21 * s0 / t24 / t23 + t30 * t40) / (0.2e1 / 0.9e1 * t37 + t30) / t40)
  res = 0.2e1 * t52
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