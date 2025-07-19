"""Generated from lda_x_erf.mpl."""

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
  t7 = t1 * t3 * t6
  t8 = jnp.cbrt(2)
  t9 = t8 ** 2
  t11 = r0 + r1
  t13 = (r0 - r1) / t11
  t14 = 0.1e1 + t13
  t15 = t14 <= p.zeta_threshold
  t16 = jnp.cbrt(p.zeta_threshold)
  t17 = t16 * p.zeta_threshold
  t18 = jnp.cbrt(t14)
  t20 = jnp.where(t15, t17, t18 * t14)
  t22 = jnp.cbrt(t11)
  t23 = jnp.cbrt(9)
  t24 = t23 ** 2
  t25 = t3 ** 2
  t27 = t24 * t25 * p.cam_omega
  t29 = t1 / t22
  t30 = jnp.where(t15, t16, t18)
  t34 = t27 * t29 / t30 / 0.18e2
  t36 = 0.135e1 < t34
  t37 = jnp.where(t36, t34, 0.135e1)
  t38 = t37 ** 2
  t41 = t38 ** 2
  t44 = t41 * t38
  t47 = t41 ** 2
  t59 = t47 ** 2
  t63 = jnp.where(t36, 0.135e1, t34)
  t64 = jnp.sqrt(jnp.pi)
  t67 = jax.lax.erf(0.1e1 / t63 / 0.2e1)
  t69 = t63 ** 2
  t72 = jnp.exp(-0.1e1 / t69 / 0.4e1)
  t83 = jnp.where(0.135e1 <= t34, 0.1e1 / t38 / 0.36e2 - 0.1e1 / t41 / 0.96e3 + 0.1e1 / t44 / 0.2688e5 - 0.1e1 / t47 / 0.82944e6 + 0.1e1 / t47 / t38 / 0.2838528e8 - 0.1e1 / t47 / t41 / 0.107347968e10 + 0.1e1 / t47 / t44 / 0.445906944e11 - 0.1e1 / t59 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t63 * (t64 * t67 + 0.2e1 * t63 * (t72 - 0.3e1 / 0.2e1 - 0.2e1 * t69 * (t72 - 0.1e1))))
  t87 = 0.1e1 - t13
  t88 = t87 <= p.zeta_threshold
  t89 = jnp.cbrt(t87)
  t91 = jnp.where(t88, t17, t89 * t87)
  t93 = jnp.where(t88, t16, t89)
  t97 = t27 * t29 / t93 / 0.18e2
  t99 = 0.135e1 < t97
  t100 = jnp.where(t99, t97, 0.135e1)
  t101 = t100 ** 2
  t104 = t101 ** 2
  t107 = t104 * t101
  t110 = t104 ** 2
  t122 = t110 ** 2
  t126 = jnp.where(t99, 0.135e1, t97)
  t129 = jax.lax.erf(0.1e1 / t126 / 0.2e1)
  t131 = t126 ** 2
  t134 = jnp.exp(-0.1e1 / t131 / 0.4e1)
  t145 = jnp.where(0.135e1 <= t97, 0.1e1 / t101 / 0.36e2 - 0.1e1 / t104 / 0.96e3 + 0.1e1 / t107 / 0.2688e5 - 0.1e1 / t110 / 0.82944e6 + 0.1e1 / t110 / t101 / 0.2838528e8 - 0.1e1 / t110 / t104 / 0.107347968e10 + 0.1e1 / t110 / t107 / 0.445906944e11 - 0.1e1 / t122 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t126 * (t64 * t129 + 0.2e1 * t126 * (t134 - 0.3e1 / 0.2e1 - 0.2e1 * t131 * (t134 - 0.1e1))))
  res = -0.3e1 / 0.32e2 * t7 * t9 * t91 * t22 * t145 - 0.3e1 / 0.32e2 * t7 * t9 * t20 * t22 * t83
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(3)
  t3 = jnp.cbrt(0.1e1 / jnp.pi)
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t8 = jnp.cbrt(2)
  t9 = t8 ** 2
  t10 = 0.1e1 <= p.zeta_threshold
  t11 = jnp.cbrt(p.zeta_threshold)
  t13 = jnp.where(t10, t11 * p.zeta_threshold, 1)
  t15 = jnp.cbrt(r0)
  t16 = jnp.cbrt(9)
  t17 = t16 ** 2
  t18 = t3 ** 2
  t23 = jnp.where(t10, t11, 1)
  t27 = t17 * t18 * p.cam_omega * t1 / t15 / t23 / 0.18e2
  t29 = 0.135e1 < t27
  t30 = jnp.where(t29, t27, 0.135e1)
  t31 = t30 ** 2
  t34 = t31 ** 2
  t37 = t34 * t31
  t40 = t34 ** 2
  t52 = t40 ** 2
  t56 = jnp.where(t29, 0.135e1, t27)
  t57 = jnp.sqrt(jnp.pi)
  t60 = jax.lax.erf(0.1e1 / t56 / 0.2e1)
  t62 = t56 ** 2
  t65 = jnp.exp(-0.1e1 / t62 / 0.4e1)
  t76 = jnp.where(0.135e1 <= t27, 0.1e1 / t31 / 0.36e2 - 0.1e1 / t34 / 0.96e3 + 0.1e1 / t37 / 0.2688e5 - 0.1e1 / t40 / 0.82944e6 + 0.1e1 / t40 / t31 / 0.2838528e8 - 0.1e1 / t40 / t34 / 0.107347968e10 + 0.1e1 / t40 / t37 / 0.445906944e11 - 0.1e1 / t52 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t56 * (t57 * t60 + 0.2e1 * t56 * (t65 - 0.3e1 / 0.2e1 - 0.2e1 * t62 * (t65 - 0.1e1))))
  res = -0.3e1 / 0.16e2 * t1 * t3 * t6 * t9 * t13 * t15 * t76
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