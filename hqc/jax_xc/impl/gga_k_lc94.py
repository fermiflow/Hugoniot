"""Generated from gga_k_lc94.mpl."""

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
  t3 = t2 ** 2
  t4 = jnp.cbrt(jnp.pi)
  t6 = t3 * t4 * jnp.pi
  t7 = r0 + r1
  t8 = 0.1e1 / t7
  t11 = 0.2e1 * r0 * t8 <= p.zeta_threshold
  t12 = p.zeta_threshold - 0.1e1
  t15 = 0.2e1 * r1 * t8 <= p.zeta_threshold
  t16 = -t12
  t18 = (r0 - r1) * t8
  t19 = jnp.where(t15, t16, t18)
  t20 = jnp.where(t11, t12, t19)
  t21 = 0.1e1 + t20
  t23 = jnp.cbrt(p.zeta_threshold)
  t24 = t23 ** 2
  t25 = t24 * p.zeta_threshold
  t26 = jnp.cbrt(t21)
  t27 = t26 ** 2
  t29 = jnp.where(t21 <= p.zeta_threshold, t25, t27 * t21)
  t30 = jnp.cbrt(t7)
  t31 = t30 ** 2
  t33 = jnp.cbrt(6)
  t34 = params.alpha * t33
  t35 = jnp.pi ** 2
  t36 = jnp.cbrt(t35)
  t37 = t36 ** 2
  t38 = 0.1e1 / t37
  t40 = r0 ** 2
  t41 = jnp.cbrt(r0)
  t42 = t41 ** 2
  t45 = t38 * s0 / t42 / t40
  t48 = jnp.exp(-t34 * t45 / 0.24e2)
  t54 = t33 ** 2
  t55 = 0.1e1 / t36
  t56 = t54 * t55
  t57 = jnp.sqrt(s0)
  t59 = 0.1e1 / t41 / r0
  t63 = (t56 * t57 * t59 / 0.12e2) ** params.expo
  t64 = params.f * t63
  t68 = params.b * t54
  t73 = jnp.arcsinh(t68 * t55 * t57 * t59 / 0.12e2)
  t84 = jnp.where(r0 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t29 * t31 * (0.1e1 + ((params.d * t48 + params.c) * t33 * t45 / 0.24e2 - t64) / (0.1e1 + t56 * t57 * t59 * params.a * t73 / 0.12e2 + t64)))
  t86 = jnp.where(t11, t16, -t18)
  t87 = jnp.where(t15, t12, t86)
  t88 = 0.1e1 + t87
  t90 = jnp.cbrt(t88)
  t91 = t90 ** 2
  t93 = jnp.where(t88 <= p.zeta_threshold, t25, t91 * t88)
  t96 = r1 ** 2
  t97 = jnp.cbrt(r1)
  t98 = t97 ** 2
  t101 = t38 * s2 / t98 / t96
  t104 = jnp.exp(-t34 * t101 / 0.24e2)
  t110 = jnp.sqrt(s2)
  t112 = 0.1e1 / t97 / r1
  t116 = (t56 * t110 * t112 / 0.12e2) ** params.expo
  t117 = params.f * t116
  t125 = jnp.arcsinh(t68 * t55 * t110 * t112 / 0.12e2)
  t136 = jnp.where(r1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t93 * t31 * (0.1e1 + ((params.d * t104 + params.c) * t33 * t101 / 0.24e2 - t117) / (0.1e1 + t56 * t110 * t112 * params.a * t125 / 0.12e2 + t117)))
  res = t84 + t136
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = jnp.cbrt(3)
  t4 = t3 ** 2
  t5 = jnp.cbrt(jnp.pi)
  t8 = 0.1e1 <= p.zeta_threshold
  t9 = p.zeta_threshold - 0.1e1
  t11 = jnp.where(t8, -t9, 0)
  t12 = jnp.where(t8, t9, t11)
  t13 = 0.1e1 + t12
  t15 = jnp.cbrt(p.zeta_threshold)
  t16 = t15 ** 2
  t18 = jnp.cbrt(t13)
  t19 = t18 ** 2
  t21 = jnp.where(t13 <= p.zeta_threshold, t16 * p.zeta_threshold, t19 * t13)
  t22 = jnp.cbrt(r0)
  t23 = t22 ** 2
  t25 = jnp.cbrt(6)
  t27 = jnp.pi ** 2
  t28 = jnp.cbrt(t27)
  t29 = t28 ** 2
  t30 = 0.1e1 / t29
  t32 = jnp.cbrt(2)
  t33 = t32 ** 2
  t35 = r0 ** 2
  t38 = s0 * t33 / t23 / t35
  t41 = jnp.exp(-params.alpha * t25 * t30 * t38 / 0.24e2)
  t48 = t25 ** 2
  t49 = 0.1e1 / t28
  t50 = t48 * t49
  t51 = jnp.sqrt(s0)
  t54 = 0.1e1 / t22 / r0
  t55 = t51 * t32 * t54
  t58 = (t50 * t55 / 0.12e2) ** params.expo
  t59 = params.f * t58
  t67 = jnp.arcsinh(params.b * t48 * t49 * t55 / 0.12e2)
  t79 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t4 * t5 * jnp.pi * t21 * t23 * (0.1e1 + ((params.d * t41 + params.c) * t25 * t30 * t38 / 0.24e2 - t59) / (0.1e1 + t50 * t51 * t32 * t54 * params.a * t67 / 0.12e2 + t59)))
  res = 0.2e1 * t79
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