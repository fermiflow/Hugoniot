"""Generated from mgga_k_pgslb.mpl."""

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
  t34 = jnp.pi ** 2
  t35 = jnp.cbrt(t34)
  t36 = t35 ** 2
  t37 = 0.1e1 / t36
  t38 = t33 * t37
  t39 = r0 ** 2
  t40 = jnp.cbrt(r0)
  t41 = t40 ** 2
  t43 = 0.1e1 / t41 / t39
  t47 = params.pgslb_mu * t33
  t52 = jnp.exp(-t47 * t37 * s0 * t43 / 0.24e2)
  t53 = t33 ** 2
  t54 = params.pgslb_beta * t53
  t56 = 0.1e1 / t35 / t34
  t57 = l0 ** 2
  t69 = jnp.where(r0 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t29 * t31 * (0.5e1 / 0.72e2 * t38 * s0 * t43 + t52 + t54 * t56 * t57 / t40 / t39 / r0 / 0.576e3))
  t71 = jnp.where(t11, t16, -t18)
  t72 = jnp.where(t15, t12, t71)
  t73 = 0.1e1 + t72
  t75 = jnp.cbrt(t73)
  t76 = t75 ** 2
  t78 = jnp.where(t73 <= p.zeta_threshold, t25, t76 * t73)
  t80 = r1 ** 2
  t81 = jnp.cbrt(r1)
  t82 = t81 ** 2
  t84 = 0.1e1 / t82 / t80
  t92 = jnp.exp(-t47 * t37 * s2 * t84 / 0.24e2)
  t93 = l1 ** 2
  t105 = jnp.where(r1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t78 * t31 * (0.5e1 / 0.72e2 * t38 * s2 * t84 + t92 + t54 * t56 * t93 / t81 / t80 / r1 / 0.576e3))
  res = t69 + t105
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
  t26 = jnp.pi ** 2
  t27 = jnp.cbrt(t26)
  t28 = t27 ** 2
  t29 = 0.1e1 / t28
  t31 = jnp.cbrt(2)
  t32 = t31 ** 2
  t34 = r0 ** 2
  t37 = s0 * t32 / t23 / t34
  t44 = jnp.exp(-params.pgslb_mu * t25 * t29 * t37 / 0.24e2)
  t45 = t25 ** 2
  t50 = l0 ** 2
  t62 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t4 * t5 * jnp.pi * t21 * t23 * (0.5e1 / 0.72e2 * t25 * t29 * t37 + t44 + params.pgslb_beta * t45 / t27 / t26 * t50 * t31 / t22 / t34 / r0 / 0.288e3))
  res = 0.2e1 * t62
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