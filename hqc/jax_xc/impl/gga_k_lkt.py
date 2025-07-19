"""Generated from gga_k_lkt.mpl."""

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
  t34 = t33 ** 2
  t35 = jnp.pi ** 2
  t36 = jnp.cbrt(t35)
  t38 = t34 / t36
  t39 = jnp.sqrt(s0)
  t40 = jnp.cbrt(r0)
  t45 = t38 * t39 / t40 / r0 / 0.12e2
  t47 = jnp.where(t45 < 0.2e3, t45, 200)
  t49 = jnp.cosh(params.a * t47)
  t51 = t36 ** 2
  t53 = t33 / t51
  t54 = r0 ** 2
  t55 = t40 ** 2
  t65 = jnp.where(r0 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t29 * t31 * (0.1e1 / t49 + 0.5e1 / 0.72e2 * t53 * s0 / t55 / t54))
  t67 = jnp.where(t11, t16, -t18)
  t68 = jnp.where(t15, t12, t67)
  t69 = 0.1e1 + t68
  t71 = jnp.cbrt(t69)
  t72 = t71 ** 2
  t74 = jnp.where(t69 <= p.zeta_threshold, t25, t72 * t69)
  t76 = jnp.sqrt(s2)
  t77 = jnp.cbrt(r1)
  t82 = t38 * t76 / t77 / r1 / 0.12e2
  t84 = jnp.where(t82 < 0.2e3, t82, 200)
  t86 = jnp.cosh(params.a * t84)
  t88 = r1 ** 2
  t89 = t77 ** 2
  t99 = jnp.where(r1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t74 * t31 * (0.1e1 / t86 + 0.5e1 / 0.72e2 * t53 * s2 / t89 / t88))
  res = t65 + t99
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
  t26 = t25 ** 2
  t27 = jnp.pi ** 2
  t28 = jnp.cbrt(t27)
  t31 = jnp.sqrt(s0)
  t32 = jnp.cbrt(2)
  t38 = t26 / t28 * t31 * t32 / t22 / r0 / 0.12e2
  t40 = jnp.where(t38 < 0.2e3, t38, 200)
  t42 = jnp.cosh(params.a * t40)
  t44 = t28 ** 2
  t47 = t32 ** 2
  t49 = r0 ** 2
  t59 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t4 * t5 * jnp.pi * t21 * t23 * (0.1e1 / t42 + 0.5e1 / 0.72e2 * t25 / t44 * s0 * t47 / t23 / t49))
  res = 0.2e1 * t59
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