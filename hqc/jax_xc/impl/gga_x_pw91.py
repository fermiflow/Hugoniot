"""Generated from gga_x_pw91.mpl."""

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
  t20 = 0.1e1 + t19
  t22 = jnp.cbrt(p.zeta_threshold)
  t23 = t22 * p.zeta_threshold
  t24 = jnp.cbrt(t20)
  t26 = jnp.where(t20 <= p.zeta_threshold, t23, t24 * t20)
  t27 = jnp.cbrt(t6)
  t29 = jnp.cbrt(6)
  t30 = params.alpha * t29
  t31 = jnp.pi ** 2
  t32 = jnp.cbrt(t31)
  t33 = t32 ** 2
  t34 = 0.1e1 / t33
  t36 = r0 ** 2
  t37 = jnp.cbrt(r0)
  t38 = t37 ** 2
  t41 = t34 * s0 / t38 / t36
  t44 = jnp.exp(-t30 * t41 / 0.24e2)
  t50 = t29 ** 2
  t51 = 0.1e1 / t32
  t52 = t50 * t51
  t53 = jnp.sqrt(s0)
  t55 = 0.1e1 / t37 / r0
  t59 = (t52 * t53 * t55 / 0.12e2) ** params.expo
  t60 = params.f * t59
  t64 = params.b * t50
  t69 = jnp.arcsinh(t64 * t51 * t53 * t55 / 0.12e2)
  t80 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + ((params.d * t44 + params.c) * t29 * t41 / 0.24e2 - t60) / (0.1e1 + t52 * t53 * t55 * params.a * t69 / 0.12e2 + t60)))
  t82 = jnp.where(t10, t15, -t17)
  t83 = jnp.where(t14, t11, t82)
  t84 = 0.1e1 + t83
  t86 = jnp.cbrt(t84)
  t88 = jnp.where(t84 <= p.zeta_threshold, t23, t86 * t84)
  t91 = r1 ** 2
  t92 = jnp.cbrt(r1)
  t93 = t92 ** 2
  t96 = t34 * s2 / t93 / t91
  t99 = jnp.exp(-t30 * t96 / 0.24e2)
  t105 = jnp.sqrt(s2)
  t107 = 0.1e1 / t92 / r1
  t111 = (t52 * t105 * t107 / 0.12e2) ** params.expo
  t112 = params.f * t111
  t120 = jnp.arcsinh(t64 * t51 * t105 * t107 / 0.12e2)
  t131 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t88 * t27 * (0.1e1 + ((params.d * t99 + params.c) * t29 * t96 / 0.24e2 - t112) / (0.1e1 + t52 * t105 * t107 * params.a * t120 / 0.12e2 + t112)))
  res = t80 + t131
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
  t12 = 0.1e1 + t11
  t14 = jnp.cbrt(p.zeta_threshold)
  t16 = jnp.cbrt(t12)
  t18 = jnp.where(t12 <= p.zeta_threshold, t14 * p.zeta_threshold, t16 * t12)
  t19 = jnp.cbrt(r0)
  t21 = jnp.cbrt(6)
  t23 = jnp.pi ** 2
  t24 = jnp.cbrt(t23)
  t25 = t24 ** 2
  t26 = 0.1e1 / t25
  t28 = jnp.cbrt(2)
  t29 = t28 ** 2
  t31 = r0 ** 2
  t32 = t19 ** 2
  t35 = s0 * t29 / t32 / t31
  t38 = jnp.exp(-params.alpha * t21 * t26 * t35 / 0.24e2)
  t45 = t21 ** 2
  t46 = 0.1e1 / t24
  t47 = t45 * t46
  t48 = jnp.sqrt(s0)
  t51 = 0.1e1 / t19 / r0
  t52 = t48 * t28 * t51
  t55 = (t47 * t52 / 0.12e2) ** params.expo
  t56 = params.f * t55
  t64 = jnp.arcsinh(params.b * t45 * t46 * t52 / 0.12e2)
  t76 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + ((params.d * t38 + params.c) * t21 * t26 * t35 / 0.24e2 - t56) / (0.1e1 + t47 * t48 * t28 * t51 * params.a * t64 / 0.12e2 + t56)))
  res = 0.2e1 * t76
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