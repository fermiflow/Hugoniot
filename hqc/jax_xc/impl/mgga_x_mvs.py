"""Generated from mgga_x_mvs.mpl."""

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
  t28 = jnp.cbrt(t6)
  t29 = jnp.cbrt(r0)
  t30 = t29 ** 2
  t34 = r0 ** 2
  t39 = tau0 / t30 / r0 - s0 / t30 / t34 / 0.8e1
  t40 = jnp.cbrt(6)
  t42 = jnp.pi ** 2
  t43 = jnp.cbrt(t42)
  t44 = t43 ** 2
  t45 = 0.1e1 / t44
  t50 = t39 ** 2
  t52 = t40 ** 2
  t54 = 0.1e1 / t43 / t42
  t55 = t52 * t54
  t59 = (0.1e1 + 0.25e2 / 0.81e2 * params.e1 * t50 * t55) ** 2
  t60 = t50 ** 2
  t62 = t42 ** 2
  t65 = t40 / t44 / t62
  t69 = (t59 + 0.125e4 / 0.2187e4 * params.c1 * t60 * t65) ** (0.1e1 / 0.4e1)
  t74 = params.b * t52
  t75 = s0 ** 2
  t77 = t34 ** 2
  t85 = (0.1e1 + t74 * t54 * t75 / t29 / t77 / r0 / 0.576e3) ** (0.1e1 / 0.8e1)
  t90 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * (0.1e1 + params.k0 * (0.1e1 - 0.5e1 / 0.9e1 * t39 * t40 * t45) / t69) / t85)
  t92 = jnp.where(t10, t15, -t17)
  t93 = jnp.where(t14, t11, t92)
  t94 = 0.1e1 + t93
  t96 = jnp.cbrt(t94)
  t98 = jnp.where(t94 <= p.zeta_threshold, t23, t96 * t94)
  t100 = jnp.cbrt(r1)
  t101 = t100 ** 2
  t105 = r1 ** 2
  t110 = tau1 / t101 / r1 - s2 / t101 / t105 / 0.8e1
  t116 = t110 ** 2
  t121 = (0.1e1 + 0.25e2 / 0.81e2 * params.e1 * t116 * t55) ** 2
  t122 = t116 ** 2
  t127 = (t121 + 0.125e4 / 0.2187e4 * params.c1 * t122 * t65) ** (0.1e1 / 0.4e1)
  t132 = s2 ** 2
  t134 = t105 ** 2
  t142 = (0.1e1 + t74 * t54 * t132 / t100 / t134 / r1 / 0.576e3) ** (0.1e1 / 0.8e1)
  t147 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t98 * t28 * (0.1e1 + params.k0 * (0.1e1 - 0.5e1 / 0.9e1 * t110 * t40 * t45) / t127) / t142)
  res = t90 + t147
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
  t20 = jnp.cbrt(r0)
  t21 = jnp.cbrt(2)
  t22 = t21 ** 2
  t24 = t20 ** 2
  t29 = r0 ** 2
  t34 = tau0 * t22 / t24 / r0 - s0 * t22 / t24 / t29 / 0.8e1
  t35 = jnp.cbrt(6)
  t37 = jnp.pi ** 2
  t38 = jnp.cbrt(t37)
  t39 = t38 ** 2
  t45 = t34 ** 2
  t47 = t35 ** 2
  t49 = 0.1e1 / t38 / t37
  t54 = (0.1e1 + 0.25e2 / 0.81e2 * params.e1 * t45 * t47 * t49) ** 2
  t55 = t45 ** 2
  t57 = t37 ** 2
  t64 = (t54 + 0.125e4 / 0.2187e4 * params.c1 * t55 * t35 / t39 / t57) ** (0.1e1 / 0.4e1)
  t71 = s0 ** 2
  t73 = t29 ** 2
  t81 = (0.1e1 + params.b * t47 * t49 * t71 * t21 / t20 / t73 / r0 / 0.288e3) ** (0.1e1 / 0.8e1)
  t86 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * (0.1e1 + params.k0 * (0.1e1 - 0.5e1 / 0.9e1 * t34 * t35 / t39) / t64) / t81)
  res = 0.2e1 * t86
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