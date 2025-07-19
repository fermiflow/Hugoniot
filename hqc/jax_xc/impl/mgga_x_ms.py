"""Generated from mgga_x_ms.mpl."""

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
  t30 = jnp.pi ** 2
  t31 = jnp.cbrt(t30)
  t32 = t31 ** 2
  t34 = t29 / t32
  t35 = r0 ** 2
  t36 = jnp.cbrt(r0)
  t37 = t36 ** 2
  t40 = s0 / t37 / t35
  t42 = 0.5e1 / 0.972e3 * t34 * t40
  t47 = params.kappa * (0.1e1 - params.kappa / (params.kappa + t42))
  t52 = tau0 / t37 / r0 - t40 / 0.8e1
  t53 = t52 ** 2
  t54 = t29 ** 2
  t57 = 0.1e1 / t31 / t30
  t60 = 0.1e1 - 0.25e2 / 0.81e2 * t53 * t54 * t57
  t61 = t60 ** 2
  t64 = t30 ** 2
  t65 = 0.1e1 / t64
  t68 = t53 ** 2
  t71 = t64 ** 2
  t72 = 0.1e1 / t71
  t89 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + t47 + t61 * t60 / (0.1e1 + 0.25e3 / 0.243e3 * t53 * t52 * t65 + 0.625e5 / 0.59049e5 * params.b * t68 * t53 * t72) * (params.kappa * (0.1e1 - params.kappa / (params.kappa + t42 + params.c)) - t47)))
  t91 = jnp.where(t10, t15, -t17)
  t92 = jnp.where(t14, t11, t91)
  t93 = 0.1e1 + t92
  t95 = jnp.cbrt(t93)
  t97 = jnp.where(t93 <= p.zeta_threshold, t23, t95 * t93)
  t99 = r1 ** 2
  t100 = jnp.cbrt(r1)
  t101 = t100 ** 2
  t104 = s2 / t101 / t99
  t106 = 0.5e1 / 0.972e3 * t34 * t104
  t111 = params.kappa * (0.1e1 - params.kappa / (params.kappa + t106))
  t116 = tau1 / t101 / r1 - t104 / 0.8e1
  t117 = t116 ** 2
  t121 = 0.1e1 - 0.25e2 / 0.81e2 * t117 * t54 * t57
  t122 = t121 ** 2
  t127 = t117 ** 2
  t146 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t97 * t27 * (0.1e1 + t111 + t122 * t121 / (0.1e1 + 0.25e3 / 0.243e3 * t117 * t116 * t65 + 0.625e5 / 0.59049e5 * params.b * t127 * t117 * t72) * (params.kappa * (0.1e1 - params.kappa / (params.kappa + t106 + params.c)) - t111)))
  res = t89 + t146
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
  t22 = jnp.pi ** 2
  t23 = jnp.cbrt(t22)
  t24 = t23 ** 2
  t27 = jnp.cbrt(2)
  t28 = t27 ** 2
  t30 = r0 ** 2
  t31 = t19 ** 2
  t34 = s0 * t28 / t31 / t30
  t36 = 0.5e1 / 0.972e3 * t21 / t24 * t34
  t41 = params.kappa * (0.1e1 - params.kappa / (params.kappa + t36))
  t47 = tau0 * t28 / t31 / r0 - t34 / 0.8e1
  t48 = t47 ** 2
  t49 = t21 ** 2
  t55 = 0.1e1 - 0.25e2 / 0.81e2 * t48 * t49 / t23 / t22
  t56 = t55 ** 2
  t59 = t22 ** 2
  t63 = t48 ** 2
  t66 = t59 ** 2
  t84 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + t41 + t56 * t55 / (0.1e1 + 0.25e3 / 0.243e3 * t48 * t47 / t59 + 0.625e5 / 0.59049e5 * params.b * t63 * t48 / t66) * (params.kappa * (0.1e1 - params.kappa / (params.kappa + t36 + params.c)) - t41)))
  res = 0.2e1 * t84
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