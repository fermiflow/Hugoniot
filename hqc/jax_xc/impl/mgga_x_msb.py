"""Generated from mgga_x_msb.mpl."""

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
  t50 = tau0 / t37 / r0
  t52 = t50 - t40 / 0.8e1
  t53 = t52 ** 2
  t54 = t29 ** 2
  t56 = 0.3e1 / 0.1e2 * t54 * t32
  t57 = t50 + t56
  t58 = t57 ** 2
  t62 = 0.1e1 - 0.4e1 * t53 / t58
  t63 = t62 ** 2
  t70 = t53 ** 2
  t73 = t58 ** 2
  t92 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + t47 + t63 * t62 / (0.1e1 + 0.8e1 * t53 * t52 / t58 / t57 + 0.64e2 * params.b * t70 * t53 / t73 / t58) * (params.kappa * (0.1e1 - params.kappa / (params.kappa + t42 + params.c)) - t47)))
  t94 = jnp.where(t10, t15, -t17)
  t95 = jnp.where(t14, t11, t94)
  t96 = 0.1e1 + t95
  t98 = jnp.cbrt(t96)
  t100 = jnp.where(t96 <= p.zeta_threshold, t23, t98 * t96)
  t102 = r1 ** 2
  t103 = jnp.cbrt(r1)
  t104 = t103 ** 2
  t107 = s2 / t104 / t102
  t109 = 0.5e1 / 0.972e3 * t34 * t107
  t114 = params.kappa * (0.1e1 - params.kappa / (params.kappa + t109))
  t117 = tau1 / t104 / r1
  t119 = t117 - t107 / 0.8e1
  t120 = t119 ** 2
  t121 = t117 + t56
  t122 = t121 ** 2
  t126 = 0.1e1 - 0.4e1 * t120 / t122
  t127 = t126 ** 2
  t134 = t120 ** 2
  t137 = t122 ** 2
  t156 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t100 * t27 * (0.1e1 + t114 + t127 * t126 / (0.1e1 + 0.8e1 * t120 * t119 / t122 / t121 + 0.64e2 * params.b * t134 * t120 / t137 / t122) * (params.kappa * (0.1e1 - params.kappa / (params.kappa + t109 + params.c)) - t114)))
  res = t92 + t156
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
  t45 = tau0 * t28 / t31 / r0
  t47 = t45 - t34 / 0.8e1
  t48 = t47 ** 2
  t49 = t21 ** 2
  t52 = t45 + 0.3e1 / 0.1e2 * t49 * t24
  t53 = t52 ** 2
  t57 = 0.1e1 - 0.4e1 * t48 / t53
  t58 = t57 ** 2
  t65 = t48 ** 2
  t68 = t53 ** 2
  t87 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + t41 + t58 * t57 / (0.1e1 + 0.8e1 * t48 * t47 / t53 / t52 + 0.64e2 * params.b * t65 * t48 / t68 / t53) * (params.kappa * (0.1e1 - params.kappa / (params.kappa + t36 + params.c)) - t41)))
  res = 0.2e1 * t87
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