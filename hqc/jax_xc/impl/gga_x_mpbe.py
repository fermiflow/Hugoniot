"""Generated from gga_x_mpbe.mpl."""

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
  t31 = jnp.pi ** 2
  t32 = jnp.cbrt(t31)
  t33 = t32 ** 2
  t34 = 0.1e1 / t33
  t35 = params.c1 * t29 * t34
  t36 = r0 ** 2
  t37 = jnp.cbrt(r0)
  t38 = t37 ** 2
  t40 = 0.1e1 / t38 / t36
  t42 = params.a * t29
  t47 = 0.1e1 + t42 * t34 * s0 * t40 / 0.24e2
  t52 = t29 ** 2
  t56 = params.c2 * t52 / t32 / t31
  t57 = s0 ** 2
  t58 = t36 ** 2
  t63 = t47 ** 2
  t68 = t31 ** 2
  t70 = params.c3 / t68
  t72 = t58 ** 2
  t84 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + t35 * s0 * t40 / t47 / 0.24e2 + t56 * t57 / t37 / t58 / r0 / t63 / 0.576e3 + t70 * t57 * s0 / t72 / t63 / t47 / 0.2304e4))
  t86 = jnp.where(t10, t15, -t17)
  t87 = jnp.where(t14, t11, t86)
  t88 = 0.1e1 + t87
  t90 = jnp.cbrt(t88)
  t92 = jnp.where(t88 <= p.zeta_threshold, t23, t90 * t88)
  t94 = r1 ** 2
  t95 = jnp.cbrt(r1)
  t96 = t95 ** 2
  t98 = 0.1e1 / t96 / t94
  t104 = 0.1e1 + t42 * t34 * s2 * t98 / 0.24e2
  t109 = s2 ** 2
  t110 = t94 ** 2
  t115 = t104 ** 2
  t121 = t110 ** 2
  t133 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t92 * t27 * (0.1e1 + t35 * s2 * t98 / t104 / 0.24e2 + t56 * t109 / t95 / t110 / r1 / t115 / 0.576e3 + t70 * t109 * s2 / t121 / t115 / t104 / 0.2304e4))
  res = t84 + t133
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
  t30 = s0 * t29
  t31 = r0 ** 2
  t32 = t19 ** 2
  t34 = 0.1e1 / t32 / t31
  t40 = 0.1e1 + params.a * t21 * t26 * t30 * t34 / 0.24e2
  t46 = t21 ** 2
  t51 = s0 ** 2
  t53 = t31 ** 2
  t57 = t40 ** 2
  t63 = t23 ** 2
  t67 = t53 ** 2
  t79 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + params.c1 * t21 * t26 * t30 * t34 / t40 / 0.24e2 + params.c2 * t46 / t24 / t23 * t51 * t28 / t19 / t53 / r0 / t57 / 0.288e3 + params.c3 / t63 * t51 * s0 / t67 / t57 / t40 / 0.576e3))
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