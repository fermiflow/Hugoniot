"""Generated from gga_x_ssb_sw.mpl."""

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
  t35 = params.B * t29 * t34
  t36 = r0 ** 2
  t37 = jnp.cbrt(r0)
  t38 = t37 ** 2
  t40 = 0.1e1 / t38 / t36
  t41 = s0 * t40
  t42 = params.C * t29
  t53 = params.D * t29 * t34
  t54 = t29 ** 2
  t55 = params.E * t54
  t57 = 0.1e1 / t32 / t31
  t58 = s0 ** 2
  t60 = t36 ** 2
  t76 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (params.A + t35 * t41 / (0.1e1 + t42 * t34 * s0 * t40 / 0.24e2) / 0.24e2 - t53 * t41 / (0.1e1 + t55 * t57 * t58 / t37 / t60 / r0 / 0.576e3) / 0.24e2))
  t78 = jnp.where(t10, t15, -t17)
  t79 = jnp.where(t14, t11, t78)
  t80 = 0.1e1 + t79
  t82 = jnp.cbrt(t80)
  t84 = jnp.where(t80 <= p.zeta_threshold, t23, t82 * t80)
  t86 = r1 ** 2
  t87 = jnp.cbrt(r1)
  t88 = t87 ** 2
  t90 = 0.1e1 / t88 / t86
  t91 = s2 * t90
  t101 = s2 ** 2
  t103 = t86 ** 2
  t119 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t84 * t27 * (params.A + t35 * t91 / (0.1e1 + t42 * t34 * s2 * t90 / 0.24e2) / 0.24e2 - t53 * t91 / (0.1e1 + t55 * t57 * t101 / t87 / t103 / r1 / 0.576e3) / 0.24e2))
  res = t76 + t119
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
  t48 = t21 ** 2
  t53 = s0 ** 2
  t55 = t31 ** 2
  t72 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (params.A + params.B * t21 * t26 * t30 * t34 / (0.1e1 + params.C * t21 * t26 * t30 * t34 / 0.24e2) / 0.24e2 - params.D * t21 * t26 * t30 * t34 / (0.1e1 + params.E * t48 / t24 / t23 * t53 * t28 / t19 / t55 / r0 / 0.288e3) / 0.24e2))
  res = 0.2e1 * t72
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