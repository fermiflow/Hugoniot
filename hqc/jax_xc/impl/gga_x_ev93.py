"""Generated from gga_x_ev93.mpl."""

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
  t29 = jnp.cbrt(6)
  t30 = params.a1 * t29
  t31 = jnp.pi ** 2
  t32 = jnp.cbrt(t31)
  t33 = t32 ** 2
  t34 = 0.1e1 / t33
  t36 = r0 ** 2
  t37 = jnp.cbrt(r0)
  t38 = t37 ** 2
  t41 = t34 * s0 / t38 / t36
  t44 = t29 ** 2
  t45 = params.a2 * t44
  t47 = 0.1e1 / t32 / t31
  t48 = s0 ** 2
  t50 = t36 ** 2
  t54 = t47 * t48 / t37 / t50 / r0
  t57 = t31 ** 2
  t58 = 0.1e1 / t57
  t59 = params.a3 * t58
  t61 = t50 ** 2
  t63 = t48 * s0 / t61
  t68 = params.b1 * t29
  t71 = params.b2 * t44
  t74 = params.b3 * t58
  t82 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * (0.1e1 + t30 * t41 / 0.24e2 + t45 * t54 / 0.576e3 + t59 * t63 / 0.2304e4) / (0.1e1 + t68 * t41 / 0.24e2 + t71 * t54 / 0.576e3 + t74 * t63 / 0.2304e4))
  t84 = jnp.where(t10, t15, -t17)
  t85 = jnp.where(t14, t11, t84)
  t86 = 0.1e1 + t85
  t88 = jnp.cbrt(t86)
  t90 = jnp.where(t86 <= p.zeta_threshold, t23, t88 * t86)
  t93 = r1 ** 2
  t94 = jnp.cbrt(r1)
  t95 = t94 ** 2
  t98 = t34 * s2 / t95 / t93
  t101 = s2 ** 2
  t103 = t93 ** 2
  t107 = t47 * t101 / t94 / t103 / r1
  t111 = t103 ** 2
  t113 = t101 * s2 / t111
  t129 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t90 * t28 * (0.1e1 + t30 * t98 / 0.24e2 + t45 * t107 / 0.576e3 + t59 * t113 / 0.2304e4) / (0.1e1 + t68 * t98 / 0.24e2 + t71 * t107 / 0.576e3 + t74 * t113 / 0.2304e4))
  res = t82 + t129
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
  t21 = jnp.cbrt(6)
  t23 = jnp.pi ** 2
  t24 = jnp.cbrt(t23)
  t25 = t24 ** 2
  t26 = 0.1e1 / t25
  t28 = jnp.cbrt(2)
  t29 = t28 ** 2
  t31 = r0 ** 2
  t32 = t20 ** 2
  t35 = s0 * t29 / t32 / t31
  t38 = t21 ** 2
  t41 = 0.1e1 / t24 / t23
  t43 = s0 ** 2
  t45 = t31 ** 2
  t49 = t43 * t28 / t20 / t45 / r0
  t52 = t23 ** 2
  t53 = 0.1e1 / t52
  t56 = t45 ** 2
  t58 = t43 * s0 / t56
  t79 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * (0.1e1 + params.a1 * t21 * t26 * t35 / 0.24e2 + params.a2 * t38 * t41 * t49 / 0.288e3 + params.a3 * t53 * t58 / 0.576e3) / (0.1e1 + params.b1 * t21 * t26 * t35 / 0.24e2 + params.b2 * t38 * t41 * t49 / 0.288e3 + params.b3 * t53 * t58 / 0.576e3))
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