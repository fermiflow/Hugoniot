"""Generated from gga_x_vmt84.mpl."""

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
  t30 = params.mu * t29
  t31 = jnp.pi ** 2
  t32 = jnp.cbrt(t31)
  t33 = t32 ** 2
  t34 = 0.1e1 / t33
  t35 = t30 * t34
  t36 = r0 ** 2
  t37 = jnp.cbrt(r0)
  t38 = t37 ** 2
  t39 = t38 * t36
  t40 = 0.1e1 / t39
  t42 = params.alpha * t29
  t44 = t34 * s0 * t40
  t47 = jnp.exp(-t42 * t44 / 0.24e2)
  t56 = t29 ** 2
  t57 = params.alpha * t56
  t59 = 0.1e1 / t32 / t31
  t60 = s0 ** 2
  t62 = t36 ** 2
  t69 = jnp.exp(-t57 * t59 * t60 / t37 / t62 / r0 / 0.576e3)
  t81 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (t35 * s0 * t40 * t47 / (0.1e1 + t30 * t44 / 0.24e2) / 0.24e2 + 0.4e1 * (0.1e1 - t69) * t56 * t33 / s0 * t39 + t69))
  t83 = jnp.where(t10, t15, -t17)
  t84 = jnp.where(t14, t11, t83)
  t85 = 0.1e1 + t84
  t87 = jnp.cbrt(t85)
  t89 = jnp.where(t85 <= p.zeta_threshold, t23, t87 * t85)
  t91 = r1 ** 2
  t92 = jnp.cbrt(r1)
  t93 = t92 ** 2
  t94 = t93 * t91
  t95 = 0.1e1 / t94
  t98 = t34 * s2 * t95
  t101 = jnp.exp(-t42 * t98 / 0.24e2)
  t110 = s2 ** 2
  t112 = t91 ** 2
  t119 = jnp.exp(-t57 * t59 * t110 / t92 / t112 / r1 / 0.576e3)
  t131 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t89 * t27 * (t35 * s2 * t95 * t101 / (0.1e1 + t30 * t98 / 0.24e2) / 0.24e2 + 0.4e1 * (0.1e1 - t119) * t56 * t33 / s2 * t94 + t119))
  res = t81 + t131
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
  t22 = params.mu * t21
  t23 = jnp.pi ** 2
  t24 = jnp.cbrt(t23)
  t25 = t24 ** 2
  t26 = 0.1e1 / t25
  t29 = jnp.cbrt(2)
  t30 = t29 ** 2
  t31 = r0 ** 2
  t32 = t19 ** 2
  t33 = t32 * t31
  t34 = 0.1e1 / t33
  t39 = s0 * t30 * t34
  t42 = jnp.exp(-params.alpha * t21 * t26 * t39 / 0.24e2)
  t52 = t21 ** 2
  t57 = s0 ** 2
  t59 = t31 ** 2
  t66 = jnp.exp(-params.alpha * t52 / t24 / t23 * t57 * t29 / t19 / t59 / r0 / 0.288e3)
  t79 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (t22 * t26 * s0 * t30 * t34 * t42 / (0.1e1 + t22 * t26 * t39 / 0.24e2) / 0.24e2 + 0.2e1 * (0.1e1 - t66) * t52 * t25 / s0 * t29 * t33 + t66))
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