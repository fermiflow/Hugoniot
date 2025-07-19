"""Generated from gga_k_dk.mpl."""

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
  t31 = jnp.cbrt(t7)
  t32 = t31 ** 2
  t33 = params.aa[0]
  t34 = params.aa[1]
  t36 = r0 ** 2
  t37 = jnp.cbrt(r0)
  t38 = t37 ** 2
  t40 = 0.1e1 / t38 / t36
  t42 = params.aa[2]
  t43 = s0 ** 2
  t45 = t36 ** 2
  t48 = 0.1e1 / t37 / t45 / r0
  t50 = params.aa[3]
  t51 = t43 * s0
  t53 = t45 ** 2
  t54 = 0.1e1 / t53
  t56 = params.aa[4]
  t57 = t43 ** 2
  t61 = 0.1e1 / t38 / t53 / t36
  t65 = params.bb[0]
  t66 = params.bb[1]
  t69 = params.bb[2]
  t72 = params.bb[3]
  t75 = params.bb[4]
  t83 = jnp.where(r0 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t29 * t32 * (t34 * s0 * t40 + t42 * t43 * t48 + t50 * t51 * t54 + t56 * t57 * t61 + t33) / (t66 * s0 * t40 + t69 * t43 * t48 + t72 * t51 * t54 + t75 * t57 * t61 + t65))
  t85 = jnp.where(t11, t16, -t18)
  t86 = jnp.where(t15, t12, t85)
  t87 = 0.1e1 + t86
  t89 = jnp.cbrt(t87)
  t90 = t89 ** 2
  t92 = jnp.where(t87 <= p.zeta_threshold, t25, t90 * t87)
  t95 = r1 ** 2
  t96 = jnp.cbrt(r1)
  t97 = t96 ** 2
  t99 = 0.1e1 / t97 / t95
  t101 = s2 ** 2
  t103 = t95 ** 2
  t106 = 0.1e1 / t96 / t103 / r1
  t108 = t101 * s2
  t110 = t103 ** 2
  t111 = 0.1e1 / t110
  t113 = t101 ** 2
  t117 = 0.1e1 / t97 / t110 / t95
  t134 = jnp.where(r1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t6 * t92 * t32 * (t34 * s2 * t99 + t42 * t101 * t106 + t50 * t108 * t111 + t56 * t113 * t117 + t33) / (t66 * s2 * t99 + t69 * t101 * t106 + t72 * t108 * t111 + t75 * t113 * t117 + t65))
  res = t83 + t134
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
  t23 = jnp.cbrt(r0)
  t24 = t23 ** 2
  t28 = jnp.cbrt(2)
  t29 = t28 ** 2
  t30 = r0 ** 2
  t33 = t29 / t24 / t30
  t36 = s0 ** 2
  t38 = t30 ** 2
  t42 = t28 / t23 / t38 / r0
  t46 = t36 * s0
  t48 = t38 ** 2
  t49 = 0.1e1 / t48
  t53 = t36 ** 2
  t58 = t29 / t24 / t48 / t30
  t84 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, 0.3e1 / 0.2e2 * t4 * t5 * jnp.pi * t21 * t24 * (params.aa[1] * s0 * t33 + 0.2e1 * params.aa[2] * t36 * t42 + 0.4e1 * params.aa[3] * t46 * t49 + 0.4e1 * params.aa[4] * t53 * t58 + params.aa[0]) / (params.bb[1] * s0 * t33 + 0.2e1 * params.bb[2] * t36 * t42 + 0.4e1 * params.bb[3] * t46 * t49 + 0.4e1 * params.bb[4] * t53 * t58 + params.bb[0]))
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