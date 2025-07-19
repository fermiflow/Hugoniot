"""Generated from mgga_x_mvsb.mpl."""

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
  t33 = tau0 / t30 / r0
  t34 = r0 ** 2
  t39 = t33 - s0 / t30 / t34 / 0.8e1
  t40 = jnp.cbrt(6)
  t41 = t40 ** 2
  t42 = jnp.pi ** 2
  t43 = jnp.cbrt(t42)
  t44 = t43 ** 2
  t46 = 0.3e1 / 0.1e2 * t41 * t44
  t47 = t33 - t46
  t52 = t39 ** 2
  t54 = t47 ** 2
  t58 = (0.1e1 + params.e1 * t52 / t54) ** 2
  t59 = t52 ** 2
  t61 = t54 ** 2
  t65 = (t58 + params.c1 * t59 / t61) ** (0.1e1 / 0.4e1)
  t70 = params.b * t41
  t72 = 0.1e1 / t43 / t42
  t73 = s0 ** 2
  t75 = t34 ** 2
  t83 = (0.1e1 + t70 * t72 * t73 / t29 / t75 / r0 / 0.576e3) ** (0.1e1 / 0.8e1)
  t88 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * (0.1e1 + params.k0 * (0.1e1 - t39 / t47) / t65) / t83)
  t90 = jnp.where(t10, t15, -t17)
  t91 = jnp.where(t14, t11, t90)
  t92 = 0.1e1 + t91
  t94 = jnp.cbrt(t92)
  t96 = jnp.where(t92 <= p.zeta_threshold, t23, t94 * t92)
  t98 = jnp.cbrt(r1)
  t99 = t98 ** 2
  t102 = tau1 / t99 / r1
  t103 = r1 ** 2
  t108 = t102 - s2 / t99 / t103 / 0.8e1
  t109 = t102 - t46
  t114 = t108 ** 2
  t116 = t109 ** 2
  t120 = (0.1e1 + params.e1 * t114 / t116) ** 2
  t121 = t114 ** 2
  t123 = t116 ** 2
  t127 = (t120 + params.c1 * t121 / t123) ** (0.1e1 / 0.4e1)
  t132 = s2 ** 2
  t134 = t103 ** 2
  t142 = (0.1e1 + t70 * t72 * t132 / t98 / t134 / r1 / 0.576e3) ** (0.1e1 / 0.8e1)
  t147 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t96 * t28 * (0.1e1 + params.k0 * (0.1e1 - t108 / t109) / t127) / t142)
  res = t88 + t147
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
  t27 = tau0 * t22 / t24 / r0
  t29 = r0 ** 2
  t34 = t27 - s0 * t22 / t24 / t29 / 0.8e1
  t35 = jnp.cbrt(6)
  t36 = t35 ** 2
  t37 = jnp.pi ** 2
  t38 = jnp.cbrt(t37)
  t39 = t38 ** 2
  t42 = t27 - 0.3e1 / 0.1e2 * t36 * t39
  t47 = t34 ** 2
  t49 = t42 ** 2
  t53 = (0.1e1 + params.e1 * t47 / t49) ** 2
  t54 = t47 ** 2
  t56 = t49 ** 2
  t60 = (t53 + params.c1 * t54 / t56) ** (0.1e1 / 0.4e1)
  t69 = s0 ** 2
  t71 = t29 ** 2
  t79 = (0.1e1 + params.b * t36 / t38 / t37 * t69 * t21 / t20 / t71 / r0 / 0.288e3) ** (0.1e1 / 0.8e1)
  t84 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * (0.1e1 + params.k0 * (0.1e1 - t34 / t42) / t60) / t79)
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