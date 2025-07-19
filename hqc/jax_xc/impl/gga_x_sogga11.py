"""Generated from gga_x_sogga11.mpl."""

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
  t29 = params.a[0]
  t30 = params.a[1]
  t31 = jnp.cbrt(6)
  t33 = jnp.pi ** 2
  t34 = jnp.cbrt(t33)
  t35 = t34 ** 2
  t37 = params.mu * t31 / t35
  t38 = 0.1e1 / params.kappa
  t40 = r0 ** 2
  t41 = jnp.cbrt(r0)
  t42 = t41 ** 2
  t47 = t37 * t38 * s0 / t42 / t40 / 0.24e2
  t50 = 0.1e1 - 0.1e1 / (0.1e1 + t47)
  t52 = params.a[2]
  t53 = t50 ** 2
  t55 = params.a[3]
  t58 = params.a[4]
  t59 = t53 ** 2
  t61 = params.a[5]
  t64 = params.b[0]
  t65 = params.b[1]
  t66 = jnp.exp(-t47)
  t67 = 0.1e1 - t66
  t69 = params.b[2]
  t70 = t67 ** 2
  t72 = params.b[3]
  t75 = params.b[4]
  t76 = t70 ** 2
  t78 = params.b[5]
  t81 = t55 * t53 * t50 + t61 * t59 * t50 + t72 * t70 * t67 + t78 * t76 * t67 + t30 * t50 + t52 * t53 + t58 * t59 + t65 * t67 + t69 * t70 + t75 * t76 + t29 + t64
  t85 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * t81)
  t87 = jnp.where(t10, t15, -t17)
  t88 = jnp.where(t14, t11, t87)
  t89 = 0.1e1 + t88
  t91 = jnp.cbrt(t89)
  t93 = jnp.where(t89 <= p.zeta_threshold, t23, t91 * t89)
  t96 = r1 ** 2
  t97 = jnp.cbrt(r1)
  t98 = t97 ** 2
  t103 = t37 * t38 * s2 / t98 / t96 / 0.24e2
  t106 = 0.1e1 - 0.1e1 / (0.1e1 + t103)
  t108 = t106 ** 2
  t112 = t108 ** 2
  t116 = jnp.exp(-t103)
  t117 = 0.1e1 - t116
  t119 = t117 ** 2
  t123 = t119 ** 2
  t127 = t55 * t108 * t106 + t61 * t112 * t106 + t72 * t119 * t117 + t78 * t123 * t117 + t30 * t106 + t52 * t108 + t58 * t112 + t65 * t117 + t69 * t119 + t75 * t123 + t29 + t64
  t131 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t93 * t27 * t127)
  res = t85 + t131
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
  t23 = jnp.cbrt(6)
  t25 = jnp.pi ** 2
  t26 = jnp.cbrt(t25)
  t27 = t26 ** 2
  t32 = jnp.cbrt(2)
  t33 = t32 ** 2
  t34 = r0 ** 2
  t35 = t19 ** 2
  t41 = params.mu * t23 / t27 / params.kappa * s0 * t33 / t35 / t34 / 0.24e2
  t44 = 0.1e1 - 0.1e1 / (0.1e1 + t41)
  t47 = t44 ** 2
  t53 = t47 ** 2
  t60 = jnp.exp(-t41)
  t61 = 0.1e1 - t60
  t64 = t61 ** 2
  t70 = t64 ** 2
  t75 = params.a[3] * t47 * t44 + params.a[5] * t53 * t44 + params.b[3] * t64 * t61 + params.b[5] * t70 * t61 + params.a[1] * t44 + params.a[2] * t47 + params.a[4] * t53 + params.b[1] * t61 + params.b[2] * t64 + params.b[4] * t70 + params.a[0] + params.b[0]
  t79 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * t75)
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