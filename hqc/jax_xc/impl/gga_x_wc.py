"""Generated from gga_x_wc.mpl."""

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
  t41 = t34 * t40
  t44 = jnp.exp(-t41 / 0.24e2)
  t48 = t29 ** 2
  t51 = t48 / t31 / t30
  t52 = s0 ** 2
  t53 = t35 ** 2
  t61 = jnp.log(0.1e1 + 0.13780328706878157639e-4 * t51 * t52 / t36 / t53 / r0)
  t69 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1804e1 - 0.646416 / (0.804 + 0.5e1 / 0.972e3 * t41 + 0.4002424276710846245e-2 * t34 * t40 * t44 + t61)))
  t71 = jnp.where(t10, t15, -t17)
  t72 = jnp.where(t14, t11, t71)
  t73 = 0.1e1 + t72
  t75 = jnp.cbrt(t73)
  t77 = jnp.where(t73 <= p.zeta_threshold, t23, t75 * t73)
  t79 = r1 ** 2
  t80 = jnp.cbrt(r1)
  t81 = t80 ** 2
  t84 = s2 / t81 / t79
  t85 = t34 * t84
  t88 = jnp.exp(-t85 / 0.24e2)
  t92 = s2 ** 2
  t93 = t79 ** 2
  t101 = jnp.log(0.1e1 + 0.13780328706878157639e-4 * t51 * t92 / t80 / t93 / r1)
  t109 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t77 * t27 * (0.1804e1 - 0.646416 / (0.804 + 0.5e1 / 0.972e3 * t85 + 0.4002424276710846245e-2 * t34 * t84 * t88 + t101)))
  res = t69 + t109
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
  t26 = t21 / t24
  t27 = jnp.cbrt(2)
  t28 = t27 ** 2
  t30 = r0 ** 2
  t31 = t19 ** 2
  t33 = 0.1e1 / t31 / t30
  t35 = t26 * s0 * t28 * t33
  t40 = jnp.exp(-t35 / 0.24e2)
  t44 = t21 ** 2
  t48 = s0 ** 2
  t50 = t30 ** 2
  t58 = jnp.log(0.1e1 + 0.27560657413756315278e-4 * t44 / t23 / t22 * t48 * t27 / t19 / t50 / r0)
  t66 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1804e1 - 0.646416 / (0.804 + 0.5e1 / 0.972e3 * t35 + 0.4002424276710846245e-2 * t26 * s0 * t28 * t33 * t40 + t58)))
  res = 0.2e1 * t66
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