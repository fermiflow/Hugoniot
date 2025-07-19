"""Generated from gga_x_q1d.mpl."""

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
  t33 = 0.1e1 / t32
  t34 = t29 * t33
  t35 = r0 ** 2
  t36 = jnp.cbrt(r0)
  t37 = t36 ** 2
  t39 = 0.1e1 / t37 / t35
  t41 = t34 * s0 * t39
  t45 = 0.646416 / (0.804 + 0.5e1 / 0.972e3 * t41)
  t47 = t29 ** 2
  t50 = t47 / t31 / t30
  t51 = s0 ** 2
  t52 = t35 ** 2
  t58 = t50 * t51 / t36 / t52 / r0 / 0.576e3
  t60 = t30 ** 2
  t61 = 0.1e1 / t60
  t64 = t52 ** 2
  t83 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1804e1 - t45 + (t41 / 0.24e2 + t58) / (0.1e1 + t58 + t61 * t51 * s0 / t64 / 0.2304e4) * (-(0.1804e1 - t45) * t29 * t33 * s0 * t39 / 0.24e2 + 0.6525e-1)))
  t85 = jnp.where(t10, t15, -t17)
  t86 = jnp.where(t14, t11, t85)
  t87 = 0.1e1 + t86
  t89 = jnp.cbrt(t87)
  t91 = jnp.where(t87 <= p.zeta_threshold, t23, t89 * t87)
  t93 = r1 ** 2
  t94 = jnp.cbrt(r1)
  t95 = t94 ** 2
  t97 = 0.1e1 / t95 / t93
  t99 = t34 * s2 * t97
  t103 = 0.646416 / (0.804 + 0.5e1 / 0.972e3 * t99)
  t105 = s2 ** 2
  t106 = t93 ** 2
  t112 = t50 * t105 / t94 / t106 / r1 / 0.576e3
  t116 = t106 ** 2
  t135 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t91 * t27 * (0.1804e1 - t103 + (t99 / 0.24e2 + t112) / (0.1e1 + t112 + t61 * t105 * s2 / t116 / 0.2304e4) * (-(0.1804e1 - t103) * t29 * t33 * s2 * t97 / 0.24e2 + 0.6525e-1)))
  res = t83 + t135
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
  t25 = 0.1e1 / t24
  t27 = jnp.cbrt(2)
  t28 = t27 ** 2
  t30 = r0 ** 2
  t31 = t19 ** 2
  t34 = s0 * t28 / t31 / t30
  t35 = t21 * t25 * t34
  t39 = 0.646416 / (0.804 + 0.5e1 / 0.972e3 * t35)
  t41 = t21 ** 2
  t45 = s0 ** 2
  t47 = t30 ** 2
  t53 = t41 / t23 / t22 * t45 * t27 / t19 / t47 / r0 / 0.288e3
  t55 = t22 ** 2
  t59 = t47 ** 2
  t77 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1804e1 - t39 + (t35 / 0.24e2 + t53) / (0.1e1 + t53 + 0.1e1 / t55 * t45 * s0 / t59 / 0.576e3) * (-(0.1804e1 - t39) * t21 * t25 * t34 / 0.24e2 + 0.6525e-1)))
  res = 0.2e1 * t77
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