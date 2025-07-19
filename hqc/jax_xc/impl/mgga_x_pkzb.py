"""Generated from mgga_x_pkzb.mpl."""

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
  t49 = t34 * tau0 / t37 / r0 / 0.4e1 - 0.9e1 / 0.2e2 - t41 / 0.288e3
  t50 = t49 ** 2
  t57 = t29 ** 2
  t60 = t57 / t31 / t30
  t61 = s0 ** 2
  t62 = t35 ** 2
  t76 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1804e1 - 0.646416 / (0.804 + 0.5e1 / 0.972e3 * t41 + 0.146e3 / 0.2025e4 * t50 - 0.73e2 / 0.972e4 * t49 * t29 * t33 * s0 * t39 + 0.22909234000912809658e-3 * t60 * t61 / t36 / t62 / r0)))
  t78 = jnp.where(t10, t15, -t17)
  t79 = jnp.where(t14, t11, t78)
  t80 = 0.1e1 + t79
  t82 = jnp.cbrt(t80)
  t84 = jnp.where(t80 <= p.zeta_threshold, t23, t82 * t80)
  t86 = r1 ** 2
  t87 = jnp.cbrt(r1)
  t88 = t87 ** 2
  t90 = 0.1e1 / t88 / t86
  t92 = t34 * s2 * t90
  t100 = t34 * tau1 / t88 / r1 / 0.4e1 - 0.9e1 / 0.2e2 - t92 / 0.288e3
  t101 = t100 ** 2
  t108 = s2 ** 2
  t109 = t86 ** 2
  t123 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t84 * t27 * (0.1804e1 - 0.646416 / (0.804 + 0.5e1 / 0.972e3 * t92 + 0.146e3 / 0.2025e4 * t101 - 0.73e2 / 0.972e4 * t100 * t29 * t33 * s2 * t90 + 0.22909234000912809658e-3 * t60 * t108 / t87 / t109 / r1)))
  res = t76 + t123
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
  t26 = t21 * t25
  t27 = jnp.cbrt(2)
  t28 = t27 ** 2
  t30 = r0 ** 2
  t31 = t19 ** 2
  t34 = s0 * t28 / t31 / t30
  t35 = t26 * t34
  t44 = t26 * tau0 * t28 / t31 / r0 / 0.4e1 - 0.9e1 / 0.2e2 - t35 / 0.288e3
  t45 = t44 ** 2
  t51 = t21 ** 2
  t55 = s0 ** 2
  t57 = t30 ** 2
  t71 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1804e1 - 0.646416 / (0.804 + 0.5e1 / 0.972e3 * t35 + 0.146e3 / 0.2025e4 * t45 - 0.73e2 / 0.972e4 * t44 * t21 * t25 * t34 + 0.45818468001825619316e-3 * t51 / t23 / t22 * t55 * t27 / t19 / t57 / r0)))
  res = 0.2e1 * t71
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