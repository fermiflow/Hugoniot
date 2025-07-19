"""Generated from gga_x_ncap.mpl."""

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
  t30 = t29 ** 2
  t31 = jnp.pi ** 2
  t32 = jnp.cbrt(t31)
  t33 = 0.1e1 / t32
  t34 = t30 * t33
  t35 = jnp.sqrt(s0)
  t36 = jnp.cbrt(r0)
  t38 = 0.1e1 / t36 / r0
  t39 = t35 * t38
  t41 = t34 * t39 / 0.12e2
  t42 = jnp.tanh(t41)
  t44 = jnp.arcsinh(t41)
  t47 = (0.1e1 - params.zeta) * t30 * t33
  t49 = jnp.log(0.1e1 + t41)
  t52 = params.zeta * t30
  t71 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + params.mu * t42 * t44 * (0.1e1 + params.alpha * (t52 * t33 * t35 * t38 / 0.12e2 + t47 * t39 * t49 / 0.12e2)) / (params.beta * t42 * t44 + 0.1e1)))
  t73 = jnp.where(t10, t15, -t17)
  t74 = jnp.where(t14, t11, t73)
  t75 = 0.1e1 + t74
  t77 = jnp.cbrt(t75)
  t79 = jnp.where(t75 <= p.zeta_threshold, t23, t77 * t75)
  t81 = jnp.sqrt(s2)
  t82 = jnp.cbrt(r1)
  t84 = 0.1e1 / t82 / r1
  t85 = t81 * t84
  t87 = t34 * t85 / 0.12e2
  t88 = jnp.tanh(t87)
  t90 = jnp.arcsinh(t87)
  t92 = jnp.log(0.1e1 + t87)
  t113 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t79 * t27 * (0.1e1 + params.mu * t88 * t90 * (0.1e1 + params.alpha * (t52 * t33 * t81 * t84 / 0.12e2 + t47 * t85 * t92 / 0.12e2)) / (params.beta * t88 * t90 + 0.1e1)))
  res = t71 + t113
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
  t22 = t21 ** 2
  t23 = jnp.pi ** 2
  t24 = jnp.cbrt(t23)
  t25 = 0.1e1 / t24
  t27 = jnp.sqrt(s0)
  t28 = jnp.cbrt(2)
  t29 = t27 * t28
  t31 = 0.1e1 / t19 / r0
  t32 = t29 * t31
  t34 = t22 * t25 * t32 / 0.12e2
  t35 = jnp.tanh(t34)
  t37 = jnp.arcsinh(t34)
  t42 = jnp.log(0.1e1 + t34)
  t64 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + params.mu * t35 * t37 * (0.1e1 + params.alpha * ((0.1e1 - params.zeta) * t22 * t25 * t29 * t31 * t42 / 0.12e2 + params.zeta * t22 * t25 * t32 / 0.12e2)) / (params.beta * t35 * t37 + 0.1e1)))
  res = 0.2e1 * t64
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