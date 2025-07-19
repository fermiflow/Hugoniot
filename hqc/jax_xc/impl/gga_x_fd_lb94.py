"""Generated from gga_x_fd_lb94.mpl."""

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
  t34 = t30 / t32
  t35 = jnp.sqrt(s0)
  t36 = jnp.cbrt(r0)
  t39 = t35 / t36 / r0
  t41 = t34 * t39 / 0.12e2
  t42 = fd_int0(t41)
  t43 = jnp.log(t41)
  t45 = fd_int1(t41)
  t54 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 - t34 * t39 * (t42 * t43 - t45) / 0.12e2))
  t56 = jnp.where(t10, t15, -t17)
  t57 = jnp.where(t14, t11, t56)
  t58 = 0.1e1 + t57
  t60 = jnp.cbrt(t58)
  t62 = jnp.where(t58 <= p.zeta_threshold, t23, t60 * t58)
  t64 = jnp.sqrt(s2)
  t65 = jnp.cbrt(r1)
  t68 = t64 / t65 / r1
  t70 = t34 * t68 / 0.12e2
  t71 = fd_int0(t70)
  t72 = jnp.log(t70)
  t74 = fd_int1(t70)
  t83 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t62 * t27 * (0.1e1 - t34 * t68 * (t71 * t72 - t74) / 0.12e2))
  res = t54 + t83
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
  t26 = t22 / t24
  t27 = jnp.sqrt(s0)
  t29 = jnp.cbrt(2)
  t31 = 0.1e1 / t19 / r0
  t36 = t26 * t27 * t29 * t31 / 0.12e2
  t37 = fd_int0(t36)
  t38 = jnp.log(t36)
  t40 = fd_int1(t36)
  t49 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 - t26 * t27 * t29 * t31 * (t37 * t38 - t40) / 0.12e2))
  res = 0.2e1 * t49
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