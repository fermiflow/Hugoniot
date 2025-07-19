"""Generated from gga_c_lypr.mpl."""

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
  t2 = r0 + r1
  t3 = jnp.cbrt(t2)
  t4 = 0.1e1 / t3
  t6 = jax.lax.erfc(params.m1 * params.omega * t4)
  t7 = r0 - r1
  t8 = t7 ** 2
  t9 = t2 ** 2
  t12 = 0.1e1 - t8 / t9
  t16 = 0.1e1 / (params.d * t4 + 0.1e1)
  t20 = jax.lax.erfc(params.m2 * params.omega * t4)
  t23 = jnp.exp(-params.c * t4)
  t27 = t3 ** 2
  t30 = (s0 + 0.2e1 * s1 + s2) / t27 / t9
  t33 = (params.d * t16 + params.c) * t4
  t40 = jnp.cbrt(3)
  t41 = t40 ** 2
  t42 = jnp.pi ** 2
  t43 = jnp.cbrt(t42)
  t44 = t43 ** 2
  t47 = t7 / t2
  t48 = 0.1e1 + t47
  t49 = t48 <= p.zeta_threshold
  t50 = p.zeta_threshold ** 2
  t51 = jnp.cbrt(p.zeta_threshold)
  t52 = t51 ** 2
  t53 = t52 * t50
  t54 = t48 ** 2
  t55 = jnp.cbrt(t48)
  t56 = t55 ** 2
  t58 = jnp.where(t49, t53, t56 * t54)
  t59 = 0.1e1 - t47
  t60 = t59 <= p.zeta_threshold
  t61 = t59 ** 2
  t62 = jnp.cbrt(t59)
  t63 = t62 ** 2
  t65 = jnp.where(t60, t53, t63 * t61)
  t70 = jnp.cbrt(2)
  t71 = t70 * t12
  t74 = r0 ** 2
  t75 = jnp.cbrt(r0)
  t76 = t75 ** 2
  t78 = 0.1e1 / t76 / t74
  t79 = s0 * t78
  t80 = t79 * t58
  t81 = r1 ** 2
  t82 = jnp.cbrt(r1)
  t83 = t82 ** 2
  t85 = 0.1e1 / t83 / t81
  t86 = s2 * t85
  t87 = t86 * t65
  t88 = t80 + t87
  t94 = t52 * t50 * p.zeta_threshold
  t97 = jnp.where(t49, t94, t56 * t54 * t48)
  t101 = jnp.where(t60, t94, t63 * t61 * t59)
  t109 = jnp.where(t49, t50, t54)
  t114 = jnp.where(t60, t50, t61)
  t126 = jnp.sqrt(jnp.pi)
  t131 = params.m2 ** 2
  t132 = params.omega ** 2
  t136 = jnp.exp(-t131 * t132 / t27)
  t142 = t47 / 0.6e1
  res = params.a * (-t6 * t12 * t16 + t20 * params.b * t23 * t16 * (-t30 * (t12 * (0.47e2 - 0.7e1 * t33) / 0.72e2 - 0.2e1 / 0.3e1) - 0.3e1 / 0.2e2 * t41 * t44 * t12 * (t58 + t65) + t71 * (0.5e1 / 0.2e1 - t33 / 0.18e2) * t88 / 0.32e2 + t71 * (t33 - 0.11e2) * (t86 * t101 + t79 * t97) / 0.576e3 - t70 * (0.2e1 / 0.3e1 * t80 + 0.2e1 / 0.3e1 * t87 - t109 * s2 * t85 * t65 / 0.4e1 - t114 * s0 * t78 * t58 / 0.4e1) / 0.8e1) + params.b * t23 * t16 / t126 * params.m2 * params.omega * t136 * t4 * t12 * (0.7e1 / 0.6e1 * t30 - 0.7e1 / 0.48e2 * t70 * t88 + (0.7e1 / 0.6e1 + t142) * s0 * t78 * t70 * t58 / 0.8e1 + (0.7e1 / 0.6e1 - t142) * s2 * t85 * t70 * t65 / 0.8e1) / 0.6e1)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t2 = jnp.cbrt(r0)
  t3 = 0.1e1 / t2
  t5 = jax.lax.erfc(params.m1 * params.omega * t3)
  t8 = 0.1e1 / (params.d * t3 + 0.1e1)
  t10 = params.m2 * params.omega
  t12 = jax.lax.erfc(t10 * t3)
  t15 = jnp.exp(-params.c * t3)
  t17 = r0 ** 2
  t18 = t2 ** 2
  t20 = 0.1e1 / t18 / t17
  t24 = (params.d * t8 + params.c) * t3
  t28 = jnp.cbrt(3)
  t29 = t28 ** 2
  t30 = jnp.pi ** 2
  t31 = jnp.cbrt(t30)
  t32 = t31 ** 2
  t34 = 0.1e1 <= p.zeta_threshold
  t35 = p.zeta_threshold ** 2
  t36 = jnp.cbrt(p.zeta_threshold)
  t37 = t36 ** 2
  t39 = jnp.where(t34, t37 * t35, 1)
  t45 = t20 * t39
  t52 = jnp.where(t34, t37 * t35 * p.zeta_threshold, 1)
  t56 = jnp.cbrt(2)
  t57 = t56 ** 2
  t61 = jnp.where(t34, t35, 1)
  t74 = jnp.sqrt(jnp.pi)
  t78 = params.m2 ** 2
  t79 = params.omega ** 2
  t83 = jnp.exp(-t78 * t79 / t18)
  res = params.a * (-t5 * t8 + t12 * params.b * t15 * t8 * (-s0 * t20 * (-0.1e1 / 0.72e2 - 0.7e1 / 0.72e2 * t24) - 0.3e1 / 0.1e2 * t29 * t32 * t39 + (0.5e1 / 0.2e1 - t24 / 0.18e2) * s0 * t45 / 0.8e1 + (t24 - 0.11e2) * s0 * t20 * t52 / 0.144e3 - t56 * (0.4e1 / 0.3e1 * s0 * t57 * t45 - t61 * s0 * t57 * t20 * t39 / 0.2e1) / 0.8e1) + 0.7e1 / 0.36e2 * params.b * t15 * t8 / t74 * t10 * t83 / t17 / r0 * s0)
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