"""Generated from lda_c_w20.mpl."""

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
  t1 = jnp.log(0.2e1)
  t2 = 0.1e1 - t1
  t3 = jnp.pi ** 2
  t4 = 0.1e1 / t3
  t5 = t2 * t4
  t6 = t1 / 0.6e1
  t7 = 0.90154267736969571405 * t4
  t9 = 0.1e1 / t2
  t13 = jnp.exp(-0.2e1 * (-0.711e-1 + t6 - t7) * t9 * t3)
  t14 = jnp.cbrt(3)
  t15 = t14 ** 2
  t17 = jnp.cbrt(0.1e1 / jnp.pi)
  t18 = t17 ** 2
  t20 = jnp.cbrt(4)
  t21 = r0 + r1
  t22 = jnp.cbrt(t21)
  t23 = t22 ** 2
  t28 = jnp.exp(-t15 * t18 * t20 / t23 / 0.4e5)
  t29 = 0.1e1 - t28
  t30 = jnp.cbrt(jnp.pi)
  t31 = t30 ** 2
  t33 = jnp.cbrt(9)
  t34 = 0.1e1 / t31 * t33
  t35 = t20 ** 2
  t41 = t13 / 0.2e1
  t47 = 0.1e1 / t17
  t49 = t47 * t20 * t22
  t53 = jnp.sqrt(0.4e1)
  t55 = t14 * t17
  t56 = 0.1e1 / t22
  t58 = t55 * t35 * t56
  t59 = jnp.sqrt(t58)
  t63 = t29 * t9 * t3 * t53 / t59 / t58
  t65 = t33 ** 2
  t66 = t65 * t20
  t67 = t31 * t3
  t78 = 0.1e1 / t18 * t35 * t23
  t82 = jnp.log(0.1e1 + (t13 - 0.2e1 * t29 * ((-0.9 + 0.3e1 / 0.16e2 * t34 * t35) * t9 * t3 + t41)) * t15 * t49 / 0.3e1 - 0.12e2 * t63 + (t13 - 0.2e1 * t29 * (-0.3e1 / 0.4e2 * t66 * t67 * t9 + t41)) * t14 * t78 / 0.3e1)
  t84 = t5 * t82 / 0.2e1
  t86 = t56 * t28
  t87 = 4 ** (0.1e1 / 0.4e1)
  t88 = t87 ** 2
  t90 = t58 ** (0.1e1 / 0.4e1)
  t95 = 0.1e1 / (t28 + 0.5e1 / 0.8e1 * t88 * t87 * t90 * t58)
  t98 = 0.1e1 / t30 / t3 / jnp.pi
  t100 = 0.12e2 * t1
  t108 = jnp.log(0.1e1 + t15 * t47 * t20 * t22 / 0.3e1)
  t116 = t55 * t35 * t86 * t95 * (-t66 * t98 * (0.7e1 / 0.6e1 * t3 - t100 - 0.1e1) * t108 / 0.36e2 - 0.1e-1) / 0.4e1
  t121 = jnp.exp(-0.4e1 * (-0.49917e-1 + t6 - t7) * t9 * t3)
  t122 = jnp.cbrt(2)
  t130 = t121 / 0.2e1
  t139 = t122 ** 2
  t152 = jnp.log(0.1e1 + (t121 - 0.2e1 * t29 * (0.2e1 * (-0.9 + 0.3e1 / 0.16e2 * t34 * t35 * t122) * t9 * t3 + t130)) * t15 * t49 / 0.3e1 - 0.24e2 * t63 + (t121 - 0.2e1 * t29 * (-0.3e1 / 0.2e2 * t66 * t67 * t139 * t9 + t130)) * t14 * t78 / 0.3e1)
  t168 = (r0 - r1) / t21
  t169 = 0.1e1 + t168
  t171 = jnp.cbrt(p.zeta_threshold)
  t172 = t171 * p.zeta_threshold
  t173 = jnp.cbrt(t169)
  t175 = jnp.where(t169 <= p.zeta_threshold, t172, t173 * t169)
  t176 = 0.1e1 - t168
  t178 = jnp.cbrt(t176)
  t180 = jnp.where(t176 <= p.zeta_threshold, t172, t178 * t176)
  res = -t84 + t116 + (-t5 * t152 / 0.4e1 - t55 * t86 * t95 * t139 * t65 * t98 * (0.13e2 / 0.12e2 * t3 - t100 + 0.1e1 / 0.2e1) * t108 / 0.144e3 + t84 - t116) * (t175 + t180 - 0.2e1) / (0.2e1 * t122 - 0.2e1)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.log(0.2e1)
  t2 = 0.1e1 - t1
  t3 = jnp.pi ** 2
  t4 = 0.1e1 / t3
  t5 = t2 * t4
  t6 = t1 / 0.6e1
  t7 = 0.90154267736969571405 * t4
  t9 = 0.1e1 / t2
  t13 = jnp.exp(-0.2e1 * (-0.711e-1 + t6 - t7) * t9 * t3)
  t14 = jnp.cbrt(3)
  t15 = t14 ** 2
  t17 = jnp.cbrt(0.1e1 / jnp.pi)
  t18 = t17 ** 2
  t20 = jnp.cbrt(4)
  t21 = jnp.cbrt(r0)
  t22 = t21 ** 2
  t27 = jnp.exp(-t15 * t18 * t20 / t22 / 0.4e5)
  t28 = 0.1e1 - t27
  t29 = jnp.cbrt(jnp.pi)
  t30 = t29 ** 2
  t32 = jnp.cbrt(9)
  t33 = 0.1e1 / t30 * t32
  t34 = t20 ** 2
  t40 = t13 / 0.2e1
  t46 = 0.1e1 / t17
  t48 = t46 * t20 * t21
  t52 = jnp.sqrt(0.4e1)
  t54 = t14 * t17
  t55 = 0.1e1 / t21
  t57 = t54 * t34 * t55
  t58 = jnp.sqrt(t57)
  t62 = t28 * t9 * t3 * t52 / t58 / t57
  t64 = t32 ** 2
  t65 = t64 * t20
  t66 = t30 * t3
  t77 = 0.1e1 / t18 * t34 * t22
  t81 = jnp.log(0.1e1 + (t13 - 0.2e1 * t28 * ((-0.9 + 0.3e1 / 0.16e2 * t33 * t34) * t9 * t3 + t40)) * t15 * t48 / 0.3e1 - 0.12e2 * t62 + (t13 - 0.2e1 * t28 * (-0.3e1 / 0.4e2 * t65 * t66 * t9 + t40)) * t14 * t77 / 0.3e1)
  t83 = t5 * t81 / 0.2e1
  t85 = t55 * t27
  t86 = 4 ** (0.1e1 / 0.4e1)
  t87 = t86 ** 2
  t89 = t57 ** (0.1e1 / 0.4e1)
  t94 = 0.1e1 / (t27 + 0.5e1 / 0.8e1 * t87 * t86 * t89 * t57)
  t97 = 0.1e1 / t29 / t3 / jnp.pi
  t99 = 0.12e2 * t1
  t107 = jnp.log(0.1e1 + t15 * t46 * t20 * t21 / 0.3e1)
  t115 = t54 * t34 * t85 * t94 * (-t65 * t97 * (0.7e1 / 0.6e1 * t3 - t99 - 0.1e1) * t107 / 0.36e2 - 0.1e-1) / 0.4e1
  t120 = jnp.exp(-0.4e1 * (-0.49917e-1 + t6 - t7) * t9 * t3)
  t121 = jnp.cbrt(2)
  t129 = t120 / 0.2e1
  t138 = t121 ** 2
  t151 = jnp.log(0.1e1 + (t120 - 0.2e1 * t28 * (0.2e1 * (-0.9 + 0.3e1 / 0.16e2 * t33 * t34 * t121) * t9 * t3 + t129)) * t15 * t48 / 0.3e1 - 0.24e2 * t62 + (t120 - 0.2e1 * t28 * (-0.3e1 / 0.2e2 * t65 * t66 * t138 * t9 + t129)) * t14 * t77 / 0.3e1)
  t166 = jnp.cbrt(p.zeta_threshold)
  t168 = jnp.where(0.1e1 <= p.zeta_threshold, t166 * p.zeta_threshold, 1)
  res = -t83 + t115 + (-t5 * t151 / 0.4e1 - t54 * t85 * t94 * t138 * t64 * t97 * (0.13e2 / 0.12e2 * t3 - t99 + 0.1e1 / 0.2e1) * t107 / 0.144e3 + t83 - t115) * (0.2e1 * t168 - 0.2e1) / (0.2e1 * t121 - 0.2e1)
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