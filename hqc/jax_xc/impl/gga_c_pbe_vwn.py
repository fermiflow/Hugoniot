"""Generated from gga_c_pbe_vwn.mpl."""

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
  t1 = jnp.cbrt(3)
  t3 = jnp.cbrt(0.1e1 / jnp.pi)
  t4 = t1 * t3
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = r0 + r1
  t8 = jnp.cbrt(t7)
  t10 = t6 / t8
  t11 = t4 * t10
  t12 = t11 / 0.4e1
  t13 = jnp.sqrt(t11)
  t16 = 0.1e1 / (t12 + 0.186372e1 * t13 + 0.129352e2)
  t20 = jnp.log(t4 * t10 * t16 / 0.4e1)
  t21 = 0.310907e-1 * t20
  t25 = jnp.arctan(0.61519908197590802322e1 / (t13 + 0.372744e1))
  t26 = 0.38783294878113014393e-1 * t25
  t27 = t13 / 0.2e1
  t29 = (t27 + 0.10498) ** 2
  t31 = jnp.log(t29 * t16)
  t32 = 0.96902277115443742139e-3 * t31
  t33 = jnp.pi ** 2
  t37 = 0.1e1 / (t12 + 0.565535 * t13 + 0.130045e2)
  t41 = jnp.log(t4 * t10 * t37 / 0.4e1)
  t45 = jnp.arctan(0.71231089178181179908e1 / (t13 + 0.113107e1))
  t48 = (t27 + 0.47584e-2) ** 2
  t50 = jnp.log(t48 * t37)
  t54 = r0 - r1
  t56 = t54 / t7
  t57 = 0.1e1 + t56
  t58 = t57 <= p.zeta_threshold
  t59 = jnp.cbrt(p.zeta_threshold)
  t60 = t59 * p.zeta_threshold
  t61 = jnp.cbrt(t57)
  t63 = jnp.where(t58, t60, t61 * t57)
  t64 = 0.1e1 - t56
  t65 = t64 <= p.zeta_threshold
  t66 = jnp.cbrt(t64)
  t68 = jnp.where(t65, t60, t66 * t64)
  t69 = t63 + t68 - 0.2e1
  t71 = jnp.cbrt(2)
  t72 = t71 - 0.1e1
  t74 = 0.1e1 / t72 / 0.2e1
  t75 = t54 ** 2
  t76 = t75 ** 2
  t77 = t7 ** 2
  t78 = t77 ** 2
  t79 = 0.1e1 / t78
  t86 = 0.3e1 / 0.8e1 / t33 * (t41 + 0.317708004743941464 * t45 + 0.41403379428206274608e-3 * t50) * t69 * t74 * (-t76 * t79 + 0.1e1) * t72
  t89 = 0.1e1 / (t12 + 0.353021e1 * t13 + 0.180578e2)
  t93 = jnp.log(t4 * t10 * t89 / 0.4e1)
  t98 = jnp.arctan(0.473092690956011283e1 / (t13 + 0.706042e1))
  t101 = (t27 + 0.325) ** 2
  t103 = jnp.log(t101 * t89)
  t109 = (0.1554535e-1 * t93 + 0.52491393169780936218e-1 * t98 + 0.22478670955426118383e-2 * t103 - t21 - t26 - t32) * t69 * t74 * t76 * t79
  t110 = t59 ** 2
  t111 = t61 ** 2
  t112 = jnp.where(t58, t110, t111)
  t113 = t66 ** 2
  t114 = jnp.where(t65, t110, t113)
  t116 = t112 / 0.2e1 + t114 / 0.2e1
  t117 = t116 ** 2
  t118 = t117 * t116
  t121 = s0 + 0.2e1 * s1 + s2
  t127 = t1 ** 2
  t135 = 0.1e1 / params.gamma
  t140 = jnp.exp(-(t21 + t26 + t32 - t86 + t109) * t135 / t118)
  t142 = 0.1e1 / (t140 - 0.1e1)
  t144 = t121 ** 2
  t147 = t8 ** 2
  t150 = t71 ** 2
  t152 = t117 ** 2
  t155 = t3 ** 2
  t162 = t121 / t8 / t77 * t71 / t117 * t127 / t3 * t5 / 0.96e2 + params.BB * params.beta * t135 * t142 * t144 / t147 / t78 * t150 / t152 * t1 / t155 * t6 / 0.3072e4
  t172 = jnp.log(0.1e1 + params.beta * t162 * t135 / (params.beta * t135 * t142 * t162 + 0.1e1))
  res = params.gamma * t118 * t172 + t109 + t21 + t26 + t32 - t86
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(3)
  t3 = jnp.cbrt(0.1e1 / jnp.pi)
  t4 = t1 * t3
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = jnp.cbrt(r0)
  t9 = t6 / t7
  t10 = t4 * t9
  t11 = t10 / 0.4e1
  t12 = jnp.sqrt(t10)
  t15 = 0.1e1 / (t11 + 0.186372e1 * t12 + 0.129352e2)
  t19 = jnp.log(t4 * t9 * t15 / 0.4e1)
  t20 = 0.310907e-1 * t19
  t24 = jnp.arctan(0.61519908197590802322e1 / (t12 + 0.372744e1))
  t25 = 0.38783294878113014393e-1 * t24
  t26 = t12 / 0.2e1
  t28 = (t26 + 0.10498) ** 2
  t30 = jnp.log(t28 * t15)
  t31 = 0.96902277115443742139e-3 * t30
  t32 = jnp.pi ** 2
  t36 = 0.1e1 / (t11 + 0.565535 * t12 + 0.130045e2)
  t40 = jnp.log(t4 * t9 * t36 / 0.4e1)
  t44 = jnp.arctan(0.71231089178181179908e1 / (t12 + 0.113107e1))
  t47 = (t26 + 0.47584e-2) ** 2
  t49 = jnp.log(t47 * t36)
  t53 = 0.1e1 <= p.zeta_threshold
  t54 = jnp.cbrt(p.zeta_threshold)
  t56 = jnp.where(t53, t54 * p.zeta_threshold, 1)
  t59 = jnp.cbrt(2)
  t60 = t59 - 0.1e1
  t67 = 0.1e1 / t32 * (t40 + 0.317708004743941464 * t44 + 0.41403379428206274608e-3 * t49) * (0.9e1 * t56 - 0.9e1) / 0.24e2
  t68 = t54 ** 2
  t69 = jnp.where(t53, t68, 1)
  t70 = t69 ** 2
  t71 = t70 * t69
  t73 = r0 ** 2
  t79 = t1 ** 2
  t87 = 0.1e1 / params.gamma
  t92 = jnp.exp(-(t20 + t25 + t31 - t67) * t87 / t71)
  t94 = 0.1e1 / (t92 - 0.1e1)
  t96 = s0 ** 2
  t99 = t73 ** 2
  t100 = t7 ** 2
  t103 = t59 ** 2
  t105 = t70 ** 2
  t108 = t3 ** 2
  t115 = s0 / t7 / t73 * t59 / t70 * t79 / t3 * t5 / 0.96e2 + params.BB * params.beta * t87 * t94 * t96 / t100 / t99 * t103 / t105 * t1 / t108 * t6 / 0.3072e4
  t125 = jnp.log(0.1e1 + params.beta * t115 * t87 / (params.beta * t87 * t94 * t115 + 0.1e1))
  res = params.gamma * t71 * t125 + t20 + t25 + t31 - t67
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