"""Generated from gga_c_p86vwn.mpl."""

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
  t2 = 0.1e1 / jnp.pi
  t3 = jnp.cbrt(t2)
  t4 = t1 * t3
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = r0 + r1
  t8 = jnp.cbrt(t7)
  t9 = 0.1e1 / t8
  t10 = t6 * t9
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
  t55 = 0.1e1 / t7
  t56 = t54 * t55
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
  t89 = 0.1e1 / (t12 + 0.353021e1 * t13 + 0.180578e2)
  t93 = jnp.log(t4 * t10 * t89 / 0.4e1)
  t98 = jnp.arctan(0.473092690956011283e1 / (t13 + 0.706042e1))
  t101 = (t27 + 0.325) ** 2
  t103 = jnp.log(t101 * t89)
  t111 = s0 + 0.2e1 * s1 + s2
  t119 = t3 * t6 * t9
  t122 = t1 ** 2
  t124 = t3 ** 2
  t126 = t8 ** 2
  t128 = t124 * t5 / t126
  t144 = params.aa + (params.bb + params.malpha * t1 * t119 / 0.4e1 + params.mbeta * t122 * t128 / 0.4e1) / (0.1e1 + params.mgamma * t1 * t119 / 0.4e1 + params.mdelta * t122 * t128 / 0.4e1 + 0.75e4 * params.mbeta * t2 * t55)
  t146 = jnp.sqrt(t111)
  t148 = t7 ** (0.1e1 / 0.6e1)
  t153 = jnp.exp(-params.ftilde * (params.aa + params.bb) / t144 * t146 / t148 / t7)
  t155 = t59 ** 2
  t156 = t155 * p.zeta_threshold
  t157 = t61 ** 2
  t159 = jnp.where(t58, t156, t157 * t57)
  t160 = t66 ** 2
  t162 = jnp.where(t65, t156, t160 * t64)
  t164 = jnp.sqrt(t159 + t162)
  t167 = jnp.sqrt(0.2e1)
  res = t21 + t26 + t32 - 0.3e1 / 0.8e1 / t33 * (t41 + 0.317708004743941464 * t45 + 0.41403379428206274608e-3 * t50) * t69 * t74 * (-t76 * t79 + 0.1e1) * t72 + (0.1554535e-1 * t93 + 0.52491393169780936218e-1 * t98 + 0.22478670955426118383e-2 * t103 - t21 - t26 - t32) * t69 * t74 * t76 * t79 + t111 / t8 / t77 * t153 * t144 / t164 * t167
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(3)
  t2 = 0.1e1 / jnp.pi
  t3 = jnp.cbrt(t2)
  t4 = t1 * t3
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = jnp.cbrt(r0)
  t8 = 0.1e1 / t7
  t9 = t6 * t8
  t10 = t4 * t9
  t11 = t10 / 0.4e1
  t12 = jnp.sqrt(t10)
  t15 = 0.1e1 / (t11 + 0.186372e1 * t12 + 0.129352e2)
  t19 = jnp.log(t4 * t9 * t15 / 0.4e1)
  t24 = jnp.arctan(0.61519908197590802322e1 / (t12 + 0.372744e1))
  t26 = t12 / 0.2e1
  t28 = (t26 + 0.10498) ** 2
  t30 = jnp.log(t28 * t15)
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
  t68 = r0 ** 2
  t76 = t3 * t6 * t8
  t79 = t1 ** 2
  t81 = t3 ** 2
  t83 = t7 ** 2
  t85 = t81 * t5 / t83
  t102 = params.aa + (params.bb + params.malpha * t1 * t76 / 0.4e1 + params.mbeta * t79 * t85 / 0.4e1) / (0.1e1 + params.mgamma * t1 * t76 / 0.4e1 + params.mdelta * t79 * t85 / 0.4e1 + 0.75e4 * params.mbeta * t2 / r0)
  t104 = jnp.sqrt(s0)
  t106 = r0 ** (0.1e1 / 0.6e1)
  t111 = jnp.exp(-params.ftilde * (params.aa + params.bb) / t102 * t104 / t106 / r0)
  t113 = t54 ** 2
  t115 = jnp.where(t53, t113 * p.zeta_threshold, 1)
  t116 = jnp.sqrt(t115)
  res = 0.310907e-1 * t19 + 0.38783294878113014393e-1 * t24 + 0.96902277115443742139e-3 * t30 - 0.1e1 / t32 * (t40 + 0.317708004743941464 * t44 + 0.41403379428206274608e-3 * t49) * (0.9e1 * t56 - 0.9e1) / 0.24e2 + s0 / t7 / t68 * t111 * t102 / t116
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