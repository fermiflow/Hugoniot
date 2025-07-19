"""Generated from lda_c_vwn_3.mpl."""

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
  t35 = 0.1e1 / (t12 + 0.353021e1 * t13 + 0.180578e2)
  t39 = jnp.log(t4 * t10 * t35 / 0.4e1)
  t44 = jnp.arctan(0.473092690956011283e1 / (t13 + 0.706042e1))
  t47 = (t27 + 0.325) ** 2
  t49 = jnp.log(t47 * t35)
  t51 = 0.1554535e-1 * t39 + 0.52491393169780936218e-1 * t44 + 0.22478670955426118383e-2 * t49 - t21 - t26 - t32
  t54 = 0.1e1 / (t12 + 0.1006155e2 * t13 + 0.101578e3)
  t58 = jnp.log(t4 * t10 * t54 / 0.4e1)
  t63 = jnp.arctan(0.11716852777089929792e1 / (t13 + 0.201231e2))
  t66 = (t27 + 0.743294) ** 2
  t68 = jnp.log(t66 * t54)
  t72 = 0.1e1 / (t12 + 0.6536e1 * t13 + 0.427198e2)
  t76 = jnp.log(t4 * t10 * t72 / 0.4e1)
  t81 = jnp.arctan(0.44899888641287296627e-1 / (t13 + 0.13072e2))
  t84 = (t27 + 0.409286) ** 2
  t86 = jnp.log(t84 * t72)
  t91 = jnp.pi ** 2
  t95 = 0.1e1 / (t12 + 0.534175 * t13 + 0.114813e2)
  t99 = jnp.log(t4 * t10 * t95 / 0.4e1)
  t103 = jnp.arctan(0.6692072046645941483e1 / (t13 + 0.106835e1))
  t106 = (t27 + 0.228344) ** 2
  t108 = jnp.log(t106 * t95)
  t113 = r0 - r1
  t115 = t113 / t7
  t116 = 0.1e1 + t115
  t118 = jnp.cbrt(p.zeta_threshold)
  t119 = t118 * p.zeta_threshold
  t120 = jnp.cbrt(t116)
  t122 = jnp.where(t116 <= p.zeta_threshold, t119, t120 * t116)
  t123 = 0.1e1 - t115
  t125 = jnp.cbrt(t123)
  t127 = jnp.where(t123 <= p.zeta_threshold, t119, t125 * t123)
  t128 = t122 + t127 - 0.2e1
  t129 = jnp.cbrt(2)
  t130 = t129 - 0.1e1
  t132 = 0.1e1 / t130 / 0.2e1
  t134 = t113 ** 2
  t135 = t134 ** 2
  t136 = t7 ** 2
  t137 = t136 ** 2
  t138 = 0.1e1 / t137
  res = t21 + t26 + t32 - 0.3e1 / 0.8e1 * t51 / (0.1554535e-1 * t58 + 0.61881802979060631482 * t63 + 0.26673100072733151594e-2 * t68 - 0.310907e-1 * t76 - 0.20521972937837502661e2 * t81 - 0.44313737677495382697e-2 * t86) / t91 * (t99 + 0.32323836906055067299 * t103 + 0.21608710360898267022e-1 * t108) * t128 * t132 * (-t135 * t138 + 0.1e1) * t130 + t51 * t128 * t132 * t135 * t138
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
  t34 = 0.1e1 / (t11 + 0.353021e1 * t12 + 0.180578e2)
  t38 = jnp.log(t4 * t9 * t34 / 0.4e1)
  t43 = jnp.arctan(0.473092690956011283e1 / (t12 + 0.706042e1))
  t46 = (t26 + 0.325) ** 2
  t48 = jnp.log(t46 * t34)
  t53 = 0.1e1 / (t11 + 0.1006155e2 * t12 + 0.101578e3)
  t57 = jnp.log(t4 * t9 * t53 / 0.4e1)
  t62 = jnp.arctan(0.11716852777089929792e1 / (t12 + 0.201231e2))
  t65 = (t26 + 0.743294) ** 2
  t67 = jnp.log(t65 * t53)
  t71 = 0.1e1 / (t11 + 0.6536e1 * t12 + 0.427198e2)
  t75 = jnp.log(t4 * t9 * t71 / 0.4e1)
  t80 = jnp.arctan(0.44899888641287296627e-1 / (t12 + 0.13072e2))
  t83 = (t26 + 0.409286) ** 2
  t85 = jnp.log(t83 * t71)
  t90 = jnp.pi ** 2
  t95 = 0.1e1 / (t11 + 0.534175 * t12 + 0.114813e2)
  t99 = jnp.log(t4 * t9 * t95 / 0.4e1)
  t103 = jnp.arctan(0.6692072046645941483e1 / (t12 + 0.106835e1))
  t106 = (t26 + 0.228344) ** 2
  t108 = jnp.log(t106 * t95)
  t112 = jnp.cbrt(p.zeta_threshold)
  t114 = jnp.where(0.1e1 <= p.zeta_threshold, t112 * p.zeta_threshold, 1)
  t118 = jnp.cbrt(2)
  t119 = t118 - 0.1e1
  res = t20 + t25 + t31 - 0.3e1 / 0.16e2 * (0.1554535e-1 * t38 + 0.52491393169780936218e-1 * t43 + 0.22478670955426118383e-2 * t48 - t20 - t25 - t31) / (0.1554535e-1 * t57 + 0.61881802979060631482 * t62 + 0.26673100072733151594e-2 * t67 - 0.310907e-1 * t75 - 0.20521972937837502661e2 * t80 - 0.44313737677495382697e-2 * t85) / t90 * (t99 + 0.32323836906055067299 * t103 + 0.21608710360898267022e-1 * t108) * (0.2e1 * t114 - 0.2e1)
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