"""Generated from gga_x_ityh_pbe.mpl."""

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
  t28 = jnp.cbrt(t6)
  t29 = t2 ** 2
  t30 = jnp.pi * t29
  t32 = jnp.cbrt(0.1e1 / jnp.pi)
  t34 = jnp.cbrt(4)
  t35 = 0.1e1 / t32 * t34
  t36 = jnp.cbrt(6)
  t37 = params.mu * t36
  t38 = jnp.pi ** 2
  t39 = jnp.cbrt(t38)
  t40 = t39 ** 2
  t41 = 0.1e1 / t40
  t43 = r0 ** 2
  t44 = jnp.cbrt(r0)
  t45 = t44 ** 2
  t56 = 0.1e1 + params.kappa * (0.1e1 - params.kappa / (params.kappa + t37 * t41 * s0 / t45 / t43 / 0.24e2))
  t60 = jnp.sqrt(t30 * t35 / t56)
  t63 = jnp.cbrt(2)
  t65 = jnp.cbrt(t20 * t6)
  t69 = p.cam_omega / t60 * t63 / t65 / 0.2e1
  t71 = 0.135e1 < t69
  t72 = jnp.where(t71, t69, 0.135e1)
  t73 = t72 ** 2
  t76 = t73 ** 2
  t79 = t76 * t73
  t82 = t76 ** 2
  t94 = t82 ** 2
  t98 = jnp.where(t71, 0.135e1, t69)
  t99 = jnp.sqrt(jnp.pi)
  t102 = jax.lax.erf(0.1e1 / t98 / 0.2e1)
  t104 = t98 ** 2
  t107 = jnp.exp(-0.1e1 / t104 / 0.4e1)
  t118 = jnp.where(0.135e1 <= t69, 0.1e1 / t73 / 0.36e2 - 0.1e1 / t76 / 0.96e3 + 0.1e1 / t79 / 0.2688e5 - 0.1e1 / t82 / 0.82944e6 + 0.1e1 / t82 / t73 / 0.2838528e8 - 0.1e1 / t82 / t76 / 0.107347968e10 + 0.1e1 / t82 / t79 / 0.445906944e11 - 0.1e1 / t94 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t98 * (t99 * t102 + 0.2e1 * t98 * (t107 - 0.3e1 / 0.2e1 - 0.2e1 * t104 * (t107 - 0.1e1))))
  t123 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * t118 * t56)
  t125 = jnp.where(t10, t15, -t17)
  t126 = jnp.where(t14, t11, t125)
  t127 = 0.1e1 + t126
  t129 = jnp.cbrt(t127)
  t131 = jnp.where(t127 <= p.zeta_threshold, t23, t129 * t127)
  t134 = r1 ** 2
  t135 = jnp.cbrt(r1)
  t136 = t135 ** 2
  t147 = 0.1e1 + params.kappa * (0.1e1 - params.kappa / (params.kappa + t37 * t41 * s2 / t136 / t134 / 0.24e2))
  t151 = jnp.sqrt(t30 * t35 / t147)
  t155 = jnp.cbrt(t127 * t6)
  t159 = p.cam_omega / t151 * t63 / t155 / 0.2e1
  t161 = 0.135e1 < t159
  t162 = jnp.where(t161, t159, 0.135e1)
  t163 = t162 ** 2
  t166 = t163 ** 2
  t169 = t166 * t163
  t172 = t166 ** 2
  t184 = t172 ** 2
  t188 = jnp.where(t161, 0.135e1, t159)
  t191 = jax.lax.erf(0.1e1 / t188 / 0.2e1)
  t193 = t188 ** 2
  t196 = jnp.exp(-0.1e1 / t193 / 0.4e1)
  t207 = jnp.where(0.135e1 <= t159, 0.1e1 / t163 / 0.36e2 - 0.1e1 / t166 / 0.96e3 + 0.1e1 / t169 / 0.2688e5 - 0.1e1 / t172 / 0.82944e6 + 0.1e1 / t172 / t163 / 0.2838528e8 - 0.1e1 / t172 / t166 / 0.107347968e10 + 0.1e1 / t172 / t169 / 0.445906944e11 - 0.1e1 / t184 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t188 * (t99 * t191 + 0.2e1 * t188 * (t196 - 0.3e1 / 0.2e1 - 0.2e1 * t193 * (t196 - 0.1e1))))
  t212 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t131 * t28 * t207 * t147)
  res = t123 + t212
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
  t20 = jnp.cbrt(r0)
  t21 = t3 ** 2
  t24 = jnp.cbrt(0.1e1 / jnp.pi)
  t26 = jnp.cbrt(4)
  t28 = jnp.cbrt(6)
  t30 = jnp.pi ** 2
  t31 = jnp.cbrt(t30)
  t32 = t31 ** 2
  t35 = jnp.cbrt(2)
  t36 = t35 ** 2
  t38 = r0 ** 2
  t39 = t20 ** 2
  t50 = 0.1e1 + params.kappa * (0.1e1 - params.kappa / (params.kappa + params.mu * t28 / t32 * s0 * t36 / t39 / t38 / 0.24e2))
  t54 = jnp.sqrt(jnp.pi * t21 / t24 * t26 / t50)
  t58 = jnp.cbrt(t12 * r0)
  t62 = p.cam_omega / t54 * t35 / t58 / 0.2e1
  t64 = 0.135e1 < t62
  t65 = jnp.where(t64, t62, 0.135e1)
  t66 = t65 ** 2
  t69 = t66 ** 2
  t72 = t69 * t66
  t75 = t69 ** 2
  t87 = t75 ** 2
  t91 = jnp.where(t64, 0.135e1, t62)
  t92 = jnp.sqrt(jnp.pi)
  t95 = jax.lax.erf(0.1e1 / t91 / 0.2e1)
  t97 = t91 ** 2
  t100 = jnp.exp(-0.1e1 / t97 / 0.4e1)
  t111 = jnp.where(0.135e1 <= t62, 0.1e1 / t66 / 0.36e2 - 0.1e1 / t69 / 0.96e3 + 0.1e1 / t72 / 0.2688e5 - 0.1e1 / t75 / 0.82944e6 + 0.1e1 / t75 / t66 / 0.2838528e8 - 0.1e1 / t75 / t69 / 0.107347968e10 + 0.1e1 / t75 / t72 / 0.445906944e11 - 0.1e1 / t87 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t91 * (t92 * t95 + 0.2e1 * t91 * (t100 - 0.3e1 / 0.2e1 - 0.2e1 * t97 * (t100 - 0.1e1))))
  t116 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * t111 * t50)
  res = 0.2e1 * t116
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