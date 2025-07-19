"""Generated from gga_x_ityh.mpl."""

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
  t33 = 0.1e1 / t32
  t34 = jnp.cbrt(4)
  t35 = t33 * t34
  t37 = t29 * t33 * t34
  t38 = r0 ** 2
  t39 = jnp.cbrt(r0)
  t40 = t39 ** 2
  t44 = jnp.sqrt(s0)
  t47 = t44 / t39 / r0
  t48 = jnp.arcsinh(t47)
  t56 = 0.1e1 + 0.93333333333333333332e-3 * t37 * s0 / t40 / t38 / (0.1e1 + 0.252e-1 * t47 * t48)
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
  t133 = r1 ** 2
  t134 = jnp.cbrt(r1)
  t135 = t134 ** 2
  t139 = jnp.sqrt(s2)
  t142 = t139 / t134 / r1
  t143 = jnp.arcsinh(t142)
  t151 = 0.1e1 + 0.93333333333333333332e-3 * t37 * s2 / t135 / t133 / (0.1e1 + 0.252e-1 * t142 * t143)
  t155 = jnp.sqrt(t30 * t35 / t151)
  t159 = jnp.cbrt(t127 * t6)
  t163 = p.cam_omega / t155 * t63 / t159 / 0.2e1
  t165 = 0.135e1 < t163
  t166 = jnp.where(t165, t163, 0.135e1)
  t167 = t166 ** 2
  t170 = t167 ** 2
  t173 = t170 * t167
  t176 = t170 ** 2
  t188 = t176 ** 2
  t192 = jnp.where(t165, 0.135e1, t163)
  t195 = jax.lax.erf(0.1e1 / t192 / 0.2e1)
  t197 = t192 ** 2
  t200 = jnp.exp(-0.1e1 / t197 / 0.4e1)
  t211 = jnp.where(0.135e1 <= t163, 0.1e1 / t167 / 0.36e2 - 0.1e1 / t170 / 0.96e3 + 0.1e1 / t173 / 0.2688e5 - 0.1e1 / t176 / 0.82944e6 + 0.1e1 / t176 / t167 / 0.2838528e8 - 0.1e1 / t176 / t170 / 0.107347968e10 + 0.1e1 / t176 / t173 / 0.445906944e11 - 0.1e1 / t188 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t192 * (t99 * t195 + 0.2e1 * t192 * (t200 - 0.3e1 / 0.2e1 - 0.2e1 * t197 * (t200 - 0.1e1))))
  t216 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t131 * t28 * t211 * t151)
  res = t123 + t216
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
  t25 = 0.1e1 / t24
  t26 = jnp.cbrt(4)
  t30 = jnp.cbrt(2)
  t31 = t30 ** 2
  t33 = r0 ** 2
  t34 = t20 ** 2
  t37 = jnp.sqrt(s0)
  t38 = t37 * t30
  t40 = 0.1e1 / t20 / r0
  t42 = jnp.arcsinh(t38 * t40)
  t52 = 0.1e1 + 0.93333333333333333332e-3 * t21 * t25 * t26 * s0 * t31 / t34 / t33 / (0.1e1 + 0.252e-1 * t38 * t40 * t42)
  t56 = jnp.sqrt(jnp.pi * t21 * t25 * t26 / t52)
  t60 = jnp.cbrt(t12 * r0)
  t64 = p.cam_omega / t56 * t30 / t60 / 0.2e1
  t66 = 0.135e1 < t64
  t67 = jnp.where(t66, t64, 0.135e1)
  t68 = t67 ** 2
  t71 = t68 ** 2
  t74 = t71 * t68
  t77 = t71 ** 2
  t89 = t77 ** 2
  t93 = jnp.where(t66, 0.135e1, t64)
  t94 = jnp.sqrt(jnp.pi)
  t97 = jax.lax.erf(0.1e1 / t93 / 0.2e1)
  t99 = t93 ** 2
  t102 = jnp.exp(-0.1e1 / t99 / 0.4e1)
  t113 = jnp.where(0.135e1 <= t64, 0.1e1 / t68 / 0.36e2 - 0.1e1 / t71 / 0.96e3 + 0.1e1 / t74 / 0.2688e5 - 0.1e1 / t77 / 0.82944e6 + 0.1e1 / t77 / t68 / 0.2838528e8 - 0.1e1 / t77 / t71 / 0.107347968e10 + 0.1e1 / t77 / t74 / 0.445906944e11 - 0.1e1 / t89 / 0.20214448128e13, 0.1e1 - 0.8e1 / 0.3e1 * t93 * (t94 * t97 + 0.2e1 * t93 * (t102 - 0.3e1 / 0.2e1 - 0.2e1 * t99 * (t102 - 0.1e1))))
  t118 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * t113 * t52)
  res = 0.2e1 * t118
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