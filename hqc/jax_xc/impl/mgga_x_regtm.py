"""Generated from mgga_x_regtm.mpl."""

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
  t29 = jnp.cbrt(r0)
  t30 = t29 ** 2
  t33 = tau0 / t30 / r0
  t34 = r0 ** 2
  t36 = 0.1e1 / t30 / t34
  t37 = s0 * t36
  t39 = t33 - t37 / 0.8e1
  t40 = jnp.cbrt(6)
  t41 = t39 * t40
  t42 = jnp.pi ** 2
  t43 = jnp.cbrt(t42)
  t44 = t43 ** 2
  t45 = 0.1e1 / t44
  t46 = t40 * t45
  t47 = t46 * t37
  t49 = t41 * t45
  t51 = 0.1e1 - 0.5e1 / 0.9e1 * t49
  t52 = t51 ** 2
  t54 = t39 ** 2
  t55 = t40 ** 2
  t58 = 0.1e1 / t43 / t42
  t61 = 0.1e1 + 0.67148919753086419753 * t54 * t55 * t58
  t62 = jnp.sqrt(t61)
  t67 = jnp.exp(-t47 / 0.8e1)
  t74 = 0.1e1 + t41 * t45 / (t47 / 0.24e2 + t52 * t51 / t62 / t61 * t67) / 0.3e1
  t75 = t74 ** 2
  t78 = 0.1e1 / t75 / t74
  t82 = (0.1e1 + t78) ** 2
  t84 = (0.1e1 / t75 + 0.3e1 * t78) / t82
  t86 = t55 * t58
  t87 = s0 ** 2
  t88 = t34 ** 2
  t96 = (0.1e1 + 0.15045488888888888889 * t47 + 0.26899490462262948e-2 * t86 * t87 / t29 / t88 / r0) ** (0.1e1 / 0.5e1)
  t101 = 0.256337604 * t55 * t44
  t108 = t96 ** 2
  t124 = t49 / 0.4e1 - 0.9e1 / 0.2e2 + t47 / 0.36e2
  t125 = t124 ** 2
  t131 = s0 / r0 / tau0 / 0.8e1
  t133 = jnp.where(t131 < 0.1e1, t131, 0.1e1)
  t139 = (0.1e1 + 0.5e1 / 0.12e2 * (0.1e2 / 0.81e2 + 0.25e2 / 0.8748e4 * t47) * t40 * t45 * s0 * t36 + 0.292e3 / 0.405e3 * t125 - 0.146e3 / 0.135e3 * t124 * t133 * (0.1e1 - t133)) ** (0.1e1 / 0.1e2)
  t145 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (t84 * (0.1e1 / t96 + 0.7e1 / 0.9e1 * (0.1e1 + 0.63943327777777777778e-1 * t47 - 0.5e1 / 0.9e1 * (0.14554132 * t33 + t101 + 0.11867481666666666667e-1 * t37) * t40 * t45) / t108) + (0.1e1 - t84) * t139))
  t147 = jnp.where(t10, t15, -t17)
  t148 = jnp.where(t14, t11, t147)
  t149 = 0.1e1 + t148
  t151 = jnp.cbrt(t149)
  t153 = jnp.where(t149 <= p.zeta_threshold, t23, t151 * t149)
  t155 = jnp.cbrt(r1)
  t156 = t155 ** 2
  t159 = tau1 / t156 / r1
  t160 = r1 ** 2
  t162 = 0.1e1 / t156 / t160
  t163 = s2 * t162
  t165 = t159 - t163 / 0.8e1
  t166 = t165 * t40
  t167 = t46 * t163
  t169 = t166 * t45
  t171 = 0.1e1 - 0.5e1 / 0.9e1 * t169
  t172 = t171 ** 2
  t174 = t165 ** 2
  t178 = 0.1e1 + 0.67148919753086419753 * t174 * t55 * t58
  t179 = jnp.sqrt(t178)
  t184 = jnp.exp(-t167 / 0.8e1)
  t191 = 0.1e1 + t166 * t45 / (t167 / 0.24e2 + t172 * t171 / t179 / t178 * t184) / 0.3e1
  t192 = t191 ** 2
  t195 = 0.1e1 / t192 / t191
  t199 = (0.1e1 + t195) ** 2
  t201 = (0.1e1 / t192 + 0.3e1 * t195) / t199
  t203 = s2 ** 2
  t204 = t160 ** 2
  t212 = (0.1e1 + 0.15045488888888888889 * t167 + 0.26899490462262948e-2 * t86 * t203 / t155 / t204 / r1) ** (0.1e1 / 0.5e1)
  t222 = t212 ** 2
  t238 = t169 / 0.4e1 - 0.9e1 / 0.2e2 + t167 / 0.36e2
  t239 = t238 ** 2
  t245 = s2 / r1 / tau1 / 0.8e1
  t247 = jnp.where(t245 < 0.1e1, t245, 0.1e1)
  t253 = (0.1e1 + 0.5e1 / 0.12e2 * (0.1e2 / 0.81e2 + 0.25e2 / 0.8748e4 * t167) * t40 * t45 * s2 * t162 + 0.292e3 / 0.405e3 * t239 - 0.146e3 / 0.135e3 * t238 * t247 * (0.1e1 - t247)) ** (0.1e1 / 0.1e2)
  t259 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t153 * t27 * (t201 * (0.1e1 / t212 + 0.7e1 / 0.9e1 * (0.1e1 + 0.63943327777777777778e-1 * t167 - 0.5e1 / 0.9e1 * (0.14554132 * t159 + t101 + 0.11867481666666666667e-1 * t163) * t40 * t45) / t222) + (0.1e1 - t201) * t253))
  res = t145 + t259
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
  t21 = jnp.cbrt(2)
  t22 = t21 ** 2
  t24 = t19 ** 2
  t27 = tau0 * t22 / t24 / r0
  t29 = r0 ** 2
  t32 = s0 * t22 / t24 / t29
  t34 = t27 - t32 / 0.8e1
  t35 = jnp.cbrt(6)
  t36 = t34 * t35
  t37 = jnp.pi ** 2
  t38 = jnp.cbrt(t37)
  t39 = t38 ** 2
  t40 = 0.1e1 / t39
  t42 = t35 * t40 * t32
  t44 = t36 * t40
  t46 = 0.1e1 - 0.5e1 / 0.9e1 * t44
  t47 = t46 ** 2
  t49 = t34 ** 2
  t50 = t35 ** 2
  t53 = 0.1e1 / t38 / t37
  t56 = 0.1e1 + 0.67148919753086419753 * t49 * t50 * t53
  t57 = jnp.sqrt(t56)
  t62 = jnp.exp(-t42 / 0.8e1)
  t69 = 0.1e1 + t36 * t40 / (t42 / 0.24e2 + t47 * t46 / t57 / t56 * t62) / 0.3e1
  t70 = t69 ** 2
  t73 = 0.1e1 / t70 / t69
  t77 = (0.1e1 + t73) ** 2
  t79 = (0.1e1 / t70 + 0.3e1 * t73) / t77
  t82 = s0 ** 2
  t84 = t29 ** 2
  t92 = (0.1e1 + 0.15045488888888888889 * t42 + 0.53798980924525896e-2 * t50 * t53 * t82 * t21 / t19 / t84 / r0) ** (0.1e1 / 0.5e1)
  t104 = t92 ** 2
  t119 = t44 / 0.4e1 - 0.9e1 / 0.2e2 + t42 / 0.36e2
  t120 = t119 ** 2
  t126 = s0 / r0 / tau0 / 0.8e1
  t128 = jnp.where(t126 < 0.1e1, t126, 0.1e1)
  t134 = (0.1e1 + 0.5e1 / 0.12e2 * (0.1e2 / 0.81e2 + 0.25e2 / 0.8748e4 * t42) * t35 * t40 * t32 + 0.292e3 / 0.405e3 * t120 - 0.146e3 / 0.135e3 * t119 * t128 * (0.1e1 - t128)) ** (0.1e1 / 0.1e2)
  t140 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (t79 * (0.1e1 / t92 + 0.7e1 / 0.9e1 * (0.1e1 + 0.63943327777777777778e-1 * t42 - 0.5e1 / 0.9e1 * (0.14554132 * t27 + 0.256337604 * t50 * t39 + 0.11867481666666666667e-1 * t32) * t35 * t40) / t104) + (0.1e1 - t79) * t134))
  res = 0.2e1 * t140
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