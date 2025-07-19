"""Generated from mgga_x_mbeefvdw.mpl."""

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
  t20 = t19 + 0.1e1
  t22 = jnp.cbrt(p.zeta_threshold)
  t23 = t22 * p.zeta_threshold
  t24 = jnp.cbrt(t20)
  t26 = jnp.where(t20 <= p.zeta_threshold, t23, t24 * t20)
  t27 = jnp.cbrt(t6)
  t29 = jnp.cbrt(r0)
  t30 = t29 ** 2
  t34 = r0 ** 2
  t37 = s0 / t30 / t34
  t40 = jnp.cbrt(6)
  t42 = jnp.pi ** 2
  t43 = jnp.cbrt(t42)
  t44 = t43 ** 2
  t45 = 0.1e1 / t44
  t47 = 0.5e1 / 0.9e1 * (tau0 / t30 / r0 - t37 / 0.8e1) * t40 * t45
  t49 = 0.1e5 < t47
  t50 = jnp.where(t49, t47, 0.1e5)
  t51 = t50 ** 2
  t56 = t51 ** 2
  t60 = jnp.where(t49, 0.1e5, t47)
  t61 = t60 ** 2
  t62 = 0.1e1 - t61
  t63 = t62 ** 2
  t65 = t61 * t60
  t71 = jnp.where(0.1e5 <= t47, 0.1e1 - 0.3e1 / t51 - 0.1e1 / t51 / t50 + 0.3e1 / t56, -t63 * t62 / (0.1e1 + t65 * (0.1e1 + t65)))
  t72 = t71 ** 2
  t73 = t72 * t71
  t77 = t40 * t45
  t83 = t77 * t37 / (0.65124e1 + t77 * t37 / 0.24e2)
  t85 = t83 / 0.12e2 - 0.1e1
  t86 = t85 ** 2
  t88 = t72 ** 2
  t90 = t86 * t85
  t92 = t86 ** 2
  t96 = 0.5e1 / 0.2e1 * t90 - t83 / 0.8e1 + 0.3e1 / 0.2e1
  t100 = -0.1e1 / 0.2e1 + 0.3e1 / 0.2e1 * t72
  t105 = 0.5e1 / 0.2e1 * t73 - 0.3e1 / 0.2e1 * t71
  t110 = 0.3e1 / 0.8e1 + 0.35e2 / 0.8e1 * t88 - 0.15e2 / 0.4e1 * t72
  t115 = 0.3e1 / 0.8e1 + 0.35e2 / 0.8e1 * t92 - 0.15e2 / 0.4e1 * t86
  t118 = 0.351985355e-2 * t73 + 0.217681859775e-1 * t72 - 0.6972770593e-1 * t71 - 0.851282539125e-1 * t86 + 0.61919587625e-3 * t88 - 0.50282912e-1 * t90 + 0.618699843125e-2 * t92 + 0.192374554e-1 * t96 * t71 + 0.133707403e-6 * t96 * t100 - 0.549909413e-7 * t96 * t105 + 0.397324768e-8 * t96 * t110 + 0.919317034e-6 * t115 * t71
  t134 = -0.1e1 / 0.2e1 + 0.3e1 / 0.2e1 * t86
  t144 = 0.10451438955835e1 - 0.500749348e-6 * t115 * t100 + 0.574317889e-7 * t115 * t105 - 0.340722258e-8 * t115 * t110 + 0.453837246e-1 * t85 * t71 + 0.318024096e-1 * t85 * t100 - 0.608338264e-2 * t85 * t105 - 0.100478906e-6 * t85 * t110 - 0.222650139e-1 * t134 * t71 - 0.521818079e-2 * t134 * t100 - 0.657949254e-6 * t134 * t105 + 0.201895739e-6 * t134 * t110 + 0.1214700985e-1 * t83
  t149 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (t118 + t144))
  t151 = jnp.where(t10, t15, -t17)
  t152 = jnp.where(t14, t11, t151)
  t153 = t152 + 0.1e1
  t155 = jnp.cbrt(t153)
  t157 = jnp.where(t153 <= p.zeta_threshold, t23, t155 * t153)
  t159 = r1 ** 2
  t160 = jnp.cbrt(r1)
  t161 = t160 ** 2
  t164 = s2 / t161 / t159
  t170 = t77 * t164 / (0.65124e1 + t77 * t164 / 0.24e2)
  t172 = t170 / 0.12e2 - 0.1e1
  t173 = t172 ** 2
  t174 = t173 * t172
  t183 = 0.5e1 / 0.9e1 * (tau1 / t161 / r1 - t164 / 0.8e1) * t40 * t45
  t185 = 0.1e5 < t183
  t186 = jnp.where(t185, t183, 0.1e5)
  t187 = t186 ** 2
  t192 = t187 ** 2
  t196 = jnp.where(t185, 0.1e5, t183)
  t197 = t196 ** 2
  t198 = 0.1e1 - t197
  t199 = t198 ** 2
  t201 = t197 * t196
  t207 = jnp.where(0.1e5 <= t183, 0.1e1 - 0.3e1 / t187 - 0.1e1 / t187 / t186 + 0.3e1 / t192, -t199 * t198 / (0.1e1 + t201 * (0.1e1 + t201)))
  t208 = t207 ** 2
  t209 = t208 ** 2
  t214 = t208 * t207
  t216 = t173 ** 2
  t219 = -0.1e1 / 0.2e1 + 0.3e1 / 0.2e1 * t173
  t221 = -0.1e1 / 0.2e1 + 0.3e1 / 0.2e1 * t208
  t226 = 0.5e1 / 0.2e1 * t214 - 0.3e1 / 0.2e1 * t207
  t233 = 0.3e1 / 0.8e1 + 0.35e2 / 0.8e1 * t209 - 0.15e2 / 0.4e1 * t208
  t238 = -0.50282912e-1 * t174 + 0.61919587625e-3 * t209 + 0.217681859775e-1 * t208 - 0.6972770593e-1 * t207 - 0.851282539125e-1 * t173 + 0.351985355e-2 * t214 + 0.618699843125e-2 * t216 - 0.521818079e-2 * t219 * t221 - 0.657949254e-6 * t219 * t226 - 0.222650139e-1 * t219 * t207 - 0.100478906e-6 * t172 * t233 + 0.318024096e-1 * t172 * t221
  t245 = 0.3e1 / 0.8e1 + 0.35e2 / 0.8e1 * t216 - 0.15e2 / 0.4e1 * t173
  t256 = 0.5e1 / 0.2e1 * t174 - t170 / 0.8e1 + 0.3e1 / 0.2e1
  t268 = 0.10451438955835e1 - 0.608338264e-2 * t172 * t226 + 0.453837246e-1 * t172 * t207 + 0.574317889e-7 * t245 * t226 - 0.340722258e-8 * t245 * t233 - 0.500749348e-6 * t245 * t221 + 0.919317034e-6 * t245 * t207 - 0.549909413e-7 * t256 * t226 + 0.397324768e-8 * t256 * t233 + 0.133707403e-6 * t256 * t221 + 0.192374554e-1 * t256 * t207 + 0.201895739e-6 * t219 * t233 + 0.1214700985e-1 * t170
  t273 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t157 * t27 * (t238 + t268))
  res = t149 + t273
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
  t12 = t11 + 0.1e1
  t14 = jnp.cbrt(p.zeta_threshold)
  t16 = jnp.cbrt(t12)
  t18 = jnp.where(t12 <= p.zeta_threshold, t14 * p.zeta_threshold, t16 * t12)
  t19 = jnp.cbrt(r0)
  t21 = jnp.cbrt(2)
  t22 = t21 ** 2
  t24 = t19 ** 2
  t29 = r0 ** 2
  t31 = 0.1e1 / t24 / t29
  t32 = s0 * t22 * t31
  t35 = jnp.cbrt(6)
  t37 = jnp.pi ** 2
  t38 = jnp.cbrt(t37)
  t39 = t38 ** 2
  t40 = 0.1e1 / t39
  t42 = 0.5e1 / 0.9e1 * (tau0 * t22 / t24 / r0 - t32 / 0.8e1) * t35 * t40
  t44 = 0.1e5 < t42
  t45 = jnp.where(t44, t42, 0.1e5)
  t46 = t45 ** 2
  t51 = t46 ** 2
  t55 = jnp.where(t44, 0.1e5, t42)
  t56 = t55 ** 2
  t57 = 0.1e1 - t56
  t58 = t57 ** 2
  t60 = t56 * t55
  t66 = jnp.where(0.1e5 <= t42, 0.1e1 - 0.3e1 / t46 - 0.1e1 / t46 / t45 + 0.3e1 / t51, -t58 * t57 / (0.1e1 + t60 * (0.1e1 + t60)))
  t67 = t66 ** 2
  t68 = t67 ** 2
  t70 = t67 * t66
  t74 = t35 * t40
  t82 = t74 * s0 * t22 * t31 / (0.65124e1 + t74 * t32 / 0.24e2)
  t84 = t82 / 0.12e2 - 0.1e1
  t85 = t84 ** 2
  t87 = t85 * t84
  t89 = t85 ** 2
  t93 = 0.3e1 / 0.8e1 + 0.35e2 / 0.8e1 * t89 - 0.15e2 / 0.4e1 * t85
  t97 = -0.1e1 / 0.2e1 + 0.3e1 / 0.2e1 * t67
  t102 = 0.5e1 / 0.2e1 * t70 - 0.3e1 / 0.2e1 * t66
  t107 = 0.3e1 / 0.8e1 + 0.35e2 / 0.8e1 * t68 - 0.15e2 / 0.4e1 * t67
  t112 = 0.61919587625e-3 * t68 + 0.351985355e-2 * t70 + 0.217681859775e-1 * t67 - 0.6972770593e-1 * t66 - 0.851282539125e-1 * t85 - 0.50282912e-1 * t87 + 0.618699843125e-2 * t89 + 0.919317034e-6 * t93 * t66 - 0.500749348e-6 * t93 * t97 + 0.574317889e-7 * t93 * t102 - 0.340722258e-8 * t93 * t107 + 0.453837246e-1 * t84 * t66
  t120 = -0.1e1 / 0.2e1 + 0.3e1 / 0.2e1 * t85
  t131 = 0.5e1 / 0.2e1 * t87 - t82 / 0.8e1 + 0.3e1 / 0.2e1
  t141 = 0.10451438955835e1 + 0.318024096e-1 * t84 * t97 - 0.608338264e-2 * t84 * t102 - 0.100478906e-6 * t84 * t107 - 0.222650139e-1 * t120 * t66 - 0.521818079e-2 * t120 * t97 - 0.657949254e-6 * t120 * t102 + 0.201895739e-6 * t120 * t107 + 0.192374554e-1 * t131 * t66 + 0.133707403e-6 * t131 * t97 - 0.549909413e-7 * t131 * t102 + 0.397324768e-8 * t131 * t107 + 0.1214700985e-1 * t82
  t146 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (t112 + t141))
  res = 0.2e1 * t146
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