"""Generated from mgga_x_r4scan.mpl."""

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
  t30 = 0.2e2 / 0.27e2 + 0.5e1 / 0.3e1 * params.eta
  t31 = jnp.cbrt(6)
  t32 = t31 ** 2
  t33 = jnp.pi ** 2
  t34 = jnp.cbrt(t33)
  t36 = 0.1e1 / t34 / t33
  t37 = t32 * t36
  t38 = s0 ** 2
  t39 = r0 ** 2
  t40 = t39 ** 2
  t42 = jnp.cbrt(r0)
  t44 = 0.1e1 / t42 / t40 / r0
  t45 = t38 * t44
  t46 = params.dp2 ** 2
  t47 = t46 ** 2
  t48 = 0.1e1 / t47
  t52 = jnp.exp(-t37 * t45 * t48 / 0.576e3)
  t57 = t34 ** 2
  t58 = 0.1e1 / t57
  t60 = t42 ** 2
  t62 = 0.1e1 / t60 / t39
  t63 = t58 * s0 * t62
  t70 = params.k1 * (0.1e1 - params.k1 / (params.k1 + (-0.162742215233874 * t30 * t52 + 0.1e2 / 0.81e2) * t31 * t63 / 0.24e2))
  t74 = s0 * t62
  t76 = tau0 / t60 / r0 - t74 / 0.8e1
  t78 = 0.3e1 / 0.1e2 * t32 * t57
  t82 = t78 + params.eta * s0 * t62 / 0.8e1
  t84 = t76 / t82
  t87 = jnp.where(0. < t84, 0, t84)
  t92 = jnp.exp(-params.c1 * t87 / (0.1e1 - t87))
  t94 = 0.25e1 < t84
  t95 = jnp.where(t94, 0.25e1, t84)
  t97 = t95 ** 2
  t99 = t97 * t95
  t101 = t97 ** 2
  t110 = jnp.where(t94, t84, 0.25e1)
  t114 = jnp.exp(params.c2 / (0.1e1 - t110))
  t116 = jnp.where(t84 <= 0.25e1, 0.1e1 - 0.667 * t95 - 0.4445555 * t97 - 0.663086601049 * t99 + 0.145129704449e1 * t101 - 0.887998041597 * t101 * t95 + 0.234528941479 * t101 * t97 - 0.23185843322e-1 * t101 * t99, -params.d * t114)
  t117 = jnp.where(t84 <= 0., t92, t116)
  t121 = t30 * t31
  t124 = 0.1e1 - t84
  t125 = t124 ** 2
  t130 = (0.40570770199022687793e-1 - 0.30235468026081006357 * params.eta) * t31 * t58
  t136 = (0.3e1 / 0.4e1 * params.eta + 0.2e1 / 0.3e1) ** 2
  t141 = (0.290700106132790123e-2 - 0.27123702538979 * params.eta) ** 2
  t145 = (0.146e3 / 0.2025e4 * t136 - 0.73e2 / 0.54e3 * params.eta - 0.146e3 / 0.1215e4 + t141 / params.k1) * t32
  t151 = t76 ** 2
  t153 = t82 ** 2
  t155 = t151 ** 2
  t156 = t153 ** 2
  t162 = params.da4 ** 2
  t163 = 0.1e1 / t162
  t165 = params.dp4 ** 2
  t166 = t165 ** 2
  t167 = 0.1e1 / t166
  t172 = jnp.exp(-t125 * t163 - t37 * t45 * t167 / 0.576e3)
  t178 = jnp.sqrt(0.3e1)
  t180 = t32 / t34
  t181 = jnp.sqrt(s0)
  t186 = jnp.sqrt(t180 * t181 / t42 / r0)
  t190 = jnp.exp(-0.98958e1 * t178 / t186)
  t195 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * (0.1e1 + t70 + t117 * (0.174 - t70) + 0.2e1 * (-0.162742215233874 + 0.162742215233874 * t84 + 0.678092563474475e-2 * t121 * t63 - 0.59353125082804e-1 * t125 + t130 * t74 * t124 / 0.24e2 + t145 * t36 * t38 * t44 / 0.576e3) * t151 / t153 / (0.1e1 + t155 / t156) * t172) * (0.1e1 - t190))
  t197 = jnp.where(t10, t15, -t17)
  t198 = jnp.where(t14, t11, t197)
  t199 = 0.1e1 + t198
  t201 = jnp.cbrt(t199)
  t203 = jnp.where(t199 <= p.zeta_threshold, t23, t201 * t199)
  t205 = s2 ** 2
  t206 = r1 ** 2
  t207 = t206 ** 2
  t209 = jnp.cbrt(r1)
  t211 = 0.1e1 / t209 / t207 / r1
  t212 = t205 * t211
  t216 = jnp.exp(-t37 * t212 * t48 / 0.576e3)
  t222 = t209 ** 2
  t224 = 0.1e1 / t222 / t206
  t225 = t58 * s2 * t224
  t232 = params.k1 * (0.1e1 - params.k1 / (params.k1 + (-0.162742215233874 * t30 * t216 + 0.1e2 / 0.81e2) * t31 * t225 / 0.24e2))
  t236 = s2 * t224
  t238 = tau1 / t222 / r1 - t236 / 0.8e1
  t242 = t78 + params.eta * s2 * t224 / 0.8e1
  t244 = t238 / t242
  t247 = jnp.where(0. < t244, 0, t244)
  t252 = jnp.exp(-params.c1 * t247 / (0.1e1 - t247))
  t254 = 0.25e1 < t244
  t255 = jnp.where(t254, 0.25e1, t244)
  t257 = t255 ** 2
  t259 = t257 * t255
  t261 = t257 ** 2
  t270 = jnp.where(t254, t244, 0.25e1)
  t274 = jnp.exp(params.c2 / (0.1e1 - t270))
  t276 = jnp.where(t244 <= 0.25e1, 0.1e1 - 0.667 * t255 - 0.4445555 * t257 - 0.663086601049 * t259 + 0.145129704449e1 * t261 - 0.887998041597 * t261 * t255 + 0.234528941479 * t261 * t257 - 0.23185843322e-1 * t261 * t259, -params.d * t274)
  t277 = jnp.where(t244 <= 0., t252, t276)
  t283 = 0.1e1 - t244
  t284 = t283 ** 2
  t294 = t238 ** 2
  t296 = t242 ** 2
  t298 = t294 ** 2
  t299 = t296 ** 2
  t310 = jnp.exp(-t284 * t163 - t37 * t212 * t167 / 0.576e3)
  t316 = jnp.sqrt(s2)
  t321 = jnp.sqrt(t180 * t316 / t209 / r1)
  t325 = jnp.exp(-0.98958e1 * t178 / t321)
  t330 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t203 * t28 * (0.1e1 + t232 + t277 * (0.174 - t232) + 0.2e1 * (-0.162742215233874 + 0.162742215233874 * t244 + 0.678092563474475e-2 * t121 * t225 - 0.59353125082804e-1 * t284 + t130 * t236 * t283 / 0.24e2 + t145 * t36 * t205 * t211 / 0.576e3) * t294 / t296 / (0.1e1 + t298 / t299) * t310) * (0.1e1 - t325))
  res = t195 + t330
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
  t22 = 0.2e2 / 0.27e2 + 0.5e1 / 0.3e1 * params.eta
  t23 = jnp.cbrt(6)
  t24 = t23 ** 2
  t25 = jnp.pi ** 2
  t26 = jnp.cbrt(t25)
  t28 = 0.1e1 / t26 / t25
  t30 = s0 ** 2
  t31 = t24 * t28 * t30
  t32 = jnp.cbrt(2)
  t33 = r0 ** 2
  t34 = t33 ** 2
  t37 = 0.1e1 / t20 / t34 / r0
  t38 = t32 * t37
  t39 = params.dp2 ** 2
  t40 = t39 ** 2
  t45 = jnp.exp(-t31 * t38 / t40 / 0.288e3)
  t50 = t26 ** 2
  t51 = 0.1e1 / t50
  t53 = t32 ** 2
  t54 = s0 * t53
  t55 = t20 ** 2
  t57 = 0.1e1 / t55 / t33
  t58 = t54 * t57
  t65 = params.k1 * (0.1e1 - params.k1 / (params.k1 + (-0.162742215233874 * t22 * t45 + 0.1e2 / 0.81e2) * t23 * t51 * t58 / 0.24e2))
  t71 = tau0 * t53 / t55 / r0 - t58 / 0.8e1
  t78 = 0.3e1 / 0.1e2 * t24 * t50 + params.eta * s0 * t53 * t57 / 0.8e1
  t80 = t71 / t78
  t83 = jnp.where(0. < t80, 0, t80)
  t88 = jnp.exp(-params.c1 * t83 / (0.1e1 - t83))
  t90 = 0.25e1 < t80
  t91 = jnp.where(t90, 0.25e1, t80)
  t93 = t91 ** 2
  t95 = t93 * t91
  t97 = t93 ** 2
  t106 = jnp.where(t90, t80, 0.25e1)
  t110 = jnp.exp(params.c2 / (0.1e1 - t106))
  t112 = jnp.where(t80 <= 0.25e1, 0.1e1 - 0.667 * t91 - 0.4445555 * t93 - 0.663086601049 * t95 + 0.145129704449e1 * t97 - 0.887998041597 * t97 * t91 + 0.234528941479 * t97 * t93 - 0.23185843322e-1 * t97 * t95, -params.d * t110)
  t113 = jnp.where(t80 <= 0., t88, t112)
  t121 = 0.1e1 - t80
  t122 = t121 ** 2
  t134 = (0.3e1 / 0.4e1 * params.eta + 0.2e1 / 0.3e1) ** 2
  t139 = (0.290700106132790123e-2 - 0.27123702538979 * params.eta) ** 2
  t150 = t71 ** 2
  t152 = t78 ** 2
  t154 = t150 ** 2
  t155 = t152 ** 2
  t161 = params.da4 ** 2
  t164 = params.dp4 ** 2
  t165 = t164 ** 2
  t171 = jnp.exp(-t122 / t161 - t31 * t38 / t165 / 0.288e3)
  t177 = jnp.sqrt(0.3e1)
  t180 = jnp.sqrt(s0)
  t186 = jnp.sqrt(t24 / t26 * t180 * t32 / t20 / r0)
  t190 = jnp.exp(-0.98958e1 * t177 / t186)
  t195 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * (0.1e1 + t65 + t113 * (0.174 - t65) + 0.2e1 * (-0.162742215233874 + 0.162742215233874 * t80 + 0.678092563474475e-2 * t22 * t23 * t51 * t58 - 0.59353125082804e-1 * t122 + (0.40570770199022687793e-1 - 0.30235468026081006357 * params.eta) * t23 * t51 * t54 * t57 * t121 / 0.24e2 + (0.146e3 / 0.2025e4 * t134 - 0.73e2 / 0.54e3 * params.eta - 0.146e3 / 0.1215e4 + t139 / params.k1) * t24 * t28 * t30 * t32 * t37 / 0.288e3) * t150 / t152 / (0.1e1 + t154 / t155) * t171) * (0.1e1 - t190))
  res = 0.2e1 * t195
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