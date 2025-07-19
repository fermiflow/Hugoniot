"""Generated from mgga_x_task.mpl."""

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
  t29 = jnp.cbrt(6)
  t30 = jnp.pi ** 2
  t31 = jnp.cbrt(t30)
  t32 = t31 ** 2
  t34 = t29 / t32
  t35 = r0 ** 2
  t36 = jnp.cbrt(r0)
  t37 = t36 ** 2
  t38 = t37 * t35
  t42 = t34 * s0 / t38 / 0.24e2
  t43 = 0. < t42
  t44 = jnp.where(t43, t42, 0)
  t45 = t44 ** (0.1e1 / 0.4e1)
  t48 = jnp.exp(-params.task_c / t45)
  t50 = jnp.where(t43, 0.1e1 - t48, 0)
  t52 = params.task_bnu[0]
  t53 = t35 ** 2
  t55 = t37 * t53 * t35
  t58 = t53 * r0
  t60 = r0 * tau0
  t63 = 0.1e1 / r0
  t65 = 0.1e1 / tau0
  t74 = jnp.where(0. < 0.125 * (0.79999999992e1 * t60 - s0) * t63 * t65, (0.8e1 * t60 - s0) * t63 * t65 / 0.8e1, 0.1e-9)
  t75 = t74 * tau0
  t79 = t36 * t35 * r0
  t81 = t74 ** 2
  t82 = tau0 ** 2
  t83 = t81 * t82
  t86 = t37 * r0
  t90 = t81 * t74 * t82 * tau0
  t93 = t81 ** 2
  t95 = t82 ** 2
  t98 = params.task_bnu[1]
  t110 = params.task_bnu[2]
  t119 = 0.47049607861172565388e8 * t52 * t55 + 0.41291508340807648886e8 * t52 * t58 * t75 + 0.13589289623490307943e8 * t52 * t79 * t83 + 0.19876972814512516632e7 * t52 * t86 * t90 + 0.10902723556992837954e6 * t52 * t93 * t95 - 0.47049607861172565388e8 * t98 * t55 - 0.20645754170403824442e8 * t98 * t58 * t75 + 0.99384864072562583159e6 * t98 * t86 * t90 + 0.10902723556992837954e6 * t98 * t93 * t95 + 0.47049607861172565386e8 * t110 * t55 - 0.41291508340807648885e8 * t110 * t58 * t75 - 0.22648816039150513236e8 * t110 * t79 * t83
  t126 = params.task_bnu[3]
  t141 = params.task_bnu[4]
  t156 = -0.19876972814512516631e7 * t110 * t86 * t90 + 0.10902723556992837954e6 * t110 * t93 * t95 - 0.47049607861172565384e8 * t126 * t55 + 0.14452027919282677111e9 * t126 * t58 * t75 + 0.1e-11 * t126 * t79 * t83 - 0.69569404850793808212e7 * t126 * t86 * t90 + 0.10902723556992837955e6 * t126 * t93 * t95 + 0.4704960786117256539e8 * t141 * t55 - 0.2890405583856535422e9 * t141 * t58 * t75 + 0.15854171227405359266e9 * t141 * t79 * t83 - 0.13913880970158761643e8 * t141 * t86 * t90 + 0.10902723556992837954e6 * t141 * t93 * t95
  t161 = (0.828207200604688193e2 * t86 + 0.18171205928321396589e2 * t75) ** 2
  t162 = t161 ** 2
  t167 = t3 ** 2
  t168 = t167 * t30
  t169 = t36 * t58 * t168
  t170 = params.task_anu[0]
  t173 = params.task_anu[1]
  t176 = params.task_anu[2]
  t179 = t38 * t29
  t180 = t3 * jnp.pi
  t181 = t180 * s0
  t188 = t29 ** 2
  t189 = s0 ** 2
  t190 = t188 * t189
  t199 = (t29 * s0 + 0.24e2 * t180 * t38) ** 2
  t204 = t50 ** params.task_d
  t210 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (params.task_h0x * t50 + (0.1e1 - (t119 + t156) / t162) * ((0.48e2 * t179 * t181 * t170 - 0.144e3 * t179 * t181 * t176 + 0.576e3 * t169 * t170 - 0.576e3 * t169 * t173 + 0.576e3 * t169 * t176 + t190 * t170 + t190 * t173 + t190 * t176) / t199 - params.task_h0x) * t204))
  t212 = jnp.where(t10, t15, -t17)
  t213 = jnp.where(t14, t11, t212)
  t214 = t213 + 0.1e1
  t216 = jnp.cbrt(t214)
  t218 = jnp.where(t214 <= p.zeta_threshold, t23, t216 * t214)
  t220 = r1 ** 2
  t221 = jnp.cbrt(r1)
  t222 = t221 ** 2
  t223 = t222 * t220
  t227 = t34 * s2 / t223 / 0.24e2
  t228 = 0. < t227
  t229 = jnp.where(t228, t227, 0)
  t230 = t229 ** (0.1e1 / 0.4e1)
  t233 = jnp.exp(-params.task_c / t230)
  t235 = jnp.where(t228, 0.1e1 - t233, 0)
  t237 = t220 ** 2
  t239 = t222 * t237 * t220
  t242 = t237 * r1
  t244 = r1 * tau1
  t247 = 0.1e1 / r1
  t249 = 0.1e1 / tau1
  t258 = jnp.where(0. < 0.125 * (0.79999999992e1 * t244 - s2) * t247 * t249, (0.8e1 * t244 - s2) * t247 * t249 / 0.8e1, 0.1e-9)
  t259 = tau1 * t258
  t263 = t221 * t220 * r1
  t265 = tau1 ** 2
  t266 = t258 ** 2
  t267 = t265 * t266
  t270 = t222 * r1
  t274 = t265 * tau1 * t266 * t258
  t277 = t265 ** 2
  t279 = t266 ** 2
  t301 = 0.47049607861172565388e8 * t52 * t239 + 0.41291508340807648886e8 * t52 * t242 * t259 + 0.13589289623490307943e8 * t52 * t263 * t267 + 0.19876972814512516632e7 * t52 * t270 * t274 + 0.10902723556992837954e6 * t52 * t277 * t279 - 0.47049607861172565388e8 * t98 * t239 - 0.20645754170403824442e8 * t98 * t242 * t259 + 0.99384864072562583159e6 * t98 * t270 * t274 + 0.10902723556992837954e6 * t98 * t277 * t279 + 0.47049607861172565386e8 * t110 * t239 - 0.41291508340807648885e8 * t110 * t242 * t259 - 0.22648816039150513236e8 * t110 * t263 * t267
  t336 = -0.19876972814512516631e7 * t110 * t270 * t274 + 0.10902723556992837954e6 * t110 * t277 * t279 - 0.47049607861172565384e8 * t126 * t239 + 0.14452027919282677111e9 * t126 * t242 * t259 + 0.1e-11 * t126 * t263 * t267 - 0.69569404850793808212e7 * t126 * t270 * t274 + 0.10902723556992837955e6 * t126 * t277 * t279 + 0.4704960786117256539e8 * t141 * t239 - 0.2890405583856535422e9 * t141 * t242 * t259 + 0.15854171227405359266e9 * t141 * t263 * t267 - 0.13913880970158761643e8 * t141 * t270 * t274 + 0.10902723556992837954e6 * t141 * t277 * t279
  t341 = (0.828207200604688193e2 * t270 + 0.18171205928321396589e2 * t259) ** 2
  t342 = t341 ** 2
  t347 = t221 * t242 * t168
  t354 = t223 * t29
  t355 = t180 * s2
  t362 = s2 ** 2
  t363 = t188 * t362
  t372 = (t29 * s2 + 0.24e2 * t180 * t223) ** 2
  t377 = t235 ** params.task_d
  t383 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t218 * t27 * (params.task_h0x * t235 + (0.1e1 - (t301 + t336) / t342) * ((0.48e2 * t354 * t355 * t170 - 0.144e3 * t354 * t355 * t176 + 0.576e3 * t347 * t170 + t363 * t170 - 0.576e3 * t347 * t173 + t363 * t173 + 0.576e3 * t347 * t176 + t363 * t176) / t372 - params.task_h0x) * t377))
  res = t210 + t383
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
  t21 = jnp.cbrt(6)
  t22 = jnp.pi ** 2
  t23 = jnp.cbrt(t22)
  t24 = t23 ** 2
  t27 = jnp.cbrt(2)
  t28 = t27 ** 2
  t30 = r0 ** 2
  t31 = t19 ** 2
  t32 = t31 * t30
  t36 = t21 / t24 * s0 * t28 / t32 / 0.24e2
  t37 = 0. < t36
  t38 = jnp.where(t37, t36, 0)
  t39 = t38 ** (0.1e1 / 0.4e1)
  t42 = jnp.exp(-params.task_c / t39)
  t44 = jnp.where(t37, 0.1e1 - t42, 0)
  t46 = params.task_bnu[0]
  t47 = t30 ** 2
  t49 = t31 * t47 * t30
  t52 = t47 * r0
  t54 = r0 * tau0
  t57 = 0.1e1 / r0
  t59 = 0.1e1 / tau0
  t68 = jnp.where(0. < 0.125 * (0.79999999992e1 * t54 - s0) * t57 * t59, (0.8e1 * t54 - s0) * t57 * t59 / 0.8e1, 0.1e-9)
  t69 = tau0 * t68
  t73 = t19 * t30 * r0
  t75 = tau0 ** 2
  t76 = t68 ** 2
  t77 = t75 * t76
  t80 = t31 * r0
  t84 = t75 * tau0 * t76 * t68
  t87 = t75 ** 2
  t89 = t76 ** 2
  t92 = params.task_bnu[1]
  t104 = params.task_bnu[2]
  t113 = 0.47049607861172565388e8 * t46 * t49 + 0.65546183777551744717e8 * t46 * t52 * t69 + 0.34242864099506828874e8 * t46 * t73 * t77 + 0.79507891258050066525e7 * t46 * t80 * t84 + 0.69227979374755602348e6 * t46 * t87 * t89 - 0.47049607861172565388e8 * t92 * t49 - 0.32773091888775872359e8 * t92 * t52 * t69 + 0.39753945629025033263e7 * t92 * t80 * t84 + 0.69227979374755602348e6 * t92 * t87 * t89 + 0.47049607861172565386e8 * t104 * t49 - 0.65546183777551744717e8 * t104 * t52 * t69 - 0.57071440165844714789e8 * t104 * t73 * t77
  t120 = params.task_bnu[3]
  t135 = params.task_bnu[4]
  t150 = -0.79507891258050066526e7 * t104 * t80 * t84 + 0.69227979374755602349e6 * t104 * t87 * t89 - 0.47049607861172565384e8 * t120 * t49 + 0.22941164322143110651e9 * t120 * t52 * t69 - 0.4e-11 * t120 * t73 * t77 - 0.27827761940317523285e8 * t120 * t80 * t84 + 0.69227979374755602348e6 * t120 * t87 * t89 + 0.4704960786117256539e8 * t135 * t49 - 0.45882328644286221302e9 * t135 * t52 * t69 + 0.39950008116091300353e9 * t135 * t73 * t77 - 0.55655523880635046568e8 * t135 * t80 * t84 + 0.6922797937475560235e6 * t135 * t87 * t89
  t155 = (0.828207200604688193e2 * t80 + 0.28844991406148167646e2 * t69) ** 2
  t156 = t155 ** 2
  t161 = t4 ** 2
  t163 = t19 * t52 * t161 * t22
  t164 = params.task_anu[0]
  t167 = params.task_anu[1]
  t170 = params.task_anu[2]
  t173 = t3 * t32
  t174 = t4 * jnp.pi
  t175 = t174 * s0
  t182 = t3 ** 2
  t183 = s0 ** 2
  t184 = t182 * t183
  t193 = (t3 * s0 + 0.12e2 * t174 * t32) ** 2
  t198 = t44 ** params.task_d
  t204 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (params.task_h0x * t44 + (0.1e1 - (t113 + t150) / t156) * ((0.24e2 * t173 * t175 * t164 - 0.72e2 * t173 * t175 * t170 + 0.144e3 * t163 * t164 - 0.144e3 * t163 * t167 + 0.144e3 * t163 * t170 + t184 * t164 + t184 * t167 + t184 * t170) / t193 - params.task_h0x) * t198))
  res = 0.2e1 * t204
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