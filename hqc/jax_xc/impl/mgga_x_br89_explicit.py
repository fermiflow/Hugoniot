"""Generated from mgga_x_br89_explicit.mpl."""

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
  t2 = r0 + r1
  t3 = 0.1e1 / t2
  t6 = 0.2e1 * r0 * t3 <= p.zeta_threshold
  t7 = p.zeta_threshold - 0.1e1
  t10 = 0.2e1 * r1 * t3 <= p.zeta_threshold
  t11 = -t7
  t13 = (r0 - r1) * t3
  t14 = jnp.where(t10, t11, t13)
  t15 = jnp.where(t6, t7, t14)
  t16 = 0.1e1 + t15
  t18 = jnp.cbrt(p.zeta_threshold)
  t19 = t18 * p.zeta_threshold
  t20 = jnp.cbrt(t16)
  t22 = jnp.where(t16 <= p.zeta_threshold, t19, t20 * t16)
  t23 = jnp.cbrt(t2)
  t26 = jnp.cbrt(0.1e1 / jnp.pi)
  t27 = 0.1e1 / t26
  t29 = jnp.cbrt(4)
  t30 = jnp.cbrt(jnp.pi)
  t31 = t30 ** 2
  t32 = jnp.cbrt(r0)
  t33 = t32 ** 2
  t35 = 0.1e1 / t33 / r0
  t42 = r0 ** 2
  t47 = l0 * t35 / 0.6e1 - 0.2e1 / 0.3e1 * params.gamma * tau0 * t35 + params.gamma * s0 / t33 / t42 / 0.12e2
  t48 = jnp.abs(t47)
  t51 = jnp.where(0. < t47, 0.5e-12, -0.5e-12)
  t52 = jnp.where(t48 < 0.5e-12, t51, t47)
  t55 = 0.2e1 / 0.3e1 * t31 / t52
  t58 = jnp.where(-0.5e-12 < t55, -0.5e-12, t55)
  t61 = jnp.arctan(0.1525525181200953e1 * t58 + 0.4576575543602858)
  t64 = t58 ** 2
  t66 = t64 * t58
  t68 = t64 ** 2
  t70 = t68 * t58
  t83 = jnp.where(0.5e-12 < t55, t55, 0.5e-12)
  t85 = jnp.arcsinh(0.1e1 / (0.2085749716493756e1 * t83))
  t88 = t83 ** 2
  t90 = t88 * t83
  t92 = t88 ** 2
  t94 = t92 * t83
  t106 = jnp.where(t55 <= 0., (-t61 + 0.4292036732051034) * (0.7566445420735584 - 0.2636397787137096e1 * t58 + 0.5474515996423288e1 * t64 - 0.1265730812710829e2 * t66 + 0.4125058472512136e1 * t68 - 0.3042513395716384e2 * t70) / (0.4771976183772063 - 0.1779981349455627e1 * t58 + 0.3843384186230215e1 * t64 - 0.9591205088051849e1 * t66 + 0.2173018028591672e1 * t68 - 0.3042513385160366e2 * t70), (t85 + 0.2e1) * (0.4435009886795587e-4 + 0.5812865360445791 * t83 + 0.6674276451594061e2 * t88 + 0.4342678089722977e3 * t90 + 0.8247765766052239e3 * t92 + 0.1657965273158212e4 * t94) / (0.3347285060926091e-4 + 0.4791793102397135 * t83 + 0.6239226833857424e2 * t88 + 0.4631481642793812e3 * t90 + 0.7852360350104029e3 * t92 + 0.1657962968223273e4 * t94))
  t108 = jnp.exp(t106 / 0.3e1)
  t110 = jnp.exp(-t106)
  t120 = jnp.where(r0 <= p.dens_threshold, 0, -t22 * t23 * t27 * t29 * t108 * (0.1e1 - t110 * (0.1e1 + t106 / 0.2e1)) / t106 / 0.4e1)
  t122 = jnp.where(t6, t11, -t13)
  t123 = jnp.where(t10, t7, t122)
  t124 = 0.1e1 + t123
  t126 = jnp.cbrt(t124)
  t128 = jnp.where(t124 <= p.zeta_threshold, t19, t126 * t124)
  t131 = jnp.cbrt(r1)
  t132 = t131 ** 2
  t134 = 0.1e1 / t132 / r1
  t141 = r1 ** 2
  t146 = l1 * t134 / 0.6e1 - 0.2e1 / 0.3e1 * params.gamma * tau1 * t134 + params.gamma * s2 / t132 / t141 / 0.12e2
  t147 = jnp.abs(t146)
  t150 = jnp.where(0. < t146, 0.5e-12, -0.5e-12)
  t151 = jnp.where(t147 < 0.5e-12, t150, t146)
  t154 = 0.2e1 / 0.3e1 * t31 / t151
  t157 = jnp.where(-0.5e-12 < t154, -0.5e-12, t154)
  t160 = jnp.arctan(0.1525525181200953e1 * t157 + 0.4576575543602858)
  t163 = t157 ** 2
  t165 = t163 * t157
  t167 = t163 ** 2
  t169 = t167 * t157
  t182 = jnp.where(0.5e-12 < t154, t154, 0.5e-12)
  t184 = jnp.arcsinh(0.1e1 / (0.2085749716493756e1 * t182))
  t187 = t182 ** 2
  t189 = t187 * t182
  t191 = t187 ** 2
  t193 = t191 * t182
  t205 = jnp.where(t154 <= 0., (-t160 + 0.4292036732051034) * (0.7566445420735584 - 0.2636397787137096e1 * t157 + 0.5474515996423288e1 * t163 - 0.1265730812710829e2 * t165 + 0.4125058472512136e1 * t167 - 0.3042513395716384e2 * t169) / (0.4771976183772063 - 0.1779981349455627e1 * t157 + 0.3843384186230215e1 * t163 - 0.9591205088051849e1 * t165 + 0.2173018028591672e1 * t167 - 0.3042513385160366e2 * t169), (t184 + 0.2e1) * (0.4435009886795587e-4 + 0.5812865360445791 * t182 + 0.6674276451594061e2 * t187 + 0.4342678089722977e3 * t189 + 0.8247765766052239e3 * t191 + 0.1657965273158212e4 * t193) / (0.3347285060926091e-4 + 0.4791793102397135 * t182 + 0.6239226833857424e2 * t187 + 0.4631481642793812e3 * t189 + 0.7852360350104029e3 * t191 + 0.1657962968223273e4 * t193))
  t207 = jnp.exp(t205 / 0.3e1)
  t209 = jnp.exp(-t205)
  t219 = jnp.where(r1 <= p.dens_threshold, 0, -t128 * t23 * t27 * t29 * t207 * (0.1e1 - t209 * (0.1e1 + t205 / 0.2e1)) / t205 / 0.4e1)
  res = t120 + t219
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t3 = 0.1e1 <= p.zeta_threshold
  t4 = p.zeta_threshold - 0.1e1
  t6 = jnp.where(t3, -t4, 0)
  t7 = jnp.where(t3, t4, t6)
  t8 = 0.1e1 + t7
  t10 = jnp.cbrt(p.zeta_threshold)
  t12 = jnp.cbrt(t8)
  t14 = jnp.where(t8 <= p.zeta_threshold, t10 * p.zeta_threshold, t12 * t8)
  t15 = jnp.cbrt(r0)
  t18 = jnp.cbrt(0.1e1 / jnp.pi)
  t21 = jnp.cbrt(4)
  t22 = jnp.cbrt(jnp.pi)
  t23 = t22 ** 2
  t24 = jnp.cbrt(2)
  t25 = t24 ** 2
  t27 = t15 ** 2
  t29 = 0.1e1 / t27 / r0
  t37 = r0 ** 2
  t43 = l0 * t25 * t29 / 0.6e1 - 0.2e1 / 0.3e1 * params.gamma * tau0 * t25 * t29 + params.gamma * s0 * t25 / t27 / t37 / 0.12e2
  t44 = jnp.abs(t43)
  t47 = jnp.where(0. < t43, 0.5e-12, -0.5e-12)
  t48 = jnp.where(t44 < 0.5e-12, t47, t43)
  t51 = 0.2e1 / 0.3e1 * t23 / t48
  t54 = jnp.where(-0.5e-12 < t51, -0.5e-12, t51)
  t57 = jnp.arctan(0.1525525181200953e1 * t54 + 0.4576575543602858)
  t60 = t54 ** 2
  t62 = t60 * t54
  t64 = t60 ** 2
  t66 = t64 * t54
  t79 = jnp.where(0.5e-12 < t51, t51, 0.5e-12)
  t81 = jnp.arcsinh(0.1e1 / (0.2085749716493756e1 * t79))
  t84 = t79 ** 2
  t86 = t84 * t79
  t88 = t84 ** 2
  t90 = t88 * t79
  t102 = jnp.where(t51 <= 0., (-t57 + 0.4292036732051034) * (0.7566445420735584 - 0.2636397787137096e1 * t54 + 0.5474515996423288e1 * t60 - 0.1265730812710829e2 * t62 + 0.4125058472512136e1 * t64 - 0.3042513395716384e2 * t66) / (0.4771976183772063 - 0.1779981349455627e1 * t54 + 0.3843384186230215e1 * t60 - 0.9591205088051849e1 * t62 + 0.2173018028591672e1 * t64 - 0.3042513385160366e2 * t66), (t81 + 0.2e1) * (0.4435009886795587e-4 + 0.5812865360445791 * t79 + 0.6674276451594061e2 * t84 + 0.4342678089722977e3 * t86 + 0.8247765766052239e3 * t88 + 0.1657965273158212e4 * t90) / (0.3347285060926091e-4 + 0.4791793102397135 * t79 + 0.6239226833857424e2 * t84 + 0.4631481642793812e3 * t86 + 0.7852360350104029e3 * t88 + 0.1657962968223273e4 * t90))
  t104 = jnp.exp(t102 / 0.3e1)
  t106 = jnp.exp(-t102)
  t116 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -t14 * t15 / t18 * t21 * t104 * (0.1e1 - t106 * (0.1e1 + t102 / 0.2e1)) / t102 / 0.4e1)
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