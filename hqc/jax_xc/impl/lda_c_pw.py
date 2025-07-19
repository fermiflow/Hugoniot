"""Generated from lda_c_pw.mpl."""

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
  t1 = params.a[0]
  t3 = jnp.cbrt(3)
  t6 = jnp.cbrt(0.1e1 / jnp.pi)
  t7 = jnp.cbrt(4)
  t8 = t7 ** 2
  t10 = r0 + r1
  t11 = jnp.cbrt(t10)
  t12 = 0.1e1 / t11
  t13 = t6 * t8 * t12
  t22 = t3 * t6 * t8 * t12
  t23 = jnp.sqrt(t22)
  t31 = t22 ** 0.15e1
  t35 = t22 / 0.4e1
  t38 = t35 ** (params.pp[0] + 0.1e1)
  t45 = jnp.log(0.1e1 + 0.1e1 / t1 / (params.beta1[0] * t23 / 0.2e1 + params.beta2[0] * t3 * t13 / 0.4e1 + 0.125 * params.beta3[0] * t31 + params.beta4[0] * t38) / 0.2e1)
  t46 = t1 * (0.1e1 + params.alpha1[0] * t3 * t13 / 0.4e1) * t45
  t48 = r0 - r1
  t49 = t48 ** 2
  t50 = t49 ** 2
  t51 = t10 ** 2
  t52 = t51 ** 2
  t56 = t48 / t10
  t57 = 0.1e1 + t56
  t59 = jnp.cbrt(p.zeta_threshold)
  t60 = t59 * p.zeta_threshold
  t61 = jnp.cbrt(t57)
  t63 = jnp.where(t57 <= p.zeta_threshold, t60, t61 * t57)
  t64 = 0.1e1 - t56
  t66 = jnp.cbrt(t64)
  t68 = jnp.where(t64 <= p.zeta_threshold, t60, t66 * t64)
  t70 = jnp.cbrt(2)
  t74 = (t63 + t68 - 0.2e1) / (0.2e1 * t70 - 0.2e1)
  t75 = params.a[1]
  t96 = t35 ** (params.pp[1] + 0.1e1)
  t103 = jnp.log(0.1e1 + 0.1e1 / t75 / (params.beta1[1] * t23 / 0.2e1 + params.beta2[1] * t3 * t13 / 0.4e1 + 0.125 * params.beta3[1] * t31 + params.beta4[1] * t96) / 0.2e1)
  t105 = params.a[2]
  t110 = 0.1e1 + params.alpha1[2] * t3 * t13 / 0.4e1
  t126 = t35 ** (params.pp[2] + 0.1e1)
  t133 = jnp.log(0.1e1 + 0.1e1 / t105 / (params.beta1[2] * t23 / 0.2e1 + params.beta2[2] * t3 * t13 / 0.4e1 + 0.125 * params.beta3[2] * t31 + params.beta4[2] * t126) / 0.2e1)
  t134 = 0.1e1 / params.fz20
  res = -0.2e1 * t46 + t50 / t52 * t74 * (-0.2e1 * t75 * (0.1e1 + params.alpha1[1] * t3 * t13 / 0.4e1) * t103 + 0.2e1 * t46 - 0.2e1 * t105 * t110 * t133 * t134) + 0.2e1 * t74 * t105 * t110 * t133 * t134
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = params.a[0]
  t3 = jnp.cbrt(3)
  t6 = jnp.cbrt(0.1e1 / jnp.pi)
  t7 = jnp.cbrt(4)
  t8 = t7 ** 2
  t10 = jnp.cbrt(r0)
  t11 = 0.1e1 / t10
  t12 = t6 * t8 * t11
  t21 = t3 * t6 * t8 * t11
  t22 = jnp.sqrt(t21)
  t30 = t21 ** 0.15e1
  t34 = t21 / 0.4e1
  t37 = t34 ** (params.pp[0] + 0.1e1)
  t44 = jnp.log(0.1e1 + 0.1e1 / t1 / (params.beta1[0] * t22 / 0.2e1 + params.beta2[0] * t3 * t12 / 0.4e1 + 0.125 * params.beta3[0] * t30 + params.beta4[0] * t37) / 0.2e1)
  t47 = jnp.cbrt(p.zeta_threshold)
  t49 = jnp.where(0.1e1 <= p.zeta_threshold, t47 * p.zeta_threshold, 1)
  t52 = jnp.cbrt(2)
  t57 = params.a[2]
  t78 = t34 ** (params.pp[2] + 0.1e1)
  t85 = jnp.log(0.1e1 + 0.1e1 / t57 / (params.beta1[2] * t22 / 0.2e1 + params.beta2[2] * t3 * t12 / 0.4e1 + 0.125 * params.beta3[2] * t30 + params.beta4[2] * t78) / 0.2e1)
  res = -0.2e1 * t1 * (0.1e1 + params.alpha1[0] * t3 * t12 / 0.4e1) * t44 + 0.2e1 * (0.2e1 * t49 - 0.2e1) / (0.2e1 * t52 - 0.2e1) * t57 * (0.1e1 + params.alpha1[2] * t3 * t12 / 0.4e1) * t85 / params.fz20
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