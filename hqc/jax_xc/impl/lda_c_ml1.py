"""Generated from lda_c_ml1.mpl."""

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
  t1 = r0 + r1
  t2 = r0 - r1
  t4 = t2 / t1
  t5 = jnp.abs(t4)
  t8 = t2 ** 2
  t9 = t1 ** 2
  t13 = jnp.cbrt(t1)
  t17 = p.zeta_threshold - 0.1e1
  t21 = jnp.where(0.1e1 - t4 <= p.zeta_threshold, -t17, t4)
  t22 = jnp.where(0.1e1 + t4 <= p.zeta_threshold, t17, t21)
  t23 = 0.1e1 + t22
  t24 = t23 ** params.q
  t25 = 0.1e1 - t22
  t26 = t25 ** params.q
  t27 = t24 + t26
  t28 = t22 ** 2
  t30 = jnp.cbrt(0.1e1 - t28)
  t32 = jnp.cbrt(t23)
  t33 = jnp.cbrt(t25)
  t34 = t32 + t33
  t42 = 0.1e1 / t13
  t43 = 0.1e1 / params.fc
  t48 = 0.1e1 / t27 / t30 * t34
  t49 = t42 * t43 * t48
  t52 = jnp.log(0.1e1 + 0.91959623973811018799e-1 * t49)
  t58 = t13 ** 2
  t60 = params.fc ** 2
  t63 = t27 ** 2
  t65 = t30 ** 2
  t68 = t34 ** 2
  t75 = jnp.where(0.1e1 - t5 <= p.zeta_threshold, 0, (0.1e1 - t8 / t9) * (-0.2763169e1 / (0.1e1 + 0.10874334072525e2 * t13 * params.fc * t27 * t30 / t34) + 0.28144540420067767464 * t52 * t42 * t43 * t48 + 0.25410002852601321894 * t49 - 0.49248579417833934399e-1 / t58 / t60 / t63 / t65 * t68) / 0.4e1)
  res = t1 * t75
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = 0.1e1 <= p.zeta_threshold
  t2 = jnp.cbrt(r0)
  t4 = p.zeta_threshold - 0.1e1
  t6 = jnp.where(t1, -t4, 0)
  t7 = jnp.where(t1, t4, t6)
  t8 = 0.1e1 + t7
  t9 = t8 ** params.q
  t10 = 0.1e1 - t7
  t11 = t10 ** params.q
  t12 = t9 + t11
  t13 = t7 ** 2
  t15 = jnp.cbrt(0.1e1 - t13)
  t17 = jnp.cbrt(t8)
  t18 = jnp.cbrt(t10)
  t19 = t17 + t18
  t27 = 0.1e1 / t2
  t28 = 0.1e1 / params.fc
  t33 = 0.1e1 / t12 / t15 * t19
  t34 = t27 * t28 * t33
  t37 = jnp.log(0.1e1 + 0.91959623973811018799e-1 * t34)
  t43 = t2 ** 2
  t45 = params.fc ** 2
  t48 = t12 ** 2
  t50 = t15 ** 2
  t53 = t19 ** 2
  t58 = jnp.where(t1, 0, -0.69079225 / (0.1e1 + 0.10874334072525e2 * t2 * params.fc * t12 * t15 / t19) + 0.7036135105016941866e-1 * t37 * t27 * t28 * t33 + 0.63525007131503304735e-1 * t34 - 0.123121448544584836e-1 / t43 / t45 / t48 / t50 * t53)
  res = r0 * t58
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