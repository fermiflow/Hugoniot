"""Generated from gga_c_p86.mpl."""

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
  t2 = 0.1e1 / jnp.pi
  t3 = jnp.cbrt(t2)
  t4 = t1 * t3
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = r0 + r1
  t8 = jnp.cbrt(t7)
  t9 = 0.1e1 / t8
  t10 = t6 * t9
  t11 = t4 * t10
  t12 = t11 / 0.4e1
  t13 = 0.1e1 <= t12
  t14 = jnp.sqrt(t11)
  t20 = jnp.log(t12)
  t23 = t4 * t10 * t20
  t27 = jnp.where(t13, -0.1423 / (0.1e1 + 0.52645 * t14 + 0.8335e-1 * t11), 0.311e-1 * t20 - 0.48e-1 + 0.5e-3 * t23 - 0.29e-2 * t11)
  t37 = jnp.where(t13, -0.843e-1 / (0.1e1 + 0.69905 * t14 + 0.65275e-1 * t11), 0.1555e-1 * t20 - 0.269e-1 + 0.175e-3 * t23 - 0.12e-2 * t11)
  t40 = 0.1e1 / t7
  t41 = (r0 - r1) * t40
  t42 = 0.1e1 + t41
  t43 = t42 <= p.zeta_threshold
  t44 = jnp.cbrt(p.zeta_threshold)
  t45 = t44 * p.zeta_threshold
  t46 = jnp.cbrt(t42)
  t48 = jnp.where(t43, t45, t46 * t42)
  t49 = 0.1e1 - t41
  t50 = t49 <= p.zeta_threshold
  t51 = jnp.cbrt(t49)
  t53 = jnp.where(t50, t45, t51 * t49)
  t56 = jnp.cbrt(2)
  t62 = s0 + 0.2e1 * s1 + s2
  t63 = t7 ** 2
  t71 = t3 * t6 * t9
  t74 = t1 ** 2
  t76 = t3 ** 2
  t78 = t8 ** 2
  t80 = t76 * t5 / t78
  t96 = params.aa + (params.bb + params.malpha * t1 * t71 / 0.4e1 + params.mbeta * t74 * t80 / 0.4e1) / (0.1e1 + params.mgamma * t1 * t71 / 0.4e1 + params.mdelta * t74 * t80 / 0.4e1 + 0.75e4 * params.mbeta * t2 * t40)
  t98 = jnp.sqrt(t62)
  t100 = t7 ** (0.1e1 / 0.6e1)
  t105 = jnp.exp(-params.ftilde * (params.aa + params.bb) / t96 * t98 / t100 / t7)
  t107 = t44 ** 2
  t108 = t107 * p.zeta_threshold
  t109 = t46 ** 2
  t111 = jnp.where(t43, t108, t109 * t42)
  t112 = t51 ** 2
  t114 = jnp.where(t50, t108, t112 * t49)
  t116 = jnp.sqrt(t111 + t114)
  t119 = jnp.sqrt(0.2e1)
  res = t27 + (t37 - t27) * (t48 + t53 - 0.2e1) / (0.2e1 * t56 - 0.2e1) + t62 / t8 / t63 * t105 * t96 / t116 * t119
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(3)
  t2 = 0.1e1 / jnp.pi
  t3 = jnp.cbrt(t2)
  t4 = t1 * t3
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = jnp.cbrt(r0)
  t8 = 0.1e1 / t7
  t9 = t6 * t8
  t10 = t4 * t9
  t11 = t10 / 0.4e1
  t12 = 0.1e1 <= t11
  t13 = jnp.sqrt(t10)
  t19 = jnp.log(t11)
  t22 = t4 * t9 * t19
  t26 = jnp.where(t12, -0.1423 / (0.1e1 + 0.52645 * t13 + 0.8335e-1 * t10), 0.311e-1 * t19 - 0.48e-1 + 0.5e-3 * t22 - 0.29e-2 * t10)
  t36 = jnp.where(t12, -0.843e-1 / (0.1e1 + 0.69905 * t13 + 0.65275e-1 * t10), 0.1555e-1 * t19 - 0.269e-1 + 0.175e-3 * t22 - 0.12e-2 * t10)
  t38 = 0.1e1 <= p.zeta_threshold
  t39 = jnp.cbrt(p.zeta_threshold)
  t41 = jnp.where(t38, t39 * p.zeta_threshold, 1)
  t45 = jnp.cbrt(2)
  t50 = r0 ** 2
  t58 = t3 * t6 * t8
  t61 = t1 ** 2
  t63 = t3 ** 2
  t65 = t7 ** 2
  t67 = t63 * t5 / t65
  t84 = params.aa + (params.bb + params.malpha * t1 * t58 / 0.4e1 + params.mbeta * t61 * t67 / 0.4e1) / (0.1e1 + params.mgamma * t1 * t58 / 0.4e1 + params.mdelta * t61 * t67 / 0.4e1 + 0.75e4 * params.mbeta * t2 / r0)
  t86 = jnp.sqrt(s0)
  t88 = r0 ** (0.1e1 / 0.6e1)
  t93 = jnp.exp(-params.ftilde * (params.aa + params.bb) / t84 * t86 / t88 / r0)
  t95 = t39 ** 2
  t97 = jnp.where(t38, t95 * p.zeta_threshold, 1)
  t98 = jnp.sqrt(t97)
  res = t26 + (t36 - t26) * (0.2e1 * t41 - 0.2e1) / (0.2e1 * t45 - 0.2e1) + s0 / t7 / t50 * t93 * t84 / t98
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