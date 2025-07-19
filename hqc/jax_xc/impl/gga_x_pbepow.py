"""Generated from gga_x_pbepow.mpl."""

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
  t29 = jnp.cbrt(6)
  t30 = jnp.pi ** 2
  t31 = jnp.cbrt(t30)
  t32 = t31 ** 2
  t34 = t29 / t32
  t35 = r0 ** 2
  t36 = jnp.cbrt(r0)
  t37 = t36 ** 2
  t40 = s0 / t37 / t35
  t47 = (t34 * t40 / (0.91464571985215458336 * t34 * t40 + 0.804)) ** 0.1e3
  t57 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 - 0.91464571985215458336e-2 * t34 * t40 * (0.13344141567995010044e-3 * t47 - 0.1e1)))
  t59 = jnp.where(t10, t15, -t17)
  t60 = jnp.where(t14, t11, t59)
  t61 = 0.1e1 + t60
  t63 = jnp.cbrt(t61)
  t65 = jnp.where(t61 <= p.zeta_threshold, t23, t63 * t61)
  t67 = r1 ** 2
  t68 = jnp.cbrt(r1)
  t69 = t68 ** 2
  t72 = s2 / t69 / t67
  t79 = (t34 * t72 / (0.91464571985215458336 * t34 * t72 + 0.804)) ** 0.1e3
  t89 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t65 * t27 * (0.1e1 - 0.91464571985215458336e-2 * t34 * t72 * (0.13344141567995010044e-3 * t79 - 0.1e1)))
  res = t57 + t89
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
  t21 = jnp.cbrt(6)
  t22 = jnp.pi ** 2
  t23 = jnp.cbrt(t22)
  t24 = t23 ** 2
  t26 = t21 / t24
  t27 = t26 * s0
  t28 = jnp.cbrt(2)
  t29 = t28 ** 2
  t30 = r0 ** 2
  t31 = t19 ** 2
  t33 = 0.1e1 / t31 / t30
  t34 = t29 * t33
  t43 = (t27 * t34 / (0.91464571985215458336 * t26 * s0 * t29 * t33 + 0.804)) ** 0.1e3
  t53 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 - 0.91464571985215458336e-2 * t27 * t34 * (0.13344141567995010044e-3 * t43 - 0.1e1)))
  res = 0.2e1 * t53
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