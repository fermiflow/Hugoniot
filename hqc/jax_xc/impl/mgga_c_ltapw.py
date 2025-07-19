"""Generated from mgga_c_ltapw.mpl."""

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
  t3 = jnp.cbrt(0.1e1 / jnp.pi)
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = jnp.cbrt(r0)
  t8 = t7 ** 2
  t12 = jnp.cbrt(6)
  t13 = jnp.pi ** 2
  t14 = jnp.cbrt(t13)
  t15 = t14 ** 2
  t17 = t12 / t15
  t20 = 0.3e1 / 0.5e1 * params.ltafrac
  t21 = (0.5e1 / 0.9e1 * tau0 / t8 / r0 * t17) ** t20
  t22 = r0 * t21
  t23 = jnp.cbrt(r1)
  t24 = t23 ** 2
  t30 = (0.5e1 / 0.9e1 * tau1 / t24 / r1 * t17) ** t20
  t31 = r1 * t30
  t32 = t22 + t31
  t33 = jnp.cbrt(t32)
  t36 = t1 * t3 * t6 / t33
  t39 = jnp.sqrt(t36)
  t42 = t36 ** 0.15e1
  t44 = t1 ** 2
  t45 = t3 ** 2
  t47 = t33 ** 2
  t50 = t44 * t45 * t5 / t47
  t56 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t39 + 0.8969 * t36 + 0.204775 * t42 + 0.123235 * t50))
  t58 = 0.62182e-1 * (0.1e1 + 0.53425e-1 * t36) * t56
  t59 = t22 - t31
  t60 = t59 ** 2
  t61 = t60 ** 2
  t62 = t32 ** 2
  t63 = t62 ** 2
  t67 = t59 / t32
  t68 = 0.1e1 + t67
  t70 = jnp.cbrt(p.zeta_threshold)
  t71 = t70 * p.zeta_threshold
  t72 = jnp.cbrt(t68)
  t74 = jnp.where(t68 <= p.zeta_threshold, t71, t72 * t68)
  t75 = 0.1e1 - t67
  t77 = jnp.cbrt(t75)
  t79 = jnp.where(t75 <= p.zeta_threshold, t71, t77 * t75)
  t81 = jnp.cbrt(2)
  t85 = (t74 + t79 - 0.2e1) / (0.2e1 * t81 - 0.2e1)
  t96 = jnp.log(0.1e1 + 0.32164683177870697974e2 / (0.705945e1 * t39 + 0.1549425e1 * t36 + 0.420775 * t42 + 0.1562925 * t50))
  t109 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t39 + 0.905775 * t36 + 0.1100325 * t42 + 0.1241775 * t50))
  t110 = (0.1e1 + 0.278125e-1 * t36) * t109
  res = -t58 + t61 / t63 * t85 * (-0.3109e-1 * (0.1e1 + 0.5137e-1 * t36) * t96 + t58 - 0.19751789702565206229e-1 * t110) + 0.19751789702565206229e-1 * t85 * t110
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t1 = jnp.cbrt(3)
  t3 = jnp.cbrt(0.1e1 / jnp.pi)
  t5 = jnp.cbrt(4)
  t6 = t5 ** 2
  t7 = jnp.cbrt(2)
  t8 = t7 ** 2
  t10 = jnp.cbrt(r0)
  t11 = t10 ** 2
  t14 = jnp.cbrt(6)
  t16 = jnp.pi ** 2
  t17 = jnp.cbrt(t16)
  t18 = t17 ** 2
  t24 = (0.5e1 / 0.9e1 * tau0 * t8 / t11 / r0 * t14 / t18) ** (0.3e1 / 0.5e1 * params.ltafrac)
  t26 = jnp.cbrt(r0 * t24)
  t29 = t1 * t3 * t6 / t26
  t32 = jnp.sqrt(t29)
  t35 = t29 ** 0.15e1
  t37 = t1 ** 2
  t38 = t3 ** 2
  t40 = t26 ** 2
  t43 = t37 * t38 * t5 / t40
  t49 = jnp.log(0.1e1 + 0.16081824322151104822e2 / (0.379785e1 * t32 + 0.8969 * t29 + 0.204775 * t35 + 0.123235 * t43))
  t53 = jnp.cbrt(p.zeta_threshold)
  t55 = jnp.where(0.1e1 <= p.zeta_threshold, t53 * p.zeta_threshold, 1)
  t72 = jnp.log(0.1e1 + 0.29608574643216675549e2 / (0.51785e1 * t32 + 0.905775 * t29 + 0.1100325 * t35 + 0.1241775 * t43))
  res = -0.62182e-1 * (0.1e1 + 0.53425e-1 * t29) * t49 + 0.19751789702565206229e-1 * (0.2e1 * t55 - 0.2e1) / (0.2e1 * t7 - 0.2e1) * (0.1e1 + 0.278125e-1 * t29) * t72
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