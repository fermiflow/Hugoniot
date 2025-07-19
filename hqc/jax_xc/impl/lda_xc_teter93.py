"""Generated from lda_xc_teter93.mpl."""

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
  t4 = (r0 - r1) * t3
  t5 = 0.1e1 + t4
  t7 = jnp.cbrt(p.zeta_threshold)
  t8 = t7 * p.zeta_threshold
  t9 = jnp.cbrt(t5)
  t11 = jnp.where(t5 <= p.zeta_threshold, t8, t9 * t5)
  t12 = 0.1e1 - t4
  t14 = jnp.cbrt(t12)
  t16 = jnp.where(t12 <= p.zeta_threshold, t8, t14 * t12)
  t18 = jnp.cbrt(2)
  t22 = (t11 + t16 - 0.2e1) / (0.2e1 * t18 - 0.2e1)
  t26 = jnp.cbrt(3)
  t28 = 0.1e1 / jnp.pi
  t29 = jnp.cbrt(t28)
  t30 = jnp.cbrt(4)
  t31 = t30 ** 2
  t33 = jnp.cbrt(t2)
  t34 = 0.1e1 / t33
  t40 = t26 ** 2
  t42 = t29 ** 2
  t44 = t33 ** 2
  t46 = t42 * t30 / t44
  res = -(0.4581652932831429 + 0.119086804055547 * t22 + (0.2217058676663745e1 + 0.6157402568883345 * t22) * t26 * t29 * t31 * t34 / 0.4e1 + (0.7405551735357053 + 0.1574201515892867 * t22) * t40 * t46 / 0.4e1 + 0.3e1 / 0.4e1 * (0.1968227878617998e-1 + 0.3532336663397157e-2 * t22) * t28 * t3) / (0.25 * t26 * t29 * t31 * t34 + (0.4504130959426697e1 + 0.2673612973836267 * t22) * t40 * t46 / 0.4e1 + 0.3e1 / 0.4e1 * (0.1110667363742916e1 + 0.2052004607777787 * t22) * t28 * t3 + 0.3e1 / 0.16e2 * (0.2359291751427506e-1 + 0.4200005045691381e-2 * t22) * t26 * t29 * t28 * t31 / t33 / t2)
  return res


def unpol(p, r, s=None, l=None, tau=None):
  params = p.params
  r0, s0, l0, tau0 = r, s, l, tau
  t2 = jnp.cbrt(p.zeta_threshold)
  t4 = jnp.where(0.1e1 <= p.zeta_threshold, t2 * p.zeta_threshold, 1)
  t7 = jnp.cbrt(2)
  t11 = (0.2e1 * t4 - 0.2e1) / (0.2e1 * t7 - 0.2e1)
  t15 = jnp.cbrt(3)
  t17 = 0.1e1 / jnp.pi
  t18 = jnp.cbrt(t17)
  t19 = jnp.cbrt(4)
  t20 = t19 ** 2
  t22 = jnp.cbrt(r0)
  t23 = 0.1e1 / t22
  t29 = t15 ** 2
  t31 = t18 ** 2
  t33 = t22 ** 2
  t35 = t31 * t19 / t33
  t41 = 0.1e1 / r0
  res = -(0.4581652932831429 + 0.119086804055547 * t11 + (0.2217058676663745e1 + 0.6157402568883345 * t11) * t15 * t18 * t20 * t23 / 0.4e1 + (0.7405551735357053 + 0.1574201515892867 * t11) * t29 * t35 / 0.4e1 + 0.3e1 / 0.4e1 * (0.1968227878617998e-1 + 0.3532336663397157e-2 * t11) * t17 * t41) / (0.25 * t15 * t18 * t20 * t23 + (0.4504130959426697e1 + 0.2673612973836267 * t11) * t29 * t35 / 0.4e1 + 0.3e1 / 0.4e1 * (0.1110667363742916e1 + 0.2052004607777787 * t11) * t17 * t41 + 0.3e1 / 0.16e2 * (0.2359291751427506e-1 + 0.4200005045691381e-2 * t11) * t15 * t18 * t17 * t20 / t22 / r0)
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