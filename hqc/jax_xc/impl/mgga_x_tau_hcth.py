"""Generated from mgga_x_tau_hcth.mpl."""

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
  t29 = params.cx_local[0]
  t30 = params.cx_local[1]
  t32 = r0 ** 2
  t33 = jnp.cbrt(r0)
  t34 = t33 ** 2
  t36 = 0.1e1 / t34 / t32
  t39 = 0.1e1 + 0.4e-2 * s0 * t36
  t41 = t36 / t39
  t44 = params.cx_local[2]
  t45 = s0 ** 2
  t47 = t32 ** 2
  t51 = t39 ** 2
  t53 = 0.1e1 / t33 / t47 / r0 / t51
  t56 = params.cx_local[3]
  t57 = t45 * s0
  t59 = t47 ** 2
  t63 = 0.1e1 / t59 / t51 / t39
  t66 = params.cx_nlocal[0]
  t67 = params.cx_nlocal[1]
  t71 = params.cx_nlocal[2]
  t75 = params.cx_nlocal[3]
  t80 = jnp.cbrt(6)
  t81 = t80 ** 2
  t82 = jnp.pi ** 2
  t83 = jnp.cbrt(t82)
  t84 = t83 ** 2
  t86 = 0.3e1 / 0.1e2 * t81 * t84
  t89 = tau0 / t34 / r0
  t90 = t86 - t89
  t91 = t86 + t89
  t94 = t90 ** 2
  t96 = t91 ** 2
  t101 = t94 ** 2
  t103 = t96 ** 2
  t113 = jnp.where(r0 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (t29 + 0.4e-2 * t30 * s0 * t41 + 0.16e-4 * t44 * t45 * t53 + 0.64e-7 * t56 * t57 * t63 + (t66 + 0.4e-2 * t67 * s0 * t41 + 0.16e-4 * t71 * t45 * t53 + 0.64e-7 * t75 * t57 * t63) * (t90 / t91 - 0.2e1 * t94 * t90 / t96 / t91 + t101 * t90 / t103 / t91)))
  t115 = jnp.where(t10, t15, -t17)
  t116 = jnp.where(t14, t11, t115)
  t117 = 0.1e1 + t116
  t119 = jnp.cbrt(t117)
  t121 = jnp.where(t117 <= p.zeta_threshold, t23, t119 * t117)
  t124 = r1 ** 2
  t125 = jnp.cbrt(r1)
  t126 = t125 ** 2
  t128 = 0.1e1 / t126 / t124
  t131 = 0.1e1 + 0.4e-2 * s2 * t128
  t133 = t128 / t131
  t136 = s2 ** 2
  t138 = t124 ** 2
  t142 = t131 ** 2
  t144 = 0.1e1 / t125 / t138 / r1 / t142
  t147 = t136 * s2
  t149 = t138 ** 2
  t153 = 0.1e1 / t149 / t142 / t131
  t168 = tau1 / t126 / r1
  t169 = t86 - t168
  t170 = t86 + t168
  t173 = t169 ** 2
  t175 = t170 ** 2
  t180 = t173 ** 2
  t182 = t175 ** 2
  t192 = jnp.where(r1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t121 * t27 * (t29 + 0.4e-2 * t30 * s2 * t133 + 0.16e-4 * t44 * t136 * t144 + 0.64e-7 * t56 * t147 * t153 + (t66 + 0.4e-2 * t67 * s2 * t133 + 0.16e-4 * t71 * t136 * t144 + 0.64e-7 * t75 * t147 * t153) * (t169 / t170 - 0.2e1 * t173 * t169 / t175 / t170 + t180 * t169 / t182 / t170)))
  res = t113 + t192
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
  t24 = jnp.cbrt(2)
  t25 = t24 ** 2
  t26 = r0 ** 2
  t27 = t19 ** 2
  t29 = 0.1e1 / t27 / t26
  t34 = 0.1e1 + 0.4e-2 * s0 * t25 * t29
  t36 = t25 * t29 / t34
  t40 = s0 ** 2
  t42 = t26 ** 2
  t47 = t34 ** 2
  t49 = t24 / t19 / t42 / r0 / t47
  t53 = t40 * s0
  t55 = t42 ** 2
  t59 = 0.1e1 / t55 / t47 / t34
  t76 = jnp.cbrt(6)
  t77 = t76 ** 2
  t78 = jnp.pi ** 2
  t79 = jnp.cbrt(t78)
  t80 = t79 ** 2
  t82 = 0.3e1 / 0.1e2 * t77 * t80
  t86 = tau0 * t25 / t27 / r0
  t87 = t82 - t86
  t88 = t82 + t86
  t91 = t87 ** 2
  t93 = t88 ** 2
  t98 = t91 ** 2
  t100 = t93 ** 2
  t110 = jnp.where(r0 / 0.2e1 <= p.dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (params.cx_local[0] + 0.4e-2 * params.cx_local[1] * s0 * t36 + 0.32e-4 * params.cx_local[2] * t40 * t49 + 0.256e-6 * params.cx_local[3] * t53 * t59 + (params.cx_nlocal[0] + 0.4e-2 * params.cx_nlocal[1] * s0 * t36 + 0.32e-4 * params.cx_nlocal[2] * t40 * t49 + 0.256e-6 * params.cx_nlocal[3] * t53 * t59) * (t87 / t88 - 0.2e1 * t91 * t87 / t93 / t88 + t98 * t87 / t100 / t88)))
  res = 0.2e1 * t110
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