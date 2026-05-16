"""
Boys function implementation for Gaussian integrals.

The Boys function is defined as:
    F_n(x) = ∫₀¹ t^(2n) exp(-x·t²) dt

It's used in nuclear attraction and electron repulsion integrals.
"""

import jax
import jax.numpy as jnp


def boys_function(n, x):
    """
    Compute Boys function F_n(x) using erf-based formula and recursion.

    F_n(x) = ∫₀¹ t^(2n) exp(-x·t²) dt

    For small x: use erf-based formula with upward recursion
    For large x: use asymptotic expansion

    Args:
        n: int, order of Boys function (0, 1, 2, ...)
        x: float or array, argument

    Returns:
        F_n(x): Boys function value
    """
    x = jnp.asarray(x)
    x_safe = jnp.maximum(x, 1e-10)

    # Special case: x ≈ 0
    # F_n(0) = 1/(2n+1)
    is_zero = x < 1e-10

    # For small to moderate x: use erf-based formula
    # F_0(x) = (√π/2) * erf(√x) / √x
    sqrt_x = jnp.sqrt(x_safe)
    f0 = jnp.sqrt(jnp.pi) / (2 * sqrt_x) * jax.lax.erf(sqrt_x)
    f0 = jnp.where(is_zero, 1.0, f0)

    if n == 0:
        return f0

    # For n > 0, use upward recursion:
    # F_{n+1}(x) = ((2n+1)*F_n(x) - exp(-x)) / (2x)
    exp_neg_x = jnp.exp(-x)
    f_curr = f0

    for i in range(n):
        f_next = ((2*i + 1) * f_curr - exp_neg_x) / (2 * x_safe)
        # Handle x=0 case: F_n(0) = 1/(2n+1)
        f_next = jnp.where(is_zero, 1.0 / (2*(i+1) + 1), f_next)
        f_curr = f_next

    return f_curr


# Vectorized version for multiple n values
def boys_function_array(n_max, x):
    """
    Compute Boys function F_n(x) for n = 0, 1, ..., n_max.

    Uses upward recursion for efficiency:
        F_{n+1}(x) = ((2n+1)*F_n(x) - exp(-x)) / (2x)

    Args:
        n_max: int, maximum order
        x: float or array, argument

    Returns:
        array of shape (n_max+1,) or (n_max+1, *x.shape) containing [F_0(x), F_1(x), ..., F_{n_max}(x)]
    """
    x = jnp.asarray(x)
    x_safe = jnp.maximum(x, 1e-10)
    is_zero = x < 1e-10

    # Start with F_0(x)
    sqrt_x = jnp.sqrt(x_safe)
    f0 = jnp.sqrt(jnp.pi) / (2 * sqrt_x) * jax.lax.erf(sqrt_x)
    f0 = jnp.where(is_zero, 1.0, f0)

    if n_max == 0:
        return jnp.array([f0])

    # Use upward recursion to compute F_1, F_2, ..., F_{n_max}
    exp_neg_x = jnp.exp(-x)

    def scan_fn(f_prev, i):
        f_next = ((2*i + 1) * f_prev - exp_neg_x) / (2 * x_safe)
        # Handle x=0 case
        f_next = jnp.where(is_zero, 1.0 / (2*(i+1) + 1), f_next)
        return f_next, f_next

    _, f_rest = jax.lax.scan(scan_fn, f0, jnp.arange(n_max))
    return jnp.concatenate([jnp.array([f0]), f_rest])
