"""
Test Boys function implementation against reference values.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from hqc.gto.boys import boys_function, boys_function_array

jax.config.update("jax_enable_x64", True)


class TestBoysFunction:
    """Test Boys function accuracy."""

    def test_boys_f0_small_x(self):
        """Test F_0(x) for small x values."""
        # Reference values computed with scipy.integrate.quad
        # F_0(x) = ∫₀¹ exp(-x·t²) dt
        x_vals = jnp.array([0.0, 0.1, 0.5, 1.0, 2.0])
        expected = jnp.array([1.0000000000, 0.9676433126, 0.8556243919, 0.7468241328, 0.5981440067])

        for x, exp in zip(x_vals, expected):
            result = boys_function(0, x)
            assert jnp.abs(result - exp) < 1e-6, f"F_0({x}) = {result}, expected {exp}"

    def test_boys_f0_large_x(self):
        """Test F_0(x) for large x values."""
        # For large x: F_0(x) ≈ sqrt(π/4x)
        x_vals = jnp.array([30.0, 50.0, 100.0])

        for x in x_vals:
            result = boys_function(0, x)
            asymptotic = jnp.sqrt(jnp.pi / (4 * x))
            # Should be close to asymptotic value
            assert jnp.abs(result - asymptotic) / asymptotic < 0.01

    def test_boys_higher_orders(self):
        """Test F_n(x) for n > 0."""
        # F_1(0) = 1/3
        # F_2(0) = 1/5
        # F_3(0) = 1/7
        assert jnp.abs(boys_function(1, 0.0) - 1/3) < 1e-6
        assert jnp.abs(boys_function(2, 0.0) - 1/5) < 1e-6
        assert jnp.abs(boys_function(3, 0.0) - 1/7) < 1e-6

    def test_boys_recursion_relation(self):
        """Test Boys function satisfies recursion relation."""
        # Recursion: F_n(x) = (2x*F_{n+1}(x) + exp(-x)) / (2n+1)
        x = 1.5
        for n in range(5):
            f_n = boys_function(n, x)
            f_n1 = boys_function(n+1, x)
            f_n_from_recursion = (2*x*f_n1 + jnp.exp(-x)) / (2*n + 1)
            assert jnp.abs(f_n - f_n_from_recursion) < 1e-6

    def test_boys_array_consistency(self):
        """Test boys_function_array gives same results as individual calls."""
        x = 2.0
        n_max = 5

        # Compute using array function
        f_array = boys_function_array(n_max, x)

        # Compute individually
        for n in range(n_max + 1):
            f_single = boys_function(n, x)
            assert jnp.abs(f_array[n] - f_single) < 1e-10

    def test_boys_vectorized(self):
        """Test Boys function works with array inputs."""
        x_vals = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0])
        results = jax.vmap(lambda x: boys_function(0, x))(x_vals)

        assert results.shape == (5,)
        # Check first value
        assert jnp.abs(results[0] - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
