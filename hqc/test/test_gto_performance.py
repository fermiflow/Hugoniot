"""
Performance test for the new vectorized solver interface.

Demonstrates the efficiency of the new design where:
1. Basis data is precomputed once in make_solver
2. The returned hf function only takes atom_positions
3. Integral computation is fully vectorized
"""

import jax
import jax.numpy as jnp
import time
from hqc.gto.solver import make_solver

jax.config.update("jax_enable_x64", True)


def test_repeated_calls():
    """Test performance of repeated calls with different positions."""
    # Setup: H2 molecule
    atom_charges = jnp.array([1.0, 1.0])
    n_electrons = 2

    # Create solver once (preprocessing)
    print("Creating solver...")
    start = time.time()
    hf = make_solver(atom_charges, n_electrons, basis='gth-szv', diis=True, use_jit=True)
    print(f"Solver creation time: {time.time() - start:.4f}s\n")

    # Test different H-H distances
    distances = jnp.linspace(0.8, 3.0, 10)

    print("Running calculations for different H-H distances:")
    print("Distance (Bohr) | Energy (Ha) | Time (s)")
    print("-" * 50)

    energies = []
    times = []

    for d in distances:
        atom_positions = jnp.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])

        start = time.time()
        result = hf(atom_positions)
        elapsed = time.time() - start

        energies.append(result['energy'])
        times.append(elapsed)

        print(f"{d:14.4f} | {result['energy']:11.6f} | {elapsed:.6f}")

    print(f"\nAverage time per calculation: {sum(times)/len(times):.6f}s")
    print(f"Total time for {len(distances)} calculations: {sum(times):.4f}s")

    # Find equilibrium distance (minimum energy)
    min_idx = jnp.argmin(jnp.array(energies))
    print(f"\nEquilibrium distance: {distances[min_idx]:.4f} Bohr")
    print(f"Equilibrium energy: {energies[min_idx]:.6f} Ha")


def test_jit_speedup():
    """Compare JIT vs non-JIT performance."""
    atom_charges = jnp.array([1.0, 1.0])
    n_electrons = 2
    atom_positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])

    # Non-JIT version
    print("\n" + "="*60)
    print("Testing JIT speedup")
    print("="*60)

    hf_no_jit = make_solver(atom_charges, n_electrons, basis='gth-szv', use_jit=False)

    start = time.time()
    result1 = hf_no_jit(atom_positions)
    time_no_jit = time.time() - start
    print(f"Non-JIT time: {time_no_jit:.6f}s")

    # JIT version (first call includes compilation)
    hf_jit = make_solver(atom_charges, n_electrons, basis='gth-szv', use_jit=True)

    start = time.time()
    result2 = hf_jit(atom_positions)  # First call: compilation + execution
    time_jit_first = time.time() - start
    print(f"JIT first call (with compilation): {time_jit_first:.6f}s")

    # JIT version (second call, already compiled)
    start = time.time()
    result3 = hf_jit(atom_positions)
    time_jit_second = time.time() - start
    print(f"JIT second call (compiled): {time_jit_second:.6f}s")

    print(f"\nSpeedup (JIT vs non-JIT): {time_no_jit/time_jit_second:.2f}x")

    # Verify results are the same
    assert jnp.abs(result1['energy'] - result2['energy']) < 1e-10
    assert jnp.abs(result2['energy'] - result3['energy']) < 1e-10
    print("✓ All results match")


if __name__ == "__main__":
    test_repeated_calls()
    test_jit_speedup()
