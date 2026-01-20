"""
Benchmark: CPU vs GPU SPH Simulation
"""

import time
import numpy as np

# Import simulators
from landslide_sph import SPHLandslideSimulator, create_terrain_with_pit
from landslide_sph_gpu import SPHSimulatorGPU


def benchmark_cpu(terrain, center, duration=1.0):
    """Run CPU simulation and measure time."""
    sim = SPHLandslideSimulator(terrain, cell_size=10.0)
    sim.mu_b = 0.1
    sim.tau_y = 50.0
    sim.warmup_time = 0.0

    sim.initialize_particles(center, radius=50, thickness=8.0)
    n_particles = sim.particles.n

    n_steps = int(duration / sim.dt)

    start = time.time()
    for _ in range(n_steps):
        sim.step()
    elapsed = time.time() - start

    return n_particles, n_steps, elapsed


def benchmark_gpu(terrain, center, duration=1.0):
    """Run GPU simulation and measure time."""
    sim = SPHSimulatorGPU(terrain, cell_size=10.0)

    sim.initialize_particles(center, radius=50, thickness=8.0)
    n_particles = sim.particles.n

    n_steps = int(duration / sim.dt)

    # Warmup GPU
    for _ in range(10):
        sim.step()
    sim.time = 0.0

    import cupy as cp
    cp.cuda.Stream.null.synchronize()

    start = time.time()
    for _ in range(n_steps):
        sim.step()
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start

    return n_particles, n_steps, elapsed


def benchmark_gpu_only(terrain, center, radius, duration=0.5):
    """Run GPU simulation only for scaling test."""
    sim = SPHSimulatorGPU(terrain, cell_size=10.0)
    sim.initialize_particles(center, radius=radius, thickness=8.0)
    n_particles = sim.particles.n

    n_steps = int(duration / sim.dt)

    # Warmup
    for _ in range(5):
        sim.step()
    sim.time = 0.0

    import cupy as cp
    cp.cuda.Stream.null.synchronize()

    start = time.time()
    for _ in range(n_steps):
        sim.step()
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start

    return n_particles, n_steps, elapsed


def main():
    print("=" * 60)
    print("CPU vs GPU SPH BENCHMARK")
    print("=" * 60)

    # Create terrain
    terrain, pit_loc, peak_loc = create_terrain_with_pit()

    slope_fraction = 0.4
    center = (
        peak_loc[0] + (pit_loc[0] - peak_loc[0]) * slope_fraction,
        peak_loc[1] + (pit_loc[1] - peak_loc[1]) * slope_fraction
    )

    duration = 1.0

    print(f"\nRunning {duration}s simulation benchmark...")

    # CPU benchmark
    print("\n[CPU] Running...")
    cpu_particles, cpu_steps, cpu_time = benchmark_cpu(terrain, center, duration)
    cpu_rate = cpu_steps / cpu_time

    print(f"[CPU] {cpu_particles} particles, {cpu_steps} steps")
    print(f"[CPU] Time: {cpu_time:.2f}s ({cpu_rate:.0f} steps/sec)")

    # GPU benchmark
    print("\n[GPU] Running...")
    gpu_particles, gpu_steps, gpu_time = benchmark_gpu(terrain, center, duration)
    gpu_rate = gpu_steps / gpu_time

    print(f"[GPU] {gpu_particles} particles, {gpu_steps} steps")
    print(f"[GPU] Time: {gpu_time:.2f}s ({gpu_rate:.0f} steps/sec)")

    # Comparison
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Particles: {cpu_particles}")
    print(f"Steps: {cpu_steps}")
    print(f"")
    print(f"CPU: {cpu_time:.2f}s ({cpu_rate:.0f} steps/sec)")
    print(f"GPU: {gpu_time:.2f}s ({gpu_rate:.0f} steps/sec)")
    print(f"")
    speedup = cpu_time / gpu_time
    if speedup > 1:
        print(f"GPU is {speedup:.1f}x FASTER than CPU")
    else:
        print(f"CPU is {1/speedup:.1f}x faster than GPU")

    # GPU scaling test
    print("\n" + "=" * 60)
    print("GPU SCALING TEST (varying particle count)")
    print("=" * 60)

    radii = [30, 50, 80, 120, 180]
    print(f"{'Particles':<12} {'Steps/sec':<12} {'Time (0.5s sim)':<15}")
    print("-" * 40)

    for r in radii:
        try:
            n, steps, t = benchmark_gpu_only(terrain, center, r, duration=0.5)
            rate = steps / t
            print(f"{n:<12} {rate:<12.0f} {t:<15.2f}s")
        except Exception as e:
            print(f"radius={r}: Error - {e}")


if __name__ == '__main__':
    main()
