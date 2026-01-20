"""
Run SPH Landslide Simulation and save results to file.
"""

import numpy as np
import os
import time

from landslide_sph_gpu import SPHSimulatorGPU


def create_arbitrary_blob(cx, cy, base_radius=75, n_lobes=5, particle_spacing=1.25):
    """Create arbitrary blob shape with irregular boundary.

    Algorithm:
    1. Base ellipse (slightly elongated, rotated 0.3 rad)
    2. Add n_lobes random protrusions using Gaussian functions
    3. Add small random noise to boundary
    """
    np.random.seed(42)
    particles_x, particles_y = [], []
    grid_size = int(base_radius * 2.5 / particle_spacing)
    xs = np.linspace(cx - base_radius * 1.2, cx + base_radius * 1.2, grid_size)
    ys = np.linspace(cy - base_radius * 1.2, cy + base_radius * 1.2, grid_size)

    # Random lobe parameters
    lobe_angles = np.random.uniform(0, 2*np.pi, n_lobes)
    lobe_radii = np.random.uniform(0.3, 0.8, n_lobes) * base_radius
    lobe_widths = np.random.uniform(0.3, 0.6, n_lobes)

    for x in xs:
        for y in ys:
            dx, dy = x - cx, y - cy
            dist = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)

            # Base ellipse
            ellipse_a, ellipse_b = base_radius * 1.1, base_radius * 0.8
            ellipse_r = (ellipse_a * ellipse_b) / np.sqrt(
                (ellipse_b * np.cos(angle - 0.3))**2 + (ellipse_a * np.sin(angle - 0.3))**2
            )

            # Boundary with lobes
            boundary = ellipse_r * 0.7
            for i in range(n_lobes):
                angle_diff = angle - lobe_angles[i]
                boundary += lobe_radii[i] * np.exp(-angle_diff**2 / lobe_widths[i]**2)
            boundary += np.random.uniform(-3, 3)

            if dist < boundary:
                particles_x.append(x)
                particles_y.append(y)

    return np.array(particles_x), np.array(particles_y)


def main():
    print("=" * 60)
    print("SPH LANDSLIDE SIMULATION")
    print("=" * 60)

    # Load terrain
    terrain = np.load('D:/Claude/landslide/irwon_terrain_hires.npy')
    cell_size = 30.0
    x_min, y_min = 203461.0, 535418.0

    ny, nx = terrain.shape
    print(f"Terrain: {nx}x{ny}, cell_size={cell_size}m")
    print(f"Domain: X={x_min:.0f}~{x_min+nx*cell_size:.0f}, Y={y_min:.0f}~{y_min+ny*cell_size:.0f}")

    # Initialize GPU simulator
    sim = SPHSimulatorGPU(terrain, cell_size)
    print(f"SPH h={sim.h}m, spacing={sim.h/2}m")

    # Plume center
    plume_x, plume_y = 203800, 536000

    # Create arbitrary blob particles
    h = sim.h
    particle_spacing = h / 2
    particles_x, particles_y = create_arbitrary_blob(plume_x, plume_y,
                                                      base_radius=75,
                                                      particle_spacing=particle_spacing)

    # Initialize particles
    sim.initialize_particles_from_coords(particles_x, particles_y, x_min, y_min, thickness=5.0)

    # Run simulation
    duration = 30.0
    save_interval = 0.2
    print(f"\nRunning {duration}s simulation...")
    start = time.time()
    sim.run(duration=duration, save_interval=save_interval)
    elapsed = time.time() - start
    print(f"Simulation time: {elapsed:.1f}s")

    # Calculate initial centroid
    init_x = particles_x.mean()
    init_y = particles_y.mean()

    # Save results to file
    print("\nSaving simulation results...")

    # Convert history to arrays for saving
    n_frames = len(sim.history)
    max_particles = max(len(h['x']) for h in sim.history)

    # Pre-allocate arrays
    times = np.zeros(n_frames)
    n_active = np.zeros(n_frames, dtype=np.int32)
    x_data = np.full((n_frames, max_particles), np.nan, dtype=np.float32)
    y_data = np.full((n_frames, max_particles), np.nan, dtype=np.float32)
    vx_data = np.full((n_frames, max_particles), np.nan, dtype=np.float32)
    vy_data = np.full((n_frames, max_particles), np.nan, dtype=np.float32)

    for i, state in enumerate(sim.history):
        times[i] = state['time']
        n = len(state['x'])
        n_active[i] = state.get('n_active', n)
        x_data[i, :n] = state['x']
        y_data[i, :n] = state['y']
        vx_data[i, :n] = state['vx']
        vy_data[i, :n] = state['vy']

    # Save to npz file
    output_file = 'D:/Claude/landslide/simulation_results.npz'
    np.savez_compressed(output_file,
        # Simulation history
        times=times,
        n_active=n_active,
        x=x_data,
        y=y_data,
        vx=vx_data,
        vy=vy_data,
        # Terrain data
        terrain=terrain,
        # Metadata
        cell_size=cell_size,
        x_min=x_min,
        y_min=y_min,
        init_x=init_x,
        init_y=init_y,
        duration=duration,
        save_interval=save_interval,
        h=sim.h
    )

    print(f"Results saved to {output_file}")
    print(f"  - Frames: {n_frames}")
    print(f"  - Max particles: {max_particles}")
    print(f"  - File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
