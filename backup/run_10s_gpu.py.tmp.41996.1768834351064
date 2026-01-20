"""
Run 10 second GPU simulation and create animation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time

from landslide_sph_gpu import SPHSimulatorGPU, create_terrain_with_pit


def main():
    print("=" * 60)
    print("10 SECOND GPU SPH SIMULATION")
    print("=" * 60)

    # Create terrain
    terrain, pit_loc, peak_loc = create_terrain_with_pit()
    cell_size = 10.0

    print(f"Terrain: {terrain.shape}")
    print(f"Peak: {peak_loc}, Pit: {pit_loc}")

    # Initialize GPU simulator
    sim = SPHSimulatorGPU(terrain, cell_size)

    slope_fraction = 0.4
    center = (
        peak_loc[0] + (pit_loc[0] - peak_loc[0]) * slope_fraction,
        peak_loc[1] + (pit_loc[1] - peak_loc[1]) * slope_fraction
    )

    sim.initialize_particles(center, radius=50, thickness=8.0)

    # Run 10 second simulation
    print("\nRunning 10 second simulation...")
    start = time.time()
    sim.run(duration=10.0, save_interval=0.1)
    elapsed = time.time() - start
    print(f"Total wall time: {elapsed:.1f}s")

    # Create animation
    print("\nCreating animation...")
    create_animation(sim, terrain, pit_loc, peak_loc)


def create_animation(sim, terrain, pit_loc, peak_loc):
    """Create animation."""
    history = sim.history
    cell_size = sim.cell_size
    rows, cols = terrain.shape
    extent = [0, cols * cell_size, 0, rows * cell_size]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pre-compute ranges
    all_speeds = []
    for h in history:
        if len(h['vx']) > 0:
            speeds = np.sqrt(h['vx']**2 + h['vy']**2)
            all_speeds.extend(speeds)
    speed_max = max(all_speeds) if all_speeds else 1.0
    height_max = history[0]['height'].max() if len(history[0]['height']) > 0 else 1.0

    def update(frame):
        axes[0].clear()
        axes[1].clear()

        state = history[frame]
        x, y = state['x'], state['y']
        vx, vy = state['vx'], state['vy']
        height = state['height']

        speed = np.sqrt(vx**2 + vy**2) if len(x) > 0 else np.array([])

        # Left: particles colored by speed
        axes[0].imshow(terrain, cmap='terrain', origin='lower', extent=extent, alpha=0.7)
        axes[0].contour(np.linspace(0, cols*cell_size, cols),
                        np.linspace(0, rows*cell_size, rows),
                        terrain, levels=12, colors='gray', alpha=0.3, linewidths=0.5)

        if len(x) > 0:
            axes[0].scatter(x, y, c=speed, cmap='hot', s=25,
                           vmin=0, vmax=speed_max, alpha=0.8,
                           edgecolors='black', linewidths=0.2)

        circle = plt.Circle(pit_loc, 120, fill=False, color='blue',
                           linestyle='--', linewidth=2)
        axes[0].add_patch(circle)
        axes[0].plot(pit_loc[0], pit_loc[1], 'bx', markersize=10, markeredgewidth=2)
        axes[0].plot(peak_loc[0], peak_loc[1], 'g^', markersize=10)

        n_active = state.get('n_active', len(x))
        max_spd = speed.max() if len(speed) > 0 else 0

        axes[0].set_xlim(0, cols * cell_size)
        axes[0].set_ylim(0, rows * cell_size)
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].set_title(f'GPU SPH Landslide - t = {state["time"]:.1f}s\n'
                         f'Active: {n_active}, Max speed: {max_spd:.1f} m/s')
        axes[0].set_aspect('equal')

        # Right: thickness with velocity vectors
        axes[1].imshow(terrain, cmap='terrain', origin='lower', extent=extent, alpha=0.7)

        if len(x) > 0:
            axes[1].scatter(x, y, c=height, cmap='YlOrRd', s=25,
                           vmin=0, vmax=height_max, alpha=0.8)
            step = max(1, len(x) // 60)
            axes[1].quiver(x[::step], y[::step], vx[::step], vy[::step],
                          scale=200, alpha=0.6, color='blue', width=0.003)

        circle2 = plt.Circle(pit_loc, 120, fill=False, color='blue',
                            linestyle='--', linewidth=2)
        axes[1].add_patch(circle2)

        mean_spd = speed.mean() if len(speed) > 0 else 0

        axes[1].set_xlim(0, cols * cell_size)
        axes[1].set_ylim(0, rows * cell_size)
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Y (m)')
        axes[1].set_title(f'Debris Thickness & Velocity\nMean speed: {mean_spd:.1f} m/s')
        axes[1].set_aspect('equal')

        plt.tight_layout()
        return []

    print(f"Generating {len(history)} frames...")
    anim = animation.FuncAnimation(fig, update, frames=len(history),
                                   interval=100, blit=False)

    output_file = 'gpu_10s_animation.gif'
    print(f"Saving to {output_file}...")
    anim.save(output_file, writer='pillow', fps=10, dpi=100)
    print(f"Animation saved!")

    plt.close()


if __name__ == '__main__':
    main()
