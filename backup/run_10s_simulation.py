"""
Run 10 second simulation with drho/dt method and create animation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from landslide_sph import SPHLandslideSimulator, create_terrain_with_pit


def run_simulation():
    """Run 10 second simulation and create animation."""
    print("=" * 60)
    print("10 SECOND SPH SIMULATION WITH DRHO/DT")
    print("=" * 60)

    # Create terrain
    print("\n[1/3] Creating terrain...")
    terrain, pit_location, peak_location = create_terrain_with_pit()
    cell_size = 10.0

    print(f"Terrain: {terrain.shape}, Elevation: {terrain.min():.0f}-{terrain.max():.0f} m")
    print(f"Peak: {peak_location}, Pit: {pit_location}")

    # Initialize simulator
    print("\n[2/3] Initializing simulator...")
    simulator = SPHLandslideSimulator(terrain, cell_size)
    simulator.mu_b = 0.1
    simulator.tau_y = 50.0
    simulator.warmup_time = 0.0

    # Initialize particles
    slope_fraction = 0.4
    landslide_center = (
        peak_location[0] + (pit_location[0] - peak_location[0]) * slope_fraction,
        peak_location[1] + (pit_location[1] - peak_location[1]) * slope_fraction
    )

    simulator.initialize_particles(landslide_center, radius=50, thickness=8.0)
    print(f"Particles: {simulator.particles.n}")

    # Run simulation
    print("\n[3/3] Running 10 second simulation...")
    duration = 10.0
    save_interval = 0.1

    simulator.run(duration=duration, save_interval=save_interval)

    # Create animation
    print("\nCreating animation...")
    create_animation(simulator, terrain, pit_location, peak_location)

    return simulator


def create_animation(simulator, terrain, pit_location, peak_location):
    """Create animation of simulation."""
    history = simulator.history
    cell_size = simulator.cell_size
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

        if len(x) > 0:
            speed = np.sqrt(vx**2 + vy**2)
        else:
            speed = np.array([])

        # Left: Terrain with particles colored by speed
        axes[0].imshow(terrain, cmap='terrain', origin='lower', extent=extent, alpha=0.7)
        axes[0].contour(np.linspace(0, cols*cell_size, cols),
                       np.linspace(0, rows*cell_size, rows),
                       terrain, levels=12, colors='gray', alpha=0.3, linewidths=0.5)

        if len(x) > 0:
            scatter = axes[0].scatter(x, y, c=speed, cmap='hot', s=25,
                                      vmin=0, vmax=speed_max, alpha=0.8,
                                      edgecolors='black', linewidths=0.2)

        # Mark pit
        circle = plt.Circle(pit_location, 120, fill=False, color='blue',
                           linestyle='--', linewidth=2)
        axes[0].add_patch(circle)
        axes[0].plot(pit_location[0], pit_location[1], 'bx', markersize=10, markeredgewidth=2)
        axes[0].plot(peak_location[0], peak_location[1], 'g^', markersize=10)

        n_active = state.get('n_active', len(x))
        max_spd = speed.max() if len(speed) > 0 else 0

        axes[0].set_xlim(0, cols * cell_size)
        axes[0].set_ylim(0, rows * cell_size)
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].set_title(f'SPH Landslide (drho/dt) - t = {state["time"]:.1f}s\n'
                         f'Active: {n_active}, Max speed: {max_spd:.1f} m/s')
        axes[0].set_aspect('equal')

        # Right: Thickness with velocity vectors
        axes[1].imshow(terrain, cmap='terrain', origin='lower', extent=extent, alpha=0.7)

        if len(x) > 0:
            scatter2 = axes[1].scatter(x, y, c=height, cmap='YlOrRd', s=25,
                                       vmin=0, vmax=height_max, alpha=0.8)

            # Velocity vectors
            step = max(1, len(x) // 60)
            axes[1].quiver(x[::step], y[::step], vx[::step], vy[::step],
                          scale=200, alpha=0.6, color='blue', width=0.003)

        circle2 = plt.Circle(pit_location, 120, fill=False, color='blue',
                            linestyle='--', linewidth=2)
        axes[1].add_patch(circle2)

        mean_spd = speed.mean() if len(speed) > 0 else 0

        axes[1].set_xlim(0, cols * cell_size)
        axes[1].set_ylim(0, rows * cell_size)
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Y (m)')
        axes[1].set_title(f'Debris Thickness & Velocity\n'
                         f'Mean speed: {mean_spd:.1f} m/s')
        axes[1].set_aspect('equal')

        plt.tight_layout()
        return []

    print(f"Generating {len(history)} frames...")
    anim = animation.FuncAnimation(fig, update, frames=len(history),
                                   interval=100, blit=False)

    output_file = 'drho_dt_10s_animation.gif'
    print(f"Saving to {output_file}...")
    anim.save(output_file, writer='pillow', fps=10, dpi=100)
    print(f"Animation saved to {output_file}")

    plt.close()


if __name__ == '__main__':
    run_simulation()
