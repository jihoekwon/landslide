"""
3D SPH Landslide Animation with DEM visualization - Irwon DEM
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
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
    print("3D GPU SPH SIMULATION - IRWON DEM")
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
    print(f"\nRunning {duration}s simulation...")
    start = time.time()
    sim.run(duration=duration, save_interval=0.2)
    elapsed = time.time() - start
    print(f"Simulation time: {elapsed:.1f}s")

    # Calculate initial centroid for runout distance
    init_x = particles_x.mean()
    init_y = particles_y.mean()

    # Create 3D animation
    print("\nCreating 3D animation...")
    create_3d_animation(sim, terrain, cell_size, x_min, y_min)

    # Plot runout distance
    print("\nCreating runout distance plot...")
    plot_runout_distance(sim, init_x, init_y, x_min, y_min)


def create_3d_animation(sim, terrain, cell_size, x_min, y_min):
    """Create 3D animation with DEM surface."""
    history = sim.history
    ny, nx = terrain.shape

    # Create mesh grid for terrain (in local coordinates)
    x = np.arange(nx) * cell_size
    y = np.arange(ny) * cell_size
    X, Y = np.meshgrid(x, y)

    # Figure setup
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Pre-compute speed range
    all_speeds = []
    for h in history:
        if len(h['vx']) > 0:
            speeds = np.sqrt(h['vx']**2 + h['vy']**2)
            all_speeds.extend(speeds)
    speed_max = max(all_speeds) if all_speeds else 1.0

    VIEW_ELEV = 35
    VIEW_AZIM = 30  # Rotated 180 degrees from -150

    def get_particle_elevation(x_arr, y_arr):
        col = np.clip(x_arr / cell_size, 0, nx - 1.001)
        row = np.clip(y_arr / cell_size, 0, ny - 1.001)
        col0 = np.floor(col).astype(int)
        row0 = np.floor(row).astype(int)
        col1 = np.minimum(col0 + 1, nx - 1)
        row1 = np.minimum(row0 + 1, ny - 1)
        wx, wy = col - col0, row - row0
        z = (terrain[row0, col0] * (1 - wx) * (1 - wy) +
             terrain[row0, col1] * wx * (1 - wy) +
             terrain[row1, col0] * (1 - wx) * wy +
             terrain[row1, col1] * wx * wy)
        return z

    def update(frame):
        ax.clear()

        state = history[frame]
        px, py = state['x'], state['y']
        vx, vy = state['vx'], state['vy']

        if len(px) > 0:
            terrain_z = get_particle_elevation(px, py)
            pz = terrain_z + 10
            speed = np.sqrt(vx**2 + vy**2)
        else:
            pz = np.array([])
            speed = np.array([])

        ax.computed_zorder = False

        # Plot terrain
        ax.plot_surface(X, Y, terrain, cmap='terrain', alpha=0.8, zorder=1,
                       rstride=1, cstride=1, antialiased=False)

        # Plot particles
        if len(px) > 0:
            ax.scatter(px, py, pz, c=speed, cmap='plasma', s=1,
                      vmin=0, vmax=speed_max, edgecolors='none',
                      depthshade=False, alpha=1.0, zorder=2)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        n_active = state.get('n_active', len(px))
        max_spd = speed.max() if len(speed) > 0 else 0
        mean_spd = speed.mean() if len(speed) > 0 else 0

        ax.set_title(f'Irwon Landslide - t = {state["time"]:.1f}s\n'
                    f'Particles: {n_active}, Speed: mean={mean_spd:.1f}, max={max_spd:.1f} m/s')

        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

        # Equal aspect ratio
        x_range = X.max() - X.min()
        y_range = Y.max() - Y.min()
        z_range = terrain.max() - terrain.min()
        max_range = max(x_range, y_range, z_range)
        ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])

        return []

    print(f"Generating {len(history)} frames...")
    anim = animation.FuncAnimation(fig, update, frames=len(history),
                                   interval=100, blit=False)

    output_file = 'irwon_landslide_3d.gif'
    print(f"Saving to {output_file}...")
    anim.save(output_file, writer='pillow', fps=10, dpi=100)
    print(f"Animation saved!")

    plt.close()


def plot_runout_distance(sim, init_x, init_y, x_min, y_min):
    """Plot runout distance (head) over time."""
    history = sim.history

    times = []
    runout_distances = []
    mean_distances = []
    max_speeds = []

    # Initial position in local coordinates
    init_x_local = init_x - x_min
    init_y_local = init_y - y_min

    for state in history:
        t = state['time']
        px, py = state['x'], state['y']
        vx, vy = state['vx'], state['vy']

        if len(px) > 0:
            # Distance from initial centroid for each particle
            distances = np.sqrt((px - init_x_local)**2 + (py - init_y_local)**2)

            # Head runout = maximum distance (leading edge)
            runout_head = distances.max()

            # Mean distance (centroid movement)
            mean_dist = distances.mean()

            # Max speed
            speeds = np.sqrt(vx**2 + vy**2)
            max_spd = speeds.max()
        else:
            runout_head = 0
            mean_dist = 0
            max_spd = 0

        times.append(t)
        runout_distances.append(runout_head)
        mean_distances.append(mean_dist)
        max_speeds.append(max_spd)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Runout distance
    ax1.plot(times, runout_distances, 'b-', linewidth=2, label='Head (max)')
    ax1.plot(times, mean_distances, 'g--', linewidth=1.5, label='Centroid (mean)')
    ax1.set_ylabel('Runout Distance (m)')
    ax1.set_title('Landslide Runout Distance Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Max speed
    ax2.plot(times, max_speeds, 'r-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Max Speed (m/s)')
    ax2.set_title('Maximum Particle Speed Over Time')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('runout_distance.png', dpi=150)
    print(f"Runout plot saved to runout_distance.png")
    print(f"Final runout (head): {runout_distances[-1]:.1f} m")
    print(f"Final runout (centroid): {mean_distances[-1]:.1f} m")
    plt.close()


if __name__ == '__main__':
    main()
