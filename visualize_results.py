"""
Visualize SPH Landslide Simulation results from saved file.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LightSource
from PIL import Image


# ============================================================
# VISUALIZATION SETTINGS (modify these as needed)
# ============================================================
VIEW_ELEV = 35          # Elevation angle (degrees)
VIEW_AZIM = 30          # Azimuth angle (degrees)
FPS = 10                # Frames per second for GIF
DPI = 100               # Resolution
PARTICLE_SIZE = 1       # Scatter point size
PARTICLE_EDGE = False   # Draw particle edges
EDGE_COLOR = 'black'    # Edge color
EDGE_WIDTH = 0.3        # Edge line width
USE_SATELLITE = True    # Use satellite texture on terrain
SATELLITE_FILE = 'D:/Claude/landslide/satellite_texture.png'
# ============================================================


def load_results(filepath='D:/Claude/landslide/simulation_results.npz'):
    """Load simulation results from file."""
    data = np.load(filepath)
    return data


def create_3d_animation(data, output_file='irwon_landslide_3d.gif'):
    """Create 3D animation from saved simulation data."""

    # Extract data
    terrain = data['terrain']
    cell_size = float(data['cell_size'])
    times = data['times']
    n_active = data['n_active']
    x_data = data['x']
    y_data = data['y']
    vx_data = data['vx']
    vy_data = data['vy']

    ny, nx = terrain.shape
    n_frames = len(times)

    # Create mesh grid for terrain
    x = np.arange(nx) * cell_size
    y = np.arange(ny) * cell_size
    X, Y = np.meshgrid(x, y)

    # Load satellite texture if enabled
    satellite_colors = None
    if USE_SATELLITE:
        try:
            img = Image.open(SATELLITE_FILE)
            # Resize to (nx, ny) for the grid points
            img_resized = img.resize((nx, ny), Image.LANCZOS)
            img_array = np.array(img_resized) / 255.0  # Normalize to 0-1
            # Flip vertically to match terrain orientation
            img_array = np.flipud(img_array)
            # For facecolors, we need colors at cell centers (ny-1, nx-1)
            # Average neighboring pixels
            satellite_colors = (img_array[:-1, :-1] + img_array[1:, :-1] +
                               img_array[:-1, 1:] + img_array[1:, 1:]) / 4
            print(f"  Satellite texture loaded: {SATELLITE_FILE}")
        except Exception as e:
            print(f"  Warning: Could not load satellite texture: {e}")
            satellite_colors = None

    # Pre-compute speed range (do this FIRST, before creating figure)
    all_speeds = []
    for i in range(n_frames):
        mask = ~np.isnan(x_data[i])
        if mask.any():
            speeds = np.sqrt(vx_data[i, mask]**2 + vy_data[i, mask]**2)
            all_speeds.extend(speeds)
    speed_max = max(all_speeds) if all_speeds else 1.0
    print(f"  Speed range: 0 ~ {speed_max:.2f} m/s (fixed)")

    # Figure setup with gridspec for persistent colorbar
    from matplotlib.gridspec import GridSpec
    import matplotlib.cm as cm

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0], projection='3d')
    cax = fig.add_subplot(gs[1])

    # Create fixed colorbar (will not change during animation)
    norm = plt.Normalize(vmin=0, vmax=speed_max)
    sm = cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, label='Speed (m/s)')

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

        # Get valid particles for this frame
        mask = ~np.isnan(x_data[frame])
        px = x_data[frame, mask]
        py = y_data[frame, mask]
        vx = vx_data[frame, mask]
        vy = vy_data[frame, mask]

        if len(px) > 0:
            terrain_z = get_particle_elevation(px, py)
            pz = terrain_z + 10
            speed = np.sqrt(vx**2 + vy**2)
        else:
            pz = np.array([])
            speed = np.array([])

        ax.computed_zorder = False

        # Plot terrain
        if satellite_colors is not None:
            ax.plot_surface(X, Y, terrain, facecolors=satellite_colors,
                           rstride=1, cstride=1, antialiased=False,
                           shade=False, zorder=1)
        else:
            ax.plot_surface(X, Y, terrain, cmap='terrain', alpha=0.8, zorder=1,
                           rstride=1, cstride=1, antialiased=False)

        # Plot particles
        if len(px) > 0:
            edge_colors = EDGE_COLOR if PARTICLE_EDGE else 'none'
            line_widths = EDGE_WIDTH if PARTICLE_EDGE else 0
            ax.scatter(px, py, pz, c=speed, cmap='plasma', s=PARTICLE_SIZE,
                      vmin=0, vmax=speed_max, edgecolors=edge_colors,
                      linewidths=line_widths,
                      depthshade=False, alpha=1.0, zorder=2)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        n_act = n_active[frame]
        max_spd = speed.max() if len(speed) > 0 else 0
        mean_spd = speed.mean() if len(speed) > 0 else 0

        # TM coordinates (EPSG:5186) - center of cropped area
        x_min = float(data['x_min'])
        y_min = float(data['y_min'])
        x_center = x_min + (nx * cell_size) / 2
        y_center = y_min + (ny * cell_size) / 2

        ax.set_title(f'Irwon Landslide (TM: {x_center:.0f}, {y_center:.0f}) - t = {times[frame]:.1f}s\n'
                    f'Particles: {n_act}, Speed: mean={mean_spd:.1f}, max={max_spd:.1f} m/s')

        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

        # Equal aspect ratio
        x_range = X.max() - X.min()
        y_range = Y.max() - Y.min()
        z_range = terrain.max() - terrain.min()
        max_range = max(x_range, y_range, z_range)
        ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])

        return []

    print(f"Generating {n_frames} frames...")
    print(f"  View: elev={VIEW_ELEV}, azim={VIEW_AZIM}")
    print(f"  FPS: {FPS}, DPI: {DPI}")

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=1000//FPS, blit=False)

    print(f"Saving to {output_file}...")
    anim.save(output_file, writer='pillow', fps=FPS, dpi=DPI)
    print(f"Animation saved!")

    plt.close()


def plot_runout_distance(data, output_file='runout_distance.png'):
    """Plot runout distance over time."""

    # Extract data
    times = data['times']
    x_data = data['x']
    y_data = data['y']
    vx_data = data['vx']
    vy_data = data['vy']
    init_x_local = float(data['init_x']) - float(data['x_min'])
    init_y_local = float(data['init_y']) - float(data['y_min'])

    n_frames = len(times)
    runout_distances = []
    mean_distances = []
    max_speeds = []

    for i in range(n_frames):
        mask = ~np.isnan(x_data[i])
        px = x_data[i, mask]
        py = y_data[i, mask]
        vx = vx_data[i, mask]
        vy = vy_data[i, mask]

        if len(px) > 0:
            distances = np.sqrt((px - init_x_local)**2 + (py - init_y_local)**2)
            runout_head = distances.max()
            mean_dist = distances.mean()
            speeds = np.sqrt(vx**2 + vy**2)
            max_spd = speeds.max()
        else:
            runout_head = 0
            mean_dist = 0
            max_spd = 0

        runout_distances.append(runout_head)
        mean_distances.append(mean_dist)
        max_speeds.append(max_spd)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(times, runout_distances, 'b-', linewidth=2, label='Head (max)')
    ax1.plot(times, mean_distances, 'g--', linewidth=1.5, label='Centroid (mean)')
    ax1.set_ylabel('Runout Distance (m)')
    ax1.set_title('Landslide Runout Distance Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, max_speeds, 'r-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Max Speed (m/s)')
    ax2.set_title('Maximum Particle Speed Over Time')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Runout plot saved to {output_file}")
    print(f"Final runout (head): {runout_distances[-1]:.1f} m")
    print(f"Final runout (centroid): {mean_distances[-1]:.1f} m")
    plt.close()


def main():
    print("=" * 60)
    print("LANDSLIDE VISUALIZATION")
    print("=" * 60)

    # Load saved results
    results_file = 'D:/Claude/landslide/simulation_results.npz'
    print(f"Loading results from {results_file}...")
    data = load_results(results_file)

    print(f"  Frames: {len(data['times'])}")
    print(f"  Duration: {data['times'][-1]:.1f}s")
    print(f"  Terrain: {data['terrain'].shape}")

    # Create visualizations
    print("\nCreating 3D animation...")
    create_3d_animation(data)

    print("\nCreating runout distance plot...")
    plot_runout_distance(data)

    print("\nDone!")


if __name__ == '__main__':
    main()
