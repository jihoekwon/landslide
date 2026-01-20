"""
Test drho/dt method on the original DEM with pit.
Run 1 second simulation and report results.
"""

import numpy as np
import matplotlib.pyplot as plt
from landslide_sph import SPHLandslideSimulator, create_terrain_with_pit


def run_original_dem_test():
    """Run 1 second simulation on original DEM."""
    print("=" * 60)
    print("DRHO/DT TEST ON ORIGINAL DEM (1 second)")
    print("=" * 60)

    # Create terrain with pit (same as main())
    print("\n[1/4] Creating terrain with pit...")
    terrain, pit_location, peak_location = create_terrain_with_pit()
    cell_size = 10.0

    print(f"Terrain shape: {terrain.shape}")
    print(f"Elevation range: {terrain.min():.1f} - {terrain.max():.1f} m")
    print(f"Peak location: {peak_location}")
    print(f"Pit location: {pit_location}")

    # Initialize simulator
    print("\n[2/4] Initializing SPH simulator...")
    simulator = SPHLandslideSimulator(terrain, cell_size)

    # Reduce friction for flow (same as successful test)
    simulator.mu_b = 0.1
    simulator.tau_y = 50.0
    simulator.warmup_time = 0.0
    print(f"Friction: mu_b={simulator.mu_b}, tau_y={simulator.tau_y}")

    # Initialize landslide on the slope (same as main())
    slope_fraction = 0.4
    landslide_center = (
        peak_location[0] + (pit_location[0] - peak_location[0]) * slope_fraction,
        peak_location[1] + (pit_location[1] - peak_location[1]) * slope_fraction
    )
    landslide_radius = 50
    landslide_thickness = 8.0

    print(f"Landslide center: ({landslide_center[0]:.0f}, {landslide_center[1]:.0f})")

    simulator.initialize_particles(landslide_center, landslide_radius, landslide_thickness)

    # Record initial state
    p = simulator.particles
    initial_x = p.x.copy()
    initial_y = p.y.copy()
    initial_centroid = (initial_x.mean(), initial_y.mean())
    n_particles = p.n

    # Calculate expected flow direction (towards pit)
    dir_to_pit_x = pit_location[0] - landslide_center[0]
    dir_to_pit_y = pit_location[1] - landslide_center[1]
    dir_mag = np.sqrt(dir_to_pit_x**2 + dir_to_pit_y**2)
    expected_dir_x = dir_to_pit_x / dir_mag
    expected_dir_y = dir_to_pit_y / dir_mag

    print(f"Number of particles: {n_particles}")
    print(f"Initial centroid: ({initial_centroid[0]:.1f}, {initial_centroid[1]:.1f})")
    print(f"Expected flow direction (to pit): ({expected_dir_x:.3f}, {expected_dir_y:.3f})")

    # Store history
    positions_history = [(0.0, p.x.copy(), p.y.copy(), p.vx.copy(), p.vy.copy())]

    # Run 1 second simulation
    print("\n[3/4] Running simulation for 1.0s...")
    duration = 1.0
    dt = simulator.dt
    n_steps = int(duration / dt)

    for step in range(n_steps):
        avg_speed = simulator.step()

        # Record every 0.1 seconds
        if (step + 1) % int(0.1 / dt) == 0:
            positions_history.append((
                simulator.time,
                p.x.copy(),
                p.y.copy(),
                p.vx.copy(),
                p.vy.copy()
            ))
            speed = np.sqrt(p.vx**2 + p.vy**2)
            print(f"  t={simulator.time:.1f}s: avg_speed={avg_speed:.2f} m/s, max_speed={speed.max():.2f} m/s")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    final_x = p.x
    final_y = p.y
    final_vx = p.vx
    final_vy = p.vy
    final_centroid = (final_x.mean(), final_y.mean())

    # 1. Centroid movement
    centroid_dx = final_centroid[0] - initial_centroid[0]
    centroid_dy = final_centroid[1] - initial_centroid[1]
    centroid_displacement = np.sqrt(centroid_dx**2 + centroid_dy**2)

    print(f"\n1. Centroid Movement:")
    print(f"   Initial: ({initial_centroid[0]:.1f}, {initial_centroid[1]:.1f})")
    print(f"   Final:   ({final_centroid[0]:.1f}, {final_centroid[1]:.1f})")
    print(f"   Displacement: {centroid_displacement:.2f} m")

    # 2. Flow direction
    if centroid_displacement > 0.1:
        actual_dir_x = centroid_dx / centroid_displacement
        actual_dir_y = centroid_dy / centroid_displacement
        dot_product = actual_dir_x * expected_dir_x + actual_dir_y * expected_dir_y
        angle_deg = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))

        print(f"\n2. Flow Direction:")
        print(f"   Expected (to pit): ({expected_dir_x:.3f}, {expected_dir_y:.3f})")
        print(f"   Actual:            ({actual_dir_x:.3f}, {actual_dir_y:.3f})")
        print(f"   Angle difference: {angle_deg:.1f} degrees")
    else:
        print(f"\n2. Flow Direction: Minimal movement")
        angle_deg = None

    # 3. Spread check (explosion)
    initial_spread = np.std(np.sqrt((initial_x - initial_centroid[0])**2 +
                                     (initial_y - initial_centroid[1])**2))
    final_spread = np.std(np.sqrt((final_x - final_centroid[0])**2 +
                                   (final_y - final_centroid[1])**2))
    spread_ratio = final_spread / initial_spread if initial_spread > 0 else 1.0

    print(f"\n3. Particle Spread:")
    print(f"   Initial: {initial_spread:.2f} m")
    print(f"   Final:   {final_spread:.2f} m")
    print(f"   Change: {100*(spread_ratio-1):+.1f}%")

    # 4. Velocity analysis
    speed = np.sqrt(final_vx**2 + final_vy**2)
    print(f"\n4. Velocity at t=1.0s:")
    print(f"   Mean speed: {speed.mean():.2f} m/s")
    print(f"   Max speed:  {speed.max():.2f} m/s")

    # 5. Distance to pit
    dist_to_pit_initial = np.sqrt((initial_centroid[0] - pit_location[0])**2 +
                                   (initial_centroid[1] - pit_location[1])**2)
    dist_to_pit_final = np.sqrt((final_centroid[0] - pit_location[0])**2 +
                                 (final_centroid[1] - pit_location[1])**2)

    print(f"\n5. Distance to Pit:")
    print(f"   Initial: {dist_to_pit_initial:.1f} m")
    print(f"   Final:   {dist_to_pit_final:.1f} m")
    print(f"   Progress: {dist_to_pit_initial - dist_to_pit_final:.1f} m closer")

    # Create visualization
    print("\n[4/4] Creating visualization...")
    create_visualization(terrain, cell_size, positions_history,
                        pit_location, peak_location, expected_dir_x, expected_dir_y)

    # Verdict
    print("\n" + "=" * 60)
    print("TEST VERDICT")
    print("=" * 60)

    no_explosion = spread_ratio < 1.5
    moving_towards_pit = dist_to_pit_final < dist_to_pit_initial

    if no_explosion:
        print("[PASS] No explosion - particles stayed together")
    else:
        print("[FAIL] Explosion detected")

    if moving_towards_pit:
        print("[PASS] Moving towards pit (downslope)")
    else:
        print("[FAIL] Not moving towards pit")

    if no_explosion and moving_towards_pit:
        print("\nOVERALL: SUCCESS")
    else:
        print("\nOVERALL: NEEDS REVIEW")

    return positions_history


def create_visualization(terrain, cell_size, positions_history,
                        pit_location, peak_location, expected_dir_x, expected_dir_y):
    """Create visualization similar to original animation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    rows, cols = terrain.shape
    extent = [0, cols * cell_size, 0, rows * cell_size]

    # Get initial and final states
    t0, x0, y0, vx0, vy0 = positions_history[0]
    tf, xf, yf, vxf, vyf = positions_history[-1]

    # Plot 1: Initial vs Final with terrain
    ax = axes[0, 0]
    im = ax.imshow(terrain, cmap='terrain', origin='lower', extent=extent, alpha=0.8)
    ax.contour(np.linspace(0, cols*cell_size, cols),
               np.linspace(0, rows*cell_size, rows),
               terrain, levels=15, colors='gray', alpha=0.3, linewidths=0.5)

    ax.scatter(x0, y0, c='blue', s=20, alpha=0.6, label=f't=0s (n={len(x0)})')
    ax.scatter(xf, yf, c='red', s=20, alpha=0.6, label=f't={tf:.1f}s')

    # Mark pit and peak
    circle = plt.Circle(pit_location, 120, fill=False, color='blue',
                        linestyle='--', linewidth=2)
    ax.add_patch(circle)
    ax.plot(pit_location[0], pit_location[1], 'bx', markersize=12, markeredgewidth=3, label='Pit')
    ax.plot(peak_location[0], peak_location[1], 'g^', markersize=12, label='Peak')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Initial (blue) vs Final (red) Positions')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Elevation (m)')

    # Plot 2: Trajectories
    ax = axes[0, 1]
    ax.imshow(terrain, cmap='terrain', origin='lower', extent=extent, alpha=0.7)

    n_particles = len(x0)
    sample_indices = np.linspace(0, n_particles-1, min(40, n_particles)).astype(int)

    for idx in sample_indices:
        traj_x = [h[1][idx] for h in positions_history]
        traj_y = [h[2][idx] for h in positions_history]
        ax.plot(traj_x, traj_y, 'b-', alpha=0.5, linewidth=1)

    # Mark pit
    circle2 = plt.Circle(pit_location, 120, fill=False, color='blue',
                         linestyle='--', linewidth=2)
    ax.add_patch(circle2)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Particle Trajectories (1 second)')
    ax.set_aspect('equal')

    # Plot 3: Final velocity vectors
    ax = axes[1, 0]
    ax.imshow(terrain, cmap='terrain', origin='lower', extent=extent, alpha=0.7)

    speed = np.sqrt(vxf**2 + vyf**2)
    scatter = ax.scatter(xf, yf, c=speed, cmap='hot', s=25, alpha=0.8)

    step = max(1, len(xf) // 50)
    ax.quiver(xf[::step], yf[::step], vxf[::step], vyf[::step],
              scale=100, alpha=0.7, color='blue', width=0.003)

    circle3 = plt.Circle(pit_location, 120, fill=False, color='blue',
                         linestyle='--', linewidth=2)
    ax.add_patch(circle3)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Velocity Vectors at t={tf:.1f}s\nMax: {speed.max():.1f} m/s')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Speed (m/s)')

    # Plot 4: Centroid and speed over time
    ax = axes[1, 1]
    times = [h[0] for h in positions_history]
    centroid_x = [h[1].mean() for h in positions_history]
    centroid_y = [h[2].mean() for h in positions_history]
    mean_speeds = [np.sqrt(h[3]**2 + h[4]**2).mean() for h in positions_history]

    ax2 = ax.twinx()

    l1, = ax.plot(times, centroid_x, 'b-', linewidth=2, label='X centroid')
    l2, = ax.plot(times, centroid_y, 'r-', linewidth=2, label='Y centroid')
    l3, = ax2.plot(times, mean_speeds, 'g--', linewidth=2, label='Mean speed')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Centroid Position (m)')
    ax2.set_ylabel('Speed (m/s)', color='green')
    ax.set_title('Centroid Movement & Speed Over Time')

    lines = [l1, l2, l3]
    ax.legend(lines, [l.get_label() for l in lines], loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_original_dem_result.png', dpi=150)
    print("Visualization saved to test_original_dem_result.png")
    plt.close()


if __name__ == '__main__':
    run_original_dem_test()
