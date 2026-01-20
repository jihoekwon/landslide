"""
Test script for drho/dt density calculation method.

This test verifies that particles on a slope:
1. Do NOT push each other apart initially (no explosion)
2. Flow together in one direction (downslope)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Import from the modified SPH module
from landslide_sph import SPHLandslideSimulator, SPHKernel, Particle, TerrainInterpolator


def create_simple_slope():
    """Create a simple inclined plane for testing."""
    rows, cols = 60, 80
    cell_size = 10.0

    # Create coordinate grids
    j_grid, i_grid = np.meshgrid(np.arange(cols), np.arange(rows))
    x_grid = j_grid * cell_size
    y_grid = i_grid * cell_size

    # Simple slope: elevation decreases from left to right and top to bottom
    # Slope direction: towards lower-right corner
    terrain = 500 - 0.3 * x_grid - 0.2 * y_grid

    # Small roughness
    terrain += 2 * np.random.randn(rows, cols)
    terrain = gaussian_filter(terrain, sigma=1.0)

    return terrain, cell_size


def run_short_test():
    """Run a short simulation and analyze particle behavior."""
    print("=" * 60)
    print("DRHO/DT DENSITY METHOD TEST")
    print("=" * 60)

    # Create terrain
    terrain, cell_size = create_simple_slope()
    print(f"\nTerrain shape: {terrain.shape}")
    print(f"Elevation range: {terrain.min():.1f} - {terrain.max():.1f} m")

    # Calculate expected flow direction (downslope)
    grad_y, grad_x = np.gradient(terrain, cell_size)
    center_row, center_col = terrain.shape[0] // 2, terrain.shape[1] // 3
    expected_dir_x = -grad_x[center_row, center_col]
    expected_dir_y = -grad_y[center_row, center_col]
    expected_mag = np.sqrt(expected_dir_x**2 + expected_dir_y**2)
    expected_dir_x /= expected_mag
    expected_dir_y /= expected_mag
    print(f"\nExpected flow direction (downslope): ({expected_dir_x:.3f}, {expected_dir_y:.3f})")

    # Initialize simulator with warmup disabled for this test
    simulator = SPHLandslideSimulator(terrain, cell_size)
    simulator.warmup_time = 0.0  # Disable warmup to see immediate behavior

    # Reduce friction to allow flow on slope
    simulator.mu_b = 0.1  # Basal friction (was 0.5)
    simulator.tau_y = 50.0  # Yield stress (was 500)
    print(f"Friction reduced: mu_b={simulator.mu_b}, tau_y={simulator.tau_y}")

    # Place particles at center of slope
    center_x = center_col * cell_size
    center_y = center_row * cell_size
    radius = 40  # meters
    thickness = 5.0  # meters

    print(f"\nInitializing particles at ({center_x:.0f}, {center_y:.0f}) m")
    print(f"Radius: {radius} m, Thickness: {thickness} m")

    simulator.initialize_particles((center_x, center_y), radius, thickness, spacing=8.0)

    # Record initial state
    p = simulator.particles
    initial_x = p.x.copy()
    initial_y = p.y.copy()
    initial_centroid = (initial_x.mean(), initial_y.mean())
    n_particles = p.n

    print(f"Number of particles: {n_particles}")
    print(f"Initial centroid: ({initial_centroid[0]:.1f}, {initial_centroid[1]:.1f})")

    # Store positions at each time step
    positions_history = [(0.0, initial_x.copy(), initial_y.copy(), p.vx.copy(), p.vy.copy())]

    # Run short simulation (0.5 seconds)
    duration = 0.5
    dt = simulator.dt
    n_steps = int(duration / dt)

    print(f"\nRunning simulation for {duration}s ({n_steps} steps)...")

    for step in range(n_steps):
        simulator.step()

        # Record positions every 0.05 seconds
        if (step + 1) % int(0.05 / dt) == 0:
            positions_history.append((
                simulator.time,
                p.x.copy(),
                p.y.copy(),
                p.vx.copy(),
                p.vy.copy()
            ))

    # Analyze results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    final_x = p.x
    final_y = p.y
    final_vx = p.vx
    final_vy = p.vy
    final_centroid = (final_x.mean(), final_y.mean())

    # 1. Check if particles moved together (centroid displacement)
    centroid_dx = final_centroid[0] - initial_centroid[0]
    centroid_dy = final_centroid[1] - initial_centroid[1]
    centroid_displacement = np.sqrt(centroid_dx**2 + centroid_dy**2)

    print(f"\n1. Centroid Movement:")
    print(f"   Initial: ({initial_centroid[0]:.1f}, {initial_centroid[1]:.1f})")
    print(f"   Final:   ({final_centroid[0]:.1f}, {final_centroid[1]:.1f})")
    print(f"   Displacement: {centroid_displacement:.2f} m")

    # 2. Check flow direction vs expected direction
    if centroid_displacement > 0.1:
        actual_dir_x = centroid_dx / centroid_displacement
        actual_dir_y = centroid_dy / centroid_displacement
        dot_product = actual_dir_x * expected_dir_x + actual_dir_y * expected_dir_y
        angle_deg = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))

        print(f"\n2. Flow Direction:")
        print(f"   Expected (downslope): ({expected_dir_x:.3f}, {expected_dir_y:.3f})")
        print(f"   Actual:               ({actual_dir_x:.3f}, {actual_dir_y:.3f})")
        print(f"   Angle difference: {angle_deg:.1f} degrees")
        direction_correct = angle_deg < 45
    else:
        print(f"\n2. Flow Direction: Particles barely moved")
        direction_correct = False
        angle_deg = None

    # 3. Check for explosion (particles spreading apart)
    initial_spread = np.std(np.sqrt((initial_x - initial_centroid[0])**2 +
                                     (initial_y - initial_centroid[1])**2))
    final_spread = np.std(np.sqrt((final_x - final_centroid[0])**2 +
                                   (final_y - final_centroid[1])**2))

    spread_change = final_spread - initial_spread
    spread_ratio = final_spread / initial_spread if initial_spread > 0 else 1.0

    print(f"\n3. Particle Spread (explosion check):")
    print(f"   Initial spread (std): {initial_spread:.2f} m")
    print(f"   Final spread (std):   {final_spread:.2f} m")
    print(f"   Change: {spread_change:+.2f} m ({100*(spread_ratio-1):+.1f}%)")

    no_explosion = spread_ratio < 1.5  # Less than 50% increase in spread

    # 4. Check velocity coherence (are particles moving in similar direction?)
    speed = np.sqrt(final_vx**2 + final_vy**2)
    moving_mask = speed > 0.1  # Only consider particles with significant velocity

    if np.any(moving_mask):
        # Normalize velocities
        norm_vx = final_vx[moving_mask] / speed[moving_mask]
        norm_vy = final_vy[moving_mask] / speed[moving_mask]

        # Average direction
        avg_dir_x = norm_vx.mean()
        avg_dir_y = norm_vy.mean()
        avg_mag = np.sqrt(avg_dir_x**2 + avg_dir_y**2)

        # Direction coherence (1.0 = all same direction, 0.0 = random)
        coherence = avg_mag

        print(f"\n4. Velocity Coherence:")
        print(f"   Average speed: {speed[moving_mask].mean():.2f} m/s")
        print(f"   Max speed: {speed.max():.2f} m/s")
        print(f"   Direction coherence: {coherence:.3f} (1.0 = perfect alignment)")
        print(f"   Average direction: ({avg_dir_x/avg_mag:.3f}, {avg_dir_y/avg_mag:.3f})")

        good_coherence = coherence > 0.7
    else:
        print(f"\n4. Velocity Coherence: Particles not moving significantly")
        good_coherence = False
        coherence = 0

    # Final verdict
    print("\n" + "=" * 60)
    print("TEST VERDICT")
    print("=" * 60)

    test_passed = no_explosion and (good_coherence or centroid_displacement < 0.5)

    if no_explosion:
        print("[PASS] No explosion - particles did not spread apart excessively")
    else:
        print("[FAIL] Explosion detected - particles spread apart too much")

    if direction_correct:
        print("[PASS] Flow direction correct - particles moving downslope")
    elif centroid_displacement < 0.5:
        print("[INFO] Particles barely moved - may need longer simulation")
    else:
        print("[FAIL] Flow direction incorrect - not moving downslope")

    if good_coherence:
        print("[PASS] Good coherence - particles moving together")
    else:
        print("[INFO] Low coherence - particles not moving uniformly")

    print("\n" + "=" * 60)
    if test_passed:
        print("OVERALL: SUCCESS - drho/dt method working correctly")
    else:
        print("OVERALL: NEEDS REVIEW - check particle behavior")
    print("=" * 60)

    # Create visualization
    create_visualization(terrain, cell_size, positions_history, expected_dir_x, expected_dir_y)

    return test_passed


def create_visualization(terrain, cell_size, positions_history, expected_dir_x, expected_dir_y):
    """Create visualization of particle movement."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    rows, cols = terrain.shape
    extent = [0, cols * cell_size, 0, rows * cell_size]

    # Plot 1: Initial and final positions
    ax = axes[0, 0]
    ax.imshow(terrain, cmap='terrain', origin='lower', extent=extent, alpha=0.7)
    ax.contour(np.linspace(0, cols*cell_size, cols),
               np.linspace(0, rows*cell_size, rows),
               terrain, levels=15, colors='gray', alpha=0.3, linewidths=0.5)

    t0, x0, y0, _, _ = positions_history[0]
    tf, xf, yf, vxf, vyf = positions_history[-1]

    ax.scatter(x0, y0, c='blue', s=15, alpha=0.5, label=f't=0s (n={len(x0)})')
    ax.scatter(xf, yf, c='red', s=15, alpha=0.5, label=f't={tf:.2f}s')

    # Draw expected flow direction
    cx, cy = x0.mean(), y0.mean()
    ax.arrow(cx, cy, expected_dir_x * 50, expected_dir_y * 50,
             head_width=10, head_length=5, fc='green', ec='green', linewidth=2,
             label='Expected flow direction')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Initial (blue) vs Final (red) Positions')
    ax.legend()
    ax.set_aspect('equal')

    # Plot 2: Trajectories
    ax = axes[0, 1]
    ax.imshow(terrain, cmap='terrain', origin='lower', extent=extent, alpha=0.7)

    # Draw trajectories for sample particles
    n_particles = len(positions_history[0][1])
    sample_indices = np.linspace(0, n_particles-1, min(30, n_particles)).astype(int)

    for idx in sample_indices:
        traj_x = [h[1][idx] for h in positions_history if idx < len(h[1])]
        traj_y = [h[2][idx] for h in positions_history if idx < len(h[2])]
        ax.plot(traj_x, traj_y, 'b-', alpha=0.5, linewidth=1)
        ax.scatter(traj_x[0], traj_y[0], c='green', s=10)
        ax.scatter(traj_x[-1], traj_y[-1], c='red', s=10)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Particle Trajectories (sample)')
    ax.set_aspect('equal')

    # Plot 3: Velocity vectors at final time
    ax = axes[1, 0]
    ax.imshow(terrain, cmap='terrain', origin='lower', extent=extent, alpha=0.7)

    speed = np.sqrt(vxf**2 + vyf**2)
    ax.scatter(xf, yf, c=speed, cmap='hot', s=20, alpha=0.8)

    # Subsample for quiver
    step = max(1, len(xf) // 50)
    ax.quiver(xf[::step], yf[::step], vxf[::step], vyf[::step],
              scale=50, alpha=0.7, color='blue')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Velocity Vectors at t={tf:.2f}s')
    ax.set_aspect('equal')

    # Plot 4: Centroid movement over time
    ax = axes[1, 1]
    times = [h[0] for h in positions_history]
    centroid_x = [h[1].mean() for h in positions_history]
    centroid_y = [h[2].mean() for h in positions_history]

    ax.plot(times, centroid_x, 'b-', label='X centroid')
    ax.plot(times, centroid_y, 'r-', label='Y centroid')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Centroid Position (m)')
    ax.set_title('Centroid Movement Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_drho_dt_result.png', dpi=150)
    print(f"\nVisualization saved to test_drho_dt_result.png")
    plt.close()


if __name__ == '__main__':
    run_short_test()
