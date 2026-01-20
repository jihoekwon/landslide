"""
SPH (Smoothed Particle Hydrodynamics) Landslide Simulation
===========================================================

This module implements a 2D SPH-based landslide/debris flow simulation.

SPH represents the material as discrete particles and computes:
- Density using kernel interpolation
- Pressure from equation of state (Tait EOS)
- Pressure gradient forces
- Viscous forces (non-Newtonian Bingham-like rheology)
- Basal friction (Voellmy model)
- Gravity

References:
- Monaghan, J.J. (1992) "Smoothed Particle Hydrodynamics"
- Pastor et al. (2009) "SPH simulation of debris flows"
- Cascini et al. (2014) "SPH-based landslide modeling"

Author: Claude Code Demo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')


class SPHKernel:
    """
    SPH Kernel functions for interpolation.

    Implements the Cubic Spline (M4) kernel which is widely used
    for its good stability properties.
    """

    def __init__(self, h):
        """
        Initialize kernel with smoothing length h.

        Args:
            h: Smoothing length (influence radius = 2h)
        """
        self.h = h
        self.h2 = h * h
        # Normalization factor for 2D cubic spline
        self.alpha = 10.0 / (7.0 * np.pi * self.h2)

    def W(self, r):
        """
        Cubic spline kernel function.

        Args:
            r: Distance between particles

        Returns:
            Kernel value W(r, h)
        """
        q = r / self.h
        result = np.zeros_like(q, dtype=float)

        # 0 <= q < 1
        mask1 = (q >= 0) & (q < 1)
        result[mask1] = 1.0 - 1.5 * q[mask1]**2 + 0.75 * q[mask1]**3

        # 1 <= q < 2
        mask2 = (q >= 1) & (q < 2)
        result[mask2] = 0.25 * (2.0 - q[mask2])**3

        return self.alpha * result

    def gradW(self, r, dx, dy):
        """
        Gradient of cubic spline kernel.

        Args:
            r: Distance between particles
            dx, dy: Position difference (xi - xj, yi - yj)

        Returns:
            Tuple of (dW/dx, dW/dy)
        """
        q = r / self.h

        # Compute dW/dr
        dWdr = np.zeros_like(q, dtype=float)

        # 0 <= q < 1
        mask1 = (q > 1e-10) & (q < 1)
        dWdr[mask1] = (-3.0 * q[mask1] + 2.25 * q[mask1]**2) / self.h

        # 1 <= q < 2
        mask2 = (q >= 1) & (q < 2)
        dWdr[mask2] = -0.75 * (2.0 - q[mask2])**2 / self.h

        dWdr *= self.alpha

        # Convert to gradient: gradW = (dW/dr) * (r_vec / |r|)
        # Avoid division by zero
        r_safe = np.maximum(r, 1e-10)
        gradWx = dWdr * dx / r_safe
        gradWy = dWdr * dy / r_safe

        return gradWx, gradWy


class Particle:
    """Container for particle data using Structure of Arrays (SoA) layout."""

    def __init__(self, n_particles):
        self.n = n_particles

        # Position
        self.x = np.zeros(n_particles)
        self.y = np.zeros(n_particles)

        # Velocity
        self.vx = np.zeros(n_particles)
        self.vy = np.zeros(n_particles)

        # Acceleration
        self.ax = np.zeros(n_particles)
        self.ay = np.zeros(n_particles)

        # Physical properties
        self.mass = np.zeros(n_particles)
        self.density = np.zeros(n_particles)
        self.drho_dt = np.zeros(n_particles)  # Density rate of change for continuity equation
        self.pressure = np.zeros(n_particles)

        # Terrain height at particle position
        self.terrain_z = np.zeros(n_particles)

        # Particle height (thickness of debris)
        self.height = np.zeros(n_particles)

        # Active flag
        self.active = np.ones(n_particles, dtype=bool)


class TerrainInterpolator:
    """Interpolate terrain elevation and compute gradients."""

    def __init__(self, dem, cell_size, origin=(0, 0)):
        """
        Initialize terrain interpolator.

        Args:
            dem: 2D numpy array of elevations
            cell_size: Size of each DEM cell in meters
            origin: (x, y) coordinates of DEM origin
        """
        self.dem = dem.astype(float)
        self.cell_size = cell_size
        self.origin = origin
        self.rows, self.cols = dem.shape

        # Precompute gradients
        # np.gradient returns (gradient along rows, gradient along cols)
        # rows = y direction, cols = x direction
        self.grad_y, self.grad_x = np.gradient(dem, cell_size)

    def get_elevation(self, x, y):
        """Get interpolated elevation at (x, y) coordinates."""
        # Convert to grid indices
        col = (x - self.origin[0]) / self.cell_size
        row = (y - self.origin[1]) / self.cell_size

        # Clamp to valid range
        col = np.clip(col, 0, self.cols - 1.001)
        row = np.clip(row, 0, self.rows - 1.001)

        # Bilinear interpolation
        col0 = np.floor(col).astype(int)
        row0 = np.floor(row).astype(int)
        col1 = np.minimum(col0 + 1, self.cols - 1)
        row1 = np.minimum(row0 + 1, self.rows - 1)

        # Interpolation weights
        wx = col - col0
        wy = row - row0

        # Interpolate
        z = (self.dem[row0, col0] * (1 - wx) * (1 - wy) +
             self.dem[row0, col1] * wx * (1 - wy) +
             self.dem[row1, col0] * (1 - wx) * wy +
             self.dem[row1, col1] * wx * wy)

        return z

    def get_gradient(self, x, y):
        """Get terrain gradient (dz/dx, dz/dy) at (x, y) coordinates."""
        col = (x - self.origin[0]) / self.cell_size
        row = (y - self.origin[1]) / self.cell_size

        col = np.clip(col, 0, self.cols - 1.001)
        row = np.clip(row, 0, self.rows - 1.001)

        col0 = np.floor(col).astype(int)
        row0 = np.floor(row).astype(int)
        col1 = np.minimum(col0 + 1, self.cols - 1)
        row1 = np.minimum(row0 + 1, self.rows - 1)

        wx = col - col0
        wy = row - row0

        # Interpolate gradients
        gx = (self.grad_x[row0, col0] * (1 - wx) * (1 - wy) +
              self.grad_x[row0, col1] * wx * (1 - wy) +
              self.grad_x[row1, col0] * (1 - wx) * wy +
              self.grad_x[row1, col1] * wx * wy)

        gy = (self.grad_y[row0, col0] * (1 - wx) * (1 - wy) +
              self.grad_y[row0, col1] * wx * (1 - wy) +
              self.grad_y[row1, col0] * (1 - wx) * wy +
              self.grad_y[row1, col1] * wx * wy)

        return gx, gy


class SPHLandslideSimulator:
    """
    SPH-based landslide/debris flow simulator.

    Uses weakly compressible SPH (WCSPH) with:
    - Tait equation of state for pressure
    - Artificial viscosity for stability
    - Bingham-like rheology for debris behavior
    - Voellmy friction model for basal resistance
    """

    def __init__(self, terrain, cell_size=10.0):
        """
        Initialize the SPH simulator.

        Args:
            terrain: 2D numpy array of terrain elevation
            cell_size: Size of terrain cells in meters
        """
        self.terrain = TerrainInterpolator(terrain, cell_size)
        self.cell_size = cell_size
        self.domain_width = terrain.shape[1] * cell_size
        self.domain_height = terrain.shape[0] * cell_size

        # SPH parameters
        self.h = 15.0  # Smoothing length (meters)
        self.kernel = SPHKernel(self.h)

        # Physical parameters
        self.rho0 = 2000.0  # Reference density (kg/m³) - typical debris
        self.c0 = 50.0  # Speed of sound (m/s) - for weakly compressible
        self.gamma = 7.0  # Tait EOS exponent
        self.g = 9.81  # Gravity (m/s²)

        # Rheology parameters (Bingham-like)
        self.mu = 100.0  # Dynamic viscosity (Pa·s)
        self.tau_y = 500.0  # Yield stress (Pa) - debris starts flowing above this

        # Friction parameters (Voellmy model) - increased for stability
        self.mu_b = 0.5  # Basal friction coefficient (tan of friction angle ~27°)
        self.xi = 200.0  # Turbulent friction coefficient (m/s²) - lower = more friction

        # Velocity ceiling to prevent explosion
        self.v_max = 30.0  # Maximum velocity (m/s)

        # Artificial viscosity (for numerical stability)
        self.alpha_visc = 0.1
        self.beta_visc = 0.1

        # Time integration
        self.dt = 0.005  # Time step (seconds)

        # Warm-up period: disable pressure force initially to prevent explosion
        self.warmup_time = 0.5  # seconds - pressure force gradually enabled

        # Particles
        self.particles = None

        # History for visualization
        self.history = []
        self.time = 0.0

    def initialize_particles(self, center, radius, thickness, spacing=None):
        """
        Initialize particles in a circular region.

        Args:
            center: (x, y) center of landslide source
            radius: Radius of source region in meters
            thickness: Initial debris thickness in meters
            spacing: Particle spacing (default: h/2)
        """
        if spacing is None:
            spacing = self.h / 2.0

        # Generate particle positions in a grid within the circle
        x_range = np.arange(center[0] - radius, center[0] + radius, spacing)
        y_range = np.arange(center[1] - radius, center[1] + radius, spacing)

        positions = []
        for x in x_range:
            for y in y_range:
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if dist < radius:
                    # Gaussian thickness distribution
                    h_local = thickness * np.exp(-dist**2 / (2 * (radius/2)**2))
                    if h_local > 0.1:  # Minimum thickness threshold
                        positions.append((x, y, h_local))

        n_particles = len(positions)
        print(f"Initialized {n_particles} particles")

        self.particles = Particle(n_particles)

        for i, (x, y, h) in enumerate(positions):
            self.particles.x[i] = x
            self.particles.y[i] = y
            self.particles.height[i] = h
            self.particles.terrain_z[i] = self.terrain.get_elevation(x, y)

            # Mass = density * volume (volume = area * height)
            particle_area = spacing * spacing
            self.particles.mass[i] = self.rho0 * particle_area * h
            self.particles.density[i] = self.rho0

        self._save_state()

        total_volume = np.sum(self.particles.height * spacing * spacing)
        print(f"Total debris volume: {total_volume:.0f} m³")

    def _find_neighbors(self):
        """Find neighbors within 2h using KD-tree (only active particles)."""
        p = self.particles
        active = p.active

        if not np.any(active):
            return np.array([]).reshape(0, 2).astype(int), None, np.array([])

        # Get indices of active particles
        active_indices = np.where(active)[0]

        # Build tree with only active particles
        positions = np.column_stack([p.x[active], p.y[active]])
        tree = cKDTree(positions)

        # Query all pairs within 2h
        pairs_local = tree.query_pairs(r=2.0 * self.h, output_type='ndarray')

        # Convert local indices back to global indices
        if len(pairs_local) > 0:
            pairs_global = active_indices[pairs_local]
        else:
            pairs_global = np.array([]).reshape(0, 2).astype(int)

        return pairs_global, tree, active_indices

    def _compute_drho_dt(self, pairs):
        """
        Compute density rate of change using continuity equation.

        Uses velocity divergence approximation:
        dρ/dt = Σⱼ mⱼ (vᵢ - vⱼ) · ∇Wᵢⱼ

        This approach avoids the initial "explosion" problem of summation density
        because stationary particles have zero velocity divergence.
        """
        p = self.particles
        active = p.active

        # Reset drho_dt
        p.drho_dt[active] = 0.0

        # Neighbor contributions to density rate of change
        for i, j in pairs:
            if not (p.active[i] and p.active[j]):
                continue

            dx = p.x[i] - p.x[j]
            dy = p.y[i] - p.y[j]
            r = np.sqrt(dx*dx + dy*dy)

            if r < 1e-10:
                continue

            gradWx, gradWy = self.kernel.gradW(
                np.array([r]), np.array([dx]), np.array([dy])
            )
            gradWx, gradWy = gradWx[0], gradWy[0]

            # Velocity difference
            dvx = p.vx[i] - p.vx[j]
            dvy = p.vy[i] - p.vy[j]

            # dρᵢ/dt += mⱼ (vᵢ - vⱼ) · ∇Wᵢⱼ
            dot_product = dvx * gradWx + dvy * gradWy
            p.drho_dt[i] += p.mass[j] * dot_product
            # For particle j: ∇Wⱼᵢ = -∇Wᵢⱼ, so sign flips
            p.drho_dt[j] += p.mass[i] * dot_product  # Note: same sign because (vj-vi)·(-∇W) = (vi-vj)·∇W

    def _compute_pressure(self):
        """Compute pressure using Tait equation of state (only active particles)."""
        p = self.particles
        active = p.active

        # Tait EOS: P = B * ((rho/rho0)^gamma - 1)
        B = self.rho0 * self.c0**2 / self.gamma
        p.pressure[active] = B * ((p.density[active] / self.rho0)**self.gamma - 1.0)

        # Ensure non-negative pressure
        p.pressure[active] = np.maximum(p.pressure[active], 0.0)

    def _compute_forces(self, pairs):
        """Compute all forces on particles (only active particles)."""
        p = self.particles
        active = p.active

        # Reset acceleration for active particles
        p.ax[active] = 0.0
        p.ay[active] = 0.0

        # Gravity component along terrain slope (only for active)
        if np.any(active):
            gx, gy = self.terrain.get_gradient(p.x[active], p.y[active])

            # Gravity drives flow downslope
            # a_gravity = -g * grad(z) for horizontal components
            p.ax[active] += -self.g * gx
            p.ay[active] += -self.g * gy

        # SPH pressure and viscosity forces from neighbors
        for i, j in pairs:
            if not (p.active[i] and p.active[j]):
                continue
            dx = p.x[i] - p.x[j]
            dy = p.y[i] - p.y[j]
            r = np.sqrt(dx*dx + dy*dy)

            if r < 1e-10:
                continue

            gradWx, gradWy = self.kernel.gradW(
                np.array([r]), np.array([dx]), np.array([dy])
            )
            gradWx, gradWy = gradWx[0], gradWy[0]

            # --- Pressure force ---
            # Symmetric form: -m * (P_i/rho_i^2 + P_j/rho_j^2) * gradW
            # Gradually enable during warmup to prevent initial explosion
            pressure_factor = min(1.0, self.time / self.warmup_time) if self.warmup_time > 0 else 1.0

            pressure_term = (p.pressure[i] / (p.density[i]**2) +
                           p.pressure[j] / (p.density[j]**2))
            pressure_term *= pressure_factor  # Scale by warmup factor

            p.ax[i] -= p.mass[j] * pressure_term * gradWx
            p.ay[i] -= p.mass[j] * pressure_term * gradWy
            p.ax[j] += p.mass[i] * pressure_term * gradWx
            p.ay[j] += p.mass[i] * pressure_term * gradWy

            # --- Viscosity force (artificial + physical) ---
            dvx = p.vx[i] - p.vx[j]
            dvy = p.vy[i] - p.vy[j]

            # Dot product of velocity difference and position difference
            vr_dot = dvx * dx + dvy * dy

            if vr_dot < 0:  # Only for approaching particles
                # Artificial viscosity for stability
                rho_avg = 0.5 * (p.density[i] + p.density[j])
                mu_art = self.h * vr_dot / (r*r + 0.01*self.h*self.h)
                Pi_ij = (-self.alpha_visc * self.c0 * mu_art +
                        self.beta_visc * mu_art**2) / rho_avg

                p.ax[i] -= p.mass[j] * Pi_ij * gradWx
                p.ay[i] -= p.mass[j] * Pi_ij * gradWy
                p.ax[j] += p.mass[i] * Pi_ij * gradWx
                p.ay[j] += p.mass[i] * Pi_ij * gradWy

            # Physical viscosity (laminar)
            visc_term = 2.0 * self.mu * r / (p.density[i] * p.density[j] * (r*r + 0.01*self.h*self.h))

            p.ax[i] += p.mass[j] * visc_term * dvx * (gradWx * dx + gradWy * dy) / r
            p.ay[i] += p.mass[j] * visc_term * dvy * (gradWx * dx + gradWy * dy) / r
            p.ax[j] -= p.mass[i] * visc_term * dvx * (gradWx * dx + gradWy * dy) / r
            p.ay[j] -= p.mass[i] * visc_term * dvy * (gradWx * dx + gradWy * dy) / r

        # --- Basal friction (Voellmy model) --- (only active particles)
        # tau_b = rho * g * h * (mu_b + v²/xi)
        # a_friction = -tau_b / (rho * h) * v_hat

        if not np.any(active):
            return

        speed = np.sqrt(p.vx[active]**2 + p.vy[active]**2)
        speed_safe = np.maximum(speed, 1e-6)

        # Friction coefficient (Coulomb + turbulent)
        friction_coeff = self.mu_b + speed**2 / self.xi

        # Friction force magnitude
        friction_mag = self.g * friction_coeff

        # Apply friction (opposing velocity direction)
        p.ax[active] -= friction_mag * p.vx[active] / speed_safe
        p.ay[active] -= friction_mag * p.vy[active] / speed_safe

        # --- Yield stress (Bingham behavior) ---
        # Material doesn't flow below yield stress
        # Implemented as increased friction at low velocities
        yield_factor = self.tau_y / (self.rho0 * np.maximum(p.height[active], 0.1))
        low_speed_mask = speed < 1.0  # m/s threshold

        # Create full-size mask aligned with active particles
        active_indices = np.where(active)[0]
        low_speed_global = active_indices[low_speed_mask]

        p.ax[low_speed_global] -= yield_factor[low_speed_mask] * p.vx[low_speed_global] / speed_safe[low_speed_mask]
        p.ay[low_speed_global] -= yield_factor[low_speed_mask] * p.vy[low_speed_global] / speed_safe[low_speed_mask]

    def _integrate(self):
        """Time integration using Velocity Verlet scheme."""
        p = self.particles

        # Only update active particles
        active = p.active

        # Update velocity (half step)
        p.vx[active] += 0.5 * self.dt * p.ax[active]
        p.vy[active] += 0.5 * self.dt * p.ay[active]

        # Apply velocity ceiling to prevent explosion
        speed = np.sqrt(p.vx[active]**2 + p.vy[active]**2)
        too_fast = speed > self.v_max
        if np.any(too_fast):
            # Get indices of active particles that are too fast
            active_indices = np.where(active)[0]
            fast_indices = active_indices[too_fast]
            scale = self.v_max / speed[too_fast]
            p.vx[fast_indices] *= scale
            p.vy[fast_indices] *= scale

        # Update position
        p.x[active] += self.dt * p.vx[active]
        p.y[active] += self.dt * p.vy[active]

        # Boundary conditions: deactivate particles that leave domain
        # (no reflection - particles simply exit and are removed from calculation)
        out_left = p.x < 0
        out_right = p.x > self.domain_width
        out_bottom = p.y < 0
        out_top = p.y > self.domain_height

        out_of_bounds = out_left | out_right | out_bottom | out_top
        newly_deactivated = out_of_bounds & p.active

        if np.any(newly_deactivated):
            n_deactivated = np.sum(newly_deactivated)
            p.active[newly_deactivated] = False
            # Note: we don't print every time to avoid spam

        # Update terrain height at new positions (only for active particles)
        active = p.active  # refresh after deactivation
        if np.any(active):
            p.terrain_z[active] = self.terrain.get_elevation(p.x[active], p.y[active])

    def step(self):
        """Perform one simulation step using continuity equation for density."""
        p = self.particles
        active = p.active

        if not np.any(active):
            self.time += self.dt
            return 0.0

        # Find neighbors
        pairs, _, _ = self._find_neighbors()

        # Compute density rate of change (drho/dt) using velocity divergence
        self._compute_drho_dt(pairs)

        # Update density using time integration (half step)
        p.density[active] += 0.5 * self.dt * p.drho_dt[active]
        # Clamp density to avoid negative values
        p.density[active] = np.maximum(p.density[active], 0.5 * self.rho0)

        # Compute pressure
        self._compute_pressure()

        # Compute forces
        self._compute_forces(pairs)

        # Integrate
        self._integrate()

        # Compute new forces for velocity update
        pairs, _, _ = self._find_neighbors()

        # Compute drho/dt again with updated velocities
        self._compute_drho_dt(pairs)

        # Update density (second half step)
        p.density[active] += 0.5 * self.dt * p.drho_dt[active]
        p.density[active] = np.maximum(p.density[active], 0.5 * self.rho0)

        self._compute_pressure()
        self._compute_forces(pairs)

        # Update velocity (second half step) - only active particles
        active = p.active  # refresh after integration
        p.vx[active] += 0.5 * self.dt * p.ax[active]
        p.vy[active] += 0.5 * self.dt * p.ay[active]

        # Apply velocity ceiling again
        if np.any(active):
            speed = np.sqrt(p.vx[active]**2 + p.vy[active]**2)
            too_fast = speed > self.v_max
            if np.any(too_fast):
                active_indices = np.where(active)[0]
                fast_indices = active_indices[too_fast]
                scale = self.v_max / speed[too_fast]
                p.vx[fast_indices] *= scale
                p.vy[fast_indices] *= scale

        self.time += self.dt

        # Return kinetic energy for convergence check (only active particles)
        if np.any(active):
            speed = np.sqrt(p.vx[active]**2 + p.vy[active]**2)
            return np.mean(speed)
        return 0.0

    def _save_state(self):
        """Save current state to history."""
        p = self.particles
        # Only save active particles
        active = p.active
        self.history.append({
            'time': self.time,
            'x': p.x[active].copy(),
            'y': p.y[active].copy(),
            'vx': p.vx[active].copy(),
            'vy': p.vy[active].copy(),
            'ax': p.ax[active].copy(),
            'ay': p.ay[active].copy(),
            'height': p.height[active].copy(),
            'density': p.density[active].copy(),
            'terrain_z': p.terrain_z[active].copy(),
            'n_active': np.sum(active),
            'n_total': p.n
        })

    def run(self, duration=30.0, save_interval=0.2):
        """
        Run the simulation.

        Args:
            duration: Total simulation time in seconds
            save_interval: Time between saved frames

        Returns:
            Number of steps executed
        """
        print(f"Starting SPH simulation...")
        print(f"Duration: {duration}s, dt: {self.dt}s")
        print(f"Particles: {self.particles.n}")
        print(f"Smoothing length: {self.h}m")

        n_steps = int(duration / self.dt)
        save_steps = int(save_interval / self.dt)

        last_save_time = 0.0

        for step_num in range(n_steps):
            avg_speed = self.step()

            # Save state at intervals
            if self.time - last_save_time >= save_interval:
                self._save_state()
                last_save_time = self.time

                active = self.particles.active
                n_active = np.sum(active)

                if n_active > 0:
                    max_speed = np.sqrt(self.particles.vx[active]**2 +
                                       self.particles.vy[active]**2).max()
                else:
                    max_speed = 0.0

                print(f"t = {self.time:.2f}s: active = {n_active}/{self.particles.n}, "
                      f"avg_speed = {avg_speed:.2f} m/s, max_speed = {max_speed:.2f} m/s")

            # Early termination if flow stops or all particles deactivated
            n_active = np.sum(self.particles.active)
            if n_active == 0:
                print(f"All particles left domain at t = {self.time:.2f}s")
                self._save_state()
                break

            if step_num > 100 and avg_speed < 0.01:
                print(f"Flow stopped at t = {self.time:.2f}s")
                self._save_state()
                break

        print(f"Simulation complete. {len(self.history)} frames saved.")
        return step_num + 1


class SPHVisualizer:
    """Visualize SPH simulation results."""

    def __init__(self, simulator):
        self.simulator = simulator
        self.history = simulator.history
        self.terrain = simulator.terrain

    def create_animation(self, output_file='landslide_sph_animation.gif',
                        fps=10, dpi=100):
        """Create animated visualization."""
        fig = plt.figure(figsize=(16, 6))

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Get data ranges
        all_x = np.concatenate([h['x'] for h in self.history if len(h['x']) > 0])
        all_y = np.concatenate([h['y'] for h in self.history if len(h['y']) > 0])
        all_speeds = np.concatenate([np.sqrt(h['vx']**2 + h['vy']**2)
                                     for h in self.history if len(h['vx']) > 0])

        speed_max = max(all_speeds.max(), 1.0) if len(all_speeds) > 0 else 1.0

        # Terrain for background
        dem = self.terrain.dem
        extent = [0, self.simulator.domain_width, 0, self.simulator.domain_height]

        # Get pit and peak locations if available
        pit_loc = getattr(self, 'pit_location', None)
        peak_loc = getattr(self, 'peak_location', None)

        # Pre-create static colorbars (outside update function)
        # DEM elevation colorbar for left plot
        dem_mappable = plt.cm.ScalarMappable(cmap='terrain',
                                             norm=plt.Normalize(vmin=dem.min(), vmax=dem.max()))
        cbar_dem = fig.colorbar(dem_mappable, ax=ax1, location='left', pad=0.12, shrink=0.8)
        cbar_dem.set_label('Elevation (m)')

        # Thickness colorbar for right plot
        height_max = self.history[0]['height'].max() if len(self.history[0]['height']) > 0 else 1.0
        thick_mappable = plt.cm.ScalarMappable(cmap='YlOrRd',
                                               norm=plt.Normalize(vmin=0, vmax=height_max))
        cbar_thick = fig.colorbar(thick_mappable, ax=ax2, location='right', pad=0.02, shrink=0.8)
        cbar_thick.set_label('Thickness (m)')

        def update(frame):
            ax1.clear()
            ax2.clear()

            state = self.history[frame]
            x, y = state['x'], state['y']
            vx, vy = state['vx'], state['vy']

            # Handle empty arrays
            if len(x) == 0:
                speed = np.array([])
                height = np.array([])
            else:
                speed = np.sqrt(vx**2 + vy**2)
                height = state['height']

            # Left plot: Terrain with particles colored by speed
            dem_img = ax1.imshow(dem, cmap='terrain', origin='lower', extent=extent, alpha=0.7,
                                vmin=dem.min(), vmax=dem.max())
            ax1.contour(np.linspace(0, self.simulator.domain_width, dem.shape[1]),
                       np.linspace(0, self.simulator.domain_height, dem.shape[0]),
                       dem, levels=15, colors='gray', alpha=0.3, linewidths=0.5)

            if len(x) > 0:
                scatter1 = ax1.scatter(x, y, c=speed, cmap='hot', s=20,
                                      vmin=0, vmax=speed_max, alpha=0.8, edgecolors='black', linewidths=0.3)
            else:
                scatter1 = ax1.scatter([], [], c=[], cmap='hot', s=20,
                                      vmin=0, vmax=speed_max)

            # Mark pit location with circle
            if pit_loc is not None:
                circle = plt.Circle(pit_loc, 120, fill=False, color='blue',
                                   linestyle='--', linewidth=2, label='Pit')
                ax1.add_patch(circle)
                ax1.plot(pit_loc[0], pit_loc[1], 'bx', markersize=10, markeredgewidth=2)

            # Mark peak location
            if peak_loc is not None:
                ax1.plot(peak_loc[0], peak_loc[1], 'g^', markersize=10, label='Peak')

            ax1.set_xlim(0, self.simulator.domain_width)
            ax1.set_ylim(0, self.simulator.domain_height)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')

            n_active = state.get('n_active', len(x))
            n_total = state.get('n_total', len(x))
            max_spd = speed.max() if len(speed) > 0 else 0

            ax1.set_title(f'SPH Landslide - t = {state["time"]:.1f}s\n'
                         f'Active: {n_active}/{n_total}, Max speed: {max_spd:.1f} m/s')
            ax1.set_aspect('equal')

            # Right plot: Particles colored by height/thickness with ACCELERATION vectors
            ax2.imshow(dem, cmap='terrain', origin='lower', extent=extent, alpha=0.7)

            height_max = self.history[0]['height'].max() if len(self.history[0]['height']) > 0 else 1.0

            if len(x) > 0:
                scatter2 = ax2.scatter(x, y, c=height, cmap='YlOrRd', s=20,
                                      vmin=0, vmax=height_max, alpha=0.8)

                # Acceleration vectors (subsample for clarity)
                ax_acc = state['ax']
                ay_acc = state['ay']
                acc_mag = np.sqrt(ax_acc**2 + ay_acc**2)

                step = max(1, len(x) // 100)
                # Scale arrows by acceleration magnitude for visibility
                q = ax2.quiver(x[::step], y[::step], ax_acc[::step], ay_acc[::step],
                              acc_mag[::step], cmap='cool', alpha=0.7, scale=500)
            else:
                scatter2 = ax2.scatter([], [], c=[], cmap='YlOrRd', s=20,
                                      vmin=0, vmax=height_max)
                acc_mag = np.array([])

            # Mark pit location with circle
            if pit_loc is not None:
                circle2 = plt.Circle(pit_loc, 120, fill=False, color='blue',
                                    linestyle='--', linewidth=2)
                ax2.add_patch(circle2)
                ax2.plot(pit_loc[0], pit_loc[1], 'bx', markersize=10, markeredgewidth=2)

            ax2.set_xlim(0, self.simulator.domain_width)
            ax2.set_ylim(0, self.simulator.domain_height)
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')

            max_acc = acc_mag.max() if len(acc_mag) > 0 else 0
            mean_spd = speed.mean() if len(speed) > 0 else 0

            ax2.set_title(f'Debris Thickness & Acceleration\n'
                         f'Max accel: {max_acc:.1f} m/s², Mean speed: {mean_spd:.1f} m/s')
            ax2.set_aspect('equal')

            return []

        print(f"Creating animation with {len(self.history)} frames...")
        anim = animation.FuncAnimation(fig, update, frames=len(self.history),
                                       interval=1000/fps, blit=False)

        print(f"Saving animation to {output_file}...")
        anim.save(output_file, writer='pillow', fps=fps, dpi=dpi)
        print(f"Animation saved!")

        plt.close()
        return output_file

    def plot_trajectory(self, output_file='landslide_sph_trajectory.png'):
        """Plot particle trajectories over time."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        dem = self.terrain.dem
        extent = [0, self.simulator.domain_width, 0, self.simulator.domain_height]

        # Plot 1: Initial and final positions
        ax = axes[0, 0]
        ax.imshow(dem, cmap='terrain', origin='lower', extent=extent, alpha=0.6)

        initial = self.history[0]
        final = self.history[-1]

        ax.scatter(initial['x'], initial['y'], c='blue', s=10, alpha=0.5, label='Initial')
        ax.scatter(final['x'], final['y'], c='red', s=10, alpha=0.5, label='Final')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Initial (blue) vs Final (red) Positions')
        ax.legend()
        ax.set_aspect('equal')

        # Plot 2: Trajectory lines (sample particles)
        ax = axes[0, 1]
        ax.imshow(dem, cmap='terrain', origin='lower', extent=extent, alpha=0.6)

        n_particles = len(self.history[0]['x'])
        sample_indices = np.linspace(0, n_particles-1, min(50, n_particles)).astype(int)

        for idx in sample_indices:
            traj_x = [h['x'][idx] for h in self.history if idx < len(h['x'])]
            traj_y = [h['y'][idx] for h in self.history if idx < len(h['y'])]
            ax.plot(traj_x, traj_y, 'b-', alpha=0.3, linewidth=0.5)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Particle Trajectories (sample)')
        ax.set_aspect('equal')

        # Plot 3: Speed over time
        ax = axes[1, 0]
        times = [h['time'] for h in self.history]
        mean_speeds = [np.sqrt(h['vx']**2 + h['vy']**2).mean() for h in self.history]
        max_speeds = [np.sqrt(h['vx']**2 + h['vy']**2).max() for h in self.history]

        ax.plot(times, mean_speeds, 'b-', label='Mean speed')
        ax.plot(times, max_speeds, 'r-', label='Max speed')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Flow Speed Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Runout distance
        ax = axes[1, 1]

        # Calculate runout (distance from initial centroid)
        init_cx = initial['x'].mean()
        init_cy = initial['y'].mean()

        runouts = []
        for h in self.history:
            distances = np.sqrt((h['x'] - init_cx)**2 + (h['y'] - init_cy)**2)
            runouts.append(distances.max())

        ax.plot(times, runouts, 'g-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Max Runout Distance (m)')
        ax.set_title('Runout Distance Over Time')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"Trajectory plot saved to {output_file}")
        plt.close()

        return output_file


def create_terrain_with_pit():
    """Create synthetic terrain with a depression/pit for testing particle collection."""
    from scipy.ndimage import gaussian_filter

    np.random.seed(42)
    rows, cols = 80, 100
    cell_size = 10.0

    # Create coordinate grids
    # Note: row index i corresponds to y, col index j corresponds to x
    j_grid, i_grid = np.meshgrid(np.arange(cols), np.arange(rows))
    x_grid = j_grid * cell_size  # x in meters
    y_grid = i_grid * cell_size  # y in meters

    # Define key locations (in meters)
    # Source: upper-left area (high elevation)
    source_x, source_y = 250, 550

    # Pit: center-right area (low elevation - particles should collect here)
    pit_x, pit_y = 650, 350

    # Create base terrain: slope from source toward pit
    # Calculate direction vector from source to pit
    dx = pit_x - source_x
    dy = pit_y - source_y
    dist_total = np.sqrt(dx**2 + dy**2)

    # Base elevation: high at source, low at pit
    # Project each point onto the source-pit axis
    proj = ((x_grid - source_x) * dx + (y_grid - source_y) * dy) / (dist_total**2)
    proj = np.clip(proj, 0, 1.5)  # Allow some extension past pit

    # Elevation decreases along the path
    terrain = 900 - 400 * proj

    # Add a raised area (source region) - landslide starts here
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((x_grid[i,j] - source_x)**2 + (y_grid[i,j] - source_y)**2)
            terrain[i, j] += 200 * np.exp(-dist**2 / (2 * 80**2))

    # Add the PIT (depression) - very low elevation
    pit_depth = 200  # meters deep
    pit_radius = 100  # meters

    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((x_grid[i,j] - pit_x)**2 + (y_grid[i,j] - pit_y)**2)
            if dist < pit_radius * 2:
                terrain[i, j] -= pit_depth * np.exp(-dist**2 / (2 * (pit_radius/2)**2))

    # Add subtle surrounding ridge to keep particles from escaping
    # (raised edges except in the source direction)
    for i in range(rows):
        for j in range(cols):
            # Distance from pit
            dist_from_pit = np.sqrt((x_grid[i,j] - pit_x)**2 + (y_grid[i,j] - pit_y)**2)
            # Angle from pit
            angle = np.arctan2(y_grid[i,j] - pit_y, x_grid[i,j] - pit_x)
            # Source direction angle
            source_angle = np.arctan2(source_y - pit_y, source_x - pit_x)
            # Angular difference
            angle_diff = np.abs(np.arctan2(np.sin(angle - source_angle), np.cos(angle - source_angle)))

            # Add ridge except in source direction
            if dist_from_pit > pit_radius and dist_from_pit < pit_radius * 2.5:
                if angle_diff > np.pi/3:  # Not facing source
                    ridge_factor = (1 - (dist_from_pit - pit_radius) / (pit_radius * 1.5))
                    ridge_factor = max(0, ridge_factor)
                    terrain[i, j] += 50 * ridge_factor * (angle_diff / np.pi)

    # Small roughness
    terrain += 5 * np.random.randn(rows, cols)
    terrain = gaussian_filter(terrain, sigma=1.0)

    # Find actual pit minimum location
    pit_row = int(pit_y / cell_size)
    pit_col = int(pit_x / cell_size)

    print(f"Created terrain with pit")
    print(f"  Source location: ({source_x}, {source_y}) m, elevation: {terrain[int(source_y/cell_size), int(source_x/cell_size)]:.1f} m")
    print(f"  Pit location: ({pit_x}, {pit_y}) m, elevation: {terrain[pit_row, pit_col]:.1f} m")
    print(f"  Elevation difference: {terrain[int(source_y/cell_size), int(source_x/cell_size)] - terrain[pit_row, pit_col]:.1f} m")

    return terrain, (pit_x, pit_y), (source_x, source_y)


def main():
    """Run SPH landslide demonstration with pit collection test."""
    print("=" * 60)
    print("SPH LANDSLIDE SIMULATION")
    print("(Testing gravity direction and pit collection)")
    print("=" * 60)

    # Step 1: Create terrain with pit
    print("\n[1/4] Creating terrain DEM with depression/pit...")
    terrain, pit_location, peak_location = create_terrain_with_pit()
    cell_size = 10.0

    print(f"Terrain shape: {terrain.shape}")
    print(f"Elevation range: {terrain.min():.1f} - {terrain.max():.1f} m")
    print(f"Landslide source (peak): {peak_location}")
    print(f"Collection pit location: {pit_location}")

    # Save terrain
    np.save('terrain_dem_sph.npy', terrain)

    # Step 2: Initialize simulator
    print("\n[2/4] Initializing SPH simulator...")
    simulator = SPHLandslideSimulator(terrain, cell_size)

    # Initialize landslide on the SLOPE (between peak and pit)
    # Position: 40% of the way from peak to pit (on the slope, not at top)
    slope_fraction = 0.4
    landslide_center = (
        peak_location[0] + (pit_location[0] - peak_location[0]) * slope_fraction,
        peak_location[1] + (pit_location[1] - peak_location[1]) * slope_fraction
    )
    landslide_radius = 50  # meters
    landslide_thickness = 8.0  # meters

    print(f"Landslide initial center: ({landslide_center[0]:.0f}, {landslide_center[1]:.0f})")

    # Calculate initial velocity direction (towards pit)
    dir_x = pit_location[0] - landslide_center[0]
    dir_y = pit_location[1] - landslide_center[1]
    dir_mag = np.sqrt(dir_x**2 + dir_y**2)
    dir_x /= dir_mag
    dir_y /= dir_mag

    initial_speed = 5.0  # m/s initial velocity towards pit
    initial_vx = dir_x * initial_speed
    initial_vy = dir_y * initial_speed

    print(f"Initial velocity direction: ({dir_x:.2f}, {dir_y:.2f}) at {initial_speed} m/s")

    simulator.initialize_particles(landslide_center, landslide_radius, landslide_thickness)

    # Set initial velocity for all particles (towards pit direction)
    simulator.particles.vx[:] = initial_vx
    simulator.particles.vy[:] = initial_vy
    print(f"Set initial velocity: vx={initial_vx:.2f}, vy={initial_vy:.2f} m/s")

    # Step 3: Run simulation
    print("\n[3/4] Running SPH simulation...")
    simulator.run(duration=5.0, save_interval=0.1)

    # Step 4: Visualize
    print("\n[4/4] Creating visualizations...")
    visualizer = SPHVisualizer(simulator)

    # Store pit location for visualization
    visualizer.pit_location = pit_location
    visualizer.peak_location = peak_location

    animation_file = visualizer.create_animation('landslide_sph_animation.gif', fps=8, dpi=100)
    trajectory_file = visualizer.plot_trajectory('landslide_sph_trajectory.png')

    # Print final analysis
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Output files:")
    print(f"  - Animation: {animation_file}")
    print(f"  - Trajectory: {trajectory_file}")

    # Check if particles collected in pit
    if len(simulator.history) > 0:
        final_state = simulator.history[-1]
        pit_x, pit_y = pit_location
        pit_radius = 150  # meters - check area around pit

        distances_to_pit = np.sqrt((final_state['x'] - pit_x)**2 +
                                   (final_state['y'] - pit_y)**2)
        particles_in_pit = np.sum(distances_to_pit < pit_radius)
        total_active = final_state['n_active']

        print(f"\nPit collection analysis:")
        print(f"  - Active particles at end: {total_active}")
        print(f"  - Particles near pit (within {pit_radius}m): {particles_in_pit}")
        print(f"  - Collection rate: {100*particles_in_pit/max(total_active,1):.1f}%")


if __name__ == '__main__':
    main()
