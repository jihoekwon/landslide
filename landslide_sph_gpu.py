"""
GPU-Parallelized SPH Landslide Simulation using CuPy
=====================================================

Uses vectorized CuPy operations for GPU acceleration.
Avoids race conditions by computing contributions per-particle.

Author: Claude Code Demo
"""

import numpy as np
import cupy as cp
import time


class SPHKernelGPU:
    """SPH Kernel functions using CuPy (GPU)."""

    def __init__(self, h):
        self.h = cp.float32(h)
        self.h2 = h * h
        self.alpha = cp.float32(10.0 / (7.0 * np.pi * self.h2))

    def W(self, r):
        """Cubic spline kernel."""
        q = r / self.h
        result = cp.zeros_like(q, dtype=cp.float32)

        mask1 = (q >= 0) & (q < 1)
        result[mask1] = 1.0 - 1.5 * q[mask1]**2 + 0.75 * q[mask1]**3

        mask2 = (q >= 1) & (q < 2)
        result[mask2] = 0.25 * (2.0 - q[mask2])**3

        return self.alpha * result

    def gradW_scalar(self, r, dx, dy):
        """Gradient of kernel for distance matrices."""
        q = r / self.h
        dWdr = cp.zeros_like(q, dtype=cp.float32)

        mask1 = (q > 1e-10) & (q < 1)
        dWdr[mask1] = (-3.0 * q[mask1] + 2.25 * q[mask1]**2) / self.h

        mask2 = (q >= 1) & (q < 2)
        dWdr[mask2] = -0.75 * (2.0 - q[mask2])**2 / self.h

        dWdr *= self.alpha

        r_safe = cp.maximum(r, 1e-10)
        gradWx = dWdr * dx / r_safe
        gradWy = dWdr * dy / r_safe

        return gradWx, gradWy


class SPHParticlesGPU:
    """GPU-based particle storage."""

    def __init__(self, n_particles):
        self.n = n_particles

        self.x = cp.zeros(n_particles, dtype=cp.float32)
        self.y = cp.zeros(n_particles, dtype=cp.float32)
        self.vx = cp.zeros(n_particles, dtype=cp.float32)
        self.vy = cp.zeros(n_particles, dtype=cp.float32)
        self.ax = cp.zeros(n_particles, dtype=cp.float32)
        self.ay = cp.zeros(n_particles, dtype=cp.float32)

        self.mass = cp.zeros(n_particles, dtype=cp.float32)
        self.density = cp.zeros(n_particles, dtype=cp.float32)  # kept for compatibility
        self.dh_dt = cp.zeros(n_particles, dtype=cp.float32)  # depth change rate
        self.pressure = cp.zeros(n_particles, dtype=cp.float32)

        self.terrain_grad_x = cp.zeros(n_particles, dtype=cp.float32)
        self.terrain_grad_y = cp.zeros(n_particles, dtype=cp.float32)
        self.height = cp.zeros(n_particles, dtype=cp.float32)

        # Entrainment (Takahashi model)
        self.concentration = cp.zeros(n_particles, dtype=cp.float32)  # solid volume concentration
        self.slope = cp.zeros(n_particles, dtype=cp.float32)  # local slope angle (radians)

        self.active = cp.ones(n_particles, dtype=cp.bool_)


class SPHSimulatorGPU:
    """GPU-accelerated SPH simulator using vectorized operations."""

    def __init__(self, terrain, cell_size=10.0,
                 # SPH parameters
                 h: float = 5.0,
                 dt: float = 0.004,
                 v_max: float = 30.0,
                 # Physical parameters
                 rho0: float = 2000.0,
                 c0: float = 300.0,
                 gamma: float = 7.0,
                 g: float = 9.81,
                 # Rheology
                 mu: float = 100.0,
                 tau_y: float = 500.0,
                 mu_b: float = 0.1,
                 xi: float = 200.0,
                 alpha: float = 0.1,
                 beta: float = 0.1,
                 # Entrainment (Takahashi 2007)
                 entrainment_enabled: bool = True,
                 delta_e: float = 0.0007,
                 delta_d: float = 0.01,
                 rho_s: float = 2650.0,
                 rho_w: float = 1000.0,
                 phi_bed: float = 35.0,
                 C_init: float = 0.4,
                 C_max: float = 0.65):
        """
        Initialize SPH simulator.

        Args:
            terrain: 2D numpy array of elevation data
            cell_size: Grid cell size in meters
            h: SPH smoothing length (m)
            dt: Time step (s)
            v_max: Velocity ceiling (m/s)
            rho0: Reference density (kg/m³)
            c0: Speed of sound (m/s), use 10*v_max for ~1% compressibility
            gamma: EOS exponent
            g: Gravity (m/s²)
            mu: Dynamic viscosity (Pa·s)
            tau_y: Yield stress (Pa)
            mu_b: Basal friction coefficient
            xi: Turbulent coefficient (m/s²)
            alpha: Artificial viscosity parameter
            beta: Artificial viscosity parameter
            entrainment_enabled: Enable Takahashi entrainment model
            delta_e: Erosion coefficient
            delta_d: Deposition coefficient
            rho_s: Solid particle density (kg/m³)
            rho_w: Water density (kg/m³)
            phi_bed: Bed friction angle (degrees)
            C_init: Initial solid concentration
            C_max: Maximum packing concentration
        """
        # DEM row 0=북 → row 0=남으로 변환 (SPH local_y=0=남과 일치)
        terrain = terrain[::-1]
        self.terrain = terrain.astype(np.float32)
        self.terrain_grid = cp.asarray(self.terrain)
        self.cell_size = cell_size
        self.domain_width = terrain.shape[1] * cell_size
        self.domain_height = terrain.shape[0] * cell_size

        # Terrain gradients on GPU
        grad_y, grad_x = np.gradient(terrain, cell_size)
        self.terrain_grad_x_grid = cp.asarray(grad_x.astype(np.float32))
        self.terrain_grad_y_grid = cp.asarray(grad_y.astype(np.float32))

        # SPH parameters
        self.h = h
        self.kernel = SPHKernelGPU(self.h)
        self.cutoff = 2.0 * self.h

        # Physical parameters
        self.rho0 = rho0
        self.c0 = c0
        self.gamma = gamma
        self.g = g

        # Rheology
        self.mu = mu
        self.tau_y = tau_y
        self.mu_b = mu_b
        self.xi = xi

        # Artificial viscosity
        self.alpha_visc = alpha
        self.beta_visc = beta

        # Time/limits
        self.dt = dt
        self.v_max = v_max

        # Entrainment parameters (Takahashi 2007 model)
        self.entrainment_enabled = entrainment_enabled
        self.rho_s = rho_s
        self.rho_w = rho_w
        self.phi_bed = phi_bed
        self.C_max = C_max
        self.C_init = C_init
        self.delta_e = delta_e
        self.delta_d = delta_d

        self.particles = None
        self.time = 0.0
        self.history = []

    def initialize_particles(self, center, radius, thickness, spacing=None):
        """Initialize particles."""
        if spacing is None:
            spacing = self.h / 2.0

        x_range = np.arange(center[0] - radius, center[0] + radius, spacing)
        y_range = np.arange(center[1] - radius, center[1] + radius, spacing)

        positions = []
        for x in x_range:
            for y in y_range:
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if dist < radius:
                    h_local = thickness * np.exp(-dist**2 / (2 * (radius/2)**2))
                    if h_local > 0.1:
                        positions.append((x, y, h_local))

        n_particles = len(positions)
        print(f"Initialized {n_particles} particles on GPU")

        self.particles = SPHParticlesGPU(n_particles)
        p = self.particles

        positions_arr = np.array(positions, dtype=np.float32)
        p.x = cp.asarray(positions_arr[:, 0])
        p.y = cp.asarray(positions_arr[:, 1])
        p.height = cp.asarray(positions_arr[:, 2])

        particle_area = spacing * spacing
        p.mass = cp.asarray(self.rho0 * particle_area * positions_arr[:, 2], dtype=cp.float32)
        p.density = cp.full(n_particles, self.rho0, dtype=cp.float32)

        self._update_terrain_gradients()
        self._save_state()

    def initialize_particles_from_coords(self, x_coords, y_coords, x_min, y_min, thickness=5.0):
        """Initialize particles from given x,y coordinates."""
        n_particles = len(x_coords)
        print(f"Initialized {n_particles} particles on GPU")

        self.particles = SPHParticlesGPU(n_particles)
        p = self.particles

        # Convert to local coordinates (relative to terrain grid)
        p.x = cp.asarray((x_coords - x_min).astype(np.float32))
        p.y = cp.asarray((y_coords - y_min).astype(np.float32))
        p.height = cp.full(n_particles, thickness, dtype=cp.float32)

        spacing = self.h / 2.0
        particle_area = spacing * spacing
        p.mass = cp.full(n_particles, self.rho0 * particle_area * thickness, dtype=cp.float32)
        p.density = cp.full(n_particles, self.rho0, dtype=cp.float32)

        # Initialize concentration for entrainment model
        p.concentration = cp.full(n_particles, self.C_init, dtype=cp.float32)

        self._update_terrain_gradients()
        self._save_state()

    def _update_terrain_gradients(self):
        """Update terrain gradients at particle positions."""
        p = self.particles

        col = cp.clip(p.x / self.cell_size, 0, self.terrain.shape[1] - 1.001)
        row = cp.clip(p.y / self.cell_size, 0, self.terrain.shape[0] - 1.001)

        col_idx = cp.floor(col).astype(cp.int32)
        row_idx = cp.floor(row).astype(cp.int32)

        p.terrain_grad_x = self.terrain_grad_x_grid[row_idx, col_idx]
        p.terrain_grad_y = self.terrain_grad_y_grid[row_idx, col_idx]

        # Calculate local slope (radians)
        slope_mag = cp.sqrt(p.terrain_grad_x**2 + p.terrain_grad_y**2)
        p.slope = cp.arctan(slope_mag)

    def _compute_entrainment(self):
        """Compute entrainment/deposition using Takahashi (2007) original model.

        Takahashi equilibrium concentration:
            C_eq = rho_w * tan(theta) / [(rho_s - rho_w) * (tan(phi) - tan(theta))]

        Takahashi (2007) erosion/deposition rates:
            Erosion (C < C_eq):    i_e = delta_e * C_eq * v
            Deposition (C > C_eq): i_d = delta_d * C * v * (1 - tan(theta)/tan(phi))

        where:
            theta = bed slope angle
            phi = internal friction angle of bed material
            C_eq = equilibrium sediment concentration
            C = current sediment concentration
            v = flow velocity

        Physical meaning of (1 - tan(theta)/tan(phi)):
            - theta -> phi (steep): factor -> 0, no deposition (unstable slope)
            - theta -> 0 (flat): factor -> 1, maximum deposition
        """
        if not self.entrainment_enabled:
            return

        p = self.particles

        # Get slope angle
        theta = p.slope  # radians

        # Compute tan values
        tan_theta = cp.tan(theta)
        tan_phi = cp.tan(cp.float32(np.radians(self.phi_bed)))

        # Equilibrium concentration (Takahashi formula)
        # C_eq = rho_w * tan_theta / [(rho_s - rho_w) * (tan_phi - tan_theta)]
        denominator = (self.rho_s - self.rho_w) * (tan_phi - tan_theta)

        # Avoid division by zero and negative values (when theta > phi, no equilibrium)
        valid_slope = (tan_theta < tan_phi) & (theta > 0.01)
        denominator_safe = cp.where(valid_slope, cp.maximum(denominator, 1e-6), 1.0)

        C_eq = self.rho_w * tan_theta / denominator_safe
        C_eq = cp.where(valid_slope, C_eq, 0.0)

        # Limit C_eq to maximum packing
        C_eq = cp.minimum(C_eq, self.C_max)

        # Current concentration and speed
        C = p.concentration
        speed = cp.sqrt(p.vx**2 + p.vy**2)

        # Takahashi (2007) original erosion/deposition rates
        # Erosion: i_e = delta_e * C_eq * v  (when C < C_eq)
        # Deposition: i_d = delta_d * C * v * (1 - tan_theta/tan_phi)  (when C > C_eq)

        is_eroding = C < C_eq

        # Erosion rate (m/s) - bed lowering rate
        erosion_rate = self.delta_e * C_eq * speed

        # Deposition rate (m/s) - bed rising rate
        # Factor (1 - tan_theta/tan_phi): no deposition on steep slopes
        slope_factor = cp.maximum(1.0 - tan_theta / tan_phi, 0.0)
        deposition_rate = self.delta_d * C * speed * slope_factor

        # Apply only to active particles and valid slopes
        C_bed = self.C_max

        erosion_rate = cp.where(p.active & valid_slope, erosion_rate, 0.0)
        deposition_rate = cp.where(p.active & valid_slope, deposition_rate, 0.0)

        # Update depth (volume conservation)
        # Erosion: bed lowers by i_e, flow gains volume i_e
        # Deposition: bed rises by i_d, flow loses volume i_d
        # dh = +i_e * dt (erosion) or -i_d * dt (deposition)
        dh = cp.where(is_eroding,
                      erosion_rate * self.dt,
                      -deposition_rate * self.dt)

        # Update height
        h_old = p.height.copy()
        p.height += dh
        p.height = cp.maximum(p.height, 0.01)

        # Update concentration using mass conservation
        # Erosion: bed material (C_bed) enters flow
        #   d(h*C) = +C_bed * i_e * dt
        # Deposition: flow material settles and compacts to C_bed on bed
        #   d(h*C) = -C_bed * i_d * dt (solids deposited at bed concentration)
        mass_solid_old = h_old * C

        # Both erosion and deposition involve C_bed (bed packing concentration)
        mass_change = cp.where(is_eroding,
                               C_bed * erosion_rate * self.dt,
                               -C_bed * deposition_rate * self.dt)

        mass_solid_new = mass_solid_old + mass_change
        mass_solid_new = cp.maximum(mass_solid_new, 0.0)

        # New concentration
        C_new = mass_solid_new / p.height
        C_new = cp.clip(C_new, 0.0, self.C_max)

        p.concentration = cp.where(p.active, C_new, p.concentration)

    def _compute_distance_matrix(self):
        """Compute pairwise distances (N x N matrix on GPU)."""
        p = self.particles
        n = p.n

        # Broadcasting to compute all pairwise distances
        dx = p.x[:, None] - p.x[None, :]  # N x N
        dy = p.y[:, None] - p.y[None, :]  # N x N
        r = cp.sqrt(dx**2 + dy**2)

        return r, dx, dy

    def _compute_dh_dt_vectorized(self):
        """Compute dh/dt using depth-averaged continuity equation.

        Depth-averaged continuity: dh/dt + h * div(v) = 0
        SPH form: dh_i/dt = sum_j Omega_j * (v_i - v_j) . grad_W_ij
        where Omega_j = m_j / (rho0 * h_j) is particle area
        """
        p = self.particles

        # Distance matrix
        r, dx, dy = self._compute_distance_matrix()

        # Kernel gradient
        gradWx, gradWy = self.kernel.gradW_scalar(r, dx, dy)

        # Velocity differences: v_i - v_j for all pairs
        dvx = p.vx[:, None] - p.vx[None, :]  # N x N
        dvy = p.vy[:, None] - p.vy[None, :]  # N x N

        # Mask: within cutoff and not self
        mask = (r < self.cutoff) & (r > 1e-10)

        # Active mask (both particles must be active)
        active_mask = p.active[:, None] & p.active[None, :]
        mask = mask & active_mask

        # Particle area: Omega_j = m_j / (rho0 * h_j)
        h_j = cp.maximum(p.height[None, :], 0.01)  # avoid division by zero
        Omega_j = p.mass[None, :] / (self.rho0 * h_j)

        # Contribution: Omega_j * (v_i - v_j) . gradW
        contribution = Omega_j * (dvx * gradWx + dvy * gradWy)
        contribution = cp.where(mask, contribution, 0.0)

        # Sum contributions for each particle i
        p.dh_dt = cp.sum(contribution, axis=1)

    def _compute_pressure(self):
        """Compute depth-integrated hydrostatic pressure.

        P = (1/2) * rho * g * h^2
        """
        p = self.particles
        h_safe = cp.maximum(p.height, 0.01)
        p.pressure = 0.5 * self.rho0 * self.g * h_safe**2
        # Also update density for compatibility (rho_2D = rho0 * h)
        p.density = self.rho0 * h_safe

    def _compute_forces_vectorized(self):
        """Compute forces using vectorized operations."""
        p = self.particles
        n = p.n

        # Reset accelerations
        p.ax = cp.zeros(n, dtype=cp.float32)
        p.ay = cp.zeros(n, dtype=cp.float32)

        # Gravity along slope
        p.ax += -self.g * p.terrain_grad_x
        p.ay += -self.g * p.terrain_grad_y

        # Distance matrix
        r, dx, dy = self._compute_distance_matrix()
        gradWx, gradWy = self.kernel.gradW_scalar(r, dx, dy)

        # Masks
        mask = (r < self.cutoff) & (r > 1e-10)
        active_mask = p.active[:, None] & p.active[None, :]
        mask = mask & active_mask

        # --- Pressure force (depth-averaged) ---
        # Depth-averaged momentum: dv/dt = -g * grad(h) + other terms
        # SPH form: a_pressure = -g * sum_j Omega_j * (h_i + h_j)/2 * grad_W_ij
        h_i = p.height[:, None]  # N x 1
        h_j = p.height[None, :]  # 1 x N
        h_j_safe = cp.maximum(h_j, 0.01)

        # Particle area: Omega_j = m_j / (rho0 * h_j)
        Omega_j = p.mass[None, :] / (self.rho0 * h_j_safe)

        # Symmetric pressure gradient
        h_avg = 0.5 * (h_i + h_j)
        pressure_force_x = -self.g * Omega_j * h_avg * gradWx
        pressure_force_y = -self.g * Omega_j * h_avg * gradWy

        pressure_force_x = cp.where(mask, pressure_force_x, 0.0)
        pressure_force_y = cp.where(mask, pressure_force_y, 0.0)

        p.ax += cp.sum(pressure_force_x, axis=1)
        p.ay += cp.sum(pressure_force_y, axis=1)

        # --- Artificial viscosity ---
        dvx = p.vx[:, None] - p.vx[None, :]
        dvy = p.vy[:, None] - p.vy[None, :]
        vr_dot = dvx * dx + dvy * dy

        approaching_mask = mask & (vr_dot < 0)

        # Use density = rho0 * h for viscosity calculations
        rho_i = p.density[:, None]  # = rho0 * h_i
        rho_j = p.density[None, :]  # = rho0 * h_j
        m_j = p.mass[None, :]

        rho_avg = 0.5 * (rho_i + rho_j)
        r_sq = r**2 + 0.01 * self.h**2
        mu_art = self.h * vr_dot / r_sq
        Pi_ij = (-self.alpha_visc * self.c0 * mu_art + self.beta_visc * mu_art**2) / rho_avg

        visc_force_x = -m_j * Pi_ij * gradWx
        visc_force_y = -m_j * Pi_ij * gradWy

        visc_force_x = cp.where(approaching_mask, visc_force_x, 0.0)
        visc_force_y = cp.where(approaching_mask, visc_force_y, 0.0)

        p.ax += cp.sum(visc_force_x, axis=1)
        p.ay += cp.sum(visc_force_y, axis=1)

        # --- Physical viscosity ---
        r_safe = cp.maximum(r, 1e-10)
        visc_term = 2.0 * self.mu * r / (rho_i * rho_j * r_sq)
        grad_dot_r = (gradWx * dx + gradWy * dy) / r_safe

        phys_visc_x = m_j * visc_term * dvx * grad_dot_r
        phys_visc_y = m_j * visc_term * dvy * grad_dot_r

        phys_visc_x = cp.where(mask, phys_visc_x, 0.0)
        phys_visc_y = cp.where(mask, phys_visc_y, 0.0)

        p.ax += cp.sum(phys_visc_x, axis=1)
        p.ay += cp.sum(phys_visc_y, axis=1)

        # --- Basal friction (Voellmy) ---
        speed = cp.sqrt(p.vx**2 + p.vy**2)
        speed_safe = cp.maximum(speed, 1e-6)

        friction_coeff = self.mu_b + speed**2 / self.xi
        friction_mag = self.g * friction_coeff

        p.ax -= friction_mag * p.vx / speed_safe
        p.ay -= friction_mag * p.vy / speed_safe

        # --- Yield stress (Bingham) ---
        low_speed = speed < 1.0
        h_safe = cp.maximum(p.height, 0.1)
        yield_factor = self.tau_y / (self.rho0 * h_safe)

        p.ax -= cp.where(low_speed, yield_factor * p.vx / speed_safe, 0.0)
        p.ay -= cp.where(low_speed, yield_factor * p.vy / speed_safe, 0.0)

    def _integrate(self):
        """Integrate positions and velocities."""
        p = self.particles

        # Velocity half step
        p.vx += 0.5 * self.dt * p.ax
        p.vy += 0.5 * self.dt * p.ay

        # Velocity ceiling
        speed = cp.sqrt(p.vx**2 + p.vy**2)
        too_fast = speed > self.v_max
        scale = cp.where(too_fast, self.v_max / speed, 1.0)
        p.vx *= scale
        p.vy *= scale

        # Position update
        p.x += self.dt * p.vx
        p.y += self.dt * p.vy

        # Boundary check
        out_of_bounds = (p.x < 0) | (p.x > self.domain_width) | \
                        (p.y < 0) | (p.y > self.domain_height)
        p.active = p.active & ~out_of_bounds

        # Update terrain gradients
        self._update_terrain_gradients()

    def step(self):
        """Perform one simulation step (depth-averaged)."""
        p = self.particles

        if not cp.any(p.active):
            self.time += self.dt
            return 0.0

        # Compute dh/dt (depth-averaged continuity)
        self._compute_dh_dt_vectorized()

        # Update height (half step)
        p.height += 0.5 * self.dt * p.dh_dt
        p.height = cp.maximum(p.height, 0.01)  # minimum depth

        # Compute pressure (hydrostatic)
        self._compute_pressure()

        # Compute forces
        self._compute_forces_vectorized()

        # Integrate
        self._integrate()

        # Second half (leapfrog)
        self._compute_dh_dt_vectorized()
        p.height += 0.5 * self.dt * p.dh_dt
        p.height = cp.maximum(p.height, 0.01)

        # Entrainment/deposition (Takahashi model)
        self._compute_entrainment()

        self._compute_pressure()
        self._compute_forces_vectorized()

        # Final velocity update
        p.vx += 0.5 * self.dt * p.ax
        p.vy += 0.5 * self.dt * p.ay

        # Velocity ceiling
        speed = cp.sqrt(p.vx**2 + p.vy**2)
        too_fast = speed > self.v_max
        scale = cp.where(too_fast, self.v_max / speed, 1.0)
        p.vx *= scale
        p.vy *= scale

        self.time += self.dt

        # Return average speed
        if cp.any(p.active):
            return float(cp.mean(speed[p.active]))
        return 0.0

    def _save_state(self):
        """Save state to history."""
        p = self.particles
        active = cp.asnumpy(p.active)

        self.history.append({
            'time': self.time,
            'x': cp.asnumpy(p.x[active]),
            'y': cp.asnumpy(p.y[active]),
            'vx': cp.asnumpy(p.vx[active]),
            'vy': cp.asnumpy(p.vy[active]),
            'height': cp.asnumpy(p.height[active]),
            'density': cp.asnumpy(p.density[active]),
            'pressure': cp.asnumpy(p.pressure[active]),
            'concentration': cp.asnumpy(p.concentration[active]),
            'n_active': int(cp.sum(active)),
            'n_total': p.n
        })

    def run(self, duration=10.0, save_interval=0.1, log_file='simulation_progress.txt'):
        """Run simulation."""
        import datetime

        n_steps = int(duration / self.dt)
        last_save = 0.0

        # Write simulation log header
        with open(log_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SPH LANDSLIDE SIMULATION LOG\n")
            f.write("=" * 60 + "\n")
            f.write(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n[SIMULATION PARAMETERS]\n")
            f.write(f"  Duration: {duration} s\n")
            f.write(f"  Time step (dt): {self.dt} s\n")
            f.write(f"  Total steps: {n_steps}\n")
            f.write(f"  Save interval: {save_interval} s\n")
            f.write("\n[SPH PARAMETERS]\n")
            f.write(f"  Smoothing length (h): {self.h} m\n")
            f.write(f"  Particle spacing: {self.h/2} m\n")
            f.write(f"  Cutoff distance: {self.cutoff} m\n")
            f.write(f"  Reference density (rho0): {self.rho0} kg/m^3\n")
            f.write(f"  Speed of sound (c0): {self.c0} m/s\n")
            f.write("\n[RHEOLOGY]\n")
            f.write(f"  Dynamic viscosity (mu): {self.mu} Pa.s\n")
            f.write(f"  Yield stress (tau_y): {self.tau_y} Pa\n")
            f.write(f"  Basal friction (mu_b): {self.mu_b}\n")
            f.write(f"  Turbulent coeff (xi): {self.xi} m/s^2\n")
            f.write("\n[TERRAIN]\n")
            f.write(f"  Grid size: {self.terrain.shape[1]} x {self.terrain.shape[0]}\n")
            f.write(f"  Cell size: {self.cell_size} m\n")
            f.write(f"  Elevation range: {float(self.terrain.min()):.1f} ~ {float(self.terrain.max()):.1f} m\n")
            f.write("\n[PARTICLES]\n")
            f.write(f"  Initial count: {self.particles.n}\n")
            f.write(f"  X range: {float(self.particles.x.min()):.1f} ~ {float(self.particles.x.max()):.1f} m\n")
            f.write(f"  Y range: {float(self.particles.y.min()):.1f} ~ {float(self.particles.y.max()):.1f} m\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("SIMULATION PROGRESS\n")
            f.write("=" * 60 + "\n")

        print(f"Starting GPU SPH simulation...")
        print(f"Duration: {duration}s, dt: {self.dt}s, Particles: {self.particles.n}")

        start = time.time()

        for step_num in range(n_steps):
            avg_speed = self.step()

            if self.time - last_save >= save_interval:
                self._save_state()
                last_save = self.time

                n_active = int(cp.sum(self.particles.active))
                speed = cp.sqrt(self.particles.vx**2 + self.particles.vy**2)
                max_spd = float(cp.max(speed[self.particles.active])) if n_active > 0 else 0

                progress = (self.time / duration) * 100
                status = f"[{progress:5.1f}%] t={self.time:.2f}s: active={n_active}, avg={avg_speed:.2f}, max={max_spd:.2f} m/s"
                print(status)

                # Append progress to log file
                with open(log_file, 'a') as f:
                    f.write(f"{status}\n")

            if not cp.any(self.particles.active):
                print(f"All particles left at t={self.time:.2f}s")
                self._save_state()
                break

        elapsed = time.time() - start
        summary = f"Done. {len(self.history)} frames, {elapsed:.2f}s ({n_steps/elapsed:.0f} steps/s)"
        print(summary)

        # Write final summary to log
        with open(log_file, 'a') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write("SIMULATION COMPLETE\n")
            f.write("=" * 60 + "\n")
            f.write(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Wall time: {elapsed:.2f} s\n")
            f.write(f"Performance: {n_steps/elapsed:.0f} steps/s\n")
            f.write(f"Frames saved: {len(self.history)}\n")

        return step_num + 1, elapsed


def create_terrain_with_pit():
    """Create terrain with pit."""
    from scipy.ndimage import gaussian_filter

    np.random.seed(42)
    rows, cols = 80, 100
    cell_size = 10.0

    j_grid, i_grid = np.meshgrid(np.arange(cols), np.arange(rows))
    x_grid = j_grid * cell_size
    y_grid = i_grid * cell_size

    source_x, source_y = 250, 550
    pit_x, pit_y = 650, 350

    dx = pit_x - source_x
    dy = pit_y - source_y
    dist_total = np.sqrt(dx**2 + dy**2)

    proj = ((x_grid - source_x) * dx + (y_grid - source_y) * dy) / (dist_total**2)
    proj = np.clip(proj, 0, 1.5)

    terrain = 900 - 400 * proj

    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((x_grid[i,j] - source_x)**2 + (y_grid[i,j] - source_y)**2)
            terrain[i, j] += 200 * np.exp(-dist**2 / (2 * 80**2))

    pit_depth, pit_radius = 200, 100
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((x_grid[i,j] - pit_x)**2 + (y_grid[i,j] - pit_y)**2)
            if dist < pit_radius * 2:
                terrain[i, j] -= pit_depth * np.exp(-dist**2 / (2 * (pit_radius/2)**2))

    terrain += 5 * np.random.randn(rows, cols)
    terrain = gaussian_filter(terrain, sigma=1.0)

    return terrain.astype(np.float32), (pit_x, pit_y), (source_x, source_y)


if __name__ == '__main__':
    print("=" * 60)
    print("GPU SPH SIMULATION TEST")
    print("=" * 60)

    terrain, pit_loc, peak_loc = create_terrain_with_pit()
    print(f"Terrain: {terrain.shape}")

    sim = SPHSimulatorGPU(terrain, cell_size=10.0)

    slope_fraction = 0.4
    center = (
        peak_loc[0] + (pit_loc[0] - peak_loc[0]) * slope_fraction,
        peak_loc[1] + (pit_loc[1] - peak_loc[1]) * slope_fraction
    )

    sim.initialize_particles(center, radius=50, thickness=8.0)

    print("\nRunning 1 second test...")
    steps, elapsed = sim.run(duration=1.0, save_interval=0.1)
    print(f"\nGPU: {steps} steps in {elapsed:.2f}s = {steps/elapsed:.0f} steps/sec")
