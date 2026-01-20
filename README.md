# Landslide SPH Simulation

GPU-accelerated landslide simulation using Smoothed Particle Hydrodynamics (SPH) method.

## Overview

This project simulates landslide dynamics using real DEM data from the Irwon region in South Korea. The simulation leverages GPU parallel computing via CuPy to efficiently calculate thousands of particles in near real-time.

## Physical Models

### SPH (Smoothed Particle Hydrodynamics)

SPH is a mesh-free Lagrangian method that discretizes the continuum into particles. Each particle carries physical properties (mass, velocity, density) and interacts with neighbors through a smoothing kernel.

**Kernel Function**: Cubic spline kernel

```
W(r,h) = α × { 1 - 1.5q² + 0.75q³    (0 ≤ q < 1)
             { 0.25(2-q)³             (1 ≤ q < 2)
             { 0                       (q ≥ 2)

where q = r/h, α = 10/(7πh²)
```

**Continuity Equation** (density evolution):

```
dρ/dt = Σⱼ mⱼ(vᵢ - vⱼ)·∇Wᵢⱼ
```

### Equation of State (Tait EOS)

Pressure is computed using the weakly compressible Tait equation:

```
P = B[(ρ/ρ₀)^γ - 1]

where B = ρ₀c₀²/γ
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| ρ₀ | 2000 kg/m³ | Reference density |
| c₀ | 50 m/s | Speed of sound |
| γ | 7 | Tait exponent |
| B | 714,286 Pa | Pressure coefficient |

### Voellmy Friction Model

The basal friction combines Coulomb friction and turbulent resistance:

```
τ = μ_b·σ_n + ρg·v²/ξ
```

Or in friction coefficient form:

```
friction = μ_b + v²/ξ
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| μ_b | 0.1 | Coulomb friction coefficient (~6° friction angle) |
| ξ | 200 m/s² | Turbulent coefficient |

**Physical Interpretation**:
- **μ_b term**: Dominates at low velocities; represents particle-bed contact friction
- **v²/ξ term**: Dominates at high velocities; represents turbulent energy dissipation from particle collisions and vortex generation

**Note**: The Voellmy model is phenomenological, originally derived from open-channel hydraulics (Chézy formula). The turbulent coefficient ξ is typically calibrated via back-analysis of real events.

### Bingham Rheology

The debris flow is modeled as a Bingham viscoplastic fluid with yield stress:

```
τ = τ_y + μ·γ̇    (if τ > τ_y)
γ̇ = 0            (if τ ≤ τ_y)
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| μ | 100 Pa·s | Dynamic viscosity |
| τ_y | 50 Pa | Yield stress |

**Stop Condition**: Flow stops when shear stress falls below yield stress:

```
ρgh·sinθ < τ_y  →  Flow stops
```

**Note**: Current τ_y = 50 Pa is relatively low for typical debris flows (literature values: 100–10,000 Pa). This results in longer runout distances.

### Artificial Viscosity

Numerical stability is maintained using Monaghan-type artificial viscosity:

```
Πᵢⱼ = (-α_visc·c₀·μᵢⱼ + β_visc·μᵢⱼ²) / ρ̄ᵢⱼ

where μᵢⱼ = h(vᵢⱼ·rᵢⱼ)/(r² + 0.01h²)  for approaching particles
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| α_visc | 0.1 | Linear artificial viscosity |
| β_visc | 0.1 | Quadratic artificial viscosity |

### Momentum Equation

The acceleration of each particle is computed as:

```
dvᵢ/dt = -Σⱼ mⱼ(Pᵢ/ρᵢ² + Pⱼ/ρⱼ²)∇Wᵢⱼ    (pressure gradient)
       + viscous terms                      (artificial + physical viscosity)
       - g·∇z                               (gravity along slope)
       - friction terms                     (Voellmy + Bingham)
```

## Numerical Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| h | 2.5 m | Smoothing length |
| Particle spacing | 2.5 m | Distance between particles (h) |
| dt | 0.005 s | Time step |
| Cutoff | 5.0 m | Neighbor search radius (2h) |
| v_max | 30 m/s | Velocity ceiling |

## File Structure

```
landslide/
├── landslide_sph_gpu.py      # GPU SPH simulator core module
├── run_simulation.py         # Simulation execution script
├── visualize_results.py      # Result visualization and animation
├── run_3d_animation.py       # 3D animation generation (alternative)
│
├── irwon.dem                 # Irwon region DEM raw data
├── irwon_terrain.npy         # Preprocessed terrain array
├── irwon_terrain_crop.npy    # Cropped terrain array
├── irwon_terrain_hires.npy   # High-resolution terrain array
│
├── simulation_results.npz    # Simulation result data
├── simulation_progress.txt   # Simulation progress log
│
├── satellite_image.png       # Satellite image (original)
├── satellite_texture.png     # Satellite texture for terrain
│
├── irwon_landslide_3d.gif    # 3D landslide animation
├── irwon_initial_setup.png   # Initial particle setup image
├── runout_distance.png       # Runout distance analysis plot
│
└── backup/                   # Previous version backups
```

## Usage

### 1. Run Simulation

```bash
python run_simulation.py
```

- Loads Irwon region DEM
- Initializes particles in the collapse area
- Runs 60-second simulation
- Saves results to `simulation_results.npz`

### 2. Visualize Results

```bash
python visualize_results.py
```

- Generates 3D animation GIF (`irwon_landslide_3d.gif`)
- Creates runout distance plot (`runout_distance.png`)

## Output Data

The simulation saves the following data to `simulation_results.npz`:

| Key | Shape | Description |
|-----|-------|-------------|
| `times` | (N_frames,) | Time stamps |
| `x`, `y` | (N_frames, N_particles) | Particle positions |
| `vx`, `vy` | (N_frames, N_particles) | Particle velocities |
| `density` | (N_frames, N_particles) | Particle densities |
| `pressure` | (N_frames, N_particles) | Particle pressures |
| `height` | (N_frames, N_particles) | Particle heights |
| `terrain` | (ny, nx) | Terrain elevation grid |
| `rho0`, `c0`, `gamma` | scalar | SPH parameters |

## Coordinate System

- **CRS**: TM (EPSG:5186) - Korea 2000 / Central Belt
- **Origin**: X=203461, Y=535418
- **Cell size**: 30 m

## Dependencies

- Python 3.8+
- NumPy
- CuPy (requires CUDA)
- Matplotlib
- Pillow
- SciPy

```bash
pip install numpy cupy-cuda12x matplotlib pillow scipy
```

## Performance

| Configuration | Particles | 60s Simulation | Speed |
|---------------|-----------|----------------|-------|
| Low resolution | ~1,800 | ~3 min | 67 steps/s |
| High resolution | ~7,200 | ~12 min | ~17 steps/s |

## Limitations and Future Work

1. **Voellmy turbulent term**: Empirical model; consider implementing μ(I) rheology for physics-based friction
2. **Bingham parameters**: Current τ_y is low for debris flows; calibration with field data needed
3. **2D depth-averaged**: Vertical flow structure not resolved
4. **No entrainment**: Bed material entrainment not modeled

## References

- Bui, H. H., et al. (2008). Lagrangian meshfree particles method (SPH) for large deformation and failure flows of geomaterial using elastic-plastic soil constitutive model.
- Pastor, M., et al. (2009). Application of a SPH depth-integrated model to landslide run-out analysis.
- Hungr, O. (1995). A model for the runout analysis of rapid flow slides, debris flows, and avalanches.
- Voellmy, A. (1955). Uber die Zerstorungskraft von Lawinen (On the destructive force of avalanches).
