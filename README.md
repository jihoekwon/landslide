# Landslide SPH Simulation

GPU-accelerated depth-averaged landslide simulation using Smoothed Particle Hydrodynamics (SPH) with non-Newtonian rheology and bed entrainment.

## Overview

This project simulates landslide and debris flow dynamics using real DEM data from Daeomsan (대모산), Irwon-dong, Gangnam-gu, Seoul, South Korea. The simulation implements:

- **Depth-averaged SPH** formulation for computational efficiency
- **Non-Newtonian rheology** (Voellmy + Bingham)
- **Takahashi (2007) entrainment model** for bed erosion/deposition

GPU parallel computing via CuPy enables efficient calculation of thousands of particles.

---

## 1. Depth-Averaged Model

### 1.1 Governing Equations

The model uses depth-averaged (shallow water) equations, integrating the 3D Navier-Stokes equations over flow depth. This assumes:
- Hydrostatic pressure distribution
- Depth-averaged velocity profile
- Horizontal length scale >> vertical length scale

**Mass Conservation (Continuity)**:
```
∂h/∂t + ∇·(h·v̄) = E - D
```
where:
- `h` = flow depth [m]
- `v̄` = depth-averaged velocity [m/s]
- `E` = entrainment rate [m/s]
- `D` = deposition rate [m/s]

**Momentum Conservation**:
```
∂(hv̄)/∂t + ∇·(hv̄⊗v̄) = -g·h·∇(h + z_b) - τ_b/ρ + S
```
where:
- `z_b` = bed elevation [m]
- `τ_b` = basal shear stress [Pa]
- `S` = source terms (viscosity, etc.)

### 1.2 SPH Discretization

**Depth-averaged continuity** in SPH form:
```
dh_i/dt = Σⱼ Ωⱼ (vᵢ - vⱼ)·∇Wᵢⱼ
```
where `Ωⱼ = mⱼ/(ρ₀·hⱼ)` is the particle area.

**Hydrostatic pressure**:
```
P = ½ ρ₀ g h²
```

**Pressure gradient force** (depth-averaged):
```
(dv/dt)_pressure = -g Σⱼ Ωⱼ (hᵢ + hⱼ)/2 · ∇Wᵢⱼ
```

### 1.3 References - Depth-Averaged SPH

- **Savage & Hutter (1989)**. The motion of a finite mass of granular material down a rough incline. *J. Fluid Mech.*, 199, 177-215.
- **Pastor, M., et al. (2009)**. Application of a SPH depth-integrated model to landslide run-out analysis. *Géotechnique*, 59(1), 45-57.
- **McDougall, S. & Hungr, O. (2004)**. A model for the analysis of rapid landslide motion across three-dimensional terrain. *Can. Geotech. J.*, 41, 1084-1097.

---

## 2. Non-Newtonian Rheology

Debris flows exhibit non-Newtonian behavior - the relationship between shear stress and strain rate is nonlinear. This simulation combines two rheological models.

### 2.1 Voellmy Friction Model

The Voellmy model (1955) combines Coulomb friction with velocity-dependent turbulent resistance:

```
τ_b = μ_b·σ_n + ρg·v²/ξ
```

In friction coefficient form:
```
f = μ_b + v²/ξ
```

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Coulomb friction | μ_b | 0.1 | Basal friction coefficient (~6°) |
| Turbulent coefficient | ξ | 200 m/s² | Velocity-squared resistance |

**Physical Interpretation**:

1. **Coulomb term (μ_b·σ_n)**:
   - Dominates at low velocities
   - Represents frictional contact between flow and bed
   - Analogous to dry granular friction

2. **Turbulent term (ρg·v²/ξ)**:
   - Dominates at high velocities
   - Represents energy dissipation from:
     - Internal collisions between particles
     - Turbulent mixing and vortex generation
   - Derived from Chézy formula in open-channel hydraulics

**Limitations**: The Voellmy model is phenomenological (empirical). The coefficient ξ must be calibrated through back-analysis of observed events. Typical values:
- Snow avalanches: ξ = 1000-2000 m/s²
- Debris flows: ξ = 100-500 m/s²
- Rock avalanches: ξ = 500-1000 m/s²

### 2.2 Bingham Viscoplastic Model

The Bingham model describes fluids with yield stress - flow only occurs when applied stress exceeds a threshold:

```
τ = τ_y + μ·γ̇    (if τ > τ_y)
γ̇ = 0            (if τ ≤ τ_y)
```

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Yield stress | τ_y | 500 Pa | Minimum stress for flow |
| Plastic viscosity | μ | 100 Pa·s | Viscous resistance |

**Physical Interpretation**:

- **Yield stress (τ_y)**: Represents the internal structure of debris (particle contacts, clay matrix). Flow only initiates when gravitational stress exceeds this threshold.

- **Viscosity (μ)**: Governs the rate of deformation once flowing. Higher viscosity → slower, more uniform flow.

**Stop Condition**: Flow stops when driving stress falls below yield stress:
```
τ_driving = ρgh·sinθ
τ_driving < τ_y → Flow stops
```

**Typical Values** (from literature):
| Material | τ_y (Pa) | μ (Pa·s) |
|----------|----------|----------|
| Mudflow | 10-100 | 1-10 |
| Debris flow | 100-1000 | 10-100 |
| Hyperconcentrated flow | 1000-10000 | 100-1000 |

### 2.3 Combined Implementation

The total basal resistance in this model:
```
a_friction = -g·(μ_b + v²/ξ)·v/|v|           # Voellmy
a_yield = -τ_y/(ρ₀·h)·v/|v|    (if |v| < 1)  # Bingham (low speed only)
```

The Bingham yield term is applied only at low velocities as a stopping criterion.

### 2.4 References - Rheology

- **Voellmy, A. (1955)**. Über die Zerstörungskraft von Lawinen. *Schweizerische Bauzeitung*, 73, 159-162.
- **Bingham, E.C. (1922)**. *Fluidity and Plasticity*. McGraw-Hill, New York.
- **Hungr, O. (1995)**. A model for the runout analysis of rapid flow slides, debris flows, and avalanches. *Can. Geotech. J.*, 32, 610-623.
- **Iverson, R.M. (1997)**. The physics of debris flows. *Rev. Geophys.*, 35(3), 245-296.
- **Coussot, P. & Meunier, M. (1996)**. Recognition, classification and mechanical description of debris flows. *Earth-Sci. Rev.*, 40, 209-227.

---

## 3. Entrainment Model (Takahashi 2007)

Debris flows can grow significantly by entraining bed material (erosion) or lose volume through deposition. This simulation implements the Takahashi equilibrium concentration model.

### 3.1 Equilibrium Concentration

Takahashi (1991, 2007) proposed that debris flows tend toward an equilibrium sediment concentration that depends on channel slope:

```
C_eq = ρ_w · tan(θ) / [(ρ_s - ρ_w) · (tan(φ) - tan(θ))]
```

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Water density | ρ_w | 1000 kg/m³ | Interstitial fluid |
| Solid density | ρ_s | 2650 kg/m³ | Sediment particles |
| Bed friction angle | φ | 35° | Internal friction of bed material |
| Slope angle | θ | variable | Local bed slope |
| Max packing | C_max | 0.65 | Maximum solid concentration |

**Physical Meaning**:
- Steeper slopes → higher C_eq (more sediment can be transported)
- When θ → φ: C_eq → ∞ (slope at failure threshold)
- When θ → 0: C_eq → 0 (flat bed, no transport capacity)

### 3.2 Erosion Rate

When flow concentration is below equilibrium (C < C_eq), erosion occurs:

```
i_e = δ_e · C_eq · v
```

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Erosion coefficient | δ_e | 0.0007 | Empirical (0.0001-0.01) |

**Physical Interpretation**:
- Erosion rate proportional to equilibrium concentration (transport capacity)
- Erosion rate proportional to velocity (shear stress on bed)
- The bed lowers at rate i_e, and bed material (at concentration C_bed) enters the flow

### 3.3 Deposition Rate

When flow concentration exceeds equilibrium (C > C_eq), deposition occurs:

```
i_d = δ_d · C · v · (1 - tan(θ)/tan(φ))
```

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Deposition coefficient | δ_d | 0.01 | Empirical (0.01-0.05) |

**Physical Interpretation of (1 - tan(θ)/tan(φ))**:
- When θ → φ (steep slope): factor → 0, **no deposition** (slope too steep to hold sediment)
- When θ → 0 (flat): factor → 1, **maximum deposition**
- This captures the physical reality that deposited material must be stable on the slope

### 3.4 Mass Conservation

**Flow depth change**:
```
dh/dt = i_e - i_d    (erosion increases h, deposition decreases h)
```

**Solid mass conservation**:
```
d(h·C)/dt = C_bed · i_e - C_bed · i_d
```

where C_bed = C_max = 0.65 (packed bed concentration).

**Concentration evolution**:
- Erosion: bed material (C_bed = 0.65) enters → flow concentration increases
- Deposition: material settles at C_bed → remaining flow becomes diluted

### 3.5 Implementation Notes

```python
# Erosion (C < C_eq)
erosion_rate = delta_e * C_eq * speed
dh = +erosion_rate * dt
d(h*C) = +C_bed * erosion_rate * dt

# Deposition (C > C_eq)
slope_factor = max(1 - tan(theta)/tan(phi), 0)
deposition_rate = delta_d * C * speed * slope_factor
dh = -deposition_rate * dt
d(h*C) = -C_bed * deposition_rate * dt
```

### 3.6 References - Entrainment

- **Takahashi, T. (1991)**. *Debris Flow*. IAHR Monograph, Balkema, Rotterdam.
- **Takahashi, T. (2007)**. *Debris Flow: Mechanics, Prediction and Countermeasures*. Taylor & Francis, London. (Primary reference)
- **Egashira, S., et al. (1997)**. Mechanism of debris flow deposition and characteristics of debris flow deposits. *J. Japan Soc. Eng. Geol.*, 38, 149-155.
- **Hungr, O., et al. (2005)**. A review of the classification of landslides of the flow type. *Environ. Eng. Geosci.*, 11(3), 167-194.

---

## 4. Model Parameters Summary

### 4.1 Physical Parameters

| Category | Parameter | Symbol | Value | Reference |
|----------|-----------|--------|-------|-----------|
| **Density** | Reference density | ρ₀ | 2000 kg/m³ | - |
| | Solid particle | ρ_s | 2650 kg/m³ | Typical mineral |
| | Water | ρ_w | 1000 kg/m³ | - |
| **Voellmy** | Basal friction | μ_b | 0.1 | Hungr (1995) |
| | Turbulent coeff. | ξ | 200 m/s² | Calibrated |
| **Bingham** | Yield stress | τ_y | 500 Pa | Coussot (1996) |
| | Viscosity | μ | 100 Pa·s | Coussot (1996) |
| **Entrainment** | Erosion coeff. | δ_e | 0.0007 | Takahashi (2007) |
| | Deposition coeff. | δ_d | 0.01 | Takahashi (2007) |
| | Bed friction angle | φ | 35° | Typical sand/gravel |
| | Initial concentration | C_init | 0.4 | - |
| | Max packing | C_max | 0.65 | Random close packing |

### 4.2 Numerical Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Smoothing length (h) | 2.5 m | SPH kernel size |
| Particle spacing | 2.5 m | Initial particle distance |
| Time step (dt) | 0.005 s | Integration step |
| Cutoff distance | 5.0 m | Neighbor search (2h) |
| Velocity ceiling | 30 m/s | Numerical stability |

---

## 5. File Structure

```
landslide/
├── landslide_sph_gpu.py      # GPU SPH simulator core module
├── run_simulation.py         # Simulation execution script
├── visualize_results.py      # Result visualization and animation
│
├── irwon_terrain_hires.npy   # DEM terrain array
├── simulation_results.npz    # Simulation output data
├── satellite_texture.png     # Satellite texture for visualization
│
├── irwon_landslide_3d.gif    # 3D animation output
├── entrainment_2panel.png    # Entrainment analysis plot
└── runout_distance.png       # Runout distance plot
```

---

## 6. Usage

### Run Simulation
```bash
python run_simulation.py
```

### Visualize Results
```bash
python visualize_results.py
```

---

## 7. Output Data

The simulation saves to `simulation_results.npz`:

| Key | Shape | Description |
|-----|-------|-------------|
| `times` | (N_frames,) | Time stamps [s] |
| `x`, `y` | (N_frames, N_particles) | Positions [m] |
| `vx`, `vy` | (N_frames, N_particles) | Velocities [m/s] |
| `height` | (N_frames, N_particles) | Flow depth [m] |
| `concentration` | (N_frames, N_particles) | Solid concentration [-] |
| `density` | (N_frames, N_particles) | ρ = ρ₀·h [kg/m²] |
| `pressure` | (N_frames, N_particles) | P = ½ρ₀gh² [Pa·m] |
| `terrain` | (ny, nx) | Bed elevation [m] |

---

## 8. References (Complete)

### Depth-Averaged Flow
1. Savage, S.B. & Hutter, K. (1989). The motion of a finite mass of granular material down a rough incline. *J. Fluid Mech.*, 199, 177-215.
2. Pastor, M., et al. (2009). Application of a SPH depth-integrated model to landslide run-out analysis. *Géotechnique*, 59(1), 45-57.
3. McDougall, S. & Hungr, O. (2004). A model for the analysis of rapid landslide motion across three-dimensional terrain. *Can. Geotech. J.*, 41, 1084-1097.

### Rheology
4. Voellmy, A. (1955). Über die Zerstörungskraft von Lawinen. *Schweizerische Bauzeitung*, 73, 159-162.
5. Bingham, E.C. (1922). *Fluidity and Plasticity*. McGraw-Hill, New York.
6. Hungr, O. (1995). A model for the runout analysis of rapid flow slides, debris flows, and avalanches. *Can. Geotech. J.*, 32, 610-623.
7. Iverson, R.M. (1997). The physics of debris flows. *Rev. Geophys.*, 35(3), 245-296.
8. Coussot, P. & Meunier, M. (1996). Recognition, classification and mechanical description of debris flows. *Earth-Sci. Rev.*, 40, 209-227.

### Entrainment
9. Takahashi, T. (1991). *Debris Flow*. IAHR Monograph, Balkema, Rotterdam.
10. Takahashi, T. (2007). *Debris Flow: Mechanics, Prediction and Countermeasures*. Taylor & Francis, London.
11. Egashira, S., et al. (1997). Mechanism of debris flow deposition and characteristics of debris flow deposits. *J. Japan Soc. Eng. Geol.*, 38, 149-155.

### SPH Method
12. Bui, H.H., et al. (2008). Lagrangian meshfree particles method (SPH) for large deformation and failure flows of geomaterial. *Int. J. Numer. Anal. Meth. Geomech.*, 32, 1537-1570.
13. Monaghan, J.J. (1992). Smoothed particle hydrodynamics. *Annu. Rev. Astron. Astrophys.*, 30, 543-574.

---

## 9. Study Area

- **Location**: Daeomsan (대모산), Irwon-dong, Gangnam-gu, Seoul, South Korea
- **Coordinate System**: TM (EPSG:5186) - Korea 2000 / Central Belt
- **Origin**: X=203461, Y=535418
- **Cell size**: 30 m

---

## 10. Dependencies

```bash
pip install numpy cupy-cuda12x matplotlib pillow scipy
```

- Python 3.8+
- NumPy
- CuPy (requires CUDA-capable GPU)
- Matplotlib
- Pillow
- SciPy
