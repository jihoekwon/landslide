"""
Check time step stability criteria for SPH simulation.

SPH stability conditions:
1. CFL condition: dt <= C_CFL * h / (c + v_max)
2. Viscous condition: dt <= C_visc * h^2 * rho / mu
3. Force condition: dt <= C_f * sqrt(h / |a_max|)
4. Body force condition: dt <= sqrt(h / g)
"""

import numpy as np

def analyze_stability():
    print("=" * 70)
    print("SPH TIME STEP STABILITY ANALYSIS")
    print("=" * 70)

    # Current simulation parameters
    h = 15.0        # Smoothing length [m]
    c0 = 50.0       # Speed of sound [m/s]
    dt = 0.005      # Current time step [s]
    rho0 = 2000.0   # Reference density [kg/m^3]
    mu = 100.0      # Dynamic viscosity [Pa.s]
    g = 9.81        # Gravity [m/s^2]

    # Observed values from simulation
    v_max_observed = 19.0   # Max velocity observed [m/s]
    a_max_observed = 15.0   # Estimated max acceleration [m/s^2]

    # Safety factors (typical values)
    C_CFL = 0.25    # CFL safety factor (0.25 ~ 0.4)
    C_visc = 0.125  # Viscous safety factor
    C_force = 0.25  # Force safety factor

    print("\n[SIMULATION PARAMETERS]")
    print(f"  Smoothing length (h):    {h} m")
    print(f"  Speed of sound (c0):     {c0} m/s")
    print(f"  Current time step (dt):  {dt} s")
    print(f"  Reference density (rho): {rho0} kg/m^3")
    print(f"  Dynamic viscosity (mu):  {mu} Pa.s")
    print(f"  Gravity (g):             {g} m/s^2")
    print(f"  Max velocity observed:   {v_max_observed} m/s")

    print("\n" + "=" * 70)
    print("STABILITY CONDITIONS")
    print("=" * 70)

    # 1. CFL Condition
    dt_CFL = C_CFL * h / (c0 + v_max_observed)
    CFL_ratio = dt / dt_CFL

    print(f"\n[1] CFL CONDITION (Courant-Friedrichs-Lewy)")
    print(f"    Formula: dt <= C_CFL * h / (c + v_max)")
    print(f"    dt_CFL = {C_CFL} * {h} / ({c0} + {v_max_observed})")
    print(f"    dt_CFL = {dt_CFL:.6f} s")
    print(f"    Current dt / dt_CFL = {CFL_ratio:.2f}")
    if CFL_ratio <= 1.0:
        print(f"    [OK] SATISFIED (ratio <= 1.0)")
    else:
        print(f"    [FAIL] VIOLATED! (ratio > 1.0)")

    Co = (c0 + v_max_observed) * dt / h
    print(f"    Courant number Co = (c + v_max) * dt / h = {Co:.3f}")
    print(f"    (Should be < {1/C_CFL:.1f} for stability)")

    # 2. Viscous Condition
    nu = mu / rho0
    dt_visc = C_visc * h**2 * rho0 / mu
    visc_ratio = dt / dt_visc

    print(f"\n[2] VISCOUS CONDITION")
    print(f"    Formula: dt <= C_visc * h^2 * rho / mu")
    print(f"    Kinematic viscosity nu = mu/rho = {nu:.4f} m^2/s")
    print(f"    dt_visc = {C_visc} * {h}^2 * {rho0} / {mu}")
    print(f"    dt_visc = {dt_visc:.6f} s")
    print(f"    Current dt / dt_visc = {visc_ratio:.4f}")
    if visc_ratio <= 1.0:
        print(f"    [OK] SATISFIED (ratio <= 1.0)")
    else:
        print(f"    [FAIL] VIOLATED! (ratio > 1.0)")

    # 3. Body Force Condition
    dt_body = np.sqrt(h / g)
    body_ratio = dt / dt_body

    print(f"\n[3] BODY FORCE CONDITION (Gravity)")
    print(f"    Formula: dt <= sqrt(h / g)")
    print(f"    dt_body = sqrt({h} / {g})")
    print(f"    dt_body = {dt_body:.6f} s")
    print(f"    Current dt / dt_body = {body_ratio:.4f}")
    if body_ratio <= 1.0:
        print(f"    [OK] SATISFIED (ratio <= 1.0)")
    else:
        print(f"    [FAIL] VIOLATED! (ratio > 1.0)")

    # 4. Acceleration Condition
    dt_force = C_force * np.sqrt(h / a_max_observed)
    force_ratio = dt / dt_force

    print(f"\n[4] ACCELERATION CONDITION")
    print(f"    Formula: dt <= C_f * sqrt(h / a_max)")
    print(f"    dt_force = {C_force} * sqrt({h} / {a_max_observed})")
    print(f"    dt_force = {dt_force:.6f} s")
    print(f"    Current dt / dt_force = {force_ratio:.4f}")
    if force_ratio <= 1.0:
        print(f"    [OK] SATISFIED (ratio <= 1.0)")
    else:
        print(f"    [FAIL] VIOLATED! (ratio > 1.0)")

    # Combined
    dt_min = min(dt_CFL, dt_visc, dt_body, dt_force)
    conditions = ["CFL", "Viscous", "Body Force", "Acceleration"]
    limits = [dt_CFL, dt_visc, dt_body, dt_force]
    limiting = conditions[limits.index(dt_min)]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Condition          dt_limit (s)    dt/dt_limit   Status")
    print(f"  " + "-" * 60)

    ratios = [CFL_ratio, visc_ratio, body_ratio, force_ratio]
    for cond, lim, ratio in zip(conditions, limits, ratios):
        status = "OK" if ratio <= 1 else "FAIL"
        print(f"  {cond:<18} {lim:.6f}       {ratio:.4f}        [{status}]")

    print(f"  " + "-" * 60)
    print(f"  MINIMUM            {dt_min:.6f}       {dt/dt_min:.4f}        (Limited by: {limiting})")

    print(f"\n  Current dt = {dt} s")
    print(f"  Recommended max dt = {dt_min:.6f} s")

    if dt <= dt_min:
        print(f"\n  >>> OVERALL: TIME STEP IS STABLE <<<")
    else:
        print(f"\n  >>> OVERALL: TIME STEP MAY BE UNSTABLE <<<")
        print(f"      Reduce dt to at most {dt_min:.6f} s")

    # Sensitivity analysis
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS (CFL vs Max Velocity)")
    print("=" * 70)
    print(f"\n  v_max (m/s)    dt_CFL (s)    Current dt OK?")
    print(f"  " + "-" * 45)
    for v in [10, 15, 20, 25, 30]:
        dt_cfl_v = C_CFL * h / (c0 + v)
        ok = "OK" if dt <= dt_cfl_v else "FAIL"
        print(f"  {v:<12}   {dt_cfl_v:.6f}      [{ok}]")

    return dt_min


if __name__ == '__main__':
    analyze_stability()
