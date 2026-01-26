"""
Run SPH Landslide Simulation from YAML configuration file.

Usage:
    python run_simulation.py <config_yaml_path>

Example:
    python run_simulation.py ./guryoung_dem_10m/simulation_config.yaml
"""

import numpy as np
import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import yaml

from landslide_sph_gpu import SPHSimulatorGPU


@dataclass
class SimulationConfig:
    """시뮬레이션 설정 데이터 클래스"""
    # 작업 디렉토리
    work_dir: Path

    # 프로젝트 정보
    name: str
    description: str

    # 지형 파일
    dem_file: str
    metadata_file: str
    satellite_file: str

    # 초기 조건
    init_type: str
    init_center: Tuple[float, float]
    init_radius: float
    init_thickness: float
    init_n_lobes: int
    init_seed: int

    # SPH 파라미터
    h: float
    particle_spacing_factor: float
    dt: float
    v_max: float

    # 물성
    rho0: float
    c0: float
    gamma: float

    # 유변학
    mu: float
    tau_y: float
    mu_b: float
    xi: float
    alpha: float
    beta: float

    # 침식
    entrainment_enabled: bool
    delta_e: float
    delta_d: float
    rho_s: float
    rho_w: float
    phi_bed: float
    C_init: float
    C_max: float

    # 시뮬레이션
    duration: float
    save_interval: float

    # 출력
    results_file: str
    log_file: str
    animation_file: str


def load_config(yaml_path: str) -> SimulationConfig:
    """YAML 설정 파일 로드"""
    yaml_path = Path(yaml_path).resolve()

    if not yaml_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없음: {yaml_path}")

    work_dir = yaml_path.parent

    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    return SimulationConfig(
        work_dir=work_dir,
        # 프로젝트
        name=cfg['project']['name'],
        description=cfg['project']['description'],
        # 지형
        dem_file=cfg['terrain']['dem_file'],
        metadata_file=cfg['terrain']['metadata_file'],
        satellite_file=cfg['terrain']['satellite_file'],
        # 초기 조건
        init_type=cfg['initial_condition']['type'],
        init_center=tuple(cfg['initial_condition']['center']),
        init_radius=cfg['initial_condition']['radius'],
        init_thickness=cfg['initial_condition']['thickness'],
        init_n_lobes=cfg['initial_condition']['n_lobes'],
        init_seed=cfg['initial_condition']['seed'],
        # SPH
        h=cfg['sph']['h'],
        particle_spacing_factor=cfg['sph']['particle_spacing_factor'],
        dt=cfg['sph']['dt'],
        v_max=cfg['sph']['v_max'],
        # 물성
        rho0=cfg['material']['rho0'],
        c0=cfg['material']['c0'],
        gamma=cfg['material']['gamma'],
        # 유변학
        mu=cfg['rheology']['mu'],
        tau_y=cfg['rheology']['tau_y'],
        mu_b=cfg['rheology']['mu_b'],
        xi=cfg['rheology']['xi'],
        alpha=cfg['rheology']['alpha'],
        beta=cfg['rheology']['beta'],
        # 침식
        entrainment_enabled=cfg['entrainment']['enabled'],
        delta_e=cfg['entrainment']['delta_e'],
        delta_d=cfg['entrainment']['delta_d'],
        rho_s=cfg['entrainment']['rho_s'],
        rho_w=cfg['entrainment']['rho_w'],
        phi_bed=cfg['entrainment']['phi_bed'],
        C_init=cfg['entrainment']['C_init'],
        C_max=cfg['entrainment']['C_max'],
        # 시뮬레이션
        duration=cfg['simulation']['duration'],
        save_interval=cfg['simulation']['save_interval'],
        # 출력
        results_file=cfg['output']['results_file'],
        log_file=cfg['output']['log_file'],
        animation_file=cfg['output']['animation_file'],
    )


def load_terrain(config: SimulationConfig) -> Tuple[np.ndarray, dict]:
    """지형 데이터 및 메타데이터 로드"""
    dem_path = config.work_dir / config.dem_file
    meta_path = config.work_dir / config.metadata_file

    if not dem_path.exists():
        raise FileNotFoundError(f"DEM 파일을 찾을 수 없음: {dem_path}")

    terrain = np.load(str(dem_path))

    # 메타데이터 로드
    metadata = {}
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    return terrain, metadata


def create_arbitrary_blob(cx: float, cy: float, base_radius: float = 75,
                          n_lobes: int = 5, particle_spacing: float = 5.0,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """불규칙 blob 형상 생성"""
    np.random.seed(seed)
    particles_x, particles_y = [], []
    grid_size = int(base_radius * 2.5 / particle_spacing)
    xs = np.linspace(cx - base_radius * 1.2, cx + base_radius * 1.2, grid_size)
    ys = np.linspace(cy - base_radius * 1.2, cy + base_radius * 1.2, grid_size)

    # 랜덤 lobe 파라미터
    lobe_angles = np.random.uniform(0, 2*np.pi, n_lobes)
    lobe_radii = np.random.uniform(0.3, 0.8, n_lobes) * base_radius
    lobe_widths = np.random.uniform(0.3, 0.6, n_lobes)

    for x in xs:
        for y in ys:
            dx, dy = x - cx, y - cy
            dist = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)

            # 기본 타원
            ellipse_a, ellipse_b = base_radius * 1.1, base_radius * 0.8
            ellipse_r = (ellipse_a * ellipse_b) / np.sqrt(
                (ellipse_b * np.cos(angle - 0.3))**2 +
                (ellipse_a * np.sin(angle - 0.3))**2
            )

            # Lobe가 추가된 경계
            boundary = ellipse_r * 0.7
            for i in range(n_lobes):
                angle_diff = angle - lobe_angles[i]
                boundary += lobe_radii[i] * np.exp(-angle_diff**2 / lobe_widths[i]**2)
            boundary += np.random.uniform(-3, 3)

            if dist < boundary:
                particles_x.append(x)
                particles_y.append(y)

    return np.array(particles_x), np.array(particles_y)


def create_circular_blob(cx: float, cy: float, radius: float,
                         particle_spacing: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """원형 blob 생성"""
    particles_x, particles_y = [], []

    n = int(2 * radius / particle_spacing) + 1
    xs = np.linspace(cx - radius, cx + radius, n)
    ys = np.linspace(cy - radius, cy + radius, n)

    for x in xs:
        for y in ys:
            if (x - cx)**2 + (y - cy)**2 <= radius**2:
                particles_x.append(x)
                particles_y.append(y)

    return np.array(particles_x), np.array(particles_y)


def create_initial_particles(config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
    """초기 조건에 따른 파티클 생성"""
    cx, cy = config.init_center
    particle_spacing = config.h * config.particle_spacing_factor

    if config.init_type == "arbitrary_blob":
        return create_arbitrary_blob(
            cx, cy,
            base_radius=config.init_radius,
            n_lobes=config.init_n_lobes,
            particle_spacing=particle_spacing,
            seed=config.init_seed
        )
    elif config.init_type == "circular":
        return create_circular_blob(
            cx, cy,
            radius=config.init_radius,
            particle_spacing=particle_spacing
        )
    else:
        raise ValueError(f"Unknown initial condition type: {config.init_type}")


def save_results(sim, config: SimulationConfig, metadata: dict,
                 particles_x: np.ndarray, particles_y: np.ndarray):
    """시뮬레이션 결과 저장"""
    output_path = config.work_dir / config.results_file

    # History를 배열로 변환
    n_frames = len(sim.history)
    max_particles = max(len(h['x']) for h in sim.history)

    # 배열 할당
    times = np.zeros(n_frames)
    n_active = np.zeros(n_frames, dtype=np.int32)
    x_data = np.full((n_frames, max_particles), np.nan, dtype=np.float32)
    y_data = np.full((n_frames, max_particles), np.nan, dtype=np.float32)
    vx_data = np.full((n_frames, max_particles), np.nan, dtype=np.float32)
    vy_data = np.full((n_frames, max_particles), np.nan, dtype=np.float32)
    height_data = np.full((n_frames, max_particles), np.nan, dtype=np.float32)
    density_data = np.full((n_frames, max_particles), np.nan, dtype=np.float32)
    pressure_data = np.full((n_frames, max_particles), np.nan, dtype=np.float32)
    concentration_data = np.full((n_frames, max_particles), np.nan, dtype=np.float32)

    for i, state in enumerate(sim.history):
        times[i] = state['time']
        n = len(state['x'])
        n_active[i] = state.get('n_active', n)
        x_data[i, :n] = state['x']
        y_data[i, :n] = state['y']
        vx_data[i, :n] = state['vx']
        vy_data[i, :n] = state['vy']
        height_data[i, :n] = state['height']
        density_data[i, :n] = state['density']
        pressure_data[i, :n] = state.get('pressure', np.nan)
        concentration_data[i, :n] = state.get('concentration', np.nan)

    # 초기 중심
    init_x = particles_x.mean()
    init_y = particles_y.mean()

    # 메타데이터에서 좌표 정보 추출
    cell_size = metadata.get('cell_size', 10.0)
    crop_bounds = metadata.get('crop_bounds', metadata.get('bounds', [0, 0, 0, 0]))
    x_min = crop_bounds[0] if crop_bounds else 0
    y_min = crop_bounds[1] if crop_bounds else 0

    # 저장
    np.savez_compressed(str(output_path),
        # 시뮬레이션 히스토리
        times=times,
        n_active=n_active,
        x=x_data,
        y=y_data,
        vx=vx_data,
        vy=vy_data,
        height=height_data,
        density=density_data,
        pressure=pressure_data,
        concentration=concentration_data,
        # 지형
        terrain=sim.terrain_grid.get() if hasattr(sim.terrain_grid, 'get') else sim.terrain_grid,
        # 메타데이터
        cell_size=cell_size,
        x_min=x_min,
        y_min=y_min,
        init_x=init_x,
        init_y=init_y,
        duration=config.duration,
        save_interval=config.save_interval,
        # SPH 파라미터
        h=config.h,
        rho0=config.rho0,
        c0=config.c0,
        gamma=config.gamma,
        # 침식 파라미터
        C_init=config.C_init,
        rho_s=config.rho_s,
        rho_w=config.rho_w,
        phi_bed=config.phi_bed,
        # 설정 파일 정보
        config_name=config.name,
    )

    return output_path


def run_simulation(config_path: str):
    """메인 시뮬레이션 실행"""
    print("=" * 60)
    print("SPH LANDSLIDE SIMULATION")
    print("=" * 60)

    # 설정 로드
    print(f"\n[1] 설정 파일 로드: {config_path}")
    config = load_config(config_path)
    print(f"    프로젝트: {config.name}")
    print(f"    작업 디렉토리: {config.work_dir}")

    # 지형 로드
    print(f"\n[2] 지형 데이터 로드")
    terrain, metadata = load_terrain(config)
    ny, nx = terrain.shape
    cell_size = metadata.get('cell_size', 10.0)
    print(f"    그리드: {nx} x {ny}")
    print(f"    셀 크기: {cell_size} m")

    crop_bounds = metadata.get('crop_bounds', metadata.get('bounds'))
    if crop_bounds:
        print(f"    X 범위: {crop_bounds[0]:.0f} ~ {crop_bounds[2]:.0f}")
        print(f"    Y 범위: {crop_bounds[1]:.0f} ~ {crop_bounds[3]:.0f}")

    # 시뮬레이터 초기화 (설정값 전달)
    print(f"\n[3] SPH 시뮬레이터 초기화")
    sim = SPHSimulatorGPU(
        terrain,
        cell_size,
        # SPH 파라미터
        h=config.h,
        dt=config.dt,
        v_max=config.v_max,
        # 물성
        rho0=config.rho0,
        c0=config.c0,
        gamma=config.gamma,
        # 유변학
        mu=config.mu,
        tau_y=config.tau_y,
        mu_b=config.mu_b,
        xi=config.xi,
        alpha=config.alpha,
        beta=config.beta,
        # 침식
        entrainment_enabled=config.entrainment_enabled,
        delta_e=config.delta_e,
        delta_d=config.delta_d,
        rho_s=config.rho_s,
        rho_w=config.rho_w,
        phi_bed=config.phi_bed,
        C_init=config.C_init,
        C_max=config.C_max,
    )
    print(f"    h = {config.h} m")
    print(f"    dt = {config.dt} s")
    print(f"    c0 = {config.c0} m/s")
    print(f"    v_max = {config.v_max} m/s")

    # 초기 파티클 생성
    print(f"\n[4] 초기 조건 생성")
    particles_x, particles_y = create_initial_particles(config)
    print(f"    타입: {config.init_type}")
    print(f"    중심: ({config.init_center[0]}, {config.init_center[1]})")
    print(f"    반경: {config.init_radius} m")
    print(f"    파티클 수: {len(particles_x)}")

    # 좌표 원점 (crop_bounds에서)
    x_min = crop_bounds[0] if crop_bounds else 0
    y_min = crop_bounds[1] if crop_bounds else 0

    # 파티클 초기화
    sim.initialize_particles_from_coords(
        particles_x, particles_y,
        x_min, y_min,
        thickness=config.init_thickness
    )

    # 시뮬레이션 실행
    print(f"\n[5] 시뮬레이션 실행")
    print(f"    지속 시간: {config.duration} s")
    print(f"    저장 간격: {config.save_interval} s")

    log_path = config.work_dir / config.log_file
    start_time = time.time()
    sim.run(
        duration=config.duration,
        save_interval=config.save_interval,
        log_file=str(log_path)
    )
    elapsed = time.time() - start_time
    print(f"    실행 시간: {elapsed:.1f} s")

    # 결과 저장
    print(f"\n[6] 결과 저장")
    output_path = save_results(sim, config, metadata, particles_x, particles_y)
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"    파일: {output_path}")
    print(f"    크기: {file_size:.1f} MB")
    print(f"    프레임 수: {len(sim.history)}")

    print("\n" + "=" * 60)
    print("시뮬레이션 완료!")
    print("=" * 60)

    return sim, config


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_simulation.py <config_yaml_path>")
        print("Example: python run_simulation.py ./guryoung_dem_10m/simulation_config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    run_simulation(config_path)
