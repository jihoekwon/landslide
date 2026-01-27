# CLAUDE.md - Landslide SPH Simulation

이 파일은 Claude Code가 이 프로젝트를 이해하고 작업할 때 참고하는 가이드입니다.

## Project Overview

GPU 가속 SPH(Smoothed Particle Hydrodynamics) 기반 산사태/토석류 시뮬레이션 프로젝트입니다.

- **핵심 기술**: Depth-averaged SPH, Non-Newtonian rheology (Voellmy + Bingham), Takahashi (2007) entrainment model
- **GPU 가속**: CuPy 사용
- **개발**: 한국지질자원연구원(KIGAM) 지질자원AI융합연구실 권지회

## Pipeline

### 1. 지형 데이터 처리
```bash
python terrain_processor.py <input_dem.tif> --output-dir <project_dir>
```
- GeoTIFF DEM → `.npy` 지형 배열 변환
- 위성 이미지 크롭 및 정합
- 출력: `*_terrain_crop.npy`, `*_satellite_crop.png`, `*_metadata.json`

### 2. 시뮬레이션 실행
```bash
python run_simulation.py <project_dir>/simulation_config.yaml
```
- `simulation_config.yaml` 기반 SPH 시뮬레이션
- 출력: `simulation_results.npz`

### 3. 결과 시각화
```bash
python visualize_results.py <project_dir>
```
- `visualization_config.yaml` 설정 사용
- PyVista/VTK 기반 3D 렌더링
- 출력: `landslide_animation.gif`, `landslide_animation.webm`

### 4. 분석 보고서 생성
```bash
python generate_report.py <project_dir>
```
- `report_config.yaml`의 LLM API 키 사용
- 시뮬레이션 결과 분석 및 위험도 평가
- 출력: `analysis_report.html`

## Project Structure

```
landslide/
├── landslide_sph_gpu.py      # 핵심 SPH 엔진 (GPU)
├── run_simulation.py         # 시뮬레이션 실행 스크립트
├── terrain_processor.py      # DEM/위성 처리
├── visualize_results.py      # 3D 시각화 (PyVista)
├── generate_report.py        # LLM 분석 보고서
├── render_frame.py           # 프레임 렌더링 헬퍼
│
└── <project_dir>/            # 프로젝트별 디렉토리 (예: guryoung_dem_10m/)
    ├── simulation_config.yaml
    ├── visualization_config.yaml
    ├── report_config.yaml
    ├── *_terrain_crop.npy
    ├── *_satellite_crop.png
    ├── simulation_results.npz
    └── landslide_animation.gif
```

## Config Files

### simulation_config.yaml
- `terrain`: DEM 파일 경로
- `initial_condition`: 붕괴 위치 (center), 반경 (radius), 두께 (thickness)
- `sph`: smoothing length (h), dt, v_max
- `rheology`: mu_b, xi, tau_y, mu
- `entrainment`: delta_e, delta_d, phi_bed, C_init
- `simulation`: duration, save_interval

### visualization_config.yaml
- `view_elev`, `view_azim`: 카메라 각도
- `fps`, `frame_skip`: 애니메이션 설정
- `particle_cmap`: 컬러맵 (plasma, viridis 등)

### report_config.yaml
- `llm.api_key`: Anthropic API 키
- `llm.model`: claude-sonnet-4-20250514 또는 claude-opus-4-20250514

## Key Classes

### `LandslideSPHGPU` (landslide_sph_gpu.py)
- `__init__(terrain, ...)`: 지형 및 파라미터 초기화
- `initialize_particles(...)`: 초기 붕괴체 입자 배치
- `step()`: 1 타임스텝 계산
- `run(duration, save_interval)`: 전체 시뮬레이션 실행

## Dependencies

```bash
pip install numpy cupy-cuda12x matplotlib pillow scipy pyvista imageio[ffmpeg] anthropic
```

## Notes

- 시뮬레이션 결과는 `.npz` 포맷 (times, x, y, vx, vy, height, concentration 등)
- 좌표계: EPSG:5186 (Korea 2000 / Central Belt)
- GPU 메모리 부족 시 `particle_spacing_factor` 증가
