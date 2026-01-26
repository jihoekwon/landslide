# SPH Landslide Simulation - 문서 가이드

GPU 가속 SPH 기반 산사태/토석류 시뮬레이션 시스템

## 모듈 구성

```
landslide/
├── landslide_sph_gpu.py    # SPH 시뮬레이터 핵심
├── run_simulation.py       # 시뮬레이션 실행기
├── terrain_processor.py    # DEM 지형 전처리
├── visualize_results.py    # 3D 시각화
├── generate_report.py      # 보고서 생성
└── docs/
    ├── README.md                  # 이 파일
    ├── landslide_sph_gpu.md       # SPH 시뮬레이터 문서
    ├── run_simulation.md          # 시뮬레이션 실행 문서
    ├── terrain_processor.md       # 지형 전처리 문서
    ├── visualize_results.md       # 시각화 문서
    └── generate_report.md         # 보고서 생성 문서
```

## 전체 워크플로우

```
1. DEM 전처리
   terrain_processor.py
   ├── 입력: GeoTIFF (.tif)
   └── 출력: terrain_crop.npy, metadata.json, satellite_crop.png

2. 시뮬레이션 실행
   run_simulation.py
   ├── 입력: simulation_config.yaml, terrain_crop.npy
   └── 출력: simulation_results.npz, simulation_log.txt

3. 결과 시각화
   visualize_results.py
   ├── 입력: simulation_results.npz, satellite_crop.png
   └── 출력: landslide_animation.gif, landslide_animation.webm

4. 보고서 생성
   generate_report.py
   ├── 입력: simulation_results.npz, simulation_config.yaml
   └── 출력: analysis_report.md, analysis_report.html
```

## 빠른 시작

### 1. 환경 설정

```bash
# 필수 패키지
pip install numpy cupy pyyaml rasterio pyvista imageio imageio-ffmpeg pillow

# 선택적 (보고서 LLM 분석)
pip install anthropic
```

### 2. DEM 전처리

```python
from terrain_processor import TerrainProcessor

tp = TerrainProcessor("./data/dem.tif")
tp.crop(center=(207000, 542000), size=(2000, 2000))
tp.fetch_satellite_texture()
tp.save()
```

### 3. 시뮬레이션 설정 (`simulation_config.yaml`)

```yaml
project:
  name: "my_landslide"

terrain:
  dem_file: "terrain_crop.npy"
  metadata_file: "metadata.json"

initial_condition:
  type: "circular"
  center: [207000, 542000]
  radius: 30
  thickness: 2.0

sph:
  h: 1.0
  dt: 0.003

simulation:
  duration: 60.0
  save_interval: 0.2
```

### 4. 시뮬레이션 실행

```bash
python run_simulation.py ./my_project/simulation_config.yaml
```

### 5. 시각화

```bash
python visualize_results.py ./my_project/simulation_results.npz
```

### 6. 보고서 생성

```bash
python generate_report.py ./my_project/simulation_results.npz
```

## 설정 파일 요약

| 파일 | 용도 | 사용 모듈 |
|------|------|----------|
| `simulation_config.yaml` | 시뮬레이션 전체 설정 | run_simulation.py |
| `visualization_config.yaml` | 시각화 설정 | visualize_results.py |
| `report_config.yaml` | 보고서/LLM 설정 | generate_report.py |

## 물리 모델

### SPH (Smoothed Particle Hydrodynamics)

- 깊이 평균(Depth-averaged) 방정식
- Cubic spline 커널
- 인공점성 (Monaghan)

### 유변학

- Voellmy 마찰 모델: `τ = ρgh(μ_b + v²/ξ)`
- Bingham 유체 (항복응력)

### 침식 모델

- Takahashi (2007) 평형 농도 모델
- 경사 기반 침식/퇴적

## CFL 안정성 조건

```
대류 CFL: (v_max + c0) × dt / h < 0.5
점성 CFL: dt < h² / (2ν)
음속 조건: c0 ≥ 10 × v_max
```

## 출력 데이터 형식

### simulation_results.npz

| 키 | 형태 | 설명 |
|----|------|------|
| `times` | (n_frames,) | 시간 배열 |
| `x`, `y` | (n_frames, max_particles) | 위치 |
| `vx`, `vy` | (n_frames, max_particles) | 속도 |
| `height` | (n_frames, max_particles) | 유동 깊이 |
| `concentration` | (n_frames, max_particles) | 토사 농도 |
| `terrain` | (ny, nx) | 지형 고도 |
| `cell_size` | scalar | 그리드 크기 |
| `x_min`, `y_min` | scalar | CRS 원점 |

## 의존성

### 필수

- Python 3.8+
- numpy
- cupy (CUDA)
- pyyaml
- rasterio
- pyvista
- imageio, imageio-ffmpeg
- pillow
- scipy
- matplotlib

### 선택적

- anthropic (LLM 분석)
- pyproj (좌표 변환)

## 라이선스

MIT License
