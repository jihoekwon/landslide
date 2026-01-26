# run_simulation.py

YAML 설정 파일 기반 SPH 산사태 시뮬레이션 실행 모듈

## 개요

`simulation_config.yaml` 파일을 읽어 시뮬레이션 파라미터를 설정하고, SPH 시뮬레이션을 실행한 후 결과를 저장합니다.

## 사용법

```bash
python run_simulation.py <config_yaml_path>

# 예시
python run_simulation.py ./guryoung_dem_10m/simulation_config.yaml
```

## 입력

### 설정 파일 (`simulation_config.yaml`)

```yaml
# 프로젝트 정보
project:
  name: "guryong_landslide"
  description: "구룡산 산사태 시뮬레이션"

# 지형 데이터 (terrain_processor 출력물)
terrain:
  dem_file: "terrain_crop.npy"
  metadata_file: "metadata.json"
  satellite_file: "satellite_crop.png"

# 초기 조건
initial_condition:
  type: "arbitrary_blob"  # circular, arbitrary_blob, from_file
  center: [207000, 542000]  # (x, y) CRS 좌표
  radius: 25               # meters
  thickness: 2.0           # initial flow depth (m)
  n_lobes: 5               # for arbitrary_blob
  seed: 42                 # random seed

# SPH 수치 파라미터
sph:
  h: 1.0          # smoothing length (m)
  particle_spacing_factor: 1.0
  dt: 0.003       # time step (s)
  v_max: 10.0     # velocity ceiling (m/s)

# 물성 파라미터
material:
  rho0: 2000.0    # reference density (kg/m³)
  c0: 100.0       # speed of sound (m/s)
  gamma: 7.0      # EOS exponent

# 유변학 (Rheology)
rheology:
  mu: 100.0       # dynamic viscosity (Pa·s)
  tau_y: 500.0    # yield stress (Pa)
  mu_b: 0.1       # basal friction coefficient
  xi: 200.0       # turbulent coefficient (m/s²)
  alpha: 0.1      # artificial viscosity
  beta: 0.1       # artificial viscosity

# 침식/퇴적 (Takahashi 2007)
entrainment:
  enabled: true
  delta_e: 0.0007
  delta_d: 0.01
  rho_s: 2650.0
  rho_w: 1000.0
  phi_bed: 35.0
  C_init: 0.4
  C_max: 0.65

# 시뮬레이션 실행
simulation:
  duration: 120.0      # total time (s)
  save_interval: 0.2   # output interval (s)

# 출력 설정
output:
  results_file: "simulation_results.npz"
  log_file: "simulation_log.txt"
  animation_file: "landslide_animation.gif"
```

### 필요 파일

| 파일 | 설명 | 생성 방법 |
|------|------|----------|
| `*_terrain_crop.npy` | 지형 고도 데이터 | `terrain_processor.py` |
| `*_metadata.json` | 좌표계/범위 메타데이터 | `terrain_processor.py` |

## 출력

### 결과 파일 (`simulation_results.npz`)

```python
import numpy as np
data = np.load('simulation_results.npz')

# 시뮬레이션 히스토리
data['times']           # 시간 배열 (n_frames,)
data['n_active']        # 활성 파티클 수 (n_frames,)
data['x']               # X 좌표 (n_frames, max_particles)
data['y']               # Y 좌표 (n_frames, max_particles)
data['vx']              # X 속도 (n_frames, max_particles)
data['vy']              # Y 속도 (n_frames, max_particles)
data['height']          # 유동 깊이 (n_frames, max_particles)
data['density']         # 밀도 (n_frames, max_particles)
data['pressure']        # 압력 (n_frames, max_particles)
data['concentration']   # 토사 농도 (n_frames, max_particles)

# 지형
data['terrain']         # 지형 고도 배열

# 메타데이터
data['cell_size']       # 그리드 셀 크기
data['x_min']           # X 원점 (CRS)
data['y_min']           # Y 원점 (CRS)
data['init_x']          # 초기 중심 X (CRS)
data['init_y']          # 초기 중심 Y (CRS)
data['duration']        # 시뮬레이션 시간
data['save_interval']   # 저장 간격

# SPH 파라미터
data['h']               # smoothing length
data['rho0']            # 기준 밀도
data['c0']              # 음속
data['gamma']           # EOS 지수

# 침식 파라미터
data['C_init']          # 초기 농도
data['rho_s']           # 고체 밀도
data['rho_w']           # 물 밀도
data['phi_bed']         # 바닥재 마찰각
```

### 로그 파일 (`simulation_log.txt`)

시뮬레이션 파라미터 및 진행 상황 기록

## 주요 함수

### `load_config(yaml_path) -> SimulationConfig`

YAML 설정 파일 로드

### `load_terrain(config) -> (np.ndarray, dict)`

지형 데이터 및 메타데이터 로드

### `create_arbitrary_blob(cx, cy, base_radius, n_lobes, particle_spacing, seed)`

불규칙 blob 형상 파티클 생성

### `create_circular_blob(cx, cy, radius, particle_spacing)`

원형 파티클 생성

### `run_simulation(config_path)`

메인 시뮬레이션 실행

## 실행 흐름

1. 설정 파일 로드 (`simulation_config.yaml`)
2. 지형 데이터 로드 (`.npy`, `.json`)
3. SPH 시뮬레이터 초기화
4. 초기 파티클 생성 (circular/arbitrary_blob)
5. 시뮬레이션 실행
6. 결과 저장 (`.npz`)

## CFL 조건 확인

시뮬레이션 전 반드시 CFL 조건 확인:

```
CFL = (v_max + c0) × dt / h < 0.5
```

예: `h=1.0`, `dt=0.003`, `c0=100`, `v_max=10` → CFL = 0.33 (안전)

## 의존성

- `numpy`
- `pyyaml`
- `landslide_sph_gpu` (SPH 시뮬레이터)

## 예시

```bash
# 1. 설정 파일 수정
vim ./guryoung_dem_10m/simulation_config.yaml

# 2. 시뮬레이션 실행
python run_simulation.py ./guryoung_dem_10m/simulation_config.yaml

# 3. 결과 확인
python -c "import numpy as np; d=np.load('./guryoung_dem_10m/simulation_results.npz'); print(d.files)"
```
