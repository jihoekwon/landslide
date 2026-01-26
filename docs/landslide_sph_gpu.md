# landslide_sph_gpu.py

GPU 가속 SPH(Smoothed Particle Hydrodynamics) 산사태 시뮬레이터 핵심 모듈

## 개요

CuPy를 사용한 GPU 병렬 연산 기반 SPH 시뮬레이터입니다. 깊이 평균(depth-averaged) SPH 방정식을 사용하여 산사태/토석류 거동을 시뮬레이션합니다.

## 주요 클래스

### `SPHKernelGPU`

SPH 커널 함수 (Cubic Spline Kernel)

```python
kernel = SPHKernelGPU(h=5.0)  # h: smoothing length
W = kernel.W(r)              # 커널 값
gradWx, gradWy = kernel.gradW_scalar(r, dx, dy)  # 커널 기울기
```

### `SPHParticlesGPU`

GPU 메모리에 저장되는 파티클 데이터 구조

| 속성 | 타입 | 설명 |
|------|------|------|
| `x`, `y` | cp.ndarray | 위치 (m) |
| `vx`, `vy` | cp.ndarray | 속도 (m/s) |
| `ax`, `ay` | cp.ndarray | 가속도 (m/s²) |
| `mass` | cp.ndarray | 질량 (kg) |
| `height` | cp.ndarray | 유동 깊이 (m) |
| `density` | cp.ndarray | 밀도 (kg/m³) |
| `pressure` | cp.ndarray | 압력 (Pa) |
| `concentration` | cp.ndarray | 토사 농도 (0~1) |
| `slope` | cp.ndarray | 지형 경사 (rad) |
| `active` | cp.ndarray | 활성 상태 (bool) |

### `SPHSimulatorGPU`

메인 시뮬레이터 클래스

## 생성자 파라미터

```python
sim = SPHSimulatorGPU(
    terrain,              # 2D numpy array, 고도 데이터
    cell_size=10.0,       # 그리드 셀 크기 (m)

    # SPH 파라미터
    h=5.0,                # smoothing length (m)
    dt=0.004,             # time step (s)
    v_max=30.0,           # 최대 속도 제한 (m/s)

    # 물성 파라미터
    rho0=2000.0,          # 기준 밀도 (kg/m³)
    c0=300.0,             # 음속 (m/s), ~10*v_max 권장
    gamma=7.0,            # 상태방정식 지수
    g=9.81,               # 중력가속도 (m/s²)

    # 유변학 (Rheology)
    mu=100.0,             # 동점성계수 (Pa·s)
    tau_y=500.0,          # 항복응력 (Pa)
    mu_b=0.1,             # 바닥 마찰계수
    xi=200.0,             # 난류계수 (m/s²)
    alpha=0.1,            # 인공점성 α
    beta=0.1,             # 인공점성 β

    # 침식 모델 (Takahashi 2007)
    entrainment_enabled=True,
    delta_e=0.0007,       # 침식계수
    delta_d=0.01,         # 퇴적계수
    rho_s=2650.0,         # 고체입자 밀도 (kg/m³)
    rho_w=1000.0,         # 물 밀도 (kg/m³)
    phi_bed=35.0,         # 바닥재 마찰각 (degrees)
    C_init=0.4,           # 초기 토사 농도
    C_max=0.65,           # 최대 토사 농도
)
```

## 주요 메서드

### `initialize_particles(center, radius, thickness, spacing=None)`

원형 영역에 파티클 초기화

```python
sim.initialize_particles(
    center=(500, 600),    # 중심 좌표 (로컬)
    radius=50,            # 반경 (m)
    thickness=8.0,        # 초기 깊이 (m)
    spacing=2.5           # 파티클 간격 (기본값: h/2)
)
```

### `initialize_particles_from_coords(x_coords, y_coords, x_min, y_min, thickness=5.0)`

좌표 배열로부터 파티클 초기화 (외부 형상 사용 시)

```python
sim.initialize_particles_from_coords(
    x_coords,             # X 좌표 배열 (CRS 좌표)
    y_coords,             # Y 좌표 배열 (CRS 좌표)
    x_min,                # 지형 그리드 X 원점
    y_min,                # 지형 그리드 Y 원점
    thickness=2.0         # 초기 깊이 (m)
)
```

### `step()`

1 타임스텝 진행

```python
avg_speed = sim.step()  # 평균 속도 반환
```

### `run(duration, save_interval, log_file)`

전체 시뮬레이션 실행

```python
n_steps, elapsed = sim.run(
    duration=60.0,                    # 총 시뮬레이션 시간 (s)
    save_interval=0.2,                # 저장 간격 (s)
    log_file='simulation_log.txt'     # 로그 파일
)
```

## 물리 모델

### 깊이 평균 SPH 방정식

- **연속 방정식**: `dh/dt + h·div(v) = 0`
- **운동량 방정식**: `dv/dt = -g·grad(h) + F_friction + F_viscosity`

### Voellmy 마찰 모델

```
τ_b = ρgh(μ_b + v²/ξ)
```

### Bingham 유체 모델

저속 영역에서 항복응력 적용

### Takahashi (2007) 침식 모델

```
평형 농도: C_eq = ρ_w·tan(θ) / [(ρ_s - ρ_w)·(tan(φ) - tan(θ))]
침식율: i_e = δ_e · C_eq · v  (C < C_eq)
퇴적율: i_d = δ_d · C · v · (1 - tan(θ)/tan(φ))  (C > C_eq)
```

## 출력 데이터 (`sim.history`)

```python
# 각 저장 시점의 상태
state = {
    'time': float,           # 시뮬레이션 시간
    'x': np.ndarray,         # 파티클 X 좌표
    'y': np.ndarray,         # 파티클 Y 좌표
    'vx': np.ndarray,        # X 속도
    'vy': np.ndarray,        # Y 속도
    'height': np.ndarray,    # 유동 깊이
    'density': np.ndarray,   # 밀도
    'pressure': np.ndarray,  # 압력
    'concentration': np.ndarray,  # 토사 농도
    'n_active': int,         # 활성 파티클 수
    'n_total': int,          # 총 파티클 수
}
```

## CFL 안정성 조건

```
CFL = (v_max + c0) × dt / h < 0.3~0.5

점성 안정성: dt < h² / (2ν)
음속 조건: c0 ≥ 10 × v_max
```

## 의존성

- `numpy`
- `cupy` (CUDA 필요)

## 사용 예시

```python
import numpy as np
from landslide_sph_gpu import SPHSimulatorGPU

# 지형 데이터 로드
terrain = np.load('terrain.npy')

# 시뮬레이터 생성
sim = SPHSimulatorGPU(terrain, cell_size=10.0, h=5.0, dt=0.005)

# 파티클 초기화
sim.initialize_particles(center=(300, 400), radius=50, thickness=5.0)

# 시뮬레이션 실행
sim.run(duration=30.0, save_interval=0.5)

# 결과 접근
for state in sim.history:
    print(f"t={state['time']:.1f}s, particles={state['n_active']}")
```

## 독립 실행

```bash
python landslide_sph_gpu.py
```

테스트용 지형을 생성하고 1초간 시뮬레이션을 수행합니다.
