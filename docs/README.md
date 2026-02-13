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
   ├── 입력: GeoTIFF (.tif) 또는 OpenTopography COP30 자동 다운로드
   ├── 외부 API: OpenTopography (DEM), ArcGIS World Imagery (위성)
   └── 출력: terrain_crop.npy, metadata.json, satellite_crop.png

2. 타겟 설정 (수동)
   simulation_config.yaml → targets 섹션
   ├── 건물(building): 주소 → 좌표(EPSG:5186) 수동 변환 후 입력
   ├── 영역(area): 지도에서 bbox 범위 추정 후 입력
   └── 좌표 변환: WGS84(위경도) → EPSG:5186 (pyproj 사용)

3. 시뮬레이션 실행
   run_simulation.py
   ├── 입력: simulation_config.yaml, terrain_crop.npy
   └── 출력: simulation_results.npz, simulation_log.txt

4. 결과 시각화
   visualize_results.py
   ├── 입력: simulation_results.npz, satellite_crop.png
   └── 출력: landslide_animation_{좌표}.gif/.webm

5. 보고서 생성
   generate_report.py
   ├── 입력: simulation_results.npz, simulation_config.yaml (targets 포함)
   ├── 외부 API: Anthropic Claude (LLM 분석)
   ├── 분석: 거리별 속도 + 건물별/영역별 충격 분석 + 도달 가능성 추정
   └── 출력: analysis_report_{좌표}.html/.md
```

## 외부 데이터소스

| 데이터 | 소스 | API/URL | 인증 |
|--------|------|---------|------|
| **DEM 지형** | Copernicus COP30 | OpenTopography API | `OPENTOPO_API_KEY` |
| **위성 이미지** | ArcGIS World Imagery | REST API (4096x4096 max) | 불필요 |
| **타겟 좌표** | 수동 입력 | 네이버/카카오 지도 → pyproj 변환 | - |
| **LLM 분석** | Anthropic Claude | `report_config.yaml` | API 키 필요 |

### 타겟 좌표 입력 방법

**현재 상태: 자동 지오코딩 파이프라인 미구축. 좌표는 수동 추정값.**

현재 `simulation_config.yaml`의 `targets` 좌표는 다음과 같은 방법으로 설정됨:
- **건물**: 시뮬레이션 영상에서 유동 경로와 위성사진을 비교하여 좌표 추정
- **영역**: 지형 분석(표고/경사 기반)으로 평탄지 범위를 bbox로 추정
- **주의**: 정확한 지오코딩이 아닌 추정값이므로, 실제 건물/행정구역 위치와 차이 있을 수 있음

#### 정확한 좌표 확보 방법 (권장, 미구현)

1. **주소 → 좌표 (지오코딩 API)**:
   - Kakao Maps Geocoding: `https://dapi.kakao.com/v2/local/search/address`
   - Naver Maps Geocoding: `https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode`
   - 결과: WGS84 (위도, 경도)

2. **WGS84 → EPSG:5186 좌표 변환**:
   ```python
   from pyproj import Transformer
   t = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
   x, y = t.transform(longitude, latitude)
   ```

3. **영역 경계 데이터 (행정구역/건물)**:
   - 국가공간정보포털 (data.nsdi.go.kr): 행정구역 경계 SHP
   - 브이월드 (vworld.kr): 건물/필지 WFS/WMS
   - SGIS (sgis.kostat.go.kr): 통계지리정보 경계

4. `simulation_config.yaml`의 `targets` 섹션에 입력:
   - 건물: `target_type: "building"`, `coordinates: [x, y]`, `proximity_radius: 30`
   - 영역: `target_type: "area"`, `bbox: [x_min, y_min, x_max, y_max]`

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

### 지배 방정식 (Continuum Form)

**질량 보존:**

$$\frac{\partial h}{\partial t} + \nabla \cdot (h \bar{v}) = E - D$$

**운동량 보존:**

$$\frac{\partial (h\bar{v})}{\partial t} + \nabla \cdot (h\bar{v} \otimes \bar{v}) = -gh\nabla(h + z_b) - \frac{\tau_b}{\rho} + S$$

### SPH Form 지배방정식

SPH는 Lagrangian 입자법이므로, 입자를 따라가며 material derivative를 직접 계산합니다. Eulerian 형태의 **convective term** $\nabla \cdot (h\bar{v} \otimes \bar{v})$ 은 입자 advection 자체로 처리되어 별도 계산이 필요하지 않습니다.

#### 연속 방정식 (Continuity)

$$\frac{dh_i}{dt} = \sum_j \Omega_j \, (\mathbf{v}_i - \mathbf{v}_j) \cdot \nabla W_{ij}$$

- $\Omega_j = \frac{m_j}{\rho_0 \, h_j}$: 입자 $j$의 면적 (depth-averaged SPH에서 질량/면밀도)
- $W_{ij} = W(|\mathbf{r}_i - \mathbf{r}_j|, h_{sml})$: cubic spline 커널
- $h_{sml}$: SPH smoothing length (유동 깊이 $h$와 별개)

#### 운동량 방정식 (Momentum)

입자 $i$의 가속도는 다음 항들의 합으로 구성됩니다:

$$\frac{d\mathbf{v}_i}{dt} = \mathbf{a}_i^{grav} + \mathbf{a}_i^{press} + \mathbf{a}_i^{art.visc} + \mathbf{a}_i^{phys.visc} + \mathbf{a}_i^{friction} + \mathbf{a}_i^{yield}$$

**① 중력 (경사면 구동력)**

$$\mathbf{a}_i^{grav} = -g \, \nabla z_b \Big|_i$$

지형 경사 $\nabla z_b$를 DEM에서 유한 차분으로 계산합니다.

**② 압력 구배력 (정수압)**

정수압: $P = \frac{1}{2} \rho_0 g h^2$

$$\mathbf{a}_i^{press} = -g \sum_j \Omega_j \, \frac{h_i + h_j}{2} \, \nabla W_{ij}$$

대칭 평균 $(h_i + h_j)/2$를 사용하여 운동량 보존을 보장합니다.

**③ Artificial Viscosity (Monaghan 1992)**

수치적 충격파 처리 및 안정성을 위한 항입니다. 접근하는 입자 쌍($\mathbf{v}_{ij} \cdot \mathbf{r}_{ij} < 0$)에만 적용합니다:

$$\mu_{ij} = \frac{h_{sml} \, (\mathbf{v}_{ij} \cdot \mathbf{r}_{ij})}{|\mathbf{r}_{ij}|^2 + \eta^2}, \quad \eta^2 = 0.01 \, h_{sml}^2$$

$$\Pi_{ij} = \frac{-\alpha \, c_0 \, \mu_{ij} + \beta \, \mu_{ij}^2}{\bar{\rho}_{ij}}$$

$$\mathbf{a}_i^{art.visc} = -\sum_j m_j \, \Pi_{ij} \, \nabla W_{ij} \quad (\mathbf{v}_{ij} \cdot \mathbf{r}_{ij} < 0 \text{ 인 경우만})$$

- $\mathbf{v}_{ij} = \mathbf{v}_i - \mathbf{v}_j$, $\mathbf{r}_{ij} = \mathbf{r}_i - \mathbf{r}_j$
- $\bar{\rho}_{ij} = \frac{1}{2}(\rho_i + \rho_j)$, 여기서 $\rho = \rho_0 h$ (면밀도)
- $\alpha, \beta$: artificial viscosity 계수 (기본값 0.1)
- $c_0$: 수치적 음속

**④ Physical Viscosity (Morris et al. 1997)**

동적 점성 계수 $\mu$에 의한 내부 전단 저항입니다:

$$\mathbf{a}_i^{phys.visc} = \sum_j m_j \, \frac{2\mu \, |\mathbf{r}_{ij}|}{\rho_i \, \rho_j \, (|\mathbf{r}_{ij}|^2 + \eta^2)} \, \frac{\nabla W_{ij} \cdot \mathbf{r}_{ij}}{|\mathbf{r}_{ij}|} \, \mathbf{v}_{ij}$$

- 방사대칭 커널에서 $\frac{\nabla W_{ij} \cdot \mathbf{r}_{ij}}{|\mathbf{r}_{ij}|} = \frac{\partial W}{\partial r}$
- Artificial viscosity와 달리, 모든 이웃 입자 쌍에 적용 (접근/이격 무관)
- 각 속도 성분($v_x, v_y$)이 독립적으로 확산

**⑤ Basal Friction (Voellmy)**

$$\mathbf{a}_i^{friction} = -g \left( \mu_b + \frac{|\mathbf{v}_i|^2}{\xi} \right) \frac{\mathbf{v}_i}{|\mathbf{v}_i|}$$

- Coulomb 항 ($\mu_b$): 저속에서 지배적, 바닥면과의 마찰
- Turbulent 항 ($v^2/\xi$): 고속에서 지배적, 내부 충돌·난류 소산

**⑥ Yield Stress (Bingham)**

저속 ($|\mathbf{v}| < 1$ m/s) 영역에서만 활성화되는 정지 조건:

$$\mathbf{a}_i^{yield} = -\frac{\tau_y}{\rho_0 \, h_i} \, \frac{\mathbf{v}_i}{|\mathbf{v}_i|} \quad (|\mathbf{v}_i| < 1 \text{ m/s 인 경우만})$$

#### 시간 적분 (Leapfrog)

Half-step leapfrog 방식으로 적분합니다:

$$h_i^{n+1/2} = h_i^n + \frac{\Delta t}{2} \left.\frac{dh}{dt}\right|^n$$

$$\mathbf{v}_i^{n+1/2} = \mathbf{v}_i^n + \frac{\Delta t}{2} \, \mathbf{a}_i^n$$

$$\mathbf{r}_i^{n+1} = \mathbf{r}_i^n + \Delta t \, \mathbf{v}_i^{n+1/2}$$

$$h_i^{n+1} = h_i^{n+1/2} + \frac{\Delta t}{2} \left.\frac{dh}{dt}\right|^{n+1}$$

$$\mathbf{v}_i^{n+1} = \mathbf{v}_i^{n+1/2} + \frac{\Delta t}{2} \, \mathbf{a}_i^{n+1}$$

시간 적분 후, Takahashi (2007) 모델에 의한 침식/퇴적이 $h$와 $C$를 갱신합니다.

#### Continuum ↔ SPH Form 대응 관계

| Continuum (Eulerian) | SPH (Lagrangian) | 비고 |
|---|---|---|
| $\frac{\partial h}{\partial t} + \nabla \cdot (h\mathbf{v})$ | $\frac{dh_i}{dt} = \sum_j \Omega_j (\mathbf{v}_i - \mathbf{v}_j) \cdot \nabla W_{ij}$ | convective term이 입자 이동으로 흡수 |
| $\nabla \cdot (h\mathbf{v} \otimes \mathbf{v})$ | (입자 advection으로 자동 처리) | Lagrangian 좌표계의 장점 |
| $-gh\nabla(h + z_b)$ | $\mathbf{a}^{grav} + \mathbf{a}^{press}$ | 경사 구동력 + 정수압 구배 |
| $-\frac{\tau_b}{\rho}$ | $\mathbf{a}^{friction} + \mathbf{a}^{yield}$ | Voellmy + Bingham |
| $\nu \nabla^2 \mathbf{v}$ | $\mathbf{a}^{phys.visc}$ | Morris (1997) 근사 |
| (수치 안정성) | $\mathbf{a}^{art.visc}$ | Monaghan (1992) |

### Entrainment 모델

- Takahashi (2007) 평형 농도 모델
- 경사 기반 침식/퇴적
- 상세 내용은 메인 [README.md](../README.md) Section 3 참조

## CFL 안정성 조건

**대류 CFL:**

$$\text{CFL} = \frac{(v_{max} + c_0) \cdot \Delta t}{h} < 0.5$$

**점성 CFL:**

$$\Delta t < \frac{h^2}{2\nu}$$

**음속 조건:**

$$c_0 \geq 10 \times v_{max}$$

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
