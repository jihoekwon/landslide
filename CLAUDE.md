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

## Data Sources (외부 데이터 파이프라인)

### 1. DEM 지형 데이터
- **소스**: Copernicus COP30 GLO-30 (OpenTopography API)
- **해상도**: ~30m 전역, terrain_processor에서 리샘플링 가능
- **API**: `https://portal.opentopography.org/API/globaldem`
- **인증**: `OPENTOPO_API_KEY` 환경변수 또는 `report_config.yaml` / `simulation_config.yaml`에 설정
- **대안**: 사용자 제공 GeoTIFF (예: 국토지리정보원 5m DEM)
- **좌표계**: EPSG:5186 (Korea 2000 / Central Belt)로 변환하여 사용

### 2. 위성 이미지
- **소스**: ArcGIS World Imagery REST API
- **URL**: `https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export`
- **최대 해상도**: 4096x4096 픽셀
- **좌표 변환**: EPSG:5186 → EPSG:3857 (Web Mercator) via pyproj
- **인증**: 불필요 (공개 API)

### 3. 타겟 건물/영역 좌표
- **현재 방식**: 수동 추정값. 정확한 지오코딩 파이프라인 미구축 상태
- **자동 지오코딩 미지원** (향후 Kakao/Naver Maps Geocoding API 연동 필요)

#### 현재 좌표 입력 방법 (정확도 낮음)
- **건물**: 시뮬레이션 영상에서 유동 경로 확인 후 좌표 추정, 또는 지도에서 수동 확인
- **영역**: 지형 분석(표고/경사) 기반 평탄지 추정 → bbox 설정
- **주의**: 현재 설정된 좌표는 추정값이며, 정확한 건물/행정구역 위치와 차이가 있을 수 있음

#### 정확한 좌표 확인 방법 (권장)
1. **주소 → 좌표 변환 (지오코딩)**:
   - Kakao Maps API: `https://dapi.kakao.com/v2/local/search/address`
   - Naver Maps API: `https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode`
   - 결과: WGS84 (위도, 경도)
2. **WGS84 → EPSG:5186 변환**:
   ```python
   from pyproj import Transformer
   t = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
   x, y = t.transform(lon, lat)  # (경도, 위도) → (x, y)
   ```
3. **영역 경계 (행정구역)**:
   - 국가공간정보포털 (data.nsdi.go.kr): 행정구역 경계 SHP
   - 브이월드 (vworld.kr): 건물/필지 WFS/WMS
   - SGIS (sgis.kostat.go.kr): 통계지리정보 경계

### 4. LLM 분석
- **소스**: Anthropic Claude API
- **모델**: `claude-sonnet-4-20250514` (기본) 또는 `claude-opus-4-20250514`
- **인증**: `report_config.yaml`의 `llm.api_key`

## Config Files

### simulation_config.yaml
- `terrain`: DEM 파일 경로
- `initial_condition`: 붕괴 위치 (center), 반경 (radius), 두께 (thickness)
- `sph`: smoothing length (h), dt, v_max
- `rheology`: mu_b, xi, tau_y, mu
- `entrainment`: delta_e, delta_d, phi_bed, C_init
- `simulation`: duration, save_interval
- `targets`: 피해 대상 건물/영역 정보 (아래 참조)

### simulation_config.yaml - targets 섹션

```yaml
# 건물 타겟 (점 + 반경)
targets:
  - name: "서울대학교 제2공학관(302동)"
    target_type: "building"          # 기본값, 생략 가능
    coordinates: [195800, 538700]    # EPSG:5186
    address: "서울특별시 관악구 관악로 1 서울대학교 302동"
    type: "RC"                       # RC, wood, masonry
    proximity_radius: 30             # 영향권 반경 (m)

  # 영역 타겟 (bounding box)
  - name: "구룡마을"
    target_type: "area"
    address: "서울 강남구 양재대로 478 구룡마을 8지구"
    structure_type: "mixed"          # mixed, RC, wood, masonry
    bbox: [205700, 541900, 206100, 542200]  # [x_min, y_min, x_max, y_max] EPSG:5186
```

### visualization_config.yaml
- `view_elev`, `view_azim`: 카메라 각도 (기본: 70, 0)
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
