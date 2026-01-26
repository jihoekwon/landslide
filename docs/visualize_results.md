# visualize_results.py

PyVista/VTK 기반 시뮬레이션 결과 3D 시각화 모듈

## 개요

시뮬레이션 결과 파일(`.npz`)을 읽어 3D 애니메이션(GIF, WebM)을 생성합니다. 위성 텍스처를 지형에 매핑하고 파티클을 속도 기반 컬러로 렌더링합니다.

## 사용법

```bash
python visualize_results.py <results_npz_path> [options]

# 예시
python visualize_results.py ./guryoung_dem_10m/simulation_results.npz
python visualize_results.py ./guryoung_dem_10m/simulation_results.npz --workers 8
python visualize_results.py ./guryoung_dem_10m/simulation_results.npz --skip 2 --sequential
```

## 명령줄 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--workers`, `-w` | 4 | 병렬 렌더링 워커 수 |
| `--skip`, `-s` | config | 프레임 스킵 (n프레임당 1개 렌더링) |
| `--sequential` | False | 순차 처리 모드 (디버깅용) |

## 입력

### 필수 파일

| 파일 | 설명 |
|------|------|
| `simulation_results.npz` | 시뮬레이션 결과 데이터 |

### 선택적 파일

| 파일 | 설명 |
|------|------|
| `*_satellite_crop.png` | 위성 텍스처 이미지 |
| `visualization_config.yaml` | 시각화 설정 |

### 설정 파일 (`visualization_config.yaml`)

```yaml
# 제목
title_prefix: "구룡마을 비탈면홍수해"

# 카메라 설정
view_elev: 35.0           # 앙각 (degrees)
view_azim: 30.0           # 방위각 (degrees)
camera_distance_factor: 2.5

# 출력 설정
fps: 15                   # 애니메이션 FPS
frame_skip: 4             # 프레임 스킵

# 렌더링 설정
window_width: 1200
window_height: 900
particle_size: 3
particle_cmap: "plasma"   # 파티클 컬러맵
particle_height_offset: 30  # 지형 위 오프셋 (m)

# 지형 설정
use_satellite: true
terrain_cmap: "terrain"   # 위성 없을 때 컬러맵

# 출력 파일명
animation_gif: "landslide_animation.gif"
animation_video: "landslide_animation.webm"
```

## 출력

| 파일 | 설명 |
|------|------|
| `landslide_animation.gif` | GIF 애니메이션 |
| `landslide_animation.webm` | WebM 비디오 (VP9 코덱) |

## 주요 함수

### `render_frame(data_path, frame_idx, frame_data_idx, output_path, render_config)`

단일 프레임 렌더링 (PyVista)

### `create_animation(data_path, output_dir, vis_config, n_workers, frame_skip, parallel)`

애니메이션 생성 메인 함수

### `load_visualization_config(config_path) -> dict`

시각화 설정 로드

## 렌더링 파이프라인

1. 지형 데이터 → PyVista StructuredGrid
2. 위성 이미지 → 텍스처 매핑
3. 파티클 위치 → PolyData (점)
4. 속도 → 컬러 스칼라
5. 프레임 저장 → PNG
6. 프레임 병합 → GIF/WebM

## 좌표계 변환

DEM과 위성 이미지의 좌표축 정렬:
- DEM: Y축 반전 (북쪽이 array[0])
- 텍스처: UV 좌표 반전
- 파티클: 로컬 좌표 → 그리드 좌표

## 병렬 처리

- 기본: subprocess 기반 병렬 렌더링
- 각 프레임을 독립 프로세스에서 렌더링
- VTK/PyVista 메모리 누수 방지

## 의존성

- `numpy`
- `pyvista`
- `PIL` (Pillow)
- `imageio`
- `imageio-ffmpeg` (WebM 인코딩)
- `pyyaml`

## 예시

```python
from visualize_results import create_animation
from pathlib import Path

create_animation(
    data_path='./simulation_results.npz',
    output_dir=Path('./output'),
    vis_config={'fps': 20, 'frame_skip': 2},
    n_workers=4,
    parallel=True
)
```

## 트러블슈팅

### WebM 생성 실패

```bash
# imageio-ffmpeg 설치 확인
pip install imageio-ffmpeg
```

### 메모리 부족

```bash
# 워커 수 줄이기
python visualize_results.py results.npz --workers 2

# 또는 순차 처리
python visualize_results.py results.npz --sequential
```

### 프레임 수 줄이기

```bash
python visualize_results.py results.npz --skip 4
```
