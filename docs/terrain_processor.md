# terrain_processor.py

DEM 지형 데이터 전처리 모듈

## 개요

GeoTIFF/DEM 파일을 로드하고, 영역 크롭, 리샘플링, 위성 이미지 다운로드, 3D 미리보기 등의 전처리 기능을 제공합니다.

## 사용법

### Python API

```python
from terrain_processor import TerrainProcessor

# DEM 파일 로드
tp = TerrainProcessor("./data/my_dem.tif")

# 영역 크롭
tp.crop(center=(207000, 542000), size=(2000, 2000))

# 위성 이미지 다운로드
tp.fetch_satellite_texture()

# 미리보기
tp.preview_3d_crop()
tp.plot_bounds_comparison()

# 저장
tp.save()
tp.info()
```

### CLI

```bash
python terrain_processor.py <dem_path> [options]

# 예시
python terrain_processor.py ./data/guryong.tif \
    --crop-center 207000 542000 \
    --crop-size 2000 2000 \
    --satellite \
    --preview
```

## CLI 옵션

| 옵션 | 설명 |
|------|------|
| `--crop-center X Y` | 크롭 중심점 좌표 (CRS) |
| `--crop-size W H` | 크롭 크기 (미터) |
| `--cell-size` | 리샘플링 셀 크기 |
| `--satellite` | 위성 이미지 다운로드 |
| `--preview` | 3D 미리보기 |
| `--no-show` | 화면 표시 없이 저장만 |

## 입력

| 파일 형식 | 설명 |
|-----------|------|
| `.tif`, `.tiff` | GeoTIFF |
| `.dem` | USGS DEM |

## 출력

### 파일

| 파일 | 설명 |
|------|------|
| `{stem}_terrain_crop.npy` | 크롭된 지형 배열 |
| `{stem}_metadata.json` | 메타데이터 |
| `{stem}_satellite_original.png` | 원본 영역 위성 이미지 |
| `{stem}_satellite_crop.png` | 크롭 영역 위성 이미지 |
| `{stem}_preview_original.png` | 원본 3D 미리보기 |
| `{stem}_preview_crop.png` | 크롭 3D 미리보기 |
| `{stem}_bounds_comparison.png` | 범위 비교 이미지 |

### 메타데이터 (`*_metadata.json`)

```json
{
  "source_file": "/path/to/dem.tif",
  "processed_file": "/path/to/terrain_crop.npy",
  "crs": "EPSG:5186",
  "bounds": [206000, 540000, 210000, 544000],
  "crop_bounds": [206200, 540800, 208200, 542800],
  "cell_size": 10.0,
  "rows": 200,
  "cols": 200,
  "elevation_min": 120.5,
  "elevation_max": 385.2,
  "satellite_file": "/path/to/satellite_crop.png"
}
```

## 클래스

### `TerrainMetadata`

지형 메타데이터 데이터클래스

```python
@dataclass
class TerrainMetadata:
    source_file: str
    processed_file: str
    crs: str
    bounds: Tuple[float, float, float, float]
    crop_bounds: Optional[Tuple[float, float, float, float]]
    cell_size: float
    rows: int
    cols: int
    elevation_min: float
    elevation_max: float
    satellite_file: Optional[str]
```

### `TerrainProcessor`

메인 전처리 클래스

## 주요 메서드

### `load_dem(dem_path) -> TerrainProcessor`

DEM 파일 로드

### `crop(bounds=None, center=None, size=(2000, 2000)) -> TerrainProcessor`

영역 크롭

- `bounds`: (x_min, y_min, x_max, y_max) 직접 지정
- `center`: (x, y) 중심점 + size

### `resample(target_cell_size) -> TerrainProcessor`

셀 크기 변경 (scipy.ndimage.zoom)

### `fetch_satellite_texture(for_original=True, for_crop=True) -> TerrainProcessor`

ArcGIS World Imagery에서 위성 이미지 다운로드

### `save() -> TerrainProcessor`

처리된 데이터 저장 (`.npy`, `.json`)

### `preview_3d_original()`, `preview_3d_crop()`

3D 지형 미리보기 (matplotlib)

### `plot_bounds_comparison()`

원본/크롭 범위 비교 2D 플롯

### `info() -> str`

현재 상태 정보 출력

## 좌표계

- 입력: GeoTIFF CRS (자동 감지, 기본 EPSG:5186)
- 내부: 로컬 그리드 좌표 (0,0 기준)
- 위성 API: WGS84 (EPSG:4326)로 자동 변환

## 의존성

- `numpy`
- `rasterio` (GeoTIFF 로드)
- `scipy` (리샘플링)
- `pyproj` (좌표 변환)
- `matplotlib` (시각화)
- `PIL` (이미지 처리)
- `requests` (위성 이미지 다운로드)

## 예시 워크플로우

```python
from terrain_processor import TerrainProcessor

# 1. DEM 로드
tp = TerrainProcessor("./guryong_dem_10m/guryong.tif")

# 2. 관심 영역 확인
tp.info()

# 3. 크롭 (2km x 2km)
tp.crop(center=(207200, 541800), size=(2000, 2000))

# 4. 위성 이미지 다운로드
tp.fetch_satellite_texture()

# 5. 미리보기로 확인
tp.preview_3d_crop()
tp.plot_bounds_comparison()

# 6. 저장
tp.save()

# 출력 파일:
# - guryong_dem_10m_terrain_crop.npy
# - guryong_dem_10m_metadata.json
# - guryong_dem_10m_satellite_crop.png
```
