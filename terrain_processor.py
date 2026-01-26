"""
DEM Terrain Processor Module
============================

DEM 파일(.tif, .dem)을 로딩하고 전처리하는 모듈.
DEM 파일이 있는 디렉토리를 작업 디렉토리로 사용.

Usage:
    from terrain_processor import TerrainProcessor

    tp = TerrainProcessor("./data/my_dem.tif")
    tp.crop(center=(x, y), size=(1500, 1500))
    tp.fetch_satellite_texture()
    tp.preview_3d()
    tp.plot_bounds_comparison()
"""

import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, List, Union
import warnings


@dataclass
class TerrainMetadata:
    """지형 메타데이터"""
    source_file: str = ""
    processed_file: str = ""
    crs: str = ""
    bounds: Tuple[float, float, float, float] = (0, 0, 0, 0)
    crop_bounds: Optional[Tuple[float, float, float, float]] = None
    cell_size: float = 0.0
    rows: int = 0
    cols: int = 0
    elevation_min: float = 0.0
    elevation_max: float = 0.0
    satellite_file: Optional[str] = None

    def save(self, filepath: str):
        data = asdict(self)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'TerrainMetadata':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data.get('bounds'):
            data['bounds'] = tuple(data['bounds'])
        if data.get('crop_bounds'):
            data['crop_bounds'] = tuple(data['crop_bounds'])
        return cls(**data)


class TerrainProcessor:
    """
    DEM 지형 처리기

    DEM 파일이 있는 디렉토리를 작업 디렉토리로 사용.
    """

    def __init__(self, dem_path: Optional[str] = None):
        """
        Args:
            dem_path: DEM 파일 경로. 해당 파일의 디렉토리가 작업 디렉토리가 됨.
        """
        self.terrain: Optional[np.ndarray] = None
        self.terrain_original: Optional[np.ndarray] = None
        self.metadata = TerrainMetadata()
        self.metadata_original: Optional[TerrainMetadata] = None

        self.work_dir: Optional[Path] = None
        self.dem_stem: str = ""

        if dem_path:
            self.load_dem(dem_path)

    def load_dem(self, dem_path: str) -> 'TerrainProcessor':
        """DEM 파일 로드 (.tif, .dem)"""
        import rasterio

        dem_path = Path(dem_path).resolve()
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM 파일을 찾을 수 없음: {dem_path}")

        self.work_dir = dem_path.parent
        self.dem_stem = dem_path.stem

        print(f"[INFO] DEM 로딩: {dem_path}")
        print(f"[INFO] 작업 디렉토리: {self.work_dir}")

        with rasterio.open(dem_path) as src:
            self.terrain = src.read(1).astype(np.float32)
            transform = src.transform

            self.metadata.source_file = str(dem_path)
            self.metadata.crs = str(src.crs) if src.crs else "EPSG:5186"
            self.metadata.cell_size = abs(transform[0])
            self.metadata.rows, self.metadata.cols = self.terrain.shape

            x_min = transform[2]
            y_max = transform[5]
            x_max = x_min + self.metadata.cols * self.metadata.cell_size
            y_min = y_max - self.metadata.rows * self.metadata.cell_size
            self.metadata.bounds = (x_min, y_min, x_max, y_max)

            nodata = src.nodata
            if nodata is not None:
                self.terrain = np.where(self.terrain != nodata, self.terrain, np.nan)

            self.metadata.elevation_min = float(np.nanmin(self.terrain))
            self.metadata.elevation_max = float(np.nanmax(self.terrain))

        # 원본 보존
        self.terrain_original = self.terrain.copy()
        self.metadata_original = TerrainMetadata(
            source_file=self.metadata.source_file,
            crs=self.metadata.crs,
            bounds=self.metadata.bounds,
            cell_size=self.metadata.cell_size,
            rows=self.metadata.rows,
            cols=self.metadata.cols,
            elevation_min=self.metadata.elevation_min,
            elevation_max=self.metadata.elevation_max
        )

        print(f"  - 크기: {self.metadata.cols} x {self.metadata.rows}")
        print(f"  - 셀 크기: {self.metadata.cell_size} m")
        print(f"  - 좌표계: {self.metadata.crs}")
        print(f"  - 고도: {self.metadata.elevation_min:.1f} ~ {self.metadata.elevation_max:.1f} m")
        print(f"  - X 범위: {x_min:.0f} ~ {x_max:.0f}")
        print(f"  - Y 범위: {y_min:.0f} ~ {y_max:.0f}")

        return self

    def crop(self,
             bounds: Optional[Tuple[float, float, float, float]] = None,
             center: Optional[Tuple[float, float]] = None,
             size: Tuple[float, float] = (2000, 2000)) -> 'TerrainProcessor':
        """
        지형 데이터 crop

        Args:
            bounds: (x_min, y_min, x_max, y_max) 직접 지정
            center: (x, y) 중심점 좌표
            size: (width, height) 미터 단위, 기본값 2000x2000m
        """
        if self.terrain is None:
            raise ValueError("DEM이 로드되지 않음")

        orig_bounds = self.metadata_original.bounds
        cell = self.metadata.cell_size

        if bounds is not None:
            x_min, y_min, x_max, y_max = bounds
        elif center is not None:
            cx, cy = center
            w, h = size
            x_min, x_max = cx - w/2, cx + w/2
            y_min, y_max = cy - h/2, cy + h/2
        else:
            raise ValueError("bounds 또는 center를 지정하세요")

        # 클리핑
        x_min = max(x_min, orig_bounds[0])
        y_min = max(y_min, orig_bounds[1])
        x_max = min(x_max, orig_bounds[2])
        y_max = min(y_max, orig_bounds[3])

        # 인덱스 계산
        col_start = int((x_min - orig_bounds[0]) / cell)
        col_end = int((x_max - orig_bounds[0]) / cell)
        row_start = int((orig_bounds[3] - y_max) / cell)
        row_end = int((orig_bounds[3] - y_min) / cell)

        self.terrain = self.terrain_original[row_start:row_end, col_start:col_end].copy()

        self.metadata.crop_bounds = (x_min, y_min, x_max, y_max)
        self.metadata.rows, self.metadata.cols = self.terrain.shape
        self.metadata.elevation_min = float(np.nanmin(self.terrain))
        self.metadata.elevation_max = float(np.nanmax(self.terrain))

        print(f"\n[INFO] Crop 완료:")
        print(f"  - 새 크기: {self.metadata.cols} x {self.metadata.rows}")
        print(f"  - Crop X 범위: {x_min:.0f} ~ {x_max:.0f}")
        print(f"  - Crop Y 범위: {y_min:.0f} ~ {y_max:.0f}")
        print(f"  - 고도: {self.metadata.elevation_min:.1f} ~ {self.metadata.elevation_max:.1f} m")

        return self

    def resample(self, target_cell_size: float) -> 'TerrainProcessor':
        """지형 데이터 리샘플링"""
        from scipy.ndimage import zoom

        if self.terrain is None:
            raise ValueError("DEM이 로드되지 않음")

        scale = self.metadata.cell_size / target_cell_size
        self.terrain = zoom(self.terrain, scale, order=1).astype(np.float32)
        self.metadata.cell_size = target_cell_size
        self.metadata.rows, self.metadata.cols = self.terrain.shape

        print(f"[INFO] 리샘플링: {self.metadata.cols}x{self.metadata.rows}, cell={target_cell_size}m")
        return self

    def _get_satellite_image(self, bounds: Tuple[float, float, float, float],
                             width: int, height: int) -> Optional[np.ndarray]:
        """위성 이미지 다운로드 (내부 함수)"""
        try:
            import requests
            from PIL import Image
            from io import BytesIO
            from pyproj import Transformer
        except ImportError as e:
            print(f"[WARN] 필요한 패키지 없음: {e}")
            return None

        x_min, y_min, x_max, y_max = bounds
        src_crs = self.metadata.crs if "EPSG" in self.metadata.crs else "EPSG:5186"

        try:
            transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
            lon_min, lat_min = transformer.transform(x_min, y_min)
            lon_max, lat_max = transformer.transform(x_max, y_max)
        except Exception as e:
            print(f"[WARN] 좌표 변환 실패: {e}")
            return None

        # ArcGIS World Imagery
        url = (
            f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
            f"MapServer/export?"
            f"bbox={lon_min},{lat_min},{lon_max},{lat_max}"
            f"&bboxSR=4326&imageSR=4326"
            f"&size={width},{height}"
            f"&format=png&f=image"
        )

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')
            return np.array(img)
        except Exception as e:
            print(f"[WARN] 위성 이미지 다운로드 실패: {e}")
            return None

    def fetch_satellite_texture(self, for_original: bool = True, for_crop: bool = True) -> 'TerrainProcessor':
        """위성 텍스처 다운로드"""
        from PIL import Image

        # 원본용
        if for_original and self.metadata_original:
            print("[INFO] 원본 DEM 위성 이미지 다운로드...")
            orig_bounds = self.metadata_original.bounds
            img_arr = self._get_satellite_image(
                orig_bounds,
                min(1280, self.metadata_original.cols * 2),
                min(1280, self.metadata_original.rows * 2)
            )
            if img_arr is not None:
                save_path = self.work_dir / f"{self.dem_stem}_satellite_original.png"
                # 그리드 크기에 맞게 리사이즈
                img = Image.fromarray(img_arr).resize(
                    (self.metadata_original.cols, self.metadata_original.rows),
                    Image.LANCZOS
                )
                img.save(str(save_path))
                self.metadata_original.satellite_file = str(save_path)
                print(f"  저장: {save_path}")

        # Crop용
        if for_crop and self.metadata.crop_bounds:
            print("[INFO] Crop 영역 위성 이미지 다운로드...")
            crop_bounds = self.metadata.crop_bounds
            img_arr = self._get_satellite_image(
                crop_bounds,
                min(1280, self.metadata.cols * 4),
                min(1280, self.metadata.rows * 4)
            )
            if img_arr is not None:
                save_path = self.work_dir / f"{self.dem_stem}_satellite_crop.png"
                img = Image.fromarray(img_arr).resize(
                    (self.metadata.cols, self.metadata.rows),
                    Image.LANCZOS
                )
                img.save(str(save_path))
                self.metadata.satellite_file = str(save_path)
                print(f"  저장: {save_path}")

        return self

    def _fill_nan(self, data: np.ndarray) -> np.ndarray:
        """NaN 값을 주변 값으로 채우기"""
        from scipy.ndimage import distance_transform_edt

        mask = np.isnan(data)
        if not mask.any():
            return data
        indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
        return data[tuple(indices)].astype(np.float32)

    def save(self) -> 'TerrainProcessor':
        """처리된 지형 데이터 저장"""
        if self.terrain is None:
            raise ValueError("저장할 데이터 없음")

        terrain_clean = self._fill_nan(self.terrain)

        # Crop 데이터 저장
        if self.metadata.crop_bounds:
            save_path = self.work_dir / f"{self.dem_stem}_terrain_crop.npy"
        else:
            save_path = self.work_dir / f"{self.dem_stem}_terrain.npy"

        np.save(str(save_path), terrain_clean)
        self.metadata.processed_file = str(save_path)

        # 메타데이터 저장
        meta_path = self.work_dir / f"{self.dem_stem}_metadata.json"
        self.metadata.save(str(meta_path))

        print(f"[INFO] 저장 완료:")
        print(f"  - 지형: {save_path}")
        print(f"  - 메타: {meta_path}")

        return self

    def preview_3d(self,
                   terrain_data: Optional[np.ndarray] = None,
                   metadata: Optional[TerrainMetadata] = None,
                   satellite_file: Optional[str] = None,
                   title: str = "",
                   elev: float = 35,
                   azim: float = 90,  # 북쪽이 하단 정면에 오도록 (북에서 남쪽을 바라봄)
                   show: bool = True,
                   save_path: Optional[str] = None) -> None:
        """3D 지형 미리보기 (범용)"""
        import matplotlib.pyplot as plt

        if terrain_data is None:
            terrain_data = self.terrain
        if metadata is None:
            metadata = self.metadata

        if terrain_data is None:
            raise ValueError("지형 데이터 없음")

        terrain = self._fill_nan(terrain_data)
        ny, nx = terrain.shape
        cell = metadata.cell_size
        bounds = metadata.crop_bounds or metadata.bounds

        x = np.linspace(bounds[0], bounds[2], nx)
        # Y는 y_max에서 y_min으로 (북->남, DEM 배열 방향과 일치)
        y = np.linspace(bounds[3], bounds[1], ny)
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 위성 텍스처
        facecolors = None
        sat_file = satellite_file or metadata.satellite_file
        if sat_file and Path(sat_file).exists():
            try:
                from PIL import Image
                img = Image.open(sat_file).convert('RGB')
                img_array = np.array(img.resize((nx, ny), Image.LANCZOS)) / 255.0
                # flipud 불필요 - DEM과 위성 이미지 모두 [0,:]=북쪽
                facecolors = (img_array[:-1, :-1] + img_array[1:, :-1] +
                             img_array[:-1, 1:] + img_array[1:, 1:]) / 4
                print("[INFO] 위성 텍스처 적용")
            except Exception as e:
                print(f"[WARN] 텍스처 로드 실패: {e}")

        if facecolors is not None:
            ax.plot_surface(X, Y, terrain, facecolors=facecolors,
                           rstride=1, cstride=1, antialiased=True, shade=False)
        else:
            ax.plot_surface(X, Y, terrain, cmap='terrain',
                           rstride=1, cstride=1, antialiased=True, alpha=0.9)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Elevation (m)')

        if not title:
            title = f"Terrain: {nx}x{ny}, cell={cell}m\nCRS: {metadata.crs}"
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)

        x_range = bounds[2] - bounds[0]
        y_range = bounds[3] - bounds[1]
        z_range = max(terrain.max() - terrain.min(), 1)
        max_range = max(x_range, y_range, z_range)
        ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] 저장: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def preview_3d_original(self, elev: float = 35, azim: float = 90,
                            show: bool = True, save: bool = True) -> None:
        """원본 DEM 3D 미리보기"""
        if self.terrain_original is None or self.metadata_original is None:
            raise ValueError("원본 데이터 없음")

        save_path = None
        if save:
            save_path = str(self.work_dir / f"{self.dem_stem}_preview_original.png")

        # 위성 파일 자동 탐색
        sat_file = self.metadata_original.satellite_file
        if not sat_file or not Path(sat_file).exists():
            auto_sat = self.work_dir / f"{self.dem_stem}_satellite_original.png"
            if auto_sat.exists():
                sat_file = str(auto_sat)

        self.preview_3d(
            terrain_data=self.terrain_original,
            metadata=self.metadata_original,
            satellite_file=sat_file,
            title=f"Original DEM: {self.metadata_original.cols}x{self.metadata_original.rows}\n"
                  f"Bounds: X({self.metadata_original.bounds[0]:.0f}~{self.metadata_original.bounds[2]:.0f}), "
                  f"Y({self.metadata_original.bounds[1]:.0f}~{self.metadata_original.bounds[3]:.0f})",
            elev=elev, azim=azim,
            show=show, save_path=save_path
        )

    def preview_3d_crop(self, elev: float = 90, azim: float = 90,
                        show: bool = True, save: bool = True) -> None:
        """Crop된 DEM Bird View (위에서 내려다보기, 북쪽이 하단)"""
        if self.terrain is None or not self.metadata.crop_bounds:
            raise ValueError("Crop 데이터 없음")

        save_path = None
        if save:
            save_path = str(self.work_dir / f"{self.dem_stem}_preview_crop.png")

        # 위성 파일 자동 탐색
        sat_file = self.metadata.satellite_file
        if not sat_file or not Path(sat_file).exists():
            auto_sat = self.work_dir / f"{self.dem_stem}_satellite_crop.png"
            if auto_sat.exists():
                sat_file = str(auto_sat)

        cb = self.metadata.crop_bounds
        self.preview_3d(
            terrain_data=self.terrain,
            metadata=self.metadata,
            satellite_file=sat_file,
            title=f"Cropped DEM: {self.metadata.cols}x{self.metadata.rows}\n"
                  f"Bounds: X({cb[0]:.0f}~{cb[2]:.0f}), Y({cb[1]:.0f}~{cb[3]:.0f})",
            elev=elev, azim=azim,
            show=show, save_path=save_path
        )

    def plot_bounds_comparison(self, show: bool = True, save: bool = True) -> None:
        """원본 범위와 Crop 범위를 2D로 비교 시각화"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        if self.metadata_original is None:
            raise ValueError("원본 메타데이터 없음")

        orig = self.metadata_original.bounds
        crop = self.metadata.crop_bounds

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # 원본 DEM을 배경으로 표시
        terrain_show = self._fill_nan(self.terrain_original)
        extent = [orig[0], orig[2], orig[1], orig[3]]

        # 위성 이미지가 있으면 사용 (자동 탐색)
        sat_file = self.metadata_original.satellite_file
        if not sat_file or not Path(sat_file).exists():
            # 자동 탐색: {dem_stem}_satellite_original.png
            auto_sat = self.work_dir / f"{self.dem_stem}_satellite_original.png"
            if auto_sat.exists():
                sat_file = str(auto_sat)

        if sat_file and Path(sat_file).exists():
            from PIL import Image
            img = Image.open(sat_file)
            # origin='upper': data[0,0]=북서쪽이 left-top에 매핑
            # extent: [left, right, bottom, top] = [x_min, x_max, y_min, y_max]
            ax.imshow(img, extent=extent, origin='upper', aspect='equal')
        else:
            # 없으면 지형 컬러맵 사용
            im = ax.imshow(terrain_show, extent=extent, origin='upper',
                          cmap='terrain', aspect='equal')
            plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.7)

        # 원본 범위 (파란색)
        orig_rect = patches.Rectangle(
            (orig[0], orig[1]), orig[2]-orig[0], orig[3]-orig[1],
            linewidth=3, edgecolor='blue', facecolor='none',
            linestyle='-', label='Original DEM'
        )
        ax.add_patch(orig_rect)

        # Crop 범위 (빨간색)
        if crop:
            crop_rect = patches.Rectangle(
                (crop[0], crop[1]), crop[2]-crop[0], crop[3]-crop[1],
                linewidth=3, edgecolor='red', facecolor='red',
                alpha=0.2, label='Crop Area'
            )
            ax.add_patch(crop_rect)

            # Crop 범위 테두리
            crop_border = patches.Rectangle(
                (crop[0], crop[1]), crop[2]-crop[0], crop[3]-crop[1],
                linewidth=3, edgecolor='red', facecolor='none', linestyle='--'
            )
            ax.add_patch(crop_border)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='upper right')

        # 정보 텍스트
        info_text = f"Original: {self.metadata_original.cols}x{self.metadata_original.rows}, cell={self.metadata_original.cell_size}m\n"
        info_text += f"  X: {orig[0]:.0f} ~ {orig[2]:.0f} ({orig[2]-orig[0]:.0f}m)\n"
        info_text += f"  Y: {orig[1]:.0f} ~ {orig[3]:.0f} ({orig[3]-orig[1]:.0f}m)"

        if crop:
            info_text += f"\n\nCrop: {self.metadata.cols}x{self.metadata.rows}\n"
            info_text += f"  X: {crop[0]:.0f} ~ {crop[2]:.0f} ({crop[2]-crop[0]:.0f}m)\n"
            info_text += f"  Y: {crop[1]:.0f} ~ {crop[3]:.0f} ({crop[3]-crop[1]:.0f}m)"

        ax.set_title(f"DEM Bounds Comparison\n{self.dem_stem}")

        # 텍스트 박스
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, family='monospace')

        plt.tight_layout()

        if save:
            save_path = self.work_dir / f"{self.dem_stem}_bounds_comparison.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] 범위 비교 저장: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def info(self) -> str:
        """현재 상태 정보 출력"""
        lines = [
            "=" * 60,
            "TERRAIN PROCESSOR STATUS",
            "=" * 60,
            f"작업 디렉토리: {self.work_dir}",
            f"DEM 파일명: {self.dem_stem}",
            "",
            "[원본 DEM]",
        ]

        if self.metadata_original:
            orig = self.metadata_original
            lines.extend([
                f"  크기: {orig.cols} x {orig.rows}",
                f"  셀 크기: {orig.cell_size} m",
                f"  X 범위: {orig.bounds[0]:.0f} ~ {orig.bounds[2]:.0f} ({orig.bounds[2]-orig.bounds[0]:.0f}m)",
                f"  Y 범위: {orig.bounds[1]:.0f} ~ {orig.bounds[3]:.0f} ({orig.bounds[3]-orig.bounds[1]:.0f}m)",
                f"  고도: {orig.elevation_min:.1f} ~ {orig.elevation_max:.1f} m",
            ])
        else:
            lines.append("  (로드되지 않음)")

        if self.metadata.crop_bounds:
            crop = self.metadata.crop_bounds
            lines.extend([
                "",
                "[Crop 영역]",
                f"  크기: {self.metadata.cols} x {self.metadata.rows}",
                f"  X 범위: {crop[0]:.0f} ~ {crop[2]:.0f} ({crop[2]-crop[0]:.0f}m)",
                f"  Y 범위: {crop[1]:.0f} ~ {crop[3]:.0f} ({crop[3]-crop[1]:.0f}m)",
                f"  고도: {self.metadata.elevation_min:.1f} ~ {self.metadata.elevation_max:.1f} m",
            ])

        lines.append("=" * 60)
        info_str = "\n".join(lines)
        print(info_str)
        return info_str


# CLI 인터페이스
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DEM Terrain Processor")
    parser.add_argument("dem_path", help="DEM 파일 경로 (.tif, .dem)")
    parser.add_argument("--crop-center", nargs=2, type=float, metavar=('X', 'Y'),
                        help="Crop 중심점 좌표")
    parser.add_argument("--crop-size", nargs=2, type=float, metavar=('W', 'H'),
                        help="Crop 크기 (미터)")
    parser.add_argument("--cell-size", type=float, help="리샘플링 셀 크기")
    parser.add_argument("--satellite", action="store_true", help="위성 텍스처 다운로드")
    parser.add_argument("--preview", action="store_true", help="3D 미리보기")
    parser.add_argument("--no-show", action="store_true", help="화면에 표시하지 않음")

    args = parser.parse_args()

    tp = TerrainProcessor(args.dem_path)

    if args.crop_center and args.crop_size:
        tp.crop(center=tuple(args.crop_center), size=tuple(args.crop_size))

    if args.cell_size:
        tp.resample(args.cell_size)

    if args.satellite:
        tp.fetch_satellite_texture()

    tp.save()
    tp.info()

    if args.preview:
        show = not args.no_show
        tp.preview_3d_original(show=show, save=True)
        if tp.metadata.crop_bounds:
            tp.preview_3d_crop(show=show, save=True)
            tp.plot_bounds_comparison(show=show, save=True)
