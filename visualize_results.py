"""
PyVista/VTK 기반 산사태 시뮬레이션 시각화

Usage:
    python visualize_results.py <results_npz_path> [--workers N] [--sequential]
"""

import numpy as np
import pyvista as pv
from PIL import Image
import imageio.v3 as iio
import sys
import argparse
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import json
import yaml

# Offscreen 렌더링 설정
pv.OFF_SCREEN = True

# 기본 설정값
DEFAULT_CONFIG = {
    'title_prefix': 'Landslide Simulation',
    'view_elev': 35.0,
    'view_azim': 30.0,
    'camera_distance_factor': 2.5,
    'fps': 15,
    'frame_skip': 1,
    'window_width': 1200,
    'window_height': 900,
    'particle_size': 3,
    'particle_cmap': 'plasma',
    'particle_height_offset': 30,
    'use_satellite': True,
    'terrain_cmap': 'terrain',
    # animation_gif, animation_video: 자동 생성 (landslide_(좌표).gif/.webm)
}


def log(msg: str):
    """간단한 로깅"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{timestamp} | {msg}")
    sys.stdout.flush()


def load_visualization_config(config_path: Path) -> dict:
    """visualization_config.yaml 로드 및 기본값 병합"""
    config = DEFAULT_CONFIG.copy()

    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f) or {}
        config.update(user_config)
        log(f"  Loaded config: {config_path.name}")
    else:
        log(f"  Using default config (no {config_path.name} found)")

    return config


def render_frame(data_path: str, frame_idx: int, frame_data_idx: int,
                 output_path: str, render_config: dict):
    """PyVista로 단일 프레임 렌더링"""

    # 데이터 로드
    data = np.load(data_path, allow_pickle=True)
    terrain = data['terrain']
    cell_size = float(data['cell_size'])
    times = data['times']
    n_active = data['n_active']
    x_data = data['x']
    y_data = data['y']
    vx_data = data['vx']
    vy_data = data['vy']
    x_min = float(data['x_min'])
    y_min = float(data['y_min'])

    ny, nx = terrain.shape

    # 점 좌표 생성
    x = np.arange(nx) * cell_size
    y = np.arange(ny) * cell_size
    X, Y = np.meshgrid(x, y)

    # StructuredGrid 생성 - 정방향 좌표계 (X→동, Y→북)
    # npz의 terrain은 SPH 엔진에서 이미 변환됨 (row 0=남, row 증가=북)
    grid = pv.StructuredGrid()
    terrain_display = terrain
    points = np.column_stack([X.ravel(), Y.ravel(), terrain_display.ravel()])
    grid.points = points
    grid.dimensions = [nx, ny, 1]

    # 위성 텍스처 로드
    texture = None
    satellite_file = render_config.get('satellite_file')
    if satellite_file and Path(satellite_file).exists():
        try:
            img = Image.open(satellite_file)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img_array = np.array(img)
            texture = pv.numpy_to_texture(img_array)

            # 텍스처 좌표 - 정방향 매핑
            # VTK: (u=0,v=0)=이미지 좌하단(SW), (u=1,v=1)=이미지 우상단(NE)
            u = np.linspace(0, 1, nx)  # 서→동
            v = np.linspace(0, 1, ny)  # 남→북
            U_tex, V_tex = np.meshgrid(u, v)
            grid.active_texture_coordinates = np.c_[U_tex.ravel(), V_tex.ravel()]
        except Exception:
            texture = None

    # 입자 데이터
    mask = ~np.isnan(x_data[frame_data_idx])
    px_orig = x_data[frame_data_idx, mask]
    py_orig = y_data[frame_data_idx, mask]
    vx = vx_data[frame_data_idx, mask]
    vy = vy_data[frame_data_idx, mask]

    # 입자 좌표 - 정방향 (뒤집기 없음)
    # px_orig = geo_x - x_min (동쪽으로 증가)
    # py_orig = geo_y - y_min (북쪽으로 증가)
    px = px_orig
    py = py_orig

    # 입자 고도 계산
    height_offset = render_config.get('particle_height_offset', DEFAULT_CONFIG['particle_height_offset'])
    if len(px) > 0:
        col = np.clip(px / cell_size, 0, nx - 1.001)
        row = np.clip(py / cell_size, 0, ny - 1.001)
        col0 = np.floor(col).astype(int)
        row0 = np.floor(row).astype(int)
        col1 = np.minimum(col0 + 1, nx - 1)
        row1 = np.minimum(row0 + 1, ny - 1)
        wx, wy = col - col0, row - row0
        pz = (terrain_display[row0, col0] * (1 - wx) * (1 - wy) +
              terrain_display[row0, col1] * wx * (1 - wy) +
              terrain_display[row1, col0] * (1 - wx) * wy +
              terrain_display[row1, col1] * wx * wy)
        pz = pz + height_offset

        speed = np.sqrt(vx**2 + vy**2)
        particle_points = np.column_stack([px, py, pz])
        particles = pv.PolyData(particle_points)
        particles['speed'] = speed
    else:
        particles = None
        speed = np.array([])

    # 플로터 설정
    window_width = render_config.get('window_width', DEFAULT_CONFIG['window_width'])
    window_height = render_config.get('window_height', DEFAULT_CONFIG['window_height'])
    plotter = pv.Plotter(off_screen=True, window_size=[window_width, window_height])
    plotter.enable_depth_peeling(10)

    # 지형 추가
    terrain_cmap = render_config.get('terrain_cmap', DEFAULT_CONFIG['terrain_cmap'])
    if texture is not None:
        plotter.add_mesh(grid, texture=texture, smooth_shading=True)
    else:
        plotter.add_mesh(grid, cmap=terrain_cmap, smooth_shading=True)

    # 입자 추가
    speed_max = render_config['speed_max']
    particle_size = render_config.get('particle_size', DEFAULT_CONFIG['particle_size'])
    particle_cmap = render_config.get('particle_cmap', DEFAULT_CONFIG['particle_cmap'])

    if particles is not None and len(particles.points) > 0:
        plotter.add_mesh(
            particles,
            scalars='speed',
            cmap=particle_cmap,
            point_size=particle_size,
            render_points_as_spheres=True,
            clim=[0, speed_max],
            show_scalar_bar=False,
            ambient=0.5,
            diffuse=0.5,
        )

    # ── 타겟 bbox 마커 ──
    targets = render_config.get('targets', [])
    for target in targets:
        bbox = target.get('bbox_local')
        if not bbox:
            continue
        # bbox_local = [x0, y0, x1, y1] in local coords
        x0, y0_b, x1, y1_b = bbox
        # clip to terrain bounds
        max_x = (nx - 1) * cell_size
        max_y = (ny - 1) * cell_size
        x0 = max(0, min(x0, max_x))
        x1 = max(0, min(x1, max_x))
        y0_b = max(0, min(y0_b, max_y))
        y1_b = max(0, min(y1_b, max_y))

        # skip degenerate bbox (collapsed to zero area after clipping)
        if x1 - x0 < 1e-3 or y1_b - y0_b < 1e-3:
            continue

        # 4 corners → terrain height via bilinear interpolation
        corners_xy = [(x0, y0_b), (x1, y0_b), (x1, y1_b), (x0, y1_b)]
        corners_3d = []
        for cx, cy in corners_xy:
            col_f = np.clip(cx / cell_size, 0, nx - 1.001)
            row_f = np.clip(cy / cell_size, 0, ny - 1.001)
            c0, r0 = int(col_f), int(row_f)
            c1_, r1_ = min(c0 + 1, nx - 1), min(r0 + 1, ny - 1)
            wx_, wy_ = col_f - c0, row_f - r0
            z = (terrain_display[r0, c0] * (1 - wx_) * (1 - wy_) +
                 terrain_display[r0, c1_] * wx_ * (1 - wy_) +
                 terrain_display[r1_, c0] * (1 - wx_) * wy_ +
                 terrain_display[r1_, c1_] * wx_ * wy_)
            corners_3d.append([cx, cy, float(z) + 10])

        # 4 edges as tubes
        for i_edge in range(4):
            p1 = corners_3d[i_edge]
            p2 = corners_3d[(i_edge + 1) % 4]
            line = pv.Line(p1, p2)
            tube = line.tube(radius=3.0)
            plotter.add_mesh(tube, color='red', ambient=0.8)

        # label at center
        cx_label = (x0 + x1) / 2
        cy_label = (y0_b + y1_b) / 2
        col_f = np.clip(cx_label / cell_size, 0, nx - 1.001)
        row_f = np.clip(cy_label / cell_size, 0, ny - 1.001)
        c0, r0 = int(col_f), int(row_f)
        c1_, r1_ = min(c0 + 1, nx - 1), min(r0 + 1, ny - 1)
        wx_, wy_ = col_f - c0, row_f - r0
        z_label = (terrain_display[r0, c0] * (1 - wx_) * (1 - wy_) +
                   terrain_display[r0, c1_] * wx_ * (1 - wy_) +
                   terrain_display[r1_, c0] * (1 - wx_) * wy_ +
                   terrain_display[r1_, c1_] * wx_ * wy_)
        label_pt = pv.PolyData([[cx_label, cy_label, float(z_label) + 50]])
        plotter.add_point_labels(
            label_pt, [target.get('name', '')],
            font_size=14, text_color='red', point_size=0, shape=None,
            always_visible=True)

    # Colorbar 추가
    plotter.add_scalar_bar(
        title='Speed (m/s)',
        vertical=True,
        position_x=0.9,
        position_y=0.2,
        width=0.05,
        height=0.6,
        n_labels=5,
        fmt='%.1f',
    )
    plotter.update_scalar_bar_range([0, speed_max])

    # 카메라 설정
    center = grid.center
    bounds = grid.bounds
    x_range = bounds[1] - bounds[0]
    y_range = bounds[3] - bounds[2]
    z_range = bounds[5] - bounds[4]
    max_range = max(x_range, y_range, z_range)

    camera_factor = render_config.get('camera_distance_factor', DEFAULT_CONFIG['camera_distance_factor'])
    distance = max_range * camera_factor

    # 카메라 고도각/방위각 적용
    elev = np.radians(render_config.get('view_elev', DEFAULT_CONFIG['view_elev']))
    azim = np.radians(render_config.get('view_azim', DEFAULT_CONFIG['view_azim']))
    cam_x = center[0] + distance * np.cos(elev) * np.sin(azim)
    cam_y = center[1] - distance * np.cos(elev) * np.cos(azim)
    cam_z = center[2] + distance * np.sin(elev)
    plotter.camera_position = [(cam_x, cam_y, cam_z), center, (0, 0, 1)]

    # 제목
    n_act = n_active[frame_data_idx]
    max_spd = speed.max() if len(speed) > 0 else 0
    mean_spd = speed.mean() if len(speed) > 0 else 0
    x_center = x_min + (nx * cell_size) / 2
    y_center = y_min + (ny * cell_size) / 2

    title_prefix = render_config.get('title_prefix', DEFAULT_CONFIG['title_prefix'])
    title = (f"{title_prefix} (TM: {x_center:.0f}, {y_center:.0f}) - t = {times[frame_data_idx]:.1f}s\n"
             f"Particles: {n_act}, Speed: mean={mean_spd:.1f}, max={max_spd:.1f} m/s")

    # 한글 폰트 설정
    text_actor = plotter.add_text(title, font_size=12, position='upper_left')
    korean_font = Path('C:/Windows/Fonts/malgun.ttf')
    if korean_font.exists() and text_actor is not None:
        text_actor.GetTextProperty().SetFontFile(str(korean_font))
        text_actor.GetTextProperty().SetFontFamily(4)  # VTK_FONT_FILE

    # 저장
    plotter.screenshot(output_path)
    plotter.close()

    return frame_idx


def render_frame_subprocess(args):
    """subprocess로 프레임 렌더링"""
    data_path, frame_idx, frame_data_idx, output_path, config_json = args

    script = f'''
import sys
sys.path.insert(0, r"{Path(__file__).parent}")
from visualize_results import render_frame
import json
config = json.loads(r"""{config_json}""")
render_frame(r"{data_path}", {frame_idx}, {frame_data_idx}, r"{output_path}", config)
'''

    result = subprocess.run(
        ['python', '-c', script],
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        raise RuntimeError(f"Frame {frame_idx} failed: {result.stderr}")

    return frame_idx


def create_animation(data_path: str, output_dir: Path, vis_config: dict,
                     n_workers: int = 4, frame_skip: int = None, parallel: bool = True):
    """애니메이션 생성"""

    log("=" * 60)
    log("LANDSLIDE VISUALIZATION (PyVista/VTK)")
    log("=" * 60)

    # 데이터 로드
    log(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    terrain = data['terrain']
    times = data['times']

    ny, nx = terrain.shape
    n_frames = len(times)

    log(f"  Frames: {n_frames}")
    log(f"  Terrain: {terrain.shape}")
    log(f"  Duration: {times[-1]:.1f}s")

    # frame_skip 결정 (CLI > config > default)
    if frame_skip is None:
        frame_skip = vis_config.get('frame_skip', DEFAULT_CONFIG['frame_skip'])

    frame_indices = list(range(0, n_frames, frame_skip))
    actual_frames = len(frame_indices)
    log(f"  Frame skip: {frame_skip} ({actual_frames} frames to render)")

    # speed_max 계산
    log("  Computing speed range...")
    all_speeds = []
    x_data = data['x']
    vx_data = data['vx']
    vy_data = data['vy']
    for i in range(n_frames):
        mask = ~np.isnan(x_data[i])
        if mask.any():
            speeds = np.sqrt(vx_data[i, mask]**2 + vy_data[i, mask]**2)
            all_speeds.extend(speeds)
    speed_max = float(max(all_speeds)) if all_speeds else 1.0
    log(f"  Speed range: 0 ~ {speed_max:.2f} m/s")

    # 위성 이미지 찾기
    satellite_file = ''
    if vis_config.get('use_satellite', True):
        for pattern in ['*satellite*.png', '*satellite*.jpg']:
            matches = list(output_dir.glob(pattern))
            if matches:
                satellite_file = str(matches[0])
                log(f"  Satellite texture: {Path(satellite_file).name}")
                break

    # 좌표 suffix 계산
    cell_size = float(data['cell_size'])
    x_min_geo = float(data['x_min'])
    y_min_geo = float(data['y_min'])
    x_center = x_min_geo + (nx * cell_size) / 2
    y_center = y_min_geo + (ny * cell_size) / 2
    coord_suffix = f"{x_center:.0f}x{y_center:.0f}"

    # simulation_config.yaml에서 targets 로드
    targets_for_render = []
    sim_config_path = output_dir / 'simulation_config.yaml'
    if sim_config_path.exists():
        with open(sim_config_path, 'r', encoding='utf-8') as f:
            sim_config = yaml.safe_load(f) or {}
        for t in sim_config.get('targets', []):
            info = {'name': t.get('name', '')}
            if t.get('target_type') == 'area' and t.get('bbox'):
                b = t['bbox']
                info['bbox_local'] = [
                    b[0] - x_min_geo, b[1] - y_min_geo,
                    b[2] - x_min_geo, b[3] - y_min_geo
                ]
            elif t.get('coordinates'):
                c = t['coordinates']
                r = t.get('proximity_radius', 30)
                info['bbox_local'] = [
                    c[0] - r - x_min_geo, c[1] - r - y_min_geo,
                    c[0] + r - x_min_geo, c[1] + r - y_min_geo
                ]
            targets_for_render.append(info)

    # 렌더링 설정 구성
    render_config = {
        'speed_max': speed_max,
        'satellite_file': satellite_file,
        'targets': targets_for_render,
        **vis_config
    }

    # 임시 디렉토리
    temp_dir = Path(tempfile.mkdtemp(prefix='landslide_frames_'))
    log(f"  Temp dir: {temp_dir}")

    mode = "parallel" if parallel else "sequential"
    log(f"\nGenerating {actual_frames} frames [{mode}] (workers={n_workers})...")

    start_time = datetime.now()

    try:
        if parallel:
            # 병렬 처리
            config_json = json.dumps(render_config)
            tasks = []
            for i, frame_data_idx in enumerate(frame_indices):
                output_file = str(temp_dir / f"frame_{i:04d}.png")
                tasks.append((data_path, i, frame_data_idx, output_file, config_json))

            completed = 0
            failed = 0

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_to_task = {executor.submit(render_frame_subprocess, t): t for t in tasks}

                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        future.result()
                        completed += 1

                        elapsed = (datetime.now() - start_time).total_seconds()
                        fps = completed / elapsed if elapsed > 0 else 0
                        eta = (actual_frames - completed) / fps if fps > 0 else 0

                        if completed == 1 or completed == actual_frames or completed % 5 == 0:
                            log(f"  [Frame {completed}/{actual_frames}] {completed*100//actual_frames}% "
                                f"elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s, speed: {fps:.2f} fps")
                    except Exception as e:
                        failed += 1
                        log(f"  ERROR: Frame {task[1]} failed: {e}")

            elapsed_total = (datetime.now() - start_time).total_seconds()
            log(f"  Rendering complete: {elapsed_total:.1f}s ({completed}/{actual_frames} frames)")

        else:
            # 순차 처리
            for i, frame_data_idx in enumerate(frame_indices):
                output_file = str(temp_dir / f"frame_{i:04d}.png")

                render_frame(data_path, i, frame_data_idx, output_file, render_config)

                completed = i + 1
                elapsed = (datetime.now() - start_time).total_seconds()
                fps = completed / elapsed if elapsed > 0 else 0
                eta = (actual_frames - completed) / fps if fps > 0 else 0

                if completed == 1 or completed == actual_frames or completed % 5 == 0:
                    log(f"  [Frame {completed}/{actual_frames}] {completed*100//actual_frames}% "
                        f"elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s, speed: {fps:.2f} fps")

            elapsed_total = (datetime.now() - start_time).total_seconds()
            log(f"  Rendering complete: {elapsed_total:.1f}s ({actual_frames} frames)")

        # 프레임 로드
        log("  Loading rendered frames...")
        frames = []
        skipped = 0
        for i in range(actual_frames):
            frame_path = temp_dir / f"frame_{i:04d}.png"
            if frame_path.exists():
                frames.append(iio.imread(str(frame_path)))
            else:
                skipped += 1

        if skipped > 0:
            log(f"  Warning: {skipped} frames skipped")

        if not frames:
            log("  ERROR: No frames rendered!")
            return None, None

        # 출력 fps
        output_fps = vis_config.get('fps', DEFAULT_CONFIG['fps'])

        # GIF 저장
        gif_path = output_dir / f"landslide_{coord_suffix}.gif"
        log(f"Saving GIF to {gif_path} (fps={output_fps})...")
        iio.imwrite(str(gif_path), frames, fps=output_fps, loop=0)
        log("GIF saved!")

        # 비디오 저장 (imageio-ffmpeg 내장 바이너리 사용)
        video_path = output_dir / f"landslide_{coord_suffix}.webm"
        log(f"Saving video to {video_path} (fps={output_fps})...")
        try:
            import imageio_ffmpeg
            frames_rgb = [f[:, :, :3] if f.shape[-1] == 4 else f for f in frames]
            h, w = frames_rgb[0].shape[:2]

            # imageio-ffmpeg 내장 ffmpeg 바이너리 경로
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            log(f"  Using ffmpeg: {ffmpeg_exe}")

            # ffmpeg로 webm 생성 (stdin으로 프레임 전달)
            ffmpeg_cmd = [
                ffmpeg_exe, '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{w}x{h}',
                '-pix_fmt', 'rgb24',
                '-r', str(output_fps),
                '-i', '-',
                '-c:v', 'libvpx-vp9',
                '-b:v', '2M',
                '-pix_fmt', 'yuv420p',
                str(video_path)
            ]

            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for frame in frames_rgb:
                proc.stdin.write(frame.astype('uint8').tobytes())
            proc.stdin.close()
            proc.wait()

            if proc.returncode == 0:
                log("Video saved!")
            else:
                stderr = proc.stderr.read().decode()
                log(f"  Video save failed: {stderr[:200]}")
                video_path = None
        except Exception as e:
            log(f"  Video save failed: {e}")
            video_path = None

        return gif_path, video_path

    finally:
        log("  Cleaning up temp dir...")
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description='산사태 시뮬레이션 시각화 (PyVista/VTK)')
    parser.add_argument('results_file', type=str, help='시뮬레이션 결과 파일 (.npz)')
    parser.add_argument('--workers', '-w', type=int, default=4, help='병렬 워커 수')
    parser.add_argument('--skip', '-s', type=int, default=None, help='프레임 스킵 (config 우선)')
    parser.add_argument('--sequential', action='store_true', help='순차 처리 모드')

    args = parser.parse_args()

    results_path = Path(args.results_file).resolve()
    if not results_path.exists():
        print(f"Error: File not found: {results_path}")
        sys.exit(1)

    output_dir = results_path.parent

    # 설정 로드
    vis_config_path = output_dir / 'visualization_config.yaml'
    vis_config = load_visualization_config(vis_config_path)

    create_animation(
        str(results_path),
        output_dir,
        vis_config,
        n_workers=args.workers,
        frame_skip=args.skip,
        parallel=not args.sequential
    )

    log("\nDone!")


if __name__ == '__main__':
    main()
