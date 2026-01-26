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
    'animation_gif': 'landslide_animation.gif',
    'animation_video': 'landslide_animation.webm',
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

    # StructuredGrid 생성 - DEM X,Y축 모두 뒤집기 (terrain_processor preview와 일치)
    grid = pv.StructuredGrid()
    terrain_flipped = terrain[::-1, ::-1]  # Y축, X축 모두 뒤집기
    points = np.column_stack([X.ravel(), Y.ravel(), terrain_flipped.ravel()])
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

            # 텍스처 좌표 - U, V축 모두 뒤집기 (terrain_processor preview와 일치)
            u = np.linspace(1, 0, nx)  # X축 뒤집기에 맞춰 U축도 뒤집기
            v = np.linspace(1, 0, ny)
            U_tex, V_tex = np.meshgrid(u, v)
            grid.active_texture_coordinates = np.c_[U_tex.ravel(), V_tex.ravel()]
        except Exception:
            texture = None

    # 입자 데이터 - X,Y축 모두 뒤집기 (DEM/텍스처와 일치시키기)
    mask = ~np.isnan(x_data[frame_data_idx])
    px_orig = x_data[frame_data_idx, mask]
    py_orig = y_data[frame_data_idx, mask]
    vx = vx_data[frame_data_idx, mask]
    vy = vy_data[frame_data_idx, mask]

    # 입자 좌표 X,Y축 모두 뒤집기 (terrain_processor preview와 일치)
    max_x = (nx - 1) * cell_size
    max_y = (ny - 1) * cell_size
    px = max_x - px_orig  # X 뒤집기
    py = max_y - py_orig  # Y 뒤집기

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
        pz = (terrain_flipped[row0, col0] * (1 - wx) * (1 - wy) +
              terrain_flipped[row0, col1] * wx * (1 - wy) +
              terrain_flipped[row1, col0] * (1 - wx) * wy +
              terrain_flipped[row1, col1] * wx * wy)
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

    # 북쪽이 하단 정면에 오도록
    cam_x = center[0]
    cam_y = center[1] - distance * 0.8
    cam_z = center[2] + distance * 0.5
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

    # 렌더링 설정 구성
    render_config = {
        'speed_max': speed_max,
        'satellite_file': satellite_file,
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
        gif_filename = vis_config.get('animation_gif', DEFAULT_CONFIG['animation_gif'])
        gif_path = output_dir / gif_filename
        log(f"Saving GIF to {gif_path} (fps={output_fps})...")
        iio.imwrite(str(gif_path), frames, fps=output_fps, loop=0)
        log("GIF saved!")

        # 비디오 저장 (imageio-ffmpeg 내장 바이너리 사용)
        video_filename = vis_config.get('animation_video', DEFAULT_CONFIG['animation_video'])
        video_path = output_dir / video_filename
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
