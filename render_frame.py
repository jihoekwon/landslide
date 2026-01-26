"""
단일 프레임 렌더링 스크립트 (subprocess용)

Usage:
    python render_frame.py <data_file> <frame_idx> <output_path> <config_json>
"""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def render_frame(data_path: str, frame_data_idx: int, output_path: str, config: dict):
    """단일 프레임 렌더링"""

    # 데이터 로드
    data = np.load(data_path)
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

    # Mesh grid
    x = np.arange(nx) * cell_size
    y = np.arange(ny) * cell_size
    X, Y = np.meshgrid(x, y)

    # 위성 이미지 로드
    satellite_colors = None
    if config.get('satellite_file'):
        try:
            img = Image.open(config['satellite_file'])
            # RGBA → RGB 변환
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img_resized = img.resize((nx, ny), Image.LANCZOS)
            img_array = np.array(img_resized) / 255.0
            img_array = np.flipud(img_array)
            # facecolors용 평균 (plot_surface 호환)
            satellite_colors = (img_array[:-1, :-1] + img_array[1:, :-1] +
                               img_array[:-1, 1:] + img_array[1:, 1:]) / 4
        except Exception as e:
            print(f"Warning: Could not load satellite texture: {e}", file=sys.stderr)
            satellite_colors = None

    def get_particle_elevation(x_arr, y_arr):
        col = np.clip(x_arr / cell_size, 0, nx - 1.001)
        row = np.clip(y_arr / cell_size, 0, ny - 1.001)
        col0 = np.floor(col).astype(int)
        row0 = np.floor(row).astype(int)
        col1 = np.minimum(col0 + 1, nx - 1)
        row1 = np.minimum(row0 + 1, ny - 1)
        wx, wy = col - col0, row - row0
        z = (terrain[row0, col0] * (1 - wx) * (1 - wy) +
             terrain[row0, col1] * wx * (1 - wy) +
             terrain[row1, col0] * (1 - wx) * wy +
             terrain[row1, col1] * wx * wy)
        return z

    # Figure 생성
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0.02, right=0.92, top=0.95, bottom=0.05)

    # Colorbar
    speed_max = config['speed_max']
    norm = plt.Normalize(vmin=0, vmax=speed_max)
    sm = cm.ScalarMappable(cmap=config['particle_cmap'], norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(sm, cax=cax, label='Speed (m/s)')

    # Get valid particles
    mask = ~np.isnan(x_data[frame_data_idx])
    px = x_data[frame_data_idx, mask]
    py = y_data[frame_data_idx, mask]
    vx = vx_data[frame_data_idx, mask]
    vy = vy_data[frame_data_idx, mask]

    if len(px) > 0:
        terrain_z = get_particle_elevation(px, py)
        pz = terrain_z + 10
        speed = np.sqrt(vx**2 + vy**2)
    else:
        pz = np.array([])
        speed = np.array([])

    ax.computed_zorder = False

    # Plot terrain
    if satellite_colors is not None:
        ax.plot_surface(X, Y, terrain, facecolors=satellite_colors,
                       rstride=1, cstride=1, antialiased=False,
                       shade=False, zorder=1)
    else:
        ax.plot_surface(X, Y, terrain, cmap='terrain', alpha=0.8, zorder=1,
                       rstride=1, cstride=1, antialiased=False)

    # Plot particles
    if len(px) > 0:
        edge_colors = config['edge_color'] if config['particle_edge'] else 'none'
        line_widths = config['edge_width'] if config['particle_edge'] else 0
        ax.scatter(px, py, pz, c=speed, cmap=config['particle_cmap'],
                  s=config['particle_size'], vmin=0, vmax=speed_max,
                  edgecolors=edge_colors, linewidths=line_widths,
                  depthshade=False, alpha=1.0, zorder=2)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    n_act = n_active[frame_data_idx]
    max_spd = speed.max() if len(speed) > 0 else 0
    mean_spd = speed.mean() if len(speed) > 0 else 0

    x_center = x_min + (nx * cell_size) / 2
    y_center = y_min + (ny * cell_size) / 2

    ax.set_title(f'{config["title_prefix"]} (TM: {x_center:.0f}, {y_center:.0f}) - t = {times[frame_data_idx]:.1f}s\n'
                f'Particles: {n_act}, Speed: mean={mean_spd:.1f}, max={max_spd:.1f} m/s')

    ax.view_init(elev=config['view_elev'], azim=config['view_azim'])

    # Equal aspect ratio
    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()
    z_range = terrain.max() - terrain.min()
    max_range = max(x_range, y_range, z_range)
    ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])

    plt.savefig(output_path, dpi=config['dpi'])
    plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python render_frame.py <data_file> <frame_idx> <output_path> <config_json>")
        sys.exit(1)

    data_path = sys.argv[1]
    frame_idx = int(sys.argv[2])
    output_path = sys.argv[3]
    config = json.loads(sys.argv[4])

    render_frame(data_path, frame_idx, output_path, config)
