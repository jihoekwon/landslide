"""
Generate analysis report from SPH Landslide Simulation results.

Usage:
    python generate_report.py <results_npz_path> [--no-llm]

Example:
    python generate_report.py ./guryoung_dem_10m/simulation_results.npz
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
from dataclasses import dataclass
from io import BytesIO

import yaml

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class SimulationStats:
    """시뮬레이션 통계"""
    # 기본 정보
    n_frames: int = 0
    duration: float = 0.0
    save_interval: float = 0.0
    n_particles_initial: int = 0
    n_particles_final: int = 0

    # 속도 통계
    max_speed: float = 0.0
    max_speed_time: float = 0.0
    mean_speed_final: float = 0.0

    # Runout 통계
    runout_head_final: float = 0.0
    runout_centroid_final: float = 0.0
    runout_head_max: float = 0.0

    # 에너지/운동량
    total_mass: float = 0.0
    kinetic_energy_max: float = 0.0

    # 침식
    entrainment_enabled: bool = False
    concentration_initial: float = 0.0
    concentration_final: float = 0.0


def analyze_results(data) -> SimulationStats:
    """결과 데이터 분석"""
    stats = SimulationStats()

    times = data['times']
    n_active = data['n_active']
    x_data = data['x']
    y_data = data['y']
    vx_data = data['vx']
    vy_data = data['vy']

    stats.n_frames = len(times)
    stats.duration = float(times[-1])
    stats.save_interval = float(data.get('save_interval', times[1] - times[0] if len(times) > 1 else 0))
    stats.n_particles_initial = int(n_active[0])
    stats.n_particles_final = int(n_active[-1])

    # 초기 위치 (runout 계산용)
    init_x_local = float(data['init_x']) - float(data['x_min'])
    init_y_local = float(data['init_y']) - float(data['y_min'])

    max_speeds = []
    mean_speeds = []
    runout_heads = []
    runout_centroids = []

    for i in range(stats.n_frames):
        mask = ~np.isnan(x_data[i])
        px = x_data[i, mask]
        py = y_data[i, mask]
        vx = vx_data[i, mask]
        vy = vy_data[i, mask]

        if len(px) > 0:
            speeds = np.sqrt(vx**2 + vy**2)
            max_speeds.append(speeds.max())
            mean_speeds.append(speeds.mean())

            distances = np.sqrt((px - init_x_local)**2 + (py - init_y_local)**2)
            runout_heads.append(distances.max())
            runout_centroids.append(distances.mean())
        else:
            max_speeds.append(0)
            mean_speeds.append(0)
            runout_heads.append(0)
            runout_centroids.append(0)

    # 속도 통계
    stats.max_speed = max(max_speeds)
    stats.max_speed_time = times[np.argmax(max_speeds)]
    stats.mean_speed_final = mean_speeds[-1] if mean_speeds else 0

    # Runout 통계
    stats.runout_head_final = runout_heads[-1] if runout_heads else 0
    stats.runout_centroid_final = runout_centroids[-1] if runout_centroids else 0
    stats.runout_head_max = max(runout_heads) if runout_heads else 0

    # 침식 파라미터
    if 'C_init' in data.files:
        stats.entrainment_enabled = True
        stats.concentration_initial = float(data['C_init'])
        if 'concentration' in data.files:
            conc_data = data['concentration']
            mask = ~np.isnan(conc_data[-1])
            if mask.any():
                stats.concentration_final = float(np.nanmean(conc_data[-1]))

    return stats


def analyze_velocity_by_distance(data, distance_thresholds: List[float] = None) -> dict:
    """이동거리별 속도 분석 - Head 입자가 특정 거리에 도달했을 때의 유동체 속도"""
    if distance_thresholds is None:
        distance_thresholds = [100, 200, 300, 400, 500, 600]

    times = data['times']
    x_data = data['x']
    y_data = data['y']
    vx_data = data['vx']
    vy_data = data['vy']

    init_x_local = float(data['init_x']) - float(data['x_min'])
    init_y_local = float(data['init_y']) - float(data['y_min'])

    # 농도 데이터 (있는 경우)
    has_conc = 'concentration' in data.files
    conc_data = data['concentration'] if has_conc else None

    # 결과 저장
    results = {
        'thresholds': distance_thresholds,
        'velocity_at_distance': {},  # {distance: {'time': t, 'head_speed': v, 'mean_speed': v, 'max_speed': v, 'concentration': c}}
    }

    # 각 거리 임계값에 대해 Head가 도달한 시점 찾기
    for threshold in distance_thresholds:
        reached = False
        for i in range(len(times)):
            mask = ~np.isnan(x_data[i])
            if not mask.any():
                continue

            px = x_data[i, mask]
            py = y_data[i, mask]
            vx = vx_data[i, mask]
            vy = vy_data[i, mask]

            distances = np.sqrt((px - init_x_local)**2 + (py - init_y_local)**2)
            head_dist = distances.max()

            if head_dist >= threshold and not reached:
                reached = True
                speeds = np.sqrt(vx**2 + vy**2)

                # Head 입자 (가장 멀리 간 입자)의 속도
                head_idx = np.argmax(distances)
                head_speed = speeds[head_idx]

                # 전방 입자들 (상위 10%)의 평균 속도
                n_front = max(1, len(distances) // 10)
                front_indices = np.argsort(distances)[-n_front:]
                front_speed = speeds[front_indices].mean()

                # 농도
                if has_conc:
                    conc = conc_data[i, mask]
                    front_conc = conc[front_indices].mean()
                else:
                    front_conc = 0

                results['velocity_at_distance'][threshold] = {
                    'time': float(times[i]),
                    'head_speed': float(head_speed),
                    'front_mean_speed': float(front_speed),
                    'overall_mean_speed': float(speeds.mean()),
                    'overall_max_speed': float(speeds.max()),
                    'front_concentration': float(front_conc),
                }
                break

        if not reached:
            results['velocity_at_distance'][threshold] = None

    return results


def _analyze_building_target(target, times, x_data, y_data, vx_data, vy_data,
                              conc_data, height_data, x_min, y_min, rho_debris,
                              init_x=0, init_y=0):
    """건물(점) 타겟 분석 - 기존 point+radius 방식"""
    name = target.get('name', '미상')
    coords = target.get('coordinates', [0, 0])
    address = target.get('address', '')
    bldg_type = target.get('type', target.get('structure_type', 'unknown'))
    radius = target.get('proximity_radius', 30)

    has_conc = conc_data is not None
    has_height = height_data is not None

    target_x_local = coords[0] - x_min
    target_y_local = coords[1] - y_min

    arrival_frame = None
    peak_speed_frame = None
    peak_speed_val = 0.0
    min_closest_dist = float('inf')
    closest_frame = None

    for i in range(len(times)):
        mask = ~np.isnan(x_data[i])
        if not mask.any():
            continue
        px = x_data[i, mask]
        py = y_data[i, mask]
        dist_to_target = np.sqrt((px - target_x_local)**2 + (py - target_y_local)**2)
        in_radius = dist_to_target <= radius

        if in_radius.any():
            if arrival_frame is None:
                arrival_frame = i
            vx = vx_data[i, mask][in_radius]
            vy = vy_data[i, mask][in_radius]
            speeds = np.sqrt(vx**2 + vy**2)
            frame_max_speed = float(speeds.max())
            if frame_max_speed > peak_speed_val:
                peak_speed_val = frame_max_speed
                peak_speed_frame = i
        else:
            frame_min_dist = float(dist_to_target.min())
            if frame_min_dist < min_closest_dist:
                min_closest_dist = frame_min_dist
                closest_frame = i

    impact = {
        'name': name, 'address': address, 'type': bldg_type,
        'target_type': 'building',
        'coordinates': coords, 'proximity_radius': radius,
        'reached': arrival_frame is not None,
    }

    if arrival_frame is not None:
        mask_a = ~np.isnan(x_data[arrival_frame])
        px_a, py_a = x_data[arrival_frame, mask_a], y_data[arrival_frame, mask_a]
        vx_a, vy_a = vx_data[arrival_frame, mask_a], vy_data[arrival_frame, mask_a]
        dist_a = np.sqrt((px_a - target_x_local)**2 + (py_a - target_y_local)**2)
        in_r_a = dist_a <= radius
        speeds_a = np.sqrt(vx_a[in_r_a]**2 + vy_a[in_r_a]**2)

        impact['arrival_time'] = float(times[arrival_frame])
        impact['head_speed'] = float(speeds_a.max())
        impact['front_mean_speed'] = float(speeds_a.mean())
        impact['arrival_pressure'] = float(rho_debris * speeds_a.max()**2 / 2 / 1000)
        impact['n_particles_arrived'] = int(in_r_a.sum())
        impact['front_concentration'] = float(conc_data[arrival_frame, mask_a][in_r_a].mean()) if has_conc else 0.0
        impact['flow_depth'] = float(height_data[arrival_frame, mask_a][in_r_a].mean()) if has_height else 0.0

        # 유동 방향 분석 (도달 시점)
        arrived_vx = vx_a[in_r_a]
        arrived_vy = vy_a[in_r_a]
        arrived_speed = np.sqrt(arrived_vx**2 + arrived_vy**2)
        moving_mask = arrived_speed > 0.1
        if moving_mask.any():
            mean_vx_a = float(np.mean(arrived_vx[moving_mask]))
            mean_vy_a = float(np.mean(arrived_vy[moving_mask]))
        else:
            mean_vx_a = float(np.mean(arrived_vx))
            mean_vy_a = float(np.mean(arrived_vy))
        impact['flow_bearing'] = float(np.degrees(np.arctan2(mean_vx_a, mean_vy_a)) % 360)
        init_to_target_x = target_x_local - (init_x - x_min)
        init_to_target_y = target_y_local - (init_y - y_min)
        impact['target_bearing'] = float(np.degrees(np.arctan2(init_to_target_x, init_to_target_y)) % 360)

        if peak_speed_frame is not None:
            mask_p = ~np.isnan(x_data[peak_speed_frame])
            px_p, py_p = x_data[peak_speed_frame, mask_p], y_data[peak_speed_frame, mask_p]
            vx_p, vy_p = vx_data[peak_speed_frame, mask_p], vy_data[peak_speed_frame, mask_p]
            dist_p = np.sqrt((px_p - target_x_local)**2 + (py_p - target_y_local)**2)
            in_r_p = dist_p <= radius
            speeds_p = np.sqrt(vx_p[in_r_p]**2 + vy_p[in_r_p]**2)
            impact['peak_time'] = float(times[peak_speed_frame])
            impact['peak_speed'] = float(speeds_p.max())
            impact['peak_mean_speed'] = float(speeds_p.mean())
            impact['peak_pressure'] = float(rho_debris * speeds_p.max()**2 / 2 / 1000)
            impact['peak_n_particles'] = int(in_r_p.sum())
            impact['peak_concentration'] = float(conc_data[peak_speed_frame, mask_p][in_r_p].mean()) if has_conc else 0.0
            impact['peak_flow_depth'] = float(height_data[peak_speed_frame, mask_p][in_r_p].mean()) if has_height else 0.0

        last_frame = len(times) - 1
        mask_f = ~np.isnan(x_data[last_frame])
        if mask_f.any():
            px_f, py_f = x_data[last_frame, mask_f], y_data[last_frame, mask_f]
            dist_f = np.sqrt((px_f - target_x_local)**2 + (py_f - target_y_local)**2)
            in_r_f = dist_f <= radius
            if in_r_f.any():
                speeds_f = np.sqrt(vx_data[last_frame, mask_f][in_r_f]**2 + vy_data[last_frame, mask_f][in_r_f]**2)
                impact['final_speed'] = float(speeds_f.mean())
                impact['final_n_particles'] = int(in_r_f.sum())
                impact['final_flow_depth'] = float(height_data[last_frame, mask_f][in_r_f].mean()) if has_height else 0.0
            else:
                impact['final_speed'] = 0.0
                impact['final_n_particles'] = 0
                impact['final_flow_depth'] = 0.0
    else:
        impact['arrival_time'] = None
        impact['head_speed'] = 0.0
        impact['front_mean_speed'] = 0.0
        impact['arrival_pressure'] = 0.0
        impact['front_concentration'] = 0.0
        impact['flow_depth'] = 0.0
        impact['n_particles_arrived'] = 0
        impact['closest_distance'] = min_closest_dist if min_closest_dist < float('inf') else None

        # 유동 방향 분석 - closest_frame 시점의 선두 입자들 기준
        if closest_frame is not None:
            impact['closest_time'] = float(times[closest_frame])

            # 1차 판정: 접근/이탈 - 최근접 시점 이후 거리가 벌어졌는지 확인
            receding = False
            last_frame_idx = len(times) - 1
            if closest_frame < last_frame_idx:
                mask_end = ~np.isnan(x_data[last_frame_idx])
                if mask_end.any():
                    px_end = x_data[last_frame_idx, mask_end]
                    py_end = y_data[last_frame_idx, mask_end]
                    dist_end_min = float(np.sqrt((px_end - target_x_local)**2 + (py_end - target_y_local)**2).min())
                    if dist_end_min > min_closest_dist + 20:
                        receding = True

            mask_c = ~np.isnan(x_data[closest_frame])
            if mask_c.any():
                px_c = x_data[closest_frame, mask_c]
                py_c = y_data[closest_frame, mask_c]
                vx_c = vx_data[closest_frame, mask_c]
                vy_c = vy_data[closest_frame, mask_c]

                # 타겟에 가장 가까운 상위 10% 입자 선택
                dists_c = np.sqrt((px_c - target_x_local)**2 + (py_c - target_y_local)**2)
                n_front = max(1, int(len(dists_c) * 0.1))
                front_idx = np.argsort(dists_c)[:n_front]

                # 선두 입자 → 타겟 방향 벡터
                to_target_x = target_x_local - px_c[front_idx]
                to_target_y = target_y_local - py_c[front_idx]
                to_target_mag = np.sqrt(to_target_x**2 + to_target_y**2)
                to_target_mag = np.maximum(to_target_mag, 1e-6)

                # 선두 입자 속도 벡터
                front_vx = vx_c[front_idx]
                front_vy = vy_c[front_idx]
                front_speed = np.sqrt(front_vx**2 + front_vy**2)

                # 타겟 방향 성분 (dot product → cosine)
                cos_angle = (front_vx * to_target_x + front_vy * to_target_y) / (
                    np.maximum(front_speed, 1e-6) * to_target_mag)

                avg_cos = float(np.mean(cos_angle))
                pct_toward = float((cos_angle > 0.3).sum() / len(cos_angle) * 100)

                # 방향 판정 - 이탈 우선, pct_toward 동시 검증
                if receding:
                    flow_relation = "타겟 인근을 통과 후 이탈 중"
                elif avg_cos > 0.5 and pct_toward > 30:
                    flow_relation = "타겟 방향으로 직접 이동 중"
                elif avg_cos > 0.0 and pct_toward > 10:
                    flow_relation = "타겟 방향으로 비스듬히 이동 중"
                elif avg_cos > -0.3:
                    flow_relation = "타겟 측면을 지나가는 중"
                else:
                    flow_relation = "타겟 반대 방향으로 이동"

                impact['flow_direction'] = flow_relation
                impact['flow_toward_pct'] = pct_toward

    return impact


def _analyze_area_target(target, times, x_data, y_data, vx_data, vy_data,
                          conc_data, height_data, x_min, y_min, rho_debris,
                          init_x=0, init_y=0):
    """영역(area) 타겟 분석 - bounding box 방식"""
    name = target.get('name', '미상')
    address = target.get('address', '')
    bldg_type = target.get('structure_type', target.get('type', 'mixed'))
    bbox = target.get('bbox', [0, 0, 0, 0])  # [x_min, y_min, x_max, y_max] EPSG:5186

    has_conc = conc_data is not None
    has_height = height_data is not None

    # bbox → 로컬 좌표
    bx_min = bbox[0] - x_min
    by_min = bbox[1] - y_min
    bx_max = bbox[2] - x_min
    by_max = bbox[3] - y_min
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # m²

    arrival_frame = None
    peak_speed_frame = None
    peak_speed_val = 0.0
    max_particles_in_bbox = 0
    max_particles_frame = None

    # 최근접 거리 추적 (미도달 시 사용)
    min_closest_dist = float('inf')
    closest_frame = None

    for i in range(len(times)):
        mask = ~np.isnan(x_data[i])
        if not mask.any():
            continue
        px = x_data[i, mask]
        py = y_data[i, mask]

        # bbox 내 입자 체크
        in_bbox = (px >= bx_min) & (px <= bx_max) & (py >= by_min) & (py <= by_max)

        if in_bbox.any():
            if arrival_frame is None:
                arrival_frame = i

            n_in = int(in_bbox.sum())
            if n_in > max_particles_in_bbox:
                max_particles_in_bbox = n_in
                max_particles_frame = i

            vx = vx_data[i, mask][in_bbox]
            vy = vy_data[i, mask][in_bbox]
            speeds = np.sqrt(vx**2 + vy**2)
            frame_max_speed = float(speeds.max())
            if frame_max_speed > peak_speed_val:
                peak_speed_val = frame_max_speed
                peak_speed_frame = i
        else:
            # bbox까지 최근접 거리 계산
            dx = np.maximum(bx_min - px, np.maximum(0, px - bx_max))
            dy = np.maximum(by_min - py, np.maximum(0, py - by_max))
            dists = np.sqrt(dx**2 + dy**2)
            frame_min_dist = float(dists.min())
            if frame_min_dist < min_closest_dist:
                min_closest_dist = frame_min_dist
                closest_frame = i

    impact = {
        'name': name, 'address': address, 'type': bldg_type,
        'target_type': 'area',
        'bbox': bbox, 'bbox_area': bbox_area,
        'reached': arrival_frame is not None,
    }

    if arrival_frame is not None:
        # === 도달 시점 분석 ===
        mask_a = ~np.isnan(x_data[arrival_frame])
        px_a, py_a = x_data[arrival_frame, mask_a], y_data[arrival_frame, mask_a]
        in_bbox_a = (px_a >= bx_min) & (px_a <= bx_max) & (py_a >= by_min) & (py_a <= by_max)
        vx_a = vx_data[arrival_frame, mask_a][in_bbox_a]
        vy_a = vy_data[arrival_frame, mask_a][in_bbox_a]
        speeds_a = np.sqrt(vx_a**2 + vy_a**2)

        impact['arrival_time'] = float(times[arrival_frame])
        impact['head_speed'] = float(speeds_a.max())
        impact['front_mean_speed'] = float(speeds_a.mean())
        impact['arrival_pressure'] = float(rho_debris * speeds_a.max()**2 / 2 / 1000)
        impact['n_particles_arrived'] = int(in_bbox_a.sum())
        impact['front_concentration'] = float(conc_data[arrival_frame, mask_a][in_bbox_a].mean()) if has_conc else 0.0
        impact['flow_depth'] = float(height_data[arrival_frame, mask_a][in_bbox_a].mean()) if has_height else 0.0

        # 유동 방향 분석 (도달 시점)
        arrived_speed_a = np.sqrt(vx_a**2 + vy_a**2)
        moving_mask_a = arrived_speed_a > 0.1
        if moving_mask_a.any():
            mean_vx_a = float(np.mean(vx_a[moving_mask_a]))
            mean_vy_a = float(np.mean(vy_a[moving_mask_a]))
        else:
            mean_vx_a = float(np.mean(vx_a))
            mean_vy_a = float(np.mean(vy_a))
        impact['flow_bearing'] = float(np.degrees(np.arctan2(mean_vx_a, mean_vy_a)) % 360)
        target_cx = (bx_min + bx_max) / 2
        target_cy = (by_min + by_max) / 2
        init_to_target_x = target_cx - (init_x - x_min)
        init_to_target_y = target_cy - (init_y - y_min)
        impact['target_bearing'] = float(np.degrees(np.arctan2(init_to_target_x, init_to_target_y)) % 360)

        # === 최대 충격 시점 ===
        if peak_speed_frame is not None:
            mask_p = ~np.isnan(x_data[peak_speed_frame])
            px_p, py_p = x_data[peak_speed_frame, mask_p], y_data[peak_speed_frame, mask_p]
            in_bbox_p = (px_p >= bx_min) & (px_p <= bx_max) & (py_p >= by_min) & (py_p <= by_max)
            speeds_p = np.sqrt(vx_data[peak_speed_frame, mask_p][in_bbox_p]**2 +
                                vy_data[peak_speed_frame, mask_p][in_bbox_p]**2)
            impact['peak_time'] = float(times[peak_speed_frame])
            impact['peak_speed'] = float(speeds_p.max())
            impact['peak_mean_speed'] = float(speeds_p.mean())
            impact['peak_pressure'] = float(rho_debris * speeds_p.max()**2 / 2 / 1000)
            impact['peak_n_particles'] = int(in_bbox_p.sum())
            impact['peak_concentration'] = float(conc_data[peak_speed_frame, mask_p][in_bbox_p].mean()) if has_conc else 0.0
            impact['peak_flow_depth'] = float(height_data[peak_speed_frame, mask_p][in_bbox_p].mean()) if has_height else 0.0

        # === 최대 범람 시점 (가장 많은 입자 진입) ===
        if max_particles_frame is not None:
            impact['max_coverage_time'] = float(times[max_particles_frame])
            impact['max_coverage_particles'] = max_particles_in_bbox
            n_total = int((~np.isnan(x_data[max_particles_frame])).sum())
            impact['max_coverage_ratio'] = max_particles_in_bbox / max(n_total, 1)

        # === 최종 상태 ===
        last_frame = len(times) - 1
        mask_f = ~np.isnan(x_data[last_frame])
        if mask_f.any():
            px_f, py_f = x_data[last_frame, mask_f], y_data[last_frame, mask_f]
            in_bbox_f = (px_f >= bx_min) & (px_f <= bx_max) & (py_f >= by_min) & (py_f <= by_max)
            if in_bbox_f.any():
                speeds_f = np.sqrt(vx_data[last_frame, mask_f][in_bbox_f]**2 +
                                    vy_data[last_frame, mask_f][in_bbox_f]**2)
                impact['final_speed'] = float(speeds_f.mean())
                impact['final_n_particles'] = int(in_bbox_f.sum())
                impact['final_flow_depth'] = float(height_data[last_frame, mask_f][in_bbox_f].mean()) if has_height else 0.0
            else:
                impact['final_speed'] = 0.0
                impact['final_n_particles'] = 0
                impact['final_flow_depth'] = 0.0

    else:
        # 미도달 - 최근접 거리 및 도달 가능성 추정
        impact['arrival_time'] = None
        impact['head_speed'] = 0.0
        impact['front_mean_speed'] = 0.0
        impact['arrival_pressure'] = 0.0
        impact['front_concentration'] = 0.0
        impact['flow_depth'] = 0.0
        impact['n_particles_arrived'] = 0
        impact['closest_distance'] = min_closest_dist if min_closest_dist < float('inf') else None

        # 도달 가능성 추정: 마지막 프레임의 속도와 최근접 거리로 판단
        if closest_frame is not None:
            last_frame = len(times) - 1
            mask_last = ~np.isnan(x_data[last_frame])
            if mask_last.any():
                speeds_last = np.sqrt(vx_data[last_frame, mask_last]**2 + vy_data[last_frame, mask_last]**2)
                avg_speed_last = float(speeds_last.mean())
                max_speed_last = float(speeds_last.max())
                impact['final_avg_speed'] = avg_speed_last
                impact['final_max_speed'] = max_speed_last

                # 감속 추세 분석 (마지막 20% 구간)
                t80_idx = int(len(times) * 0.8)
                mask_80 = ~np.isnan(x_data[t80_idx])
                if mask_80.any():
                    speeds_80 = np.sqrt(vx_data[t80_idx, mask_80]**2 + vy_data[t80_idx, mask_80]**2)
                    avg_speed_80 = float(speeds_80.mean())
                    dt_segment = times[last_frame] - times[t80_idx]
                    if dt_segment > 0:
                        decel = (avg_speed_80 - avg_speed_last) / dt_segment  # m/s²
                        impact['deceleration'] = decel
                        # 등감속 가정 시 추가 이동 가능 거리: v²/(2a)
                        if decel > 0.001:
                            remaining_dist = avg_speed_last**2 / (2 * decel)
                            impact['estimated_remaining_travel'] = remaining_dist
                            impact['will_likely_reach'] = remaining_dist > min_closest_dist
                            # 도달 예상 시간 추정 (v/a 초 후 정지)
                            if remaining_dist > min_closest_dist and avg_speed_last > 0.1:
                                # v² = v0² - 2*a*d → t = (v0 - sqrt(v0²-2ad))/a
                                v0 = avg_speed_last
                                d = min_closest_dist
                                discriminant = v0**2 - 2 * decel * d
                                if discriminant > 0:
                                    t_reach = (v0 - np.sqrt(discriminant)) / decel
                                    impact['estimated_arrival_time'] = float(times[last_frame] + t_reach)
                        else:
                            # 거의 감속 없음 → 등속 가정
                            impact['estimated_remaining_travel'] = float('inf')
                            impact['will_likely_reach'] = True
                            if avg_speed_last > 0.1:
                                impact['estimated_arrival_time'] = float(times[last_frame] + min_closest_dist / avg_speed_last)

            # 유동 방향 분석 - closest_frame 시점의 선두 입자들 기준
            impact['closest_time'] = float(times[closest_frame])

            # 1차 판정: 접근/이탈 - 최근접 시점 이후 거리가 벌어졌는지 확인
            receding = False
            last_frame_idx = len(times) - 1
            if closest_frame < last_frame_idx:
                mask_end = ~np.isnan(x_data[last_frame_idx])
                if mask_end.any():
                    px_end = x_data[last_frame_idx, mask_end]
                    py_end = y_data[last_frame_idx, mask_end]
                    dx_end = np.maximum(bx_min - px_end, np.maximum(0, px_end - bx_max))
                    dy_end = np.maximum(by_min - py_end, np.maximum(0, py_end - by_max))
                    dist_end_min = float(np.sqrt(dx_end**2 + dy_end**2).min())
                    if dist_end_min > min_closest_dist + 20:
                        receding = True

            mask_c = ~np.isnan(x_data[closest_frame])
            if mask_c.any():
                px_c = x_data[closest_frame, mask_c]
                py_c = y_data[closest_frame, mask_c]
                vx_c = vx_data[closest_frame, mask_c]
                vy_c = vy_data[closest_frame, mask_c]

                # 타겟 중심 (로컬 좌표)
                target_cx = (bx_min + bx_max) / 2
                target_cy = (by_min + by_max) / 2

                # 타겟에 가장 가까운 상위 10% 입자 선택
                dx_c = np.maximum(bx_min - px_c, np.maximum(0, px_c - bx_max))
                dy_c = np.maximum(by_min - py_c, np.maximum(0, py_c - by_max))
                dists_c = np.sqrt(dx_c**2 + dy_c**2)
                n_front = max(1, int(len(dists_c) * 0.1))
                front_idx = np.argsort(dists_c)[:n_front]

                # 선두 입자 → 타겟 방향 벡터
                to_target_x = target_cx - px_c[front_idx]
                to_target_y = target_cy - py_c[front_idx]
                to_target_mag = np.sqrt(to_target_x**2 + to_target_y**2)
                to_target_mag = np.maximum(to_target_mag, 1e-6)

                # 선두 입자 속도 벡터
                front_vx = vx_c[front_idx]
                front_vy = vy_c[front_idx]
                front_speed = np.sqrt(front_vx**2 + front_vy**2)

                # 타겟 방향 성분 (dot product → cosine)
                cos_angle = (front_vx * to_target_x + front_vy * to_target_y) / (
                    np.maximum(front_speed, 1e-6) * to_target_mag)

                avg_cos = float(np.mean(cos_angle))
                pct_toward = float((cos_angle > 0.3).sum() / len(cos_angle) * 100)

                # 방향 판정 - 이탈 우선, pct_toward 동시 검증
                if receding:
                    flow_relation = "타겟 인근을 통과 후 이탈 중"
                elif avg_cos > 0.5 and pct_toward > 30:
                    flow_relation = "타겟 방향으로 직접 이동 중"
                elif avg_cos > 0.0 and pct_toward > 10:
                    flow_relation = "타겟 방향으로 비스듬히 이동 중"
                elif avg_cos > -0.3:
                    flow_relation = "타겟 측면을 지나가는 중"
                else:
                    flow_relation = "타겟 반대 방향으로 이동"

                impact['flow_direction'] = flow_relation
                impact['flow_toward_pct'] = pct_toward

    return impact


def analyze_impact_at_targets(data, config_info: dict) -> list:
    """대상별 충격 분석 - 건물(점) 또는 영역(bbox) 방식 지원

    Returns:
        list[dict]: 각 대상에 대한 충격 분석 결과
    """
    targets = []
    if 'simulation' in config_info and 'targets' in config_info['simulation']:
        targets = config_info['simulation']['targets']

    if not targets:
        return []

    times = data['times']
    x_data = data['x']
    y_data = data['y']
    vx_data = data['vx']
    vy_data = data['vy']

    x_min = float(data['x_min'])
    y_min = float(data['y_min'])
    init_x = float(data['init_x']) if 'init_x' in data.files else 0
    init_y = float(data['init_y']) if 'init_y' in data.files else 0

    has_conc = 'concentration' in data.files
    conc_data = data['concentration'] if has_conc else None
    has_height = 'height' in data.files
    height_data = data['height'] if has_height else None

    rho_debris = 2000  # kg/m³

    results = []
    for target in targets:
        target_type = target.get('target_type', 'building')
        if target_type == 'area':
            result = _analyze_area_target(
                target, times, x_data, y_data, vx_data, vy_data,
                conc_data, height_data, x_min, y_min, rho_debris,
                init_x, init_y)
        else:
            result = _analyze_building_target(
                target, times, x_data, y_data, vx_data, vy_data,
                conc_data, height_data, x_min, y_min, rho_debris,
                init_x, init_y)
        results.append(result)

    return results


def format_impact_at_targets(target_impacts: list) -> str:
    """건물별 충격 분석 결과를 마크다운 테이블로 포맷팅"""
    if not target_impacts:
        return "대상 건물 정보 없음"

    # 건물 유형별 피해 판정
    def assess_damage(bldg_type: str, pressure_kpa: float) -> str:
        if bldg_type == 'wood':
            if pressure_kpa >= 20:
                return "붕괴"
            elif pressure_kpa >= 10:
                return "심각손상"
            elif pressure_kpa >= 5:
                return "중간손상"
            else:
                return "경미"
        elif bldg_type == 'masonry':
            if pressure_kpa >= 20:
                return "붕괴"
            elif pressure_kpa >= 10:
                return "심각손상"
            elif pressure_kpa >= 5:
                return "중간손상"
            else:
                return "경미"
        elif bldg_type == 'RC':
            if pressure_kpa >= 100:
                return "심각손상"
            elif pressure_kpa >= 30:
                return "중간손상"
            elif pressure_kpa >= 10:
                return "경미손상"
            else:
                return "경미"
        else:
            if pressure_kpa >= 20:
                return "심각"
            elif pressure_kpa >= 10:
                return "중간"
            else:
                return "경미"

    lines = [
        "| 대상명 | 유형 | 도달시간 | 도달속도 | 충격압력 | 농도 | 유동심 | 판정 |",
        "|--------|------|---------|---------|---------|------|-------|------|"
    ]

    for t in target_impacts:
        ttype = "영역" if t.get('target_type') == 'area' else "건물"
        if not t['reached']:
            # 미도달 시 추가 정보
            extra = ""
            if t.get('closest_distance') is not None:
                extra = f" (최근접 {t['closest_distance']:.0f}m"
                if t.get('will_likely_reach') is True:
                    extra += f", 도달 예상 ~{t.get('estimated_arrival_time', 0):.0f}초"
                elif t.get('will_likely_reach') is False:
                    extra += ", 도달 전 정지 예상"
                extra += ")"
            lines.append(f"| {t['name']} | {ttype} | 미도달{extra} | - | - | - | - | 안전 |")
        else:
            pressure = t.get('peak_pressure', t['arrival_pressure'])
            speed = t.get('peak_speed', t['head_speed'])
            damage = assess_damage(t['type'], pressure)
            lines.append(
                f"| {t['name']} | {ttype} | {t['arrival_time']:.1f}초 "
                f"| {speed:.1f} m/s "
                f"| {pressure:.1f} kPa "
                f"| {t['front_concentration']:.0%} "
                f"| {t['flow_depth']:.2f} m "
                f"| {damage} |"
            )

    return "\n".join(lines)


def load_config_info(work_dir: Path) -> dict:
    """설정 파일에서 정보 로드"""
    info = {}

    # simulation_config.yaml
    sim_config_path = work_dir / 'simulation_config.yaml'
    if sim_config_path.exists():
        with open(sim_config_path, 'r', encoding='utf-8') as f:
            info['simulation'] = yaml.safe_load(f)

    # visualization_config.yaml
    viz_config_path = work_dir / 'visualization_config.yaml'
    if viz_config_path.exists():
        with open(viz_config_path, 'r', encoding='utf-8') as f:
            info['visualization'] = yaml.safe_load(f)

    # report_config.yaml (LLM 설정 포함)
    report_config_path = work_dir / 'report_config.yaml'
    if report_config_path.exists():
        with open(report_config_path, 'r', encoding='utf-8') as f:
            info['report'] = yaml.safe_load(f)

    return info


def format_velocity_by_distance(velocity_data: dict) -> str:
    """이동거리별 속도 데이터를 포맷팅"""
    if not velocity_data or 'velocity_at_distance' not in velocity_data:
        return "데이터 없음"

    lines = ["| 도달거리 | 도달시간 | Head 속도 | 전방부 평균속도 | 전방부 농도 |",
             "|---------|---------|----------|---------------|------------|"]

    for dist in sorted(velocity_data['velocity_at_distance'].keys()):
        v = velocity_data['velocity_at_distance'][dist]
        if v is None:
            lines.append(f"| {dist}m | 미도달 | - | - | - |")
        else:
            # 충격압력 계산 (kPa)
            impact = 2000 * (v['head_speed'] ** 2) / 2 / 1000
            lines.append(f"| {dist}m | {v['time']:.1f}초 | {v['head_speed']:.1f} m/s ({impact:.0f}kPa) | {v['front_mean_speed']:.1f} m/s | {v['front_concentration']:.0%} |")

    return "\n".join(lines)


def get_llm_analysis(stats: SimulationStats, data, config_info: dict,
                     velocity_by_distance: dict = None,
                     target_impacts: list = None) -> Optional[str]:
    """Anthropic API를 사용한 LLM 분석"""
    if not HAS_ANTHROPIC:
        print("  [WARN] anthropic 패키지가 설치되지 않음. pip install anthropic")
        return None

    # API 키 로드
    api_key = None
    if 'report' in config_info and 'llm' in config_info['report']:
        api_key = config_info['report']['llm'].get('api_key')

    if not api_key:
        print("  [WARN] API 키가 설정되지 않음. report_config.yaml에 llm.api_key 설정 필요")
        return None

    # 모델 설정
    model = "claude-sonnet-4-20250514"
    if 'report' in config_info and 'llm' in config_info['report']:
        model = config_info['report']['llm'].get('model', model)

    # 프롬프트 데이터 준비
    terrain = data['terrain']
    project_name = "Landslide"
    if 'simulation' in config_info and 'project' in config_info['simulation']:
        project_name = config_info['simulation']['project'].get('name', project_name)

    # 시뮬레이션 충분성 판단
    is_short_simulation = stats.duration < 10.0
    is_still_accelerating = stats.max_speed_time >= stats.duration * 0.8

    # 토석류 위험도 분류 기준 (일본 국토교통성 기준 참고)
    # 속도 기준: <2m/s 저위험, 2-5m/s 중위험, 5-10m/s 고위험, >10m/s 극고위험
    # 농도 기준: <0.3 묽은 이류, 0.3-0.5 토석류, >0.5 고농도 토석류
    speed_risk = "극고위험" if stats.max_speed > 10 else "고위험" if stats.max_speed > 5 else "중위험" if stats.max_speed > 2 else "저위험"
    conc_type = "고농도 토석류" if stats.concentration_final > 0.5 else "토석류" if stats.concentration_final > 0.3 else "묽은 이류"

    # 충격력 추정 (ρ * v² / 2, kPa)
    rho_debris = 2000  # kg/m³ (토석류 밀도)
    impact_pressure = rho_debris * (stats.max_speed ** 2) / 2 / 1000  # kPa

    prompt_data = f"""
# 산사태(토석류) 재해 위험 분석 보고서 작성 요청

## 대상 독자
서울특별시 재난안전 담당 공무원 (비전문가도 이해 가능하게)

## 지역 특성
- **서울시 도심 지역**: 주거지역(아파트, 단독주택), 상업시설, 도로, 지하철역, 학교 등 도시 인프라 밀집
- 농경지 없음, 도시 환경 기준으로 피해 분석 필요

## 프로젝트: {project_name}

## 시뮬레이션 개요
- 시뮬레이션 시간: {stats.duration:.1f}초
- 분석 대상 면적: {terrain.shape[1] * float(data['cell_size']):.0f}m x {terrain.shape[0] * float(data['cell_size']):.0f}m
- 고도 범위: {terrain.min():.1f} ~ {terrain.max():.1f}m (고도차 {terrain.max() - terrain.min():.0f}m)

## 토석류 특성 분석 결과

### 1. 유속 분석
- **최대 유속**: {stats.max_speed:.2f} m/s ({stats.max_speed * 3.6:.1f} km/h)
- **최대 유속 도달 시점**: {stats.max_speed_time:.1f}초
- **종료 시점 평균 유속**: {stats.mean_speed_final:.2f} m/s
- **유속 기반 위험등급**: {speed_risk}
  - (기준: <2m/s 저위험, 2-5m/s 중위험, 5-10m/s 고위험, >10m/s 극고위험)

### 2. 토사 농도 분석
- **초기 토사 농도**: {stats.concentration_initial:.0%}
- **종료 시점 평균 농도**: {stats.concentration_final:.0%}
- **토석류 유형**: {conc_type}
  - (기준: <30% 묽은 이류, 30-50% 토석류, >50% 고농도 토석류)

### 3. 충격력 추정
- **추정 충격압력**: {impact_pressure:.1f} kPa (= {impact_pressure * 10:.0f} tf/m²)
- (참고: 목조건물 붕괴 ~20kPa, 철근콘크리트 손상 ~100kPa, 차량 전복 ~10kPa)

### 4. 도달 범위
- **최대 도달 거리(Head)**: {stats.runout_head_final:.1f}m
- **평균 도달 거리(Centroid)**: {stats.runout_centroid_final:.1f}m

### 5. 대상별 충격 분석 (핵심 데이터)
{format_impact_at_targets(target_impacts) if target_impacts else "대상 건물/영역 정보 없음"}
"""
    # 대상별 상세 정보 추가
    if target_impacts:
        for t in target_impacts:
            ttype_str = "영역" if t.get('target_type') == 'area' else "건물"
            type_labels = {'RC': '철근콘크리트', 'wood': '목조', 'masonry': '조적조', 'mixed': '혼합구조'}
            type_label = type_labels.get(t['type'], t['type'])
            prompt_data += f"""
#### {t['name']} ({ttype_str})
- **주소**: {t['address']}
- **구조 유형**: {t['type']} ({type_label})
"""
            if t.get('target_type') == 'area' and 'bbox' in t:
                bbox = t['bbox']
                prompt_data += f"- **영역 범위**: ({bbox[0]},{bbox[1]}) ~ ({bbox[2]},{bbox[3]}) EPSG:5186, 면적 {t.get('bbox_area', 0):.0f}m²\n"
            elif 'coordinates' in t:
                prompt_data += f"- **좌표**: ({t['coordinates'][0]}, {t['coordinates'][1]}) EPSG:5186\n"

            if t['reached']:
                prompt_data += f"""- **도달 여부**: 도달 (최초 도달: {t['arrival_time']:.1f}초)
- **도달 시 Head 속도**: {t['head_speed']:.1f} m/s
- **도달 시 전방부 평균속도**: {t['front_mean_speed']:.1f} m/s
- **도달 시 충격압력**: {t['arrival_pressure']:.1f} kPa
- **도달 시 토사 농도**: {t['front_concentration']:.0%}
- **도달 시 유동심**: {t['flow_depth']:.2f} m
- **도달 입자 수**: {t['n_particles_arrived']}개
"""
                if 'peak_pressure' in t:
                    prompt_data += f"""- **최대 충격 시점**: {t['peak_time']:.1f}초
- **최대 충격 속도**: {t['peak_speed']:.1f} m/s
- **최대 충격압력**: {t['peak_pressure']:.1f} kPa
- **최대 충격 시 유동심**: {t.get('peak_flow_depth', 0):.2f} m
"""
                if 'max_coverage_particles' in t:
                    prompt_data += f"""- **최대 범람 시점**: {t['max_coverage_time']:.1f}초 ({t['max_coverage_particles']}개 입자, 전체의 {t['max_coverage_ratio']:.0%})
"""
                if 'final_speed' in t:
                    prompt_data += f"""- **최종 상태 평균 속도**: {t['final_speed']:.1f} m/s
- **최종 상태 잔류 입자 수**: {t['final_n_particles']}개
- **최종 상태 유동심**: {t.get('final_flow_depth', 0):.2f} m
"""
                if t.get('flow_bearing') is not None:
                    prompt_data += f"- **유동 방위**: {t['flow_bearing']:.0f}° (시작점→타겟: {t.get('target_bearing', 0):.0f}°)\n"
            else:
                prompt_data += "- **도달 여부**: 미도달 (시뮬레이션 시간 내 영향권 도달 안 함)\n"
                if t.get('closest_distance') is not None:
                    prompt_data += f"- **최근접 거리**: {t['closest_distance']:.0f}m\n"
                if t.get('final_avg_speed') is not None:
                    prompt_data += f"- **시뮬레이션 종료 시 평균 속도**: {t['final_avg_speed']:.2f} m/s\n"
                if t.get('deceleration') is not None:
                    prompt_data += f"- **감속률**: {t['deceleration']:.3f} m/s² (마지막 20% 구간)\n"
                if t.get('estimated_remaining_travel') is not None:
                    est = t['estimated_remaining_travel']
                    prompt_data += f"- **등감속 가정 추가 이동 가능 거리**: {est:.0f}m\n"
                if t.get('will_likely_reach') is not None:
                    if t['will_likely_reach']:
                        prompt_data += f"- **도달 가능성**: 도달 예상 (추정 도달시간 ~{t.get('estimated_arrival_time', 0):.0f}초)\n"
                    else:
                        prompt_data += "- **도달 가능성**: 도달 전 정지 예상 (추가 이동 거리 < 최근접 거리)\n"
                if t.get('flow_direction'):
                    prompt_data += f"- **유동 방향 판정**: {t['flow_direction']}\n"
                    prompt_data += f"- **타겟 방향 이동 비율**: {t['flow_toward_pct']:.0f}%\n"
                    if t.get('closest_time') is not None:
                        prompt_data += f"- **최근접 도달 시점**: {t['closest_time']:.1f}초 (시뮬레이션 종료: {stats.duration:.0f}초)\n"

    prompt_data += f"""
### 6. 이동거리별 속도/농도 분석 (보조 데이터)
{format_velocity_by_distance(velocity_by_distance) if velocity_by_distance else "데이터 없음"}

### 건물 피해 판정 기준 (참고)
- **목조/조적조**: 붕괴 ~20kPa, 심각손상 ~10kPa
- **철근콘크리트(RC)**: 심각손상 ~100kPa, 중간손상 ~30kPa, 경미손상 ~10kPa
- **차량 전복**: ~10kPa
- **보행자 사망**: ~5kPa

## 시뮬레이션 신뢰성 정보
- 시뮬레이션 길이: {"**짧음** (전체 산사태 거동 미반영)" if is_short_simulation else "적정 (120초 이상)"}
- 종료 시점 상태: {"**가속 진행 중** (최종 결과 아님)" if is_still_accelerating else "감속/안정화 추세"}

## 분석 요청
**지방자치단체 재해 예방 담당자를 위한 분석 보고서**를 작성해주세요.

### 필수 포함 내용:

1. **위험도 평가 요약** (2-3문단)
   - 토석류 유속({stats.max_speed:.1f}m/s)과 농도({stats.concentration_final:.0%}) 기반 종합 위험 평가
   - 충격압력 기준 구조물 피해 예상
   - 도달거리({stats.runout_head_final:.0f}m) 기준 영향권 평가

2. **대상별 피해 예상** (위 "대상별 충격 분석" 데이터 활용 필수!)
   - **중요**: 각 건물/영역의 이름과 주소를 명시적으로 언급하여 분석
   - 예(건물): "서울대학교 제2공학관(302동)에 XX초 후 토석류 도달, 충격압력 YY kPa로 OO 수준 피해 예상"
   - 예(영역): "구룡마을 영역에 토석류가 XX초 후 진입, 최대 N개 입자 도달로 광범위 침수 위험"
   - 각 대상별로:
     * 도달 시간 및 대피 가능 시간
     * 충격압력 기반 구조물 피해 수준 (위 판정 기준 참고)
     * 토사 농도와 유동심 기반 매몰 위험
     * 인명피해 가능성
   - 미도달 대상은: 최근접 거리, 도달 가능성 추정, 감속 추세를 반드시 언급
   - **중요**: 유동 방향 데이터를 반드시 활용하세요. "타겟 방향으로 직접 이동 중"이면 실제 위협,
     "반대 방향으로 이동"이면 현 시나리오에서는 안전하다고 명시.
     단순히 거리가 가깝다고 위험하다고 판단하면 안 됩니다.
   - 이동거리별 데이터도 보조적으로 활용하여 일반 지역 피해 분석

3. **대피 권고사항**
   - 각 건물별 대피 시간 (도달 시간 기준)
   - 위험 반경 설정 (도달거리 + 안전여유)
   - 대피 불가 지역 설정

4. **재해 예방 권고사항** (실행 가능한 조치)
   - 즉시 조치 사항 (경보, 대피, 통제)
   - 건물별 맞춤 대책 (보강, 방호벽 등)
   - 중장기 대책 (구조물 보강, 사방댐 등)

5. **데이터 신뢰성 안내**
   {"- **주의**: 본 시뮬레이션은 " + f"{stats.duration:.0f}초" + "간 분석으로 토석류 전체 거동을 반영하지 못할 수 있습니다. 실제 피해 범위는 더 클 수 있으므로 참고 자료로만 활용하시기 바랍니다." if is_short_simulation else "- 시뮬레이션 결과는 적정 수준의 신뢰성을 가집니다."}

**작성 지침:**
- 전문 용어 최소화, 쉬운 표현 사용
- 구체적인 수치와 함께 그 의미를 **일반인이 이해할 수 있도록** 상세히 설명
- **건물 이름과 주소를 반드시 명시**하여 독자가 어떤 건물인지 즉시 파악 가능하도록
- 각 섹션별로 충분히 상세하게 작성 (전체 1500-2000자)
- 위험도 평가는 근거와 함께 구체적으로
- 피해 범위는 실제 피해 유형별로 상세히 (속도/농도 수치 인용)
- 권고사항은 실행 가능한 구체적 조치로 5개 이상
- Markdown 형식, 한국어로 작성
"""

    system_prompt = """당신은 대한민국 한국지질자원연구원 소속 산사태 재해 위험 분석 전문가입니다.

핵심 원칙:
1. **데이터 기반 판단만**: 제공된 시뮬레이션 수치 데이터에만 근거하여 분석합니다.
   데이터에 없는 내용은 절대 추측하거나 지어내지 마십시오.
2. **유동 방향 중시**: 토석류가 타겟 방향으로 이동하는지, 반대/측면으로 이동하는지가
   피해 판정의 핵심입니다. 단순 거리만으로 위험도를 판단하지 마십시오.
3. **보수적 판단**: 불확실할 경우 안전 측으로 판단하되, 데이터가 명확히
   "미도달" 또는 "반대 방향"을 가리키면 과장하지 마십시오.
4. **구체적 수치 인용**: 모든 판단에 근거 수치를 반드시 함께 제시합니다.
5. **시뮬레이션 한계 명시**: SPH 시뮬레이션의 불확실성(지형 해상도, 강우 미반영,
   초기조건 민감도 등)을 적절히 언급합니다.

작성 스타일:
- 한국어, 공무원 보고서 형식
- 전문 용어 사용 시 괄호 안에 쉬운 설명 병기
- 위험도는 5단계: 매우높음/높음/보통/낮음/매우낮음"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=4000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt_data}
            ]
        )
        return message.content[0].text
    except Exception as e:
        print(f"  [ERROR] LLM API 호출 실패: {e}")
        return None


def generate_markdown_report(data, stats: SimulationStats, config_info: dict,
                              work_dir: Path, llm_analysis: Optional[str] = None) -> str:
    """Markdown 형식 보고서 생성"""
    lines = []

    # 제목
    project_name = "Landslide Simulation"
    if 'simulation' in config_info and 'project' in config_info['simulation']:
        project_name = config_info['simulation']['project'].get('name', project_name)

    lines.append(f"# {project_name} - Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Work Directory:** `{work_dir}`")
    lines.append("")

    # 1. 시뮬레이션 개요
    lines.append("## 1. Simulation Overview")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Duration | {stats.duration:.1f} s |")
    lines.append(f"| Save Interval | {stats.save_interval:.2f} s |")
    lines.append(f"| Total Frames | {stats.n_frames} |")
    lines.append(f"| Initial Particles | {stats.n_particles_initial} |")
    lines.append(f"| Final Particles | {stats.n_particles_final} |")
    lines.append("")

    # 2. 지형 정보
    terrain = data['terrain']
    cell_size = float(data['cell_size'])
    x_min, y_min = float(data['x_min']), float(data['y_min'])
    ny, nx = terrain.shape

    lines.append("## 2. Terrain Information")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Grid Size | {nx} x {ny} |")
    lines.append(f"| Cell Size | {cell_size:.1f} m |")
    lines.append(f"| X Range (TM) | {x_min:.0f} ~ {x_min + nx*cell_size:.0f} |")
    lines.append(f"| Y Range (TM) | {y_min:.0f} ~ {y_min + ny*cell_size:.0f} |")
    lines.append(f"| Elevation Range | {terrain.min():.1f} ~ {terrain.max():.1f} m |")
    lines.append("")

    # 3. 초기 조건
    init_x, init_y = float(data['init_x']), float(data['init_y'])

    lines.append("## 3. Initial Condition")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Initial Center (TM) | ({init_x:.1f}, {init_y:.1f}) |")

    if 'simulation' in config_info and 'initial_condition' in config_info['simulation']:
        ic = config_info['simulation']['initial_condition']
        lines.append(f"| Type | {ic.get('type', 'N/A')} |")
        lines.append(f"| Radius | {ic.get('radius', 'N/A')} m |")
        lines.append(f"| Thickness | {ic.get('thickness', 'N/A')} m |")
    lines.append("")

    # 4. SPH 파라미터
    lines.append("## 4. SPH Parameters")
    lines.append("")
    lines.append("| Parameter | Value | Description |")
    lines.append("|-----------|-------|-------------|")
    lines.append(f"| h | {float(data['h']):.1f} m | Smoothing length |")
    lines.append(f"| rho0 | {float(data['rho0']):.0f} kg/m³ | Reference density |")
    lines.append(f"| c0 | {float(data['c0']):.0f} m/s | Speed of sound |")
    lines.append(f"| gamma | {float(data['gamma']):.0f} | EOS exponent |")

    if 'simulation' in config_info and 'rheology' in config_info['simulation']:
        rh = config_info['simulation']['rheology']
        lines.append(f"| mu | {rh.get('mu', 'N/A')} Pa·s | Dynamic viscosity |")
        lines.append(f"| tau_y | {rh.get('tau_y', 'N/A')} Pa | Yield stress |")
        lines.append(f"| mu_b | {rh.get('mu_b', 'N/A')} | Basal friction |")
    lines.append("")

    # 5. 침식 파라미터
    if stats.entrainment_enabled:
        lines.append("## 5. Entrainment Parameters (Takahashi 2007)")
        lines.append("")
        lines.append("| Parameter | Value | Description |")
        lines.append("|-----------|-------|-------------|")
        lines.append(f"| C_init | {stats.concentration_initial:.2f} | Initial concentration |")
        lines.append(f"| rho_s | {float(data['rho_s']):.0f} kg/m³ | Solid density |")
        lines.append(f"| rho_w | {float(data['rho_w']):.0f} kg/m³ | Water density |")
        lines.append(f"| phi_bed | {float(data['phi_bed']):.1f}° | Bed friction angle |")
        lines.append("")

    # 6. 결과 요약
    lines.append("## 6. Results Summary")
    lines.append("")

    lines.append("### 6.1 Velocity Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Maximum Speed | {stats.max_speed:.2f} m/s ({stats.max_speed * 3.6:.1f} km/h) |")
    lines.append(f"| Time of Max Speed | {stats.max_speed_time:.2f} s |")
    lines.append(f"| Mean Speed (Final) | {stats.mean_speed_final:.2f} m/s |")
    lines.append("")

    lines.append("### 6.2 Runout Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Runout Head (Final) | {stats.runout_head_final:.1f} m |")
    lines.append(f"| Runout Centroid (Final) | {stats.runout_centroid_final:.1f} m |")
    lines.append(f"| Runout Head (Max) | {stats.runout_head_max:.1f} m |")
    lines.append("")

    if stats.entrainment_enabled:
        lines.append("### 6.3 Entrainment Statistics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Initial Concentration | {stats.concentration_initial:.2f} |")
        lines.append(f"| Final Concentration (Mean) | {stats.concentration_final:.2f} |")
        lines.append("")

    # 7. 출력 파일
    lines.append("## 7. Output Files")
    lines.append("")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| `simulation_results.npz` | Simulation data |")
    lines.append("| `landslide_animation.gif` | 3D animation |")
    lines.append("| `runout_distance.png` | Runout plot |")
    lines.append("| `analysis_report.md` | This report |")
    lines.append("")

    # 8. LLM 분석 (있는 경우)
    if llm_analysis:
        lines.append("## 8. AI Analysis (Claude)")
        lines.append("")
        lines.append(llm_analysis)
        lines.append("")

    # 9. 시간별 데이터 테이블
    lines.append("## 9. Time Series Data")
    lines.append("")
    lines.append("| Time (s) | Particles | Max Speed (m/s) | Runout Head (m) |")
    lines.append("|----------|-----------|-----------------|-----------------|")

    times = data['times']
    n_active = data['n_active']
    x_data, y_data = data['x'], data['y']
    vx_data, vy_data = data['vx'], data['vy']
    init_x_local = float(data['init_x']) - float(data['x_min'])
    init_y_local = float(data['init_y']) - float(data['y_min'])

    for i in range(len(times)):
        mask = ~np.isnan(x_data[i])
        px, py = x_data[i, mask], y_data[i, mask]
        vx, vy = vx_data[i, mask], vy_data[i, mask]

        if len(px) > 0:
            speeds = np.sqrt(vx**2 + vy**2)
            distances = np.sqrt((px - init_x_local)**2 + (py - init_y_local)**2)
            max_spd = speeds.max()
            runout = distances.max()
        else:
            max_spd, runout = 0, 0

        lines.append(f"| {times[i]:.2f} | {n_active[i]} | {max_spd:.2f} | {runout:.1f} |")

    lines.append("")
    lines.append("---")
    lines.append("*Report generated by `generate_report.py`*")

    return "\n".join(lines)


def image_to_base64(image_path: Path) -> Optional[str]:
    """이미지 파일을 base64로 인코딩"""
    if not image_path.exists():
        return None
    with open(image_path, 'rb') as f:
        data = f.read()
    ext = image_path.suffix.lower()
    mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'gif': 'image/gif'}.get(ext[1:], 'image/png')
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def fig_to_base64(fig) -> str:
    """matplotlib figure를 base64로 변환"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def create_initial_condition_plot(data, config_info: dict, satellite_path: Path = None) -> str:
    """초기 조건 시각화 (위성사진 배경)"""
    from PIL import Image

    terrain = data['terrain']
    cell_size = float(data['cell_size'])
    x_min, y_min = float(data['x_min']), float(data['y_min'])
    init_x, init_y = float(data['init_x']), float(data['init_y'])

    # 초기 파티클 위치
    x_data, y_data = data['x'], data['y']
    mask = ~np.isnan(x_data[0])
    px, py = x_data[0, mask], y_data[0, mask]

    ny, nx = terrain.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 좌측: 2D 평면도 (위성사진 배경)
    ax1 = axes[0]
    extent = [0, nx * cell_size, 0, ny * cell_size]

    # 위성 이미지 로드 시도
    if satellite_path and satellite_path.exists():
        sat_img = Image.open(satellite_path)
        sat_arr = np.array(sat_img)
        # 위성 이미지는 Y축이 반전되어 있을 수 있음
        ax1.imshow(sat_arr, extent=extent, origin='upper', aspect='auto')
    else:
        # 위성 이미지 없으면 DEM 사용
        im = ax1.imshow(terrain, extent=extent, origin='lower', cmap='terrain', alpha=0.8)
        plt.colorbar(im, ax=ax1, label='Elevation (m)', shrink=0.8)

    ax1.scatter(px, py, c='red', s=3, alpha=0.7, label='Initial Particles')
    ax1.scatter(init_x - x_min, init_y - y_min, c='yellow', s=100, marker='*', edgecolors='black', linewidths=1, label='Center', zorder=10)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Initial Condition - Satellite View')
    ax1.legend(loc='upper right')

    # 우측: 고도 프로파일
    ax2 = axes[1]
    # 중심선 따라 프로파일
    center_col = int((init_x - x_min) / cell_size)
    profile = terrain[:, center_col]
    y_coords = np.arange(ny) * cell_size
    ax2.plot(y_coords, profile, 'b-', linewidth=2, label='Terrain Profile')
    ax2.axvline(init_y - y_min, color='red', linestyle='--', linewidth=2, label='Initial Position')
    ax2.fill_between(y_coords, profile.min(), profile, alpha=0.3)
    ax2.set_xlabel('Y (m)')
    ax2.set_ylabel('Elevation (m)')
    ax2.set_title('Elevation Profile (along center line)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    return result


def create_velocity_evolution_plot(data) -> str:
    """속도 변화 시각화"""
    times = data['times']
    x_data, y_data = data['x'], data['y']
    vx_data, vy_data = data['vx'], data['vy']

    max_speeds = []
    mean_speeds = []

    for i in range(len(times)):
        mask = ~np.isnan(x_data[i])
        if mask.any():
            speeds = np.sqrt(vx_data[i, mask]**2 + vy_data[i, mask]**2)
            max_speeds.append(speeds.max())
            mean_speeds.append(speeds.mean())
        else:
            max_speeds.append(0)
            mean_speeds.append(0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 좌측: 속도 시계열
    ax1 = axes[0]
    ax1.plot(times, max_speeds, 'r-', linewidth=2.5, marker='o', markersize=6, label='Max Speed')
    ax1.plot(times, mean_speeds, 'b--', linewidth=2, marker='s', markersize=5, label='Mean Speed')
    ax1.fill_between(times, 0, max_speeds, alpha=0.2, color='red')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Speed (m/s)', fontsize=12)
    ax1.set_title('Velocity Evolution', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, times[-1])
    ax1.set_ylim(0, max(max_speeds) * 1.1)

    # 우측: 가속도
    ax2 = axes[1]
    if len(times) > 1:
        dt = np.diff(times)
        accel = np.diff(max_speeds) / dt
        t_mid = (times[:-1] + times[1:]) / 2
        ax2.bar(t_mid, accel, width=dt*0.8, color='green', alpha=0.7, edgecolor='darkgreen')
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Acceleration (m/s²)', fontsize=12)
        ax2.set_title('Acceleration Over Time', fontsize=14)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    return result


def create_runout_plot(data) -> str:
    """Runout 거리 시각화"""
    times = data['times']
    x_data, y_data = data['x'], data['y']
    init_x_local = float(data['init_x']) - float(data['x_min'])
    init_y_local = float(data['init_y']) - float(data['y_min'])

    runout_heads = []
    runout_centroids = []

    for i in range(len(times)):
        mask = ~np.isnan(x_data[i])
        if mask.any():
            px, py = x_data[i, mask], y_data[i, mask]
            distances = np.sqrt((px - init_x_local)**2 + (py - init_y_local)**2)
            runout_heads.append(distances.max())
            runout_centroids.append(distances.mean())
        else:
            runout_heads.append(0)
            runout_centroids.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(times, runout_heads, 'b-', linewidth=2.5, marker='o', markersize=8, label='Head (Max)')
    ax.plot(times, runout_centroids, 'g--', linewidth=2, marker='s', markersize=6, label='Centroid (Mean)')
    ax.fill_between(times, runout_centroids, runout_heads, alpha=0.2, color='blue', label='Spread')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Runout Distance (m)', fontsize=12)
    ax.set_title('Runout Distance Evolution', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    return result


def create_slope_runout_analysis_plot(data) -> str:
    """Head 이동거리에 따른 DEM 경사도 분석"""
    times = data['times']
    x_data = data['x']
    y_data = data['y']
    terrain = data['terrain']
    cell_size = float(data['cell_size'])
    x_min = float(data['x_min'])
    y_min = float(data['y_min'])
    init_x_local = float(data['init_x']) - x_min
    init_y_local = float(data['init_y']) - y_min

    ny, nx = terrain.shape

    # 지형 경사 계산 (degrees)
    grad_y, grad_x = np.gradient(terrain, cell_size)
    slope_terrain = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))

    # Head 위치 추적 (가장 멀리 이동한 입자)
    head_positions = []  # (x, y, distance, time)
    head_slopes = []
    head_elevations = []

    for i in range(len(times)):
        mask = ~np.isnan(x_data[i])
        if mask.any():
            px = x_data[i, mask]
            py = y_data[i, mask]
            distances = np.sqrt((px - init_x_local)**2 + (py - init_y_local)**2)

            # 가장 멀리 이동한 입자 (head)
            head_idx = np.argmax(distances)
            hx, hy = px[head_idx], py[head_idx]
            head_dist = distances[head_idx]

            # Head 위치에서의 경사도
            col = int(np.clip(hx / cell_size, 0, nx - 1))
            row = int(np.clip(hy / cell_size, 0, ny - 1))
            head_slope = slope_terrain[row, col]
            head_elev = terrain[row, col]

            head_positions.append((hx, hy, head_dist, times[i]))
            head_slopes.append(head_slope)
            head_elevations.append(head_elev)
        else:
            head_positions.append((init_x_local, init_y_local, 0, times[i]))
            head_slopes.append(0)
            head_elevations.append(terrain[int(init_y_local/cell_size), int(init_x_local/cell_size)])

    # Head 이동 경로 (고유한 위치만)
    runout_distances = [p[2] for p in head_positions]

    # 그래프 생성 (3패널)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Head 위치의 경사도 vs 이동거리
    ax1 = axes[0]
    scatter = ax1.scatter(runout_distances, head_slopes, c=times, cmap='viridis',
                          s=30, alpha=0.7, edgecolors='none')
    ax1.set_xlabel('Runout Distance (m)', fontsize=12)
    ax1.set_ylabel('Slope at Head Position (°)', fontsize=12)
    ax1.set_title('Slope vs Runout Distance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Time (s)', fontsize=10)

    # 이동 평균 추가
    if len(runout_distances) > 5:
        window = min(10, len(runout_distances) // 3)
        slopes_ma = np.convolve(head_slopes, np.ones(window)/window, mode='valid')
        dist_ma = np.convolve(runout_distances, np.ones(window)/window, mode='valid')
        ax1.plot(dist_ma, slopes_ma, 'r-', linewidth=2.5, label=f'Moving Avg (n={window})')
        ax1.legend(loc='upper right')

    # 2. 고도 프로파일 (Head 경로)
    ax2 = axes[1]
    ax2.plot(runout_distances, head_elevations, 'b-', linewidth=2, label='Elevation')
    ax2.fill_between(runout_distances, min(head_elevations), head_elevations, alpha=0.3)
    ax2.set_xlabel('Runout Distance (m)', fontsize=12)
    ax2.set_ylabel('Elevation (m)', fontsize=12)
    ax2.set_title('Elevation Profile along Head Path', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 경사도 구간 표시 (색상 배경)
    ax2_twin = ax2.twinx()
    ax2_twin.fill_between(runout_distances, 0, head_slopes, alpha=0.2, color='red')
    ax2_twin.set_ylabel('Slope (°)', color='red', fontsize=10)
    ax2_twin.tick_params(axis='y', labelcolor='red')

    # 3. 2D 평면도에 Head 경로 표시
    ax3 = axes[2]
    extent = [0, nx * cell_size, 0, ny * cell_size]
    im = ax3.imshow(slope_terrain, extent=extent, origin='lower', cmap='RdYlGn_r',
                    vmin=0, vmax=45, alpha=0.8)
    plt.colorbar(im, ax=ax3, label='Slope (°)', shrink=0.8)

    # Head 경로 표시
    hx_list = [p[0] for p in head_positions]
    hy_list = [p[1] for p in head_positions]
    ax3.plot(hx_list, hy_list, 'b-', linewidth=2, alpha=0.7, label='Head Path')
    ax3.scatter(hx_list[0], hy_list[0], c='green', s=100, marker='o', edgecolors='white',
                linewidths=2, zorder=10, label='Start')
    ax3.scatter(hx_list[-1], hy_list[-1], c='red', s=100, marker='s', edgecolors='white',
                linewidths=2, zorder=10, label='End')
    ax3.set_xlabel('X (m)', fontsize=12)
    ax3.set_ylabel('Y (m)', fontsize=12)
    ax3.set_title('Head Path on Slope Map', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    return result


def create_entrainment_plot(data) -> Optional[str]:
    """연행 분석 2패널 그래프 (Velocity/Runout + Slope/Entrainment/Concentration)"""
    if 'concentration' not in data.files or 'height' not in data.files:
        return None

    times = data['times']
    x_data = data['x']
    y_data = data['y']
    vx_data = data['vx']
    vy_data = data['vy']
    height_data = data['height']
    conc_data = data['concentration']
    terrain = data['terrain']
    cell_size = float(data['cell_size'])
    h = float(data['h'])

    # 초기 위치 (runout 계산용)
    init_x_local = float(data['init_x']) - float(data['x_min'])
    init_y_local = float(data['init_y']) - float(data['y_min'])

    ny, nx = terrain.shape

    # 지형 경사 계산 (degrees)
    grad_y, grad_x = np.gradient(terrain, cell_size)
    slope_terrain = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))

    # 파티클 면적 (SPH 커널 기반)
    particle_area = h * h

    # 시계열 데이터 계산
    mean_velocities = []
    runout_distances = []
    mean_slopes = []
    total_volumes = []
    mean_concentrations = []

    for i in range(len(times)):
        mask = ~np.isnan(x_data[i])
        px = x_data[i, mask]
        py = y_data[i, mask]
        vx = vx_data[i, mask]
        vy = vy_data[i, mask]
        heights = height_data[i, mask]
        conc = conc_data[i, mask]

        if len(px) > 0:
            # 평균 속도
            speeds = np.sqrt(vx**2 + vy**2)
            mean_velocities.append(speeds.mean())

            # Runout 거리 (최대)
            distances = np.sqrt((px - init_x_local)**2 + (py - init_y_local)**2)
            runout_distances.append(distances.max())

            # 평균 경사 (파티클 위치에서의 지형 경사)
            col = np.clip((px / cell_size).astype(int), 0, nx - 1)
            row = np.clip((py / cell_size).astype(int), 0, ny - 1)
            particle_slopes = slope_terrain[row, col]
            mean_slopes.append(particle_slopes.mean())

            # 총 볼륨 (height * area)
            volume = np.sum(heights * particle_area)
            total_volumes.append(volume)

            # 평균 농도
            mean_concentrations.append(conc.mean())
        else:
            mean_velocities.append(0)
            runout_distances.append(0)
            mean_slopes.append(0)
            total_volumes.append(0)
            mean_concentrations.append(0)

    # 누적 연행량 (초기 대비 변화량)
    initial_volume = total_volumes[0] if total_volumes else 0
    cumulative_dv = [v - initial_volume for v in total_volumes]

    # 2패널 그래프 생성
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # === 상단 패널: Velocity and Runout Distance ===
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    line1, = ax1.plot(times, mean_velocities, color='#8B5CF6', linewidth=2.5, label='Velocity')
    line2, = ax1_twin.plot(times, runout_distances, color='#F97316', linewidth=2.5, label='Runout')

    ax1.set_ylabel('Mean Velocity (m/s)', color='#8B5CF6', fontsize=12)
    ax1_twin.set_ylabel('Runout Distance (m)', color='#F97316', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#8B5CF6')
    ax1_twin.tick_params(axis='y', labelcolor='#F97316')
    ax1.set_title('Velocity and Runout Distance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 범례
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)

    # Y축 범위 설정
    ax1.set_ylim(0, max(mean_velocities) * 1.1 if max(mean_velocities) > 0 else 1)

    # === 하단 패널: Slope, Entrainment Volume, Concentration ===
    ax2 = axes[1]
    ax2_twin1 = ax2.twinx()
    ax2_twin2 = ax2.twinx()

    # 세 번째 Y축 오프셋
    ax2_twin2.spines['right'].set_position(('axes', 1.12))

    line3, = ax2.plot(times, mean_slopes, color='#3B82F6', linewidth=2.5, label='Slope')
    line4, = ax2_twin1.plot(times, cumulative_dv, color='#22C55E', linewidth=2.5, label='Entrainment ΔV')
    line5, = ax2_twin2.plot(times, mean_concentrations, color='#EF4444', linewidth=2.5, label='Concentration')

    # 기준선 (ΔV = 0)
    ax2_twin1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Mean Slope (°)', color='#3B82F6', fontsize=12)
    ax2_twin1.set_ylabel('Cumulative ΔV (m³)', color='#22C55E', fontsize=12)
    ax2_twin2.set_ylabel('Mean Concentration', color='#EF4444', fontsize=12)

    ax2.tick_params(axis='y', labelcolor='#3B82F6')
    ax2_twin1.tick_params(axis='y', labelcolor='#22C55E')
    ax2_twin2.tick_params(axis='y', labelcolor='#EF4444')

    ax2.set_title('Slope, Entrainment Volume, and Concentration', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 범례
    lines2 = [line3, line4, line5]
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labels2, loc='upper right', fontsize=10)

    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    return result


def create_particle_distribution_plot(data) -> str:
    """파티클 분포 변화 시각화 (멀티 프레임)"""
    times = data['times']
    x_data, y_data = data['x'], data['y']
    vx_data, vy_data = data['vx'], data['vy']
    terrain = data['terrain']
    cell_size = float(data['cell_size'])

    ny, nx = terrain.shape
    extent = [0, nx * cell_size, 0, ny * cell_size]

    # 최대 6프레임 선택
    n_frames = len(times)
    if n_frames <= 6:
        indices = list(range(n_frames))
    else:
        indices = [int(i * (n_frames - 1) / 5) for i in range(6)]

    n_plots = len(indices)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 전체 속도 범위 계산
    all_speeds = []
    for i in range(n_frames):
        mask = ~np.isnan(x_data[i])
        if mask.any():
            speeds = np.sqrt(vx_data[i, mask]**2 + vy_data[i, mask]**2)
            all_speeds.extend(speeds)
    speed_max = max(all_speeds) if all_speeds else 1

    for idx, frame_idx in enumerate(indices):
        ax = axes[idx]
        ax.imshow(terrain, extent=extent, origin='lower', cmap='terrain', alpha=0.6)

        mask = ~np.isnan(x_data[frame_idx])
        if mask.any():
            px, py = x_data[frame_idx, mask], y_data[frame_idx, mask]
            speeds = np.sqrt(vx_data[frame_idx, mask]**2 + vy_data[frame_idx, mask]**2)
            sc = ax.scatter(px, py, c=speeds, cmap='plasma', s=8, vmin=0, vmax=speed_max, alpha=0.8)

        ax.set_title(f't = {times[frame_idx]:.2f} s', fontsize=12)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

    # 빈 subplot 숨기기
    for idx in range(len(indices), len(axes)):
        axes[idx].set_visible(False)

    # 컬러바
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, speed_max))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Speed (m/s)')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    result = fig_to_base64(fig)
    plt.close(fig)
    return result


def generate_html_report(data, stats: SimulationStats, config_info: dict,
                         work_dir: Path, llm_analysis: Optional[str] = None,
                         target_impacts: list = None) -> str:
    """HTML 형식 보고서 생성 (간소화 버전)"""

    # 프로젝트 이름
    project_name = "Landslide Simulation"
    if 'simulation' in config_info and 'project' in config_info['simulation']:
        project_name = config_info['simulation']['project'].get('name', project_name)

    # 위성 이미지 경로 찾기
    satellite_path = None
    if 'simulation' in config_info and 'terrain' in config_info['simulation']:
        sat_file = config_info['simulation']['terrain'].get('satellite_file')
        if sat_file:
            satellite_path = work_dir / sat_file
    # 기본 패턴으로 검색
    if not satellite_path or not satellite_path.exists():
        for pattern in ['*_satellite_crop.png', '*satellite*.png']:
            matches = list(work_dir.glob(pattern))
            if matches:
                satellite_path = matches[0]
                break

    # 핵심 이미지만 생성
    print("  Creating plots...")
    img_initial = create_initial_condition_plot(data, config_info, satellite_path)
    img_velocity = create_velocity_evolution_plot(data)
    img_entrainment = create_entrainment_plot(data)
    img_slope_analysis = create_slope_runout_analysis_plot(data)


    # 지형 정보
    terrain = data['terrain']
    cell_size = float(data['cell_size'])
    x_min, y_min = float(data['x_min']), float(data['y_min'])
    ny, nx = terrain.shape

    # 시뮬레이션 신뢰성 판단
    is_short = stats.duration < 10.0
    reliability_class = "warning" if is_short else "success"
    reliability_text = "참고용 (시뮬레이션 시간 부족)" if is_short else "신뢰 가능"

    # LLM 분석 HTML 변환
    llm_html = ""
    if llm_analysis:
        import re
        llm_html = llm_analysis
        llm_html = re.sub(r'^### (.+)$', r'<h4>\1</h4>', llm_html, flags=re.MULTILINE)
        llm_html = re.sub(r'^## (.+)$', r'<h3>\1</h3>', llm_html, flags=re.MULTILINE)
        llm_html = re.sub(r'^# (.+)$', r'<h2>\1</h2>', llm_html, flags=re.MULTILINE)
        llm_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', llm_html)
        llm_html = re.sub(r'^- (.+)$', r'<li>\1</li>', llm_html, flags=re.MULTILINE)
        llm_html = llm_html.replace('\n\n', '</p><p>').replace('\n', '<br>')
        llm_html = f'<p>{llm_html}</p>'

    # 대상별 충격 분석 HTML 생성
    impact_cards_html = ""
    if target_impacts:
        type_labels = {'RC': '철근콘크리트', 'wood': '목조', 'masonry': '조적조', 'mixed': '혼합구조'}
        cards = []
        for t in target_impacts:
            type_label = type_labels.get(t['type'], t['type'])
            ttype_label = "영역" if t.get('target_type') == 'area' else "건물"
            if t['reached']:
                pressure = t.get('peak_pressure', t['arrival_pressure'])
                speed = t.get('peak_speed', t['head_speed'])
                # 피해 판정 색상
                if t['type'] == 'RC':
                    if pressure >= 100:
                        dmg_text, dmg_color = "심각손상", "#c53030"
                    elif pressure >= 30:
                        dmg_text, dmg_color = "중간손상", "#dd6b20"
                    elif pressure >= 10:
                        dmg_text, dmg_color = "경미손상", "#d69e2e"
                    else:
                        dmg_text, dmg_color = "경미", "#38a169"
                else:
                    if pressure >= 20:
                        dmg_text, dmg_color = "붕괴", "#c53030"
                    elif pressure >= 10:
                        dmg_text, dmg_color = "심각손상", "#dd6b20"
                    elif pressure >= 5:
                        dmg_text, dmg_color = "중간손상", "#d69e2e"
                    else:
                        dmg_text, dmg_color = "경미", "#38a169"

                card = f"""<div style="border:1px solid #e2e8f0; border-radius:10px; padding:20px; margin-bottom:15px; border-left:5px solid {dmg_color};">
                    <h3 style="color:#1a365d; margin-bottom:5px;">{t['name']} <span style="font-size:0.7em; color:#888;">({ttype_label})</span></h3>
                    <p style="color:#666; font-size:0.85em; margin-bottom:15px;">{t['address']} | {type_label}</p>
                    <div class="grid grid-4" style="gap:10px;">
                        <div style="text-align:center; padding:10px; background:#f7fafc; border-radius:8px;">
                            <span style="font-size:1.5em; font-weight:bold; color:#e53e3e;">{t['arrival_time']:.1f}</span><br>
                            <span style="font-size:0.8em; color:#666;">도달시간 (초)</span>
                        </div>
                        <div style="text-align:center; padding:10px; background:#f7fafc; border-radius:8px;">
                            <span style="font-size:1.5em; font-weight:bold; color:#dd6b20;">{speed:.1f}</span><br>
                            <span style="font-size:0.8em; color:#666;">최대속도 (m/s)</span>
                        </div>
                        <div style="text-align:center; padding:10px; background:#f7fafc; border-radius:8px;">
                            <span style="font-size:1.5em; font-weight:bold; color:#c53030;">{pressure:.1f}</span><br>
                            <span style="font-size:0.8em; color:#666;">충격압력 (kPa)</span>
                        </div>
                        <div style="text-align:center; padding:10px; background:#f7fafc; border-radius:8px;">
                            <span style="font-size:1.5em; font-weight:bold; color:{dmg_color};">{dmg_text}</span><br>
                            <span style="font-size:0.8em; color:#666;">피해 판정</span>
                        </div>
                    </div>
                    <div style="margin-top:12px; display:flex; gap:15px; flex-wrap:wrap; font-size:0.9em; color:#555;">
                        <span>농도: {t['front_concentration']:.0%}</span>
                        <span>유동심: {t['flow_depth']:.2f}m</span>
                        <span>도달입자: {t['n_particles_arrived']}개</span>
                        {"<span>최종잔류: " + str(t.get('final_n_particles', 0)) + "개</span>" if 'final_n_particles' in t else ""}
                    </div>
                </div>"""
            else:
                # 미도달 카드 - 도달 가능성 추정 포함
                extra_info = ""
                border_color = "#38a169"  # 기본 안전 (녹색)
                status_text = "시뮬레이션 시간 내 토석류 미도달 (안전)"

                if t.get('closest_distance') is not None:
                    extra_info += f'<span>최근접 거리: {t["closest_distance"]:.0f}m</span>'
                if t.get('final_avg_speed') is not None:
                    extra_info += f'<span>종료 시 평균속도: {t["final_avg_speed"]:.1f} m/s</span>'
                if t.get('deceleration') is not None:
                    extra_info += f'<span>감속률: {t["deceleration"]:.3f} m/s²</span>'

                if t.get('will_likely_reach') is True:
                    border_color = "#dd6b20"  # 주의 (주황)
                    est_time = t.get('estimated_arrival_time', 0)
                    status_text = f'미도달이나 <b>도달 가능성 높음</b> (추정 도달시간 ~{est_time:.0f}초)'
                elif t.get('will_likely_reach') is False:
                    status_text = '미도달 - 감속 추세로 <b>도달 전 정지 예상</b>'

                card = f"""<div style="border:1px solid #e2e8f0; border-radius:10px; padding:20px; margin-bottom:15px; border-left:5px solid {border_color};">
                    <h3 style="color:#1a365d; margin-bottom:5px;">{t['name']} <span style="font-size:0.7em; color:#888;">({ttype_label})</span></h3>
                    <p style="color:#666; font-size:0.85em; margin-bottom:10px;">{t['address']} | {type_label}</p>
                    <p style="color:{border_color}; font-weight:bold; margin-bottom:8px;">{status_text}</p>
                    {"<div style='display:flex; gap:15px; flex-wrap:wrap; font-size:0.9em; color:#555;'>" + extra_info + "</div>" if extra_info else ""}
                </div>"""
            cards.append(card)

        impact_cards_html = f"""
        <div class="section">
            <h2>대상별 충격 분석</h2>
            <p style="margin-bottom:15px; color:#555;">
                시뮬레이션 결과를 기반으로 각 대상 건물/영역에 대한 토석류 도달 시점, 충격 요인, 피해 수준을 분석합니다.
            </p>
            {''.join(cards)}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name} - 재해 위험 분석 보고서</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Malgun Gothic', sans-serif; background: #f5f5f5; color: #333; line-height: 1.7; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{ background: linear-gradient(135deg, #1a365d, #2c5282); color: white; padding: 30px; text-align: center; border-radius: 10px; margin-bottom: 25px; }}
        header h1 {{ font-size: 2em; margin-bottom: 5px; }}
        header .subtitle {{ opacity: 0.9; }}
        header .meta {{ font-size: 0.85em; margin-top: 10px; opacity: 0.7; }}
        .section {{ background: white; border-radius: 10px; padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .section h2 {{ color: #1a365d; border-left: 4px solid #3182ce; padding-left: 15px; margin-bottom: 20px; font-size: 1.3em; }}
        .grid {{ display: grid; gap: 15px; }}
        .grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
        .grid-4 {{ grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
        .stat-card {{ text-align: center; padding: 20px 15px; border-radius: 10px; color: white; }}
        .stat-card.blue {{ background: linear-gradient(135deg, #3182ce, #2b6cb0); }}
        .stat-card.green {{ background: linear-gradient(135deg, #38a169, #2f855a); }}
        .stat-card.red {{ background: linear-gradient(135deg, #e53e3e, #c53030); }}
        .stat-card.orange {{ background: linear-gradient(135deg, #dd6b20, #c05621); }}
        .stat-card .value {{ font-size: 2em; font-weight: bold; display: block; }}
        .stat-card .label {{ font-size: 0.85em; opacity: 0.9; }}
        .reliability {{ display: inline-block; padding: 5px 15px; border-radius: 20px; font-size: 0.9em; font-weight: bold; }}
        .reliability.warning {{ background: #fed7d7; color: #c53030; }}
        .reliability.success {{ background: #c6f6d5; color: #276749; }}
        .image-box {{ text-align: center; margin: 20px 0; }}
        .image-box img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .image-box.small img {{ max-width: 700px; }}
        .caption {{ font-size: 0.85em; color: #666; margin-top: 8px; }}
        .analysis-box {{ background: #fffaf0; border-left: 4px solid #dd6b20; padding: 20px; border-radius: 0 10px 10px 0; }}
        .analysis-box h2 {{ color: #c05621; border-left-color: #dd6b20; }}
        .analysis-box h3 {{ color: #1a365d; margin: 15px 0 10px; font-size: 1.1em; }}
        .analysis-box h4 {{ color: #2c5282; margin: 12px 0 8px; font-size: 1em; }}
        .analysis-box ul {{ margin-left: 20px; }}
        .analysis-box li {{ margin: 5px 0; }}
        footer {{ text-align: center; padding: 20px; color: #666; font-size: 0.85em; }}
        @media (max-width: 768px) {{ .grid-4 {{ grid-template-columns: repeat(2, 1fr); }} }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{project_name}</h1>
            <div class="subtitle">산사태 재해 위험 분석 보고서</div>
            <div class="meta">{datetime.now().strftime('%Y년 %m월 %d일')} | 데이터 신뢰도: <span class="reliability {reliability_class}">{reliability_text}</span></div>
        </header>

        <!-- 핵심 지표 -->
        <div class="section">
            <h2>핵심 분석 지표</h2>
            <div class="grid grid-4">
                <div class="stat-card red">
                    <span class="value">{stats.max_speed:.1f}</span>
                    <span class="label">최대 유속 (m/s)</span>
                </div>
                <div class="stat-card orange">
                    <span class="value">{stats.max_speed * 3.6:.0f}</span>
                    <span class="label">최대 유속 (km/h)</span>
                </div>
                <div class="stat-card green">
                    <span class="value">{stats.runout_head_final:.0f}</span>
                    <span class="label">도달 거리 (m)</span>
                </div>
                <div class="stat-card blue">
                    <span class="value">{stats.concentration_final:.0%}</span>
                    <span class="label">토사 농도</span>
                </div>
            </div>
        </div>

        <!-- AI 분석 (최상단 배치) -->
        {"<div class='section analysis-box'><h2>재해 위험 분석</h2>" + llm_html + "</div>" if llm_html else ""}

        <!-- 건물별 충격 분석 -->
        {impact_cards_html}

        <!-- 시뮬레이션 결과 -->
        <div class="section">
            <h2>시뮬레이션 결과</h2>
            <div class="grid grid-2">
                <div>
                    <h3 style="color:#2c5282; margin-bottom:10px;">발생 위치</h3>
                    <p>좌표 (TM): ({float(data['init_x']):.0f}, {float(data['init_y']):.0f})</p>
                    <p>분석 영역: {nx * cell_size:.0f}m x {ny * cell_size:.0f}m</p>
                    <p>고도차: {terrain.max() - terrain.min():.0f}m ({terrain.min():.0f}~{terrain.max():.0f}m)</p>
                </div>
                <div>
                    <h3 style="color:#2c5282; margin-bottom:10px;">시뮬레이션 조건</h3>
                    <p>분석 시간: {stats.duration:.1f}초</p>
                    <p>초기 붕괴 규모: 반경 {config_info.get('simulation', {}).get('initial_condition', {}).get('radius', 'N/A')}m, 두께 {config_info.get('simulation', {}).get('initial_condition', {}).get('thickness', 'N/A')}m</p>
                </div>
            </div>
            <div class="image-box">
                <img src="{img_initial}" alt="초기 조건">
                <div class="caption">발생 위치 및 지형 현황</div>
            </div>
        </div>

        <!-- 유속 및 농도 분석 -->
        <div class="section">
            <h2>유속 및 토사 농도 변화</h2>
            <div class="image-box">
                <img src="{img_velocity}" alt="유속 변화">
                <div class="caption">시간에 따른 유속 변화</div>
            </div>
            {"<div class='image-box'><img src='" + img_entrainment + "' alt='농도 변화'><div class='caption'>토사 농도 변화</div></div>" if img_entrainment else ""}
        </div>

        <!-- 경사도 분석 -->
        <div class="section">
            <h2>지형 경사도 분석</h2>
            <p style="margin-bottom:15px; color:#555;">
                산사태 Head(선두부)의 이동 경로를 따라 DEM 경사도를 분석합니다.
                경사도 변화는 유동체의 가속/감속, 퇴적 양상에 영향을 미칩니다.
            </p>
            <div class="image-box">
                <img src="{img_slope_analysis}" alt="경사도 분석">
                <div class="caption">Head 이동거리에 따른 DEM 경사도 분석 (좌: 경사도-거리 관계, 중: 고도 프로파일, 우: 경로 지도)</div>
            </div>
        </div>

        <footer>
            <p>본 보고서는 SPH 수치해석 기반 시뮬레이션 결과입니다.</p>
            <p>실제 재해 대응 시 현장 조사와 전문가 검토를 병행하시기 바랍니다.</p>
        </footer>
    </div>
</body>
</html>
"""

    return html


def generate_report(results_path: str, use_llm: bool = True):
    """메인 보고서 생성"""
    print("=" * 60)
    print("LANDSLIDE ANALYSIS REPORT GENERATOR")
    print("=" * 60)

    # 경로 처리
    results_path = Path(results_path).resolve()
    work_dir = results_path.parent

    if not results_path.exists():
        print(f"Error: 결과 파일을 찾을 수 없음: {results_path}")
        sys.exit(1)

    print(f"\nWork dir: {work_dir}")

    # 데이터 로드
    print(f"Loading: {results_path.name}")
    data = np.load(str(results_path))

    # 설정 정보 로드
    config_info = load_config_info(work_dir)

    # 분석
    print("Analyzing results...")
    stats = analyze_results(data)

    print(f"  - Frames: {stats.n_frames}")
    print(f"  - Duration: {stats.duration:.1f}s")
    print(f"  - Max Speed: {stats.max_speed:.2f} m/s")
    print(f"  - Runout (head): {stats.runout_head_final:.1f} m")

    # 이동거리별 속도 분석
    print("Analyzing velocity by distance...")
    velocity_by_distance = analyze_velocity_by_distance(data)
    for dist, v in velocity_by_distance['velocity_at_distance'].items():
        if v:
            print(f"  - {dist}m: {v['head_speed']:.1f} m/s (t={v['time']:.1f}s)")

    # 대상별 충격 분석
    print("Analyzing impact at targets...")
    target_impacts = analyze_impact_at_targets(data, config_info)
    for t in target_impacts:
        ttype = "[영역]" if t.get('target_type') == 'area' else "[건물]"
        if t['reached']:
            print(f"  - {ttype} {t['name']}: 도달 {t['arrival_time']:.1f}s, "
                  f"속도 {t['head_speed']:.1f} m/s, "
                  f"압력 {t['arrival_pressure']:.1f} kPa")
        else:
            extra = ""
            if t.get('closest_distance') is not None:
                extra = f" (최근접 {t['closest_distance']:.0f}m"
                if t.get('will_likely_reach') is True:
                    extra += f", 도달 예상 ~{t.get('estimated_arrival_time', 0):.0f}s"
                elif t.get('will_likely_reach') is False:
                    extra += ", 정지 예상"
                extra += ")"
            print(f"  - {ttype} {t['name']}: 미도달{extra}")

    # LLM 분석 (옵션)
    llm_analysis = None
    if use_llm:
        print("\nRequesting LLM analysis...")
        llm_analysis = get_llm_analysis(stats, data, config_info, velocity_by_distance, target_impacts)
        if llm_analysis:
            print("  LLM analysis received.")
        else:
            print("  LLM analysis skipped.")

    # 좌표 suffix 계산
    terrain = data['terrain']
    ny, nx = terrain.shape
    cell_size = float(data['cell_size'])
    x_min_geo = float(data['x_min'])
    y_min_geo = float(data['y_min'])
    x_center = x_min_geo + (nx * cell_size) / 2
    y_center = y_min_geo + (ny * cell_size) / 2
    coord_suffix = f"{x_center:.0f}x{y_center:.0f}"

    # Markdown 보고서 생성
    print("\nGenerating Markdown report...")
    report_content = generate_markdown_report(data, stats, config_info, work_dir, llm_analysis)
    report_md_path = work_dir / f"analysis_report_{coord_suffix}.md"
    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"  Markdown: {report_md_path}")

    # HTML 보고서 생성
    print("\nGenerating HTML report...")
    html_content = generate_html_report(data, stats, config_info, work_dir, llm_analysis, target_impacts)
    report_html_path = work_dir / f"analysis_report_{coord_suffix}.html"
    with open(report_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"  HTML: {report_html_path}")

    print("\nDone!")

    return report_html_path


def main():
    parser = argparse.ArgumentParser(
        description='SPH 산사태 시뮬레이션 결과 분석 보고서 생성',
        epilog='Example: python generate_report.py ./guryoung_dem_10m/simulation_results.npz'
    )
    parser.add_argument('results_file', type=str, help='시뮬레이션 결과 파일 (.npz)')
    parser.add_argument('--no-llm', action='store_true', help='LLM 분석 비활성화')

    args = parser.parse_args()
    generate_report(args.results_file, use_llm=not args.no_llm)


if __name__ == '__main__':
    main()
