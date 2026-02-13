#!/usr/bin/env python3
"""
resolve_targets.py - 타겟 주소를 실제 건물 영역(bbox)으로 변환

V-World API를 사용하여:
1. 도로명 주소 → 위경도 좌표 (Geocoder API)
2. 좌표 주변 건물 폴리곤 조회 (WFS API, lt_c_bldginfo)
3. PNU 기반 필터링 → 단지 전체 bbox 계산
4. simulation_config.yaml 업데이트

사용법:
    python resolve_targets.py <project_dir>
    python resolve_targets.py <project_dir>/simulation_config.yaml

필수: V-World API 키 (simulation_config.yaml의 vworld_api_key 또는 환경변수 VWORLD_API_KEY)
"""

import sys
import os
import json
import yaml
import requests
import numpy as np
from pathlib import Path
from pyproj import Transformer

# ── 좌표 변환기 ──────────────────────────────────────────────
T_4326_TO_5186 = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
T_5186_TO_900913 = Transformer.from_crs("EPSG:5186", "EPSG:900913", always_xy=True)
T_900913_TO_5186 = Transformer.from_crs("EPSG:900913", "EPSG:5186", always_xy=True)

VWORLD_WFS_URL = "https://api.vworld.kr/req/wfs"
VWORLD_GEOCODER_URL = "https://api.vworld.kr/req/address"


def get_vworld_api_key(config: dict) -> str:
    """V-World API 키를 환경변수 또는 config에서 가져옴"""
    key = os.environ.get("VWORLD_API_KEY", "")
    if not key:
        key = config.get("vworld_api_key", "")
    if not key:
        print("[ERROR] V-World API 키가 없습니다.")
        print("  환경변수 VWORLD_API_KEY 또는 simulation_config.yaml의 vworld_api_key를 설정하세요.")
        sys.exit(1)
    return key


def geocode_address(address: str, api_key: str) -> dict | None:
    """도로명 주소 → 좌표 + 주소 구조 (V-World Geocoder API)

    Returns:
        {'lon': float, 'lat': float, 'x_5186': float, 'y_5186': float,
         'refined_address': str} or None
    """
    params = {
        "service": "address",
        "request": "getcoord",
        "version": "2.0",
        "crs": "epsg:4326",
        "address": address,
        "refine": "true",
        "simple": "false",
        "format": "json",
        "type": "road",
        "key": api_key,
    }
    try:
        resp = requests.get(VWORLD_GEOCODER_URL, params=params, timeout=10)
        data = resp.json()
    except Exception as e:
        print(f"  [ERROR] 지오코딩 실패: {e}")
        return None

    status = data.get("response", {}).get("status", "")
    if status != "OK":
        # road type 실패 시 parcel type으로 재시도
        params["type"] = "parcel"
        try:
            resp = requests.get(VWORLD_GEOCODER_URL, params=params, timeout=10)
            data = resp.json()
            status = data.get("response", {}).get("status", "")
        except Exception as e:
            print(f"  [ERROR] 지오코딩 재시도 실패: {e}")
            return None

    if status != "OK":
        print(f"  [ERROR] 지오코딩 결과 없음 (status={status})")
        return None

    result = data["response"]["result"]
    lon = float(result["point"]["x"])
    lat = float(result["point"]["y"])
    x_5186, y_5186 = T_4326_TO_5186.transform(lon, lat)

    refined = data["response"].get("refined", {}).get("text", address)

    return {
        "lon": lon,
        "lat": lat,
        "x_5186": x_5186,
        "y_5186": y_5186,
        "refined_address": refined,
    }


def query_buildings_wfs(cx_5186: float, cy_5186: float, buffer_m: float,
                        api_key: str) -> list[dict]:
    """좌표 주변 건물 폴리곤 조회 (V-World WFS API)

    Returns:
        list of GeoJSON features
    """
    wx1, wy1 = T_5186_TO_900913.transform(cx_5186 - buffer_m, cy_5186 - buffer_m)
    wx2, wy2 = T_5186_TO_900913.transform(cx_5186 + buffer_m, cy_5186 + buffer_m)

    params = {
        "SERVICE": "WFS",
        "REQUEST": "GetFeature",
        "TYPENAME": "lt_c_bldginfo",
        "BBOX": f"{wx1},{wy1},{wx2},{wy2}",
        "VERSION": "1.1.0",
        "SRSNAME": "EPSG:900913",
        "OUTPUT": "application/json",
        "MAXFEATURES": "1000",
        "KEY": api_key,
    }
    try:
        resp = requests.get(VWORLD_WFS_URL, params=params, timeout=30)
        data = resp.json()
    except Exception as e:
        print(f"  [ERROR] WFS 조회 실패: {e}")
        return []

    return data.get("features", [])


def extract_polygon_coords_5186(geometry: dict) -> list[tuple[float, float]]:
    """GeoJSON geometry에서 EPSG:5186 좌표 리스트 추출"""
    coords = []
    geom_type = geometry.get("type", "")

    if geom_type == "MultiPolygon":
        for poly in geometry["coordinates"]:
            for ring in poly:
                for pt in ring:
                    x5, y5 = T_900913_TO_5186.transform(pt[0], pt[1])
                    coords.append((x5, y5))
    elif geom_type == "Polygon":
        for ring in geometry["coordinates"]:
            for pt in ring:
                x5, y5 = T_900913_TO_5186.transform(pt[0], pt[1])
                coords.append((x5, y5))
    return coords


def find_target_pnu(features: list[dict], cx_5186: float, cy_5186: float) -> str:
    """지오코딩 좌표에 가장 가까운 건물의 PNU를 찾음"""
    best_dist = float("inf")
    best_pnu = ""

    for f in features:
        pnu = f["properties"].get("pnu", "")
        if not pnu:
            continue
        coords = extract_polygon_coords_5186(f["geometry"])
        if not coords:
            continue
        # 폴리곤 중심까지 거리
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        centroid_x = sum(xs) / len(xs)
        centroid_y = sum(ys) / len(ys)
        dist = ((centroid_x - cx_5186) ** 2 + (centroid_y - cy_5186) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_pnu = pnu

    return best_pnu


def compute_bbox_from_features(features: list[dict], pnu: str) -> dict | None:
    """PNU에 해당하는 건물들의 전체 bbox 계산

    Returns:
        {'bbox': [x_min, y_min, x_max, y_max], 'n_buildings': int,
         'width': float, 'height': float, 'center': [cx, cy]}
    """
    matching = [f for f in features if f["properties"].get("pnu") == pnu]
    if not matching:
        return None

    all_x = []
    all_y = []
    building_names = []

    for f in matching:
        coords = extract_polygon_coords_5186(f["geometry"])
        all_x.extend(c[0] for c in coords)
        all_y.extend(c[1] for c in coords)
        name = f["properties"].get("bld_nm", "")
        dong = f["properties"].get("dong_nm", "")
        if name or dong:
            building_names.append(f"{name} {dong}".strip())

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    return {
        "bbox": [round(x_min), round(y_min), round(x_max), round(y_max)],
        "n_buildings": len(matching),
        "width": round(x_max - x_min),
        "height": round(y_max - y_min),
        "center": [round((x_min + x_max) / 2), round((y_min + y_max) / 2)],
        "pnu": pnu,
        "building_names": building_names[:5],  # 처음 5개만
    }


def resolve_target(target: dict, api_key: str) -> dict:
    """단일 타겟의 주소를 해석하여 좌표/bbox로 변환

    Returns:
        updated target dict
    """
    name = target.get("name", "")
    address = target.get("address", "")

    # 이미 bbox가 있으면 건너뜀
    if target.get("bbox"):
        print(f"\n── {name} ──")
        print(f"  [SKIP] bbox 이미 설정됨: {target['bbox']}")
        return target

    if not address:
        print(f"\n── {name} ──")
        print("  [SKIP] 주소 없음")
        return target

    print(f"\n── {name} ──")
    print(f"  주소: {address}")

    # Step 1: 지오코딩
    print("  [1/3] 지오코딩...")
    geo = geocode_address(address, api_key)
    if not geo:
        print("  [FAIL] 지오코딩 실패, 기존 값 유지")
        return target

    print(f"  → WGS84: ({geo['lat']:.6f}, {geo['lon']:.6f})")
    print(f"  → EPSG:5186: ({geo['x_5186']:.0f}, {geo['y_5186']:.0f})")

    # Step 2: 주변 건물 조회
    print("  [2/3] V-World WFS 건물 조회...")
    # 첫 조회: 300m 버퍼
    features = query_buildings_wfs(geo["x_5186"], geo["y_5186"], 300, api_key)
    if not features:
        # 버퍼 확대 재시도
        features = query_buildings_wfs(geo["x_5186"], geo["y_5186"], 500, api_key)
    print(f"  → {len(features)}개 건물 조회됨")

    if not features:
        print("  [FAIL] 건물 데이터 없음, 좌표만 업데이트")
        target["coordinates"] = [round(geo["x_5186"]), round(geo["y_5186"])]
        return target

    # Step 3: PNU 매칭 + bbox 계산
    print("  [3/3] PNU 매칭 및 bbox 계산...")
    pnu = find_target_pnu(features, geo["x_5186"], geo["y_5186"])
    if not pnu:
        print("  [FAIL] PNU 매칭 실패, 좌표만 업데이트")
        target["coordinates"] = [round(geo["x_5186"]), round(geo["y_5186"])]
        return target

    result = compute_bbox_from_features(features, pnu)
    if not result:
        print("  [FAIL] bbox 계산 실패")
        target["coordinates"] = [round(geo["x_5186"]), round(geo["y_5186"])]
        return target

    print(f"  → PNU: {pnu}")
    print(f"  → 건물 {result['n_buildings']}개 매칭")
    print(f"  → bbox: {result['bbox']}  ({result['width']}m x {result['height']}m)")
    if result["building_names"]:
        print(f"  → 건물: {', '.join(result['building_names'][:5])}")

    # 타겟 업데이트
    target["target_type"] = "area"
    target["bbox"] = result["bbox"]
    target.pop("coordinates", None)
    target.pop("proximity_radius", None)
    # structure_type 보존 (기존 type → structure_type)
    if "type" in target and "structure_type" not in target:
        target["structure_type"] = target.pop("type")

    return target


def build_targets_yaml(targets: list[dict]) -> str:
    """resolved targets를 YAML 텍스트로 변환 (주석 포함)"""
    lines = ["targets:"]
    for t in targets:
        lines.append(f'  - name: "{t.get("name", "")}"')
        if t.get("target_type"):
            lines.append(f'    target_type: "{t["target_type"]}"')
        if t.get("address"):
            lines.append(f'    address: "{t["address"]}"')
        if t.get("structure_type"):
            lines.append(f'    structure_type: "{t["structure_type"]}"')
        if t.get("bbox"):
            b = t["bbox"]
            w = b[2] - b[0]
            h = b[3] - b[1]
            lines.append(f"    bbox: [{b[0]}, {b[1]}, {b[2]}, {b[3]}]"
                         f"  # V-World WFS ({w}m x {h}m)")
        if t.get("coordinates"):
            c = t["coordinates"]
            lines.append(f"    coordinates: [{c[0]}, {c[1]}]")
        if t.get("proximity_radius"):
            lines.append(f"    proximity_radius: {t['proximity_radius']}")
    return "\n".join(lines) + "\n"


def update_config_targets(config_path: Path, targets: list[dict]):
    """config 파일의 targets 섹션만 교체 (나머지 주석/포맷 보존)"""
    with open(config_path, "r", encoding="utf-8") as f:
        original = f.read()

    # targets: 섹션 시작/끝 찾기
    import re
    # targets: 로 시작하는 줄 찾기
    match = re.search(r"^targets:\s*\n", original, re.MULTILINE)
    if not match:
        # targets 섹션이 없으면 파일 끝에 추가
        new_text = original.rstrip() + "\n\n" + build_targets_yaml(targets)
    else:
        start = match.start()
        # 다음 최상위 키(들여쓰기 없는 줄) 또는 파일 끝 찾기
        rest = original[match.end():]
        end_match = re.search(r"^\S", rest, re.MULTILINE)
        if end_match:
            end = match.end() + end_match.start()
        else:
            end = len(original)
        new_text = original[:start] + build_targets_yaml(targets) + "\n" + original[end:]

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(new_text)


def main():
    if len(sys.argv) < 2:
        print("사용법: python resolve_targets.py <project_dir|config.yaml>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if path.is_dir():
        config_path = path / "simulation_config.yaml"
    else:
        config_path = path

    if not config_path.exists():
        print(f"[ERROR] 파일 없음: {config_path}")
        sys.exit(1)

    print(f"Config: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    api_key = get_vworld_api_key(config)
    targets = config.get("targets", [])

    if not targets:
        print("타겟이 없습니다.")
        return

    print(f"타겟 {len(targets)}개 처리 시작")

    for i, target in enumerate(targets):
        targets[i] = resolve_target(target, api_key)

    # targets 섹션만 교체 (주석/포맷 보존)
    update_config_targets(config_path, targets)

    print(f"\n{'='*50}")
    print(f"완료! {config_path} 업데이트됨")


if __name__ == "__main__":
    main()
