#!/usr/bin/env python3

import argparse
import json
import os
import random
from typing import Optional, Tuple

import cv2
import numpy as np
import trimesh


def _load_points(ply_path: str) -> Optional[np.ndarray]:
    if not os.path.isfile(ply_path):
        return None
    mesh = trimesh.load(ply_path, process=False)
    if hasattr(mesh, "vertices"):
        points = np.asarray(mesh.vertices, dtype=np.float64)
    elif hasattr(mesh, "points"):
        points = np.asarray(mesh.points, dtype=np.float64)
    else:
        return None
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return None
    return points


def _load_scale(measurement_json: str) -> Optional[float]:
    if not measurement_json or not os.path.isfile(measurement_json):
        return None
    try:
        with open(measurement_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        scale = data.get("scale_factor_m_per_model_unit")
        if scale is None:
            return None
        value = float(scale)
        if value <= 0:
            return None
        return value
    except Exception:
        return None


def _plane_from_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return None
    n = n / norm
    d = -float(np.dot(n, p1))
    return n, d


def _fit_floor_plane(points: np.ndarray, max_iters: int = 700) -> Optional[Tuple[np.ndarray, float, np.ndarray, float]]:
    if len(points) < 400:
        return None

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    if diag <= 0:
        return None

    threshold = max(diag * 0.004, 1e-4)
    best = None
    best_count = 0
    ids = list(range(len(points)))

    for _ in range(max_iters):
        sample_ids = random.sample(ids, 3)
        plane = _plane_from_points(points[sample_ids[0]], points[sample_ids[1]], points[sample_ids[2]])
        if plane is None:
            continue
        n, d = plane

        distances = np.abs(points @ n + d)
        inliers = distances < threshold
        count = int(np.sum(inliers))
        if count > best_count and count >= 200:
            best_count = count
            best = (n, d, inliers, threshold)

    return best


def _make_plane_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, n))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis_u = np.cross(n, helper)
    axis_u = axis_u / (np.linalg.norm(axis_u) + 1e-12)
    axis_v = np.cross(n, axis_u)
    axis_v = axis_v / (np.linalg.norm(axis_v) + 1e-12)
    return axis_u, axis_v


def _points_to_polygon(points_2d: np.ndarray) -> Optional[np.ndarray]:
    if len(points_2d) < 30:
        return None

    mins = points_2d.min(axis=0)
    maxs = points_2d.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    max_span = float(max(span[0], span[1]))

    grid_size = 768
    px = max_span / float(grid_size)
    if px <= 0:
        return None

    scaled = (points_2d - mins) / px
    width = int(np.ceil(span[0] / px)) + 6
    height = int(np.ceil(span[1] / px)) + 6
    width = max(width, 32)
    height = max(height, 32)

    canvas = np.zeros((height, width), dtype=np.uint8)
    xi = np.clip(np.round(scaled[:, 0]).astype(np.int32) + 3, 0, width - 1)
    yi = np.clip(np.round(scaled[:, 1]).astype(np.int32) + 3, 0, height - 1)
    canvas[yi, xi] = 255

    kernel = np.ones((5, 5), dtype=np.uint8)
    canvas = cv2.dilate(canvas, kernel, iterations=3)
    canvas = cv2.erode(canvas, kernel, iterations=2)
    canvas = cv2.medianBlur(canvas, 5)

    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, True)
    if perimeter <= 0:
        return None

    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(largest, epsilon, True)
    if approx is None or len(approx) < 3:
        return None

    poly_px = approx[:, 0, :].astype(np.float64)
    poly_2d = np.empty_like(poly_px)
    poly_2d[:, 0] = (poly_px[:, 0] - 3.0) * px + mins[0]
    poly_2d[:, 1] = (poly_px[:, 1] - 3.0) * px + mins[1]

    return poly_2d


def _polygon_to_svg(points: np.ndarray, out_path: str, units_label: str) -> None:
    if len(points) < 3:
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    size = np.maximum(maxs - mins, 1e-6)

    margin = 20.0
    view_w = float(size[0] + 2 * margin)
    view_h = float(size[1] + 2 * margin)

    normalized = points - mins + margin

    coords = []
    for x, y in normalized:
        svg_y = view_h - y
        coords.append(f"{x:.4f},{svg_y:.4f}")
    path_data = " ".join(coords)

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_w:.4f} {view_h:.4f}">\n'
        f'  <g fill="none" stroke="#1f6feb" stroke-width="0.05">\n'
        f'    <polygon points="{path_data}" fill="#58a6ff22" />\n'
        f'  </g>\n'
        f'  <text x="6" y="16" font-size="12" fill="#24292f">Units: {units_label}</text>\n'
        f'</svg>\n'
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)


def _write_geojson(points: np.ndarray, out_path: str) -> None:
    if len(points) < 3:
        return
    ring = [[float(x), float(y)] for x, y in points]
    if ring[0] != ring[-1]:
        ring.append(ring[0])

    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "floorplan"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ring]
                }
            }
        ]
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_status(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract floorplan polygon from splat point cloud")
    parser.add_argument("--ply", required=True)
    parser.add_argument("--measurement", default="")
    parser.add_argument("--svg-out", required=True)
    parser.add_argument("--geojson-out", required=True)
    parser.add_argument("--meta-out", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.svg_out), exist_ok=True)

    result = {
        "status": "unavailable",
        "reason": "",
        "units": "model_units",
        "scale_factor_m_per_model_unit": None,
        "floor_inliers": 0,
        "polygon_points": 0
    }

    points = _load_points(args.ply)
    if points is None:
        result["reason"] = f"Invalid or missing point cloud: {args.ply}"
        _write_status(args.meta_out, result)
        return

    floor = _fit_floor_plane(points)
    if floor is None:
        result["reason"] = "Unable to find robust floor plane"
        _write_status(args.meta_out, result)
        return

    n, d, inliers, threshold = floor
    floor_points = points[inliers]

    center = floor_points.mean(axis=0)
    axis_u, axis_v = _make_plane_basis(n)
    rel = floor_points - center
    uv = np.column_stack((rel @ axis_u, rel @ axis_v))

    polygon = _points_to_polygon(uv)
    if polygon is None or len(polygon) < 3:
        result["reason"] = "Unable to derive floor contour"
        result["floor_inliers"] = int(len(floor_points))
        _write_status(args.meta_out, result)
        return

    scale = _load_scale(args.measurement)
    if scale is not None:
        polygon_scaled = polygon * scale
        units = "meters"
    else:
        polygon_scaled = polygon
        units = "model_units"

    _polygon_to_svg(polygon_scaled, args.svg_out, units)
    _write_geojson(polygon_scaled, args.geojson_out)

    result.update({
        "status": "ok",
        "reason": "",
        "units": units,
        "scale_factor_m_per_model_unit": float(scale) if scale is not None else None,
        "floor_inliers": int(len(floor_points)),
        "polygon_points": int(len(polygon_scaled)),
        "ransac_threshold": float(threshold),
        "plane_normal": [float(x) for x in n],
        "plane_offset": float(d),
        "outputs": {
            "svg": args.svg_out,
            "geojson": args.geojson_out
        }
    })
    _write_status(args.meta_out, result)


if __name__ == "__main__":
    main()
