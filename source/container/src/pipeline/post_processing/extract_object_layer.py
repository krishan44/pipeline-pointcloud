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


def _fit_floor_plane(
    points: np.ndarray,
    min_vertical_axis_component: float,
    max_iters: int = 700
) -> Optional[Tuple[np.ndarray, float, np.ndarray, float, float]]:
    if len(points) < 50:
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

        dominant_axis_component = float(np.max(np.abs(n)))
        if dominant_axis_component < float(min_vertical_axis_component):
            continue

        distances = np.abs(points @ n + d)
        inliers = distances < threshold
        inlier_count = int(np.sum(inliers))
        if inlier_count > best_count and inlier_count >= 20:
            best = (n, d, inliers, threshold, dominant_axis_component)
            best_count = inlier_count

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


def _extract_object_polygons(
    uv_points: np.ndarray,
    min_area_units2: float,
    scale_m_per_unit: Optional[float]
) -> list[dict]:
    if len(uv_points) < 50:
        return []

    mins = uv_points.min(axis=0)
    maxs = uv_points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    max_span = float(max(span[0], span[1]))

    grid_size = 900
    px = max_span / float(grid_size)
    if px <= 0:
        return []

    width = max(64, int(np.ceil(span[0] / px)) + 8)
    height = max(64, int(np.ceil(span[1] / px)) + 8)

    normalized = (uv_points - mins) / px
    xi = np.clip(np.round(normalized[:, 0]).astype(np.int32) + 4, 0, width - 1)
    yi = np.clip(np.round(normalized[:, 1]).astype(np.int32) + 4, 0, height - 1)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[yi, xi] = 255

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 5)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)

    polygons = []
    for label_id in range(1, num_labels):
        area_px = int(stats[label_id, cv2.CC_STAT_AREA])
        if area_px < 80:
            continue

        component_mask = np.where(labels == label_id, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
        if approx is None or len(approx) < 3:
            continue

        poly_px = approx[:, 0, :].astype(np.float64)
        poly_uv = np.empty_like(poly_px)
        poly_uv[:, 0] = (poly_px[:, 0] - 4.0) * px + mins[0]
        poly_uv[:, 1] = (poly_px[:, 1] - 4.0) * px + mins[1]

        area_units2 = abs(float(cv2.contourArea(poly_uv.astype(np.float32))))
        if scale_m_per_unit is not None:
            area_scaled = area_units2 * (scale_m_per_unit ** 2)
        else:
            area_scaled = area_units2

        if area_scaled < min_area_units2:
            continue

        polygons.append({
            "polygon": poly_uv,
            "area": area_scaled
        })

    return polygons


def _load_floor_polygon(geojson_path: str) -> Optional[np.ndarray]:
    if not geojson_path or not os.path.isfile(geojson_path):
        return None
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        features = data.get("features", [])
        if not features:
            return None
        coords = features[0].get("geometry", {}).get("coordinates", [])
        if not coords or not coords[0]:
            return None
        ring = np.asarray(coords[0], dtype=np.float64)
        if len(ring) >= 2 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        if ring.ndim != 2 or ring.shape[1] != 2 or len(ring) < 3:
            return None
        return ring
    except Exception:
        return None


def _write_geojson(polygons: list[dict], units: str, out_path: str) -> None:
    features = []
    for i, item in enumerate(polygons, start=1):
        poly = item["polygon"]
        ring = [[float(x), float(y)] for x, y in poly]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        features.append({
            "type": "Feature",
            "properties": {
                "object_id": f"obj_{i:03d}",
                "pattern_group": f"pattern_{i:03d}",
                "footprint_area": float(item["area"]),
                "area_units": f"{units}^2"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring]
            }
        })

    payload = {
        "type": "FeatureCollection",
        "features": features
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_svg(floor_polygon: Optional[np.ndarray], objects: list[dict], units: str, out_path: str) -> None:
    all_points = []
    if floor_polygon is not None:
        all_points.append(floor_polygon)
    for item in objects:
        all_points.append(item["polygon"])

    if not all_points:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"></svg>\n')
        return

    stacked = np.vstack(all_points)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)

    margin = 20.0
    width = float(span[0] + 2 * margin)
    height = float(span[1] + 2 * margin)

    def _svg_points(poly: np.ndarray) -> str:
        shifted = poly - mins + margin
        points = []
        for x, y in shifted:
            points.append(f"{x:.3f},{(height - y):.3f}")
        return " ".join(points)

    palette = ["#f59e0b66", "#10b98166", "#6366f166", "#ef444466", "#06b6d466", "#f9731666"]

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.3f} {height:.3f}">']
    lines.append('  <rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')

    if floor_polygon is not None and len(floor_polygon) >= 3:
        lines.append(f'  <polygon points="{_svg_points(floor_polygon)}" fill="#93c5fd33" stroke="#1d4ed8" stroke-width="0.06"/>')

    for i, item in enumerate(objects):
        poly = item["polygon"]
        fill = palette[i % len(palette)]
        lines.append(f'  <polygon points="{_svg_points(poly)}" fill="{fill}" stroke="#111827" stroke-width="0.04"/>')

    lines.append(f'  <text x="6" y="16" font-size="12" fill="#111827">Object layer ({units})</text>')
    lines.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_meta(meta_path: str, payload: dict) -> None:
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract object-aware layer from point cloud for floorplan annotation")
    parser.add_argument("--ply", required=True)
    parser.add_argument("--measurement", default="")
    parser.add_argument("--floorplan-geojson", default="")
    parser.add_argument("--images-dir", default="")
    parser.add_argument("--geojson-out", required=True)
    parser.add_argument("--svg-out", required=True)
    parser.add_argument("--meta-out", required=True)
    parser.add_argument("--min-object-area", type=float, default=0.25)
    parser.add_argument("--min-height", type=float, default=0.06)
    parser.add_argument("--max-height", type=float, default=2.8)
    parser.add_argument(
        "--min-vertical-axis-component",
        default=0.85,
        type=float,
        help="Minimum dominant absolute plane-normal component to accept floor candidates (default: 0.85)"
    )
    args = parser.parse_args()

    for output_path in (args.geojson_out, args.svg_out, args.meta_out):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    meta = {
        "status": "unavailable",
        "reason": "",
        "units": "model_units",
        "objects": 0,
        "object_points": 0,
        "sam2_note": "Not projected from SAM2 masks in this release; geometry-pattern object layer from reconstructed points."
    }

    scale = _load_scale(args.measurement)
    if scale is not None:
        meta["units"] = "meters"
        meta["scale_factor_m_per_model_unit"] = float(scale)

    floor_polygon = _load_floor_polygon(args.floorplan_geojson)

    def _write_unavailable(reason: str) -> None:
        meta["reason"] = reason
        _write_geojson([], meta["units"], args.geojson_out)
        _write_svg(floor_polygon, [], meta["units"], args.svg_out)
        _write_meta(args.meta_out, meta)

    points = _load_points(args.ply)
    if points is None:
        _write_unavailable(f"Missing or invalid PLY: {args.ply}")
        return

    if args.min_vertical_axis_component < 0 or args.min_vertical_axis_component > 1:
        _write_unavailable("min_vertical_axis_component must be in [0, 1]")
        return

    floor = _fit_floor_plane(points, float(args.min_vertical_axis_component))
    if floor is None:
        _write_unavailable("Could not detect floor plane")
        return

    n, d, floor_inliers, threshold, dominant_axis_component = floor
    signed = points @ n + d
    if float(np.median(signed)) < 0:
        n = -n
        d = -d
        signed = -signed

    non_floor = np.abs(points @ n + d) > (threshold * 1.5)
    heights = points @ n + d
    object_height_mask = (heights > args.min_height) & (heights < args.max_height)
    object_points = points[non_floor & object_height_mask]

    if len(object_points) < 40:
        _write_unavailable("Too few object points above floor")
        return

    center = points[floor_inliers].mean(axis=0)
    axis_u, axis_v = _make_plane_basis(n)
    rel = object_points - center
    uv = np.column_stack((rel @ axis_u, rel @ axis_v))

    units = "meters" if scale is not None else "model_units"

    if scale is not None:
        uv_scaled = uv * scale
        min_area = float(max(args.min_object_area, 0.01))
    else:
        uv_scaled = uv
        min_area = float(max(args.min_object_area, 0.01))

    polygons = _extract_object_polygons(
        uv_scaled,
        min_area_units2=min_area,
        scale_m_per_unit=None
    )

    _write_geojson(polygons, units, args.geojson_out)
    _write_svg(floor_polygon, polygons, units, args.svg_out)

    total_point_count = int(len(points))
    floor_inlier_count = int(np.sum(floor_inliers))
    inlier_ratio = floor_inlier_count / max(total_point_count, 1)

    if total_point_count < 500 or floor_inlier_count < 50 or inlier_ratio < 0.03:
        reconstruction_quality = "sparse"
    elif total_point_count > 20000 and floor_inlier_count > 1000:
        reconstruction_quality = "dense"
    else:
        reconstruction_quality = "normal"

    meta.update({
        "status": "ok",
        "reason": "",
        "units": units,
        "scale_factor_m_per_model_unit": float(scale) if scale is not None else None,
        "objects": int(len(polygons)),
        "object_points": int(len(object_points)),
        "reconstruction_quality": reconstruction_quality,
        "low_confidence": reconstruction_quality == "sparse",
        "scene_validity_note": (
            "Object layer extraction assumes a roughly planar floor is visible in the "
            "point cloud. Results may be unreliable for non-room scenes."
        ),
        "floor_plane": {
            "normal": [float(x) for x in n],
            "offset": float(d),
            "threshold": float(threshold),
            "inliers": floor_inlier_count,
            "inlier_ratio": round(inlier_ratio, 4),
            "dominant_axis_component": float(dominant_axis_component),
            "min_vertical_axis_component": float(args.min_vertical_axis_component)
        },
        "outputs": {
            "geojson": args.geojson_out,
            "svg": args.svg_out
        }
    })
    _write_meta(args.meta_out, meta)


if __name__ == "__main__":
    main()
