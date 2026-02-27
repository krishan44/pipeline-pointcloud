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
        count = int(np.sum(inliers))
        if count > best_count and count >= 20:
            best_count = count
            best = (n, d, inliers, threshold, dominant_axis_component)

    return best


def _fallback_plane_basis_from_pca(points: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if len(points) < 10:
        return None

    centered = points - points.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return None

    if vh.shape != (3, 3):
        return None

    axis_u = vh[0].astype(np.float64)
    axis_v = vh[1].astype(np.float64)
    axis_n = vh[2].astype(np.float64)

    axis_u = axis_u / (np.linalg.norm(axis_u) + 1e-12)
    axis_v = axis_v / (np.linalg.norm(axis_v) + 1e-12)
    axis_n = axis_n / (np.linalg.norm(axis_n) + 1e-12)
    return axis_u, axis_v, axis_n


def _fallback_rectangle_polygon(points_2d: np.ndarray) -> Optional[np.ndarray]:
    if len(points_2d) < 3:
        return None

    pts = points_2d.astype(np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    if box is None or len(box) < 3:
        return None
    return box.astype(np.float64)


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
    if len(points_2d) < 5:
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
    """Render a floorplan polygon as a rich SVG with wall-length annotations and a scale bar."""
    if len(points) < 3:
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    size = np.maximum(maxs - mins, 1e-6)

    # Margins: left/bottom for annotations, top/right for breathing room.
    margin_l = 60.0
    margin_b = 60.0
    margin_t = 30.0
    margin_r = 30.0

    view_w = float(size[0]) + margin_l + margin_r
    view_h = float(size[1]) + margin_t + margin_b

    # Coordinate helpers (SVG y-axis is flipped)
    def to_svg(p):
        x = (p[0] - mins[0]) + margin_l
        y = view_h - ((p[1] - mins[1]) + margin_b)
        return x, y

    # Build polygon points string
    svg_pts = [to_svg(p) for p in points]
    coords_str = " ".join(f"{x:.3f},{y:.3f}" for x, y in svg_pts)

    # font_sz must be computed before wall-label loop (label suppression depends on it)
    font_sz = max(4.0, min(view_w, view_h) * 0.025)

    # Wall segments with midpoint labels.
    # Labels are suppressed for walls whose screen-space length is too short to
    # fit readable text (avoids overlapping labels on alcoves / dense polygons).
    n_pts = len(points)
    wall_labels = []
    for i in range(n_pts):
        a = points[i]
        b = points[(i + 1) % n_pts]
        length = float(np.linalg.norm(b - a))
        ax, ay = to_svg(a)
        bx, by = to_svg(b)
        seg_screen_len = float(np.hypot(bx - ax, by - ay))
        # Minimum screen length required to render a label without overlapping
        # neighbouring labels (≈4× font height is a comfortable minimum).
        if seg_screen_len < font_sz * 4.0:
            continue
        mx = (ax + bx) / 2.0
        my = (ay + by) / 2.0
        # Offset the label slightly outward from the edge midpoint
        dx = bx - ax
        dy = by - ay
        seg_len_svg = max(seg_screen_len, 1e-6)
        # Normal pointing "outward" (arbitrary sign — just for readability)
        nx_n = -dy / seg_len_svg
        ny_n = dx / seg_len_svg
        offset = min(font_sz * 1.5, 8.0)
        lx = mx + nx_n * offset
        ly = my + ny_n * offset
        # Rotation angle for label to follow wall direction
        angle_deg = float(np.degrees(np.arctan2(by - ay, bx - ax)))
        if abs(angle_deg) > 90:
            angle_deg += 180
        if units_label == "meters":
            label_text = f"{length:.2f} m"
        else:
            label_text = f"{length:.3f} u"
        wall_labels.append((lx, ly, angle_deg, label_text))

    # Scale bar: pick a round number (~15 % of width)
    scene_width = float(size[0])
    raw_bar = scene_width * 0.15
    # Round to nearest 0.1 / 0.5 / 1 / 5 / 10 …
    magnitude = 10 ** int(np.floor(np.log10(max(raw_bar, 1e-6))))
    for frac in (1, 2, 5, 10):
        bar_val = frac * magnitude
        if bar_val >= raw_bar:
            break
    bar_px = bar_val / scene_width * float(size[0])  # pixels in SVG space
    if units_label == "meters":
        bar_label = f"{bar_val:.4g} m"
    else:
        bar_label = f"{bar_val:.4g} u"
    bar_x0 = margin_l
    bar_y  = view_h - margin_b * 0.35
    bar_x1 = bar_x0 + bar_px

    # font_sz already computed above; kept here for the scale-bar / units labels.

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_w:.3f} {view_h:.3f}" '
                 f'style="background:#f8f9fa">')
    lines.append(f'  <!-- Background grid -->')
    lines.append(f'  <defs>')
    lines.append(f'    <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">')
    lines.append(f'      <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#e0e0e0" stroke-width="0.3"/>')
    lines.append(f'    </pattern>')
    lines.append(f'  </defs>')
    lines.append(f'  <rect width="{view_w:.3f}" height="{view_h:.3f}" fill="url(#grid)"/>')
    lines.append(f'  <!-- Floorplan polygon -->')
    lines.append(f'  <polygon points="{coords_str}" '
                 f'fill="#cce7ff" fill-opacity="0.5" stroke="#1565c0" stroke-width="{max(0.8, view_w*0.003):.2f}"/>')
    lines.append(f'  <!-- Wall-length labels -->')
    for lx, ly, angle, text in wall_labels:
        lines.append(
            f'  <text x="{lx:.3f}" y="{ly:.3f}" '
            f'font-size="{font_sz:.2f}" font-family="sans-serif" fill="#333" '
            f'text-anchor="middle" dominant-baseline="middle" '
            f'transform="rotate({angle:.2f},{lx:.3f},{ly:.3f})">{text}</text>'
        )
    lines.append(f'  <!-- Scale bar -->')
    lines.append(f'  <line x1="{bar_x0:.3f}" y1="{bar_y:.3f}" x2="{bar_x1:.3f}" y2="{bar_y:.3f}" '
                 f'stroke="#333" stroke-width="{max(1.0, view_w*0.003):.2f}"/>')
    lines.append(f'  <line x1="{bar_x0:.3f}" y1="{bar_y - 3:.3f}" x2="{bar_x0:.3f}" y2="{bar_y + 3:.3f}" '
                 f'stroke="#333" stroke-width="{max(1.0, view_w*0.003):.2f}"/>')
    lines.append(f'  <line x1="{bar_x1:.3f}" y1="{bar_y - 3:.3f}" x2="{bar_x1:.3f}" y2="{bar_y + 3:.3f}" '
                 f'stroke="#333" stroke-width="{max(1.0, view_w*0.003):.2f}"/>')
    lines.append(f'  <text x="{(bar_x0+bar_x1)/2:.3f}" y="{bar_y + font_sz * 1.4:.3f}" '
                 f'font-size="{font_sz:.2f}" font-family="sans-serif" fill="#333" text-anchor="middle">{bar_label}</text>')
    lines.append(f'  <!-- Units label -->')
    lines.append(f'  <text x="{margin_l:.3f}" y="{font_sz * 1.4:.3f}" '
                 f'font-size="{font_sz:.2f}" font-family="sans-serif" fill="#555">Units: {units_label}</text>')
    lines.append('</svg>')

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _polygon_to_png(points: np.ndarray, out_path: str, units_label: str) -> None:
    if len(points) < 3:
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    size = np.maximum(maxs - mins, 1e-6)

    canvas_w = 1280
    canvas_h = 960
    margin_l = 120
    margin_r = 80
    margin_t = 80
    margin_b = 140

    draw_w = max(canvas_w - margin_l - margin_r, 64)
    draw_h = max(canvas_h - margin_t - margin_b, 64)

    scale_xy = min(draw_w / float(size[0]), draw_h / float(size[1]))
    if scale_xy <= 0:
        return

    x_offset = margin_l + (draw_w - float(size[0]) * scale_xy) * 0.5
    y_offset = margin_t + (draw_h - float(size[1]) * scale_xy) * 0.5

    def to_px(p):
        x = int(round((p[0] - mins[0]) * scale_xy + x_offset))
        y = int(round(canvas_h - ((p[1] - mins[1]) * scale_xy + y_offset)))
        return x, y

    image = np.full((canvas_h, canvas_w, 3), 248, dtype=np.uint8)

    grid_step = max(int(round(min(draw_w, draw_h) / 24.0)), 20)
    for x in range(0, canvas_w, grid_step):
        cv2.line(image, (x, 0), (x, canvas_h - 1), (230, 230, 230), 1)
    for y in range(0, canvas_h, grid_step):
        cv2.line(image, (0, y), (canvas_w - 1, y), (230, 230, 230), 1)

    pts = np.array([to_px(p) for p in points], dtype=np.int32)
    cv2.fillPoly(image, [pts], color=(255, 231, 204))
    cv2.polylines(image, [pts], isClosed=True, color=(192, 101, 21), thickness=3)

    walls = _wall_lengths(points)
    n_pts = len(points)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    for i in range(n_pts):
        a = points[i]
        b = points[(i + 1) % n_pts]
        ax, ay = to_px(a)
        bx, by = to_px(b)
        seg_len = float(np.hypot(bx - ax, by - ay))
        if seg_len < 80:
            continue

        mx = int(round((ax + bx) / 2.0))
        my = int(round((ay + by) / 2.0))
        dx = bx - ax
        dy = by - ay
        inv = 1.0 / max(seg_len, 1e-6)
        nx = int(round(-dy * inv * 16.0))
        ny = int(round(dx * inv * 16.0))

        if units_label == "meters":
            text = f"{walls[i]:.2f} m"
        else:
            text = f"{walls[i]:.3f} u"

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        tx = mx + nx - tw // 2
        ty = my + ny + th // 2

        cv2.rectangle(
            image,
            (tx - 6, ty - th - 6),
            (tx + tw + 6, ty + baseline + 6),
            (248, 248, 248),
            -1
        )
        cv2.putText(image, text, (tx, ty), font, font_scale, (40, 40, 40), thickness, cv2.LINE_AA)

    width_units = float(size[0])
    raw_bar = width_units * 0.15
    magnitude = 10 ** int(np.floor(np.log10(max(raw_bar, 1e-6))))
    bar_val = magnitude
    for frac in (1, 2, 5, 10):
        candidate = frac * magnitude
        if candidate >= raw_bar:
            bar_val = candidate
            break

    bar_len_px = int(round(bar_val * scale_xy))
    bar_len_px = max(bar_len_px, 40)
    bar_x0 = margin_l
    bar_y = canvas_h - margin_b // 2
    bar_x1 = min(bar_x0 + bar_len_px, canvas_w - margin_r)

    cv2.line(image, (bar_x0, bar_y), (bar_x1, bar_y), (50, 50, 50), 3)
    cv2.line(image, (bar_x0, bar_y - 8), (bar_x0, bar_y + 8), (50, 50, 50), 3)
    cv2.line(image, (bar_x1, bar_y - 8), (bar_x1, bar_y + 8), (50, 50, 50), 3)

    if units_label == "meters":
        bar_text = f"{bar_val:.4g} m"
    else:
        bar_text = f"{bar_val:.4g} u"
    cv2.putText(
        image,
        bar_text,
        (bar_x0 + max((bar_x1 - bar_x0) // 2 - 40, 0), bar_y + 38),
        font,
        0.8,
        (40, 40, 40),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        f"Units: {units_label}",
        (margin_l, margin_t // 2 + 10),
        font,
        0.8,
        (60, 60, 60),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(out_path, image)


def _polygon_area(pts: np.ndarray) -> float:
    """Shoelace formula for signed polygon area (returns absolute value)."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += float(pts[i, 0]) * float(pts[j, 1])
        area -= float(pts[j, 0]) * float(pts[i, 1])
    return abs(area) / 2.0


def _wall_lengths(pts: np.ndarray) -> list:
    """Return wall-segment lengths parallel to the polygon coordinate ring.

    The returned list has exactly ``len(pts)`` entries.  Entry *i* is the
    length of the edge from vertex *i* to vertex *(i+1) % N*, so the last
    entry is the closing edge from the last vertex back to the first.
    This 1-to-1 correspondence with the ring vertices is preserved in both
    the GeoJSON ``wall_lengths`` property and the metadata JSON.
    """
    n = len(pts)
    return [float(np.linalg.norm(pts[(i + 1) % n] - pts[i])) for i in range(n)]


def _write_geojson(points: np.ndarray, out_path: str) -> None:
    if len(points) < 3:
        return
    ring = [[float(x), float(y)] for x, y in points]
    if ring[0] != ring[-1]:
        ring.append(ring[0])

    walls = _wall_lengths(points)
    area = _polygon_area(points)
    perimeter = float(sum(walls))

    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "floorplan",
                    "area": round(area, 4),
                    "perimeter": round(perimeter, 4),
                    "wall_lengths": [round(w, 4) for w in walls]
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
    parser.add_argument(
        "--min-vertical-axis-component",
        default=0.85,
        type=float,
        help="Minimum dominant absolute plane-normal component to accept floor candidates (default: 0.85)"
    )
    parser.add_argument("--svg-out", required=True)
    parser.add_argument("--png-out", required=True)
    parser.add_argument("--geojson-out", required=True)
    parser.add_argument("--meta-out", required=True)
    args = parser.parse_args()

    for output_path in (args.svg_out, args.png_out, args.geojson_out, args.meta_out):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    result = {
        "status": "unavailable",
        "reason": "",
        "units": "model_units",
        "scale_factor_m_per_model_unit": None,
        "floor_inliers": 0,
        "polygon_points": 0
    }

    if args.min_vertical_axis_component < 0 or args.min_vertical_axis_component > 1:
        result["reason"] = "min_vertical_axis_component must be in [0, 1]"
        _write_status(args.meta_out, result)
        return

    points = _load_points(args.ply)
    if points is None:
        result["reason"] = f"Invalid or missing point cloud: {args.ply}"
        _write_status(args.meta_out, result)
        return

    floor = _fit_floor_plane(points, float(args.min_vertical_axis_component))
    used_fallback_projection = False

    if floor is None:
        pca_basis = _fallback_plane_basis_from_pca(points)
        if pca_basis is None:
            result["reason"] = "Unable to find robust floor plane or fallback PCA basis"
            _write_status(args.meta_out, result)
            return

        axis_u, axis_v, n = pca_basis
        d = -float(np.dot(n, points.mean(axis=0)))
        threshold = 0.0
        dominant_axis_component = float(np.max(np.abs(n)))
        floor_points = points
        center = points.mean(axis=0)
        rel = points - center
        uv = np.column_stack((rel @ axis_u, rel @ axis_v))
        polygon = _points_to_polygon(uv)
        if polygon is None or len(polygon) < 3:
            polygon = _fallback_rectangle_polygon(uv)
        if polygon is None or len(polygon) < 3:
            result["reason"] = "Unable to derive floor contour in fallback projection"
            _write_status(args.meta_out, result)
            return
        used_fallback_projection = True
    else:
        n, d, inliers, threshold, dominant_axis_component = floor
        floor_points = points[inliers]

        center = floor_points.mean(axis=0)
        axis_u, axis_v = _make_plane_basis(n)
        rel = floor_points - center
        uv = np.column_stack((rel @ axis_u, rel @ axis_v))

        polygon = _points_to_polygon(uv)
        if polygon is None or len(polygon) < 3:
            polygon = _fallback_rectangle_polygon(uv)
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
    _polygon_to_png(polygon_scaled, args.png_out, units)
    _write_geojson(polygon_scaled, args.geojson_out)

    walls = _wall_lengths(polygon_scaled)
    room_area = _polygon_area(polygon_scaled)
    floor_inlier_count = int(len(floor_points))
    total_point_count = int(len(points))
    inlier_ratio = floor_inlier_count / max(total_point_count, 1)

    # Reconstruction quality classification:
    #   sparse  — very few points or low inlier ratio; output may be coarse
    #   normal  — typical reconstruction
    #   dense   — high-quality point cloud
    if total_point_count < 500 or floor_inlier_count < 50 or inlier_ratio < 0.03:
        reconstruction_quality = "sparse"
    elif total_point_count > 20000 and floor_inlier_count > 1000:
        reconstruction_quality = "dense"
    else:
        reconstruction_quality = "normal"

    if used_fallback_projection:
        reconstruction_quality = "fallback_projection"

    low_confidence = reconstruction_quality == "sparse"
    if used_fallback_projection:
        low_confidence = True

    result.update({
        "status": "ok",
        "reason": "",
        "units": units,
        "scale_factor_m_per_model_unit": float(scale) if scale is not None else None,
        "floor_inliers": floor_inlier_count,
        "total_points": total_point_count,
        "floor_inlier_ratio": round(inlier_ratio, 4),
        "polygon_points": int(len(polygon_scaled)),
        "room_area": round(room_area, 4),
        "room_perimeter": round(float(sum(walls)), 4),
        # wall_lengths[i] = length of edge from vertex i to vertex (i+1)%N,
        # parallel to the GeoJSON ring coordinates.
        "wall_lengths": [round(w, 4) for w in walls],
        "reconstruction_quality": reconstruction_quality,
        "low_confidence": low_confidence,
        "scene_validity_note": (
            "Floorplan extraction assumes a roughly planar floor is visible in the "
            "point cloud. Results may be unreliable for non-room scenes (single objects, "
            "ceilings, outdoor areas, or ramps)."
        ),
        "used_fallback_projection": used_fallback_projection,
        "ransac_threshold": float(threshold),
        "plane_normal": [float(x) for x in n],
        "plane_offset": float(d),
        "plane_dominant_axis_component": float(dominant_axis_component),
        "min_vertical_axis_component": float(args.min_vertical_axis_component),
        "outputs": {
            "svg": args.svg_out,
            "png": args.png_out,
            "geojson": args.geojson_out
        }
    })
    _write_status(args.meta_out, result)


if __name__ == "__main__":
    main()
