#!/usr/bin/env python3

import argparse
import json
import os
import random
from typing import Optional, Tuple

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


def _load_camera_centers(transforms_path: str) -> Optional[np.ndarray]:
    if not os.path.isfile(transforms_path):
        return None
    with open(transforms_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", []) if isinstance(data, dict) else []
    centers = []
    for frame in frames:
        matrix = frame.get("transform_matrix") if isinstance(frame, dict) else None
        if not isinstance(matrix, list) or len(matrix) < 4:
            continue
        try:
            transform = np.array(matrix, dtype=np.float64)
            if transform.shape[0] >= 3 and transform.shape[1] >= 4:
                centers.append(transform[:3, 3])
        except Exception:
            continue

    if not centers:
        return None
    return np.vstack(centers)


def _plane_from_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        return None
    n = n / n_norm
    d = -float(np.dot(n, p1))
    return n, d


def _fit_ground_plane_ransac(
    points: np.ndarray,
    cameras: np.ndarray,
    min_vertical_axis_component: float,
    max_iters: int = 500
) -> Optional[Tuple[np.ndarray, float, int, float, float]]:
    if len(points) < 30:
        return None

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    if diag <= 0:
        return None

    threshold = max(diag * 0.005, 1e-4)
    best = None
    best_score = -1

    idxs = list(range(len(points)))
    for _ in range(max_iters):
        sample_ids = random.sample(idxs, 3)
        plane = _plane_from_points(points[sample_ids[0]], points[sample_ids[1]], points[sample_ids[2]])
        if plane is None:
            continue
        n, d = plane

        dominant_axis_component = float(np.max(np.abs(n)))
        if dominant_axis_component < float(min_vertical_axis_component):
            continue

        distances = np.abs(points @ n + d)
        inlier_mask = distances < threshold
        inlier_count = int(np.sum(inlier_mask))
        if inlier_count < 10:
            continue

        signed_camera_distances = cameras @ n + d
        # Prefer planes where cameras are mostly on one side (floor-like behavior)
        one_side_ratio = max(
            float(np.mean(signed_camera_distances > 1e-6)),
            float(np.mean(signed_camera_distances < -1e-6))
        )
        score = inlier_count * one_side_ratio
        if score > best_score:
            best_score = score
            best = (n, d, inlier_count, threshold, dominant_axis_component)

    return best


def _write_output(path: str, payload: dict) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate world scale from tripod camera height")
    parser.add_argument("--ply", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--tripod-height-m", required=True, type=float)
    parser.add_argument(
        "--min-vertical-axis-component",
        default=0.85,
        type=float,
        help="Minimum dominant absolute plane-normal component to accept floor candidates (default: 0.85)"
    )
    parser.add_argument(
        "--bbox-clip-percentile",
        default=2.0,
        type=float,
        help="Symmetric percentile clipping for robust bbox (q to 100-q, default: 2.0)"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output = {
        "status": "unavailable",
        "reference_type": "tripod_height",
        "tripod_height_m": float(args.tripod_height_m),
        "scale_factor_m_per_model_unit": None,
        "estimated_camera_height_model_units": None,
        "camera_height_std_m": None,
        "scale_confidence": "unreliable",
        "diagnostics": {}
    }

    try:
        if args.tripod_height_m <= 0:
            output["diagnostics"]["reason"] = "tripod_height_m must be > 0"
            _write_output(args.out, output)
            return

        if args.min_vertical_axis_component < 0 or args.min_vertical_axis_component > 1:
            output["diagnostics"]["reason"] = "min_vertical_axis_component must be in [0, 1]"
            _write_output(args.out, output)
            return

        if args.bbox_clip_percentile < 0 or args.bbox_clip_percentile >= 50:
            output["diagnostics"]["reason"] = "bbox_clip_percentile must be in [0, 50)"
            _write_output(args.out, output)
            return

        points = _load_points(args.ply)
        if points is None:
            output["diagnostics"]["reason"] = f"Point cloud not found or invalid: {args.ply}"
            _write_output(args.out, output)
            return

        cameras = _load_camera_centers(args.transforms)
        if cameras is None:
            output["diagnostics"]["reason"] = f"Transforms not found or invalid: {args.transforms}"
            _write_output(args.out, output)
            return

        fit = _fit_ground_plane_ransac(
            points,
            cameras,
            min_vertical_axis_component=float(args.min_vertical_axis_component)
        )
        if fit is None:
            output["diagnostics"]["reason"] = "Unable to estimate floor plane robustly"
            _write_output(args.out, output)
            return

        n, d, inlier_count, threshold, dominant_axis_component = fit
        signed = cameras @ n + d
        # Orient normal so camera distances are mostly positive
        if float(np.median(signed)) < 0:
            n = -n
            d = -d
            signed = -signed

        valid_dist = signed[signed > 1e-6]
        if len(valid_dist) == 0:
            output["diagnostics"]["reason"] = "No positive camera-to-floor distances found"
            _write_output(args.out, output)
            return

        camera_height_model = float(np.median(valid_dist))
        camera_height_std_model = float(np.std(valid_dist))
        if camera_height_model <= 1e-6:
            output["diagnostics"]["reason"] = "Estimated camera height in model units is too small"
            _write_output(args.out, output)
            return

        height_cv = float(camera_height_std_model / max(camera_height_model, 1e-12))
        if height_cv <= 0.5:
            scale_confidence = "ok"
        elif height_cv <= 1.0:
            scale_confidence = "low"
        else:
            scale_confidence = "unreliable"

        scale = float(args.tripod_height_m / camera_height_model)

        output["status"] = "ok"
        output["scale_factor_m_per_model_unit"] = scale
        output["estimated_camera_height_model_units"] = camera_height_model
        output["camera_height_std_m"] = float(camera_height_std_model * scale)
        output["scale_confidence"] = scale_confidence
        output["diagnostics"] = {
            "points_count": int(len(points)),
            "camera_count": int(len(cameras)),
            "plane_inliers": int(inlier_count),
            "ransac_threshold": float(threshold),
            "plane_normal": [float(x) for x in n],
            "plane_offset": float(d),
            "plane_dominant_axis_component": float(dominant_axis_component),
            "min_vertical_axis_component": float(args.min_vertical_axis_component),
            "camera_height_std_model_units": float(camera_height_std_model),
            "camera_height_cv": float(height_cv)
        }

        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        dims_model_raw = bbox_max - bbox_min

        q = float(args.bbox_clip_percentile)
        lower = np.percentile(points, q, axis=0)
        upper = np.percentile(points, 100.0 - q, axis=0)
        dims_model_clipped = upper - lower
        inside_mask = np.all((points >= lower) & (points <= upper), axis=1)
        bbox_outlier_fraction = float(1.0 - np.mean(inside_mask))

        output["bbox_dimensions_m"] = {
            "x": float(dims_model_clipped[0] * scale),
            "y": float(dims_model_clipped[1] * scale),
            "z": float(dims_model_clipped[2] * scale)
        }
        output["bbox_dimensions_m_raw"] = {
            "x": float(dims_model_raw[0] * scale),
            "y": float(dims_model_raw[1] * scale),
            "z": float(dims_model_raw[2] * scale)
        }
        output["diagnostics"].update({
            "bbox_clip_percentile": float(q),
            "bbox_outlier_fraction": float(bbox_outlier_fraction),
            "bbox_dimensions_model_units_raw": {
                "x": float(dims_model_raw[0]),
                "y": float(dims_model_raw[1]),
                "z": float(dims_model_raw[2])
            },
            "bbox_dimensions_model_units_clipped": {
                "x": float(dims_model_clipped[0]),
                "y": float(dims_model_clipped[1]),
                "z": float(dims_model_clipped[2])
            },
            "bbox_dimensions_m_raw": {
                "x": float(dims_model_raw[0] * scale),
                "y": float(dims_model_raw[1] * scale),
                "z": float(dims_model_raw[2] * scale)
            },
            "bbox_dimensions_m_clipped": {
                "x": float(dims_model_clipped[0] * scale),
                "y": float(dims_model_clipped[1] * scale),
                "z": float(dims_model_clipped[2] * scale)
            }
        })

        _write_output(args.out, output)
    except Exception as e:
        output["diagnostics"]["reason"] = f"Unexpected error: {str(e)}"
        _write_output(args.out, output)


if __name__ == "__main__":
    main()
