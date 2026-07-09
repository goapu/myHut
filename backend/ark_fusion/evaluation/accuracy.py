from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import open3d as o3d


def _load_geometry(path: Path) -> Tuple[o3d.geometry.PointCloud, Optional[o3d.geometry.TriangleMesh]]:
    """Load a mesh or point cloud and return a point cloud plus optional mesh."""
    suffix = path.suffix.lower()
    mesh: Optional[o3d.geometry.TriangleMesh] = None
    pcd = o3d.geometry.PointCloud()
    if suffix in {".ply", ".obj", ".stl", ".glb", ".gltf", ".off"}:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if hasattr(mesh, "vertices") and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            sample_n = min(300_000, max(20_000, len(mesh.vertices) * 8))
            pcd = mesh.sample_points_uniformly(number_of_points=int(sample_n))
            return pcd, mesh
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"Could not load usable geometry from {path}")
    return pcd, mesh


def _sample_mesh_or_cloud(path: Path, sample_points: int) -> Tuple[o3d.geometry.PointCloud, Optional[o3d.geometry.TriangleMesh]]:
    pcd, mesh = _load_geometry(path)
    if len(pcd.points) > sample_points:
        pcd = pcd.farthest_point_down_sample(sample_points) if hasattr(pcd, "farthest_point_down_sample") else pcd.random_down_sample(sample_points / len(pcd.points))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=40))
    return pcd, mesh


def _nn_distances(src: o3d.geometry.PointCloud, dst: o3d.geometry.PointCloud) -> np.ndarray:
    if len(src.points) == 0 or len(dst.points) == 0:
        return np.array([], dtype=np.float64)
    tree = o3d.geometry.KDTreeFlann(dst)
    pts = np.asarray(src.points)
    d = np.empty(len(pts), dtype=np.float64)
    for i, p in enumerate(pts):
        _, _, dist2 = tree.search_knn_vector_3d(p, 1)
        d[i] = float(np.sqrt(dist2[0])) if dist2 else np.inf
    return d


def _normal_consistency(src: o3d.geometry.PointCloud, dst: o3d.geometry.PointCloud) -> Optional[float]:
    if len(src.points) == 0 or len(dst.points) == 0 or len(src.normals) == 0 or len(dst.normals) == 0:
        return None
    tree = o3d.geometry.KDTreeFlann(dst)
    normals_src = np.asarray(src.normals)
    normals_dst = np.asarray(dst.normals)
    pts = np.asarray(src.points)
    vals: List[float] = []
    for i, p in enumerate(pts):
        _, idx, _ = tree.search_knn_vector_3d(p, 1)
        if idx:
            vals.append(abs(float(np.dot(normals_src[i], normals_dst[int(idx[0])]))))
    return float(np.mean(vals)) if vals else None


def _bbox_diag(pcd: o3d.geometry.PointCloud) -> float:
    if len(pcd.points) == 0:
        return 0.0
    extent = np.asarray(pcd.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    return float(np.linalg.norm(extent))


def _mesh_watertightness(mesh: Optional[o3d.geometry.TriangleMesh]) -> Dict[str, Any]:
    if mesh is None or len(mesh.triangles) == 0:
        return {"available": False}
    out: Dict[str, Any] = {"available": True}
    for name in ("is_watertight", "is_edge_manifold", "is_vertex_manifold", "is_self_intersecting"):
        fn = getattr(mesh, name, None)
        if callable(fn):
            try:
                out[name] = bool(fn())
            except TypeError:
                out[name] = bool(fn(True))
            except Exception:
                out[name] = None
    return out


@dataclass
class GeometryAccuracyReport:
    chamfer_l1_m: float
    predicted_to_gt_mean_m: float
    gt_to_pred_mean_m: float
    predicted_to_gt_p95_m: float
    gt_to_pred_p95_m: float
    completeness_at_thresholds: Dict[str, float]
    precision_at_thresholds: Dict[str, float]
    fscore_at_thresholds: Dict[str, float]
    normal_consistency: Optional[float]
    scale_error_ratio: Optional[float]
    watertightness: Dict[str, Any] = field(default_factory=dict)


def evaluate_geometry_accuracy(
    predicted_path: Path,
    ground_truth_path: Path,
    thresholds_m: Iterable[float] = (0.005, 0.010, 0.020),
    sample_points: int = 200_000,
) -> GeometryAccuracyReport:
    pred, pred_mesh = _sample_mesh_or_cloud(predicted_path, sample_points)
    gt, _ = _sample_mesh_or_cloud(ground_truth_path, sample_points)
    pred_to_gt = _nn_distances(pred, gt)
    gt_to_pred = _nn_distances(gt, pred)
    if pred_to_gt.size == 0 or gt_to_pred.size == 0:
        raise ValueError("Empty predicted or ground-truth geometry")
    precision: Dict[str, float] = {}
    completeness: Dict[str, float] = {}
    fscore: Dict[str, float] = {}
    for t in thresholds_m:
        key = f"{int(round(t * 1000))}mm"
        p = float(np.mean(pred_to_gt <= t))
        r = float(np.mean(gt_to_pred <= t))
        precision[key] = p
        completeness[key] = r
        fscore[key] = float(2 * p * r / max(p + r, 1e-12))
    pred_diag = _bbox_diag(pred)
    gt_diag = _bbox_diag(gt)
    scale_error = None if gt_diag <= 1e-9 else float(abs(pred_diag - gt_diag) / gt_diag)
    nc1 = _normal_consistency(pred, gt)
    nc2 = _normal_consistency(gt, pred)
    nc = None if nc1 is None or nc2 is None else float((nc1 + nc2) * 0.5)
    return GeometryAccuracyReport(
        chamfer_l1_m=float(np.mean(pred_to_gt) + np.mean(gt_to_pred)),
        predicted_to_gt_mean_m=float(np.mean(pred_to_gt)),
        gt_to_pred_mean_m=float(np.mean(gt_to_pred)),
        predicted_to_gt_p95_m=float(np.percentile(pred_to_gt, 95)),
        gt_to_pred_p95_m=float(np.percentile(gt_to_pred, 95)),
        completeness_at_thresholds=completeness,
        precision_at_thresholds=precision,
        fscore_at_thresholds=fscore,
        normal_consistency=nc,
        scale_error_ratio=scale_error,
        watertightness=_mesh_watertightness(pred_mesh),
    )


def evaluate_room_measurements(predicted_json: Path, ground_truth_json: Path) -> Dict[str, Any]:
    pred = json.loads(predicted_json.read_text(encoding="utf-8"))
    gt = json.loads(ground_truth_json.read_text(encoding="utf-8"))
    fields = ["height_m", "length_m", "width_m"]
    errors: Dict[str, Any] = {}
    abs_values = []
    for f in fields:
        if f in pred and f in gt and pred[f] is not None and gt[f] is not None:
            abs_err = abs(float(pred[f]) - float(gt[f]))
            rel_err = abs_err / max(abs(float(gt[f])), 1e-9)
            errors[f] = {"predicted_m": float(pred[f]), "ground_truth_m": float(gt[f]), "absolute_error_m": abs_err, "relative_error": rel_err}
            abs_values.append(abs_err)
    if "planes" in gt and "planes" in pred:
        errors["plane_residuals_available"] = True
    errors["mean_absolute_error_m"] = float(np.mean(abs_values)) if abs_values else None
    errors["max_absolute_error_m"] = float(np.max(abs_values)) if abs_values else None
    return errors


def write_accuracy_report(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.write_text(json.dumps(payload, indent=2, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o)), encoding="utf-8")
