from __future__ import annotations
from .common import *
from .io import *

@dataclass
class DatasetValidationReport:
    valid: bool
    frame_count: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    depth_shape_histogram: Dict[str, int] = field(default_factory=dict)
    rgb_shape_histogram: Dict[str, int] = field(default_factory=dict)
    confidence_available_count: int = 0
    timestamp_warnings: List[str] = field(default_factory=list)
    calibration_warnings: List[str] = field(default_factory=list)
    pose_convention: str = "arkit_camera_to_world_assumed"


def validate_intrinsics(K: np.ndarray, width: int, height: int) -> Optional[str]:
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        return "intrinsics are not finite 3x3"
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    if fx <= 0 or fy <= 0:
        return "intrinsics focal length must be positive"
    if not (-0.5 * width <= cx <= 1.5 * width and -0.5 * height <= cy <= 1.5 * height):
        return f"intrinsics principal point implausible: cx={cx:.2f}, cy={cy:.2f}, size={width}x{height}"
    return None


def validate_pose_matrix(T: np.ndarray) -> Optional[str]:
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return "pose is not finite 4x4"
    if not np.allclose(T[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-4):
        return "pose bottom row is not homogeneous [0,0,0,1]"
    R = T[:3, :3]
    det = float(np.linalg.det(R))
    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
    if not (0.85 <= det <= 1.15) or ortho_err > 0.08:
        return f"pose rotation is not SE(3)-like: det={det:.4f}, ortho_err={ortho_err:.4f}"
    return None


def validate_dataset_schema(dataset: Path, frames: List[FramePaths], max_probe_frames: int = 20) -> DatasetValidationReport:
    report = DatasetValidationReport(valid=True, frame_count=len(frames))
    required = ["rgb", "depth", "pose", "intrinsics"]
    for name in required:
        if not (dataset / name).exists():
            report.valid = False
            report.errors.append(f"missing required folder: {name}")
    if not frames:
        report.valid = False
        report.errors.append("no complete RGB/depth/pose/intrinsics frame groups found")
        return report

    for frame in frames[:max_probe_frames]:
        try:
            depth = read_depth_bin(frame.depth, frame.shape)
            report.depth_shape_histogram[str(tuple(depth.shape))] = report.depth_shape_histogram.get(str(tuple(depth.shape)), 0) + 1
            rgb = read_rgb(frame.rgb)
            report.rgb_shape_histogram[str(tuple(rgb.shape[:2]))] = report.rgb_shape_histogram.get(str(tuple(rgb.shape[:2])), 0) + 1
            K = load_matrix_txt(frame.intrinsics, (3, 3))
            err = validate_intrinsics(K, width=rgb.shape[1], height=rgb.shape[0])
            if err:
                report.warnings.append(f"{frame.stem}: {err}")

            if depth.shape[:2] != rgb.shape[:2]:
                report.calibration_warnings.append(
                    f"{frame.stem}: RGB {rgb.shape[:2]} and depth {depth.shape[:2]} differ; pipeline will scale intrinsics, verify exporter alignment."
                )
            fx, fy = float(K[0, 0]), float(K[1, 1])
            if not (0.2 * rgb.shape[1] <= fx <= 5.0 * rgb.shape[1]) or not (0.2 * rgb.shape[0] <= fy <= 5.0 * rgb.shape[0]):
                report.calibration_warnings.append(f"{frame.stem}: focal length is unusual for RGB resolution; check depth scale/intrinsics convention.")

            valid_depth = depth[np.isfinite(depth) & (depth > 0)]
            if valid_depth.size and float(np.nanmedian(valid_depth)) > 20.0:
                report.calibration_warnings.append(f"{frame.stem}: median depth >20m; possible millimeter-vs-meter depth scale error.")

            T = load_matrix_txt(frame.pose, (4, 4))
            err = validate_pose_matrix(T)
            if err:
                report.warnings.append(f"{frame.stem}: {err}")
            if frame.confidence is not None:
                report.confidence_available_count += 1
        except Exception as exc:
            report.warnings.append(f"{frame.stem}: probe failed: {exc}")

    numeric_stems = []
    for frame in frames[:max_probe_frames]:
        try:
            numeric_stems.append(float(frame.stem))
        except Exception:
            pass
    if len(numeric_stems) >= 3:
        deltas = np.diff(sorted(numeric_stems))
        if np.max(deltas) > 5.0 * max(float(np.median(deltas)), 1e-9):
            report.timestamp_warnings.append("irregular numeric frame timestamps; verify RGB/depth/pose synchronization.")
    if report.confidence_available_count == 0:
        report.calibration_warnings.append("no confidence maps found; depth uncertainty cannot be weighted by ARKit confidence.")
    return report
