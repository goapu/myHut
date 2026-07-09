from __future__ import annotations
from .common import *

@dataclass
class FramePaths:
    stem: str
    rgb: Path
    depth: Path
    shape: Path
    pose: Path
    intrinsics: Path
    confidence: Optional[Path] = None
    confidence_shape: Optional[Path] = None


def sorted_files(folder: Path, suffix: str) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob(f"*{suffix}"))


def files_by_stem(files: Iterable[Path]) -> Dict[str, Path]:
    return {p.stem.replace("_shape", ""): p for p in files}


def load_matrix_txt(path: Path, expected_shape: Tuple[int, int]) -> np.ndarray:
    mat = np.loadtxt(str(path), dtype=np.float64)
    if mat.shape != expected_shape:
        raise RuntimeError(
            f"Bad matrix shape in {path}: got {mat.shape}, expected {expected_shape}"
        )
    return mat


def read_depth_bin(depth_path: Path, shape_path: Path) -> np.ndarray:
    parts = shape_path.read_text().strip().split()
    height, width = int(parts[0]), int(parts[1])
    depth = np.fromfile(str(depth_path), dtype=np.float32)
    if depth.size != height * width:
        raise RuntimeError(
            f"Depth size mismatch in {depth_path}: got {depth.size}, expected {height * width}"
        )
    return depth.reshape((height, width))


def read_rgb(rgb_path: Path) -> np.ndarray:
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f"Could not read RGB image: {rgb_path}")
    return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)


def resize_rgb_to_depth(
    rgb: np.ndarray, depth: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    depth_h, depth_w = depth.shape[:2]
    rgb_h, rgb_w = rgb.shape[:2]

    if rgb_h == depth_h and rgb_w == depth_w:
        return rgb, 1.0, 1.0

    resized = cv2.resize(rgb, (depth_w, depth_h), interpolation=cv2.INTER_AREA)
    return resized, depth_w / rgb_w, depth_h / rgb_h


def scale_intrinsics(K_rgb: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    K = K_rgb.copy().astype(np.float64)
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] *= scale_x
    K[1, 2] *= scale_y
    return K


def make_open3d_intrinsic(
    K: np.ndarray, width: int, height: int
) -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        float(K[0, 0]),
        float(K[1, 1]),
        float(K[0, 2]),
        float(K[1, 2]),
    )


def arkit_camera_to_world_to_open3d_camera_to_world(
    T_arkit_c2w: np.ndarray,
) -> np.ndarray:
    return T_arkit_c2w @ OPEN3D_CAMERA_TO_ARKIT_CAMERA


def collect_complete_frames(dataset: Path) -> List[FramePaths]:
    rgb_files = (
        sorted_files(dataset / "rgb", ".jpg")
        + sorted_files(dataset / "rgb", ".jpeg")
        + sorted_files(dataset / "rgb", ".png")
    )

    rgb_by_stem = files_by_stem(rgb_files)
    depth_by_stem = files_by_stem(sorted_files(dataset / "depth", ".bin"))
    pose_by_stem = files_by_stem(sorted_files(dataset / "pose", ".txt"))
    intrinsics_by_stem = files_by_stem(sorted_files(dataset / "intrinsics", ".txt"))

    confidence_folder = dataset / "confidence"
    confidence_files = []
    for suffix in [".png", ".jpg", ".jpeg", ".bin"]:
        confidence_files.extend(sorted_files(confidence_folder, suffix))
    confidence_by_stem = files_by_stem(confidence_files)

    common_stems = sorted(
        set(rgb_by_stem)
        & set(depth_by_stem)
        & set(pose_by_stem)
        & set(intrinsics_by_stem)
    )

    frames = []
    for stem in common_stems:
        depth_path = depth_by_stem[stem]
        shape_path = depth_path.with_name(f"{stem}_shape.txt")

        if not shape_path.exists():
            continue

        confidence_path = confidence_by_stem.get(stem)
        confidence_shape = None

        if confidence_path is not None and confidence_path.suffix.lower() == ".bin":
            candidate_shape = confidence_path.with_name(f"{stem}_shape.txt")
            if candidate_shape.exists():
                confidence_shape = candidate_shape

        frames.append(
            FramePaths(
                stem=stem,
                rgb=rgb_by_stem[stem],
                depth=depth_path,
                shape=shape_path,
                pose=pose_by_stem[stem],
                intrinsics=intrinsics_by_stem[stem],
                confidence=confidence_path,
                confidence_shape=confidence_shape,
            )
        )

    return frames
