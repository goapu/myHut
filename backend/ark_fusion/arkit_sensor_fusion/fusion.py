from __future__ import annotations
from .common import *

class FusionBackend(Protocol):
    def integrate(self, rgbd: o3d.geometry.RGBDImage, intrinsic: o3d.camera.PinholeCameraIntrinsic, world_to_camera: np.ndarray) -> None:
        ...
    def extract_triangle_mesh(self) -> o3d.geometry.TriangleMesh:
        ...


@dataclass
class LegacyScalableTSDFBackend:
    voxel_length: float
    sdf_trunc: float
    color_type: Any = o3d.pipelines.integration.TSDFVolumeColorType.RGB8

    def __post_init__(self) -> None:
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(self.voxel_length),
            sdf_trunc=float(self.sdf_trunc),
            color_type=self.color_type,
        )

    def integrate(self, rgbd: o3d.geometry.RGBDImage, intrinsic: o3d.camera.PinholeCameraIntrinsic, world_to_camera: np.ndarray) -> None:
        self.volume.integrate(rgbd, intrinsic, world_to_camera)

    def extract_triangle_mesh(self) -> o3d.geometry.TriangleMesh:
        return self.volume.extract_triangle_mesh()


@dataclass
class TensorVoxelBlockGridBackend:
    """Open3D tensor VoxelBlockGrid TSDF backend for larger scans.

    This backend keeps TSDF, color, and per-voxel weight attributes in Open3D's
    sparse tensor grid. It is enabled when the installed Open3D build exposes
    ``o3d.t.geometry.VoxelBlockGrid``; otherwise construction raises a clear
    error so callers can fall back to legacy_tsdf.
    """
    voxel_length: float
    sdf_trunc: float
    block_resolution: int = 16
    block_count: int = 50000

    def __post_init__(self) -> None:
        if not hasattr(o3d, "t") or not hasattr(o3d.t, "geometry") or not hasattr(o3d.t.geometry, "VoxelBlockGrid"):
            raise NotImplementedError("This Open3D build does not provide o3d.t.geometry.VoxelBlockGrid; use legacy_tsdf.")
        self.device = o3d.core.Device("CPU:0")
        self.volume = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3d.core.Dtype.Float32, o3d.core.Dtype.Float32, o3d.core.Dtype.Float32),
            attr_channels=((1), (1), (3)),
            voxel_size=float(self.voxel_length),
            block_resolution=int(self.block_resolution),
            block_count=int(self.block_count),
            device=self.device,
        )
        self.integrated_frames = 0

    @staticmethod
    def _intrinsic_tensor(intrinsic: o3d.camera.PinholeCameraIntrinsic) -> Any:
        return o3d.core.Tensor(np.asarray(intrinsic.intrinsic_matrix, dtype=np.float64), o3d.core.Dtype.Float64)

    @staticmethod
    def _extrinsic_tensor(world_to_camera: np.ndarray) -> Any:
        return o3d.core.Tensor(np.asarray(world_to_camera, dtype=np.float64), o3d.core.Dtype.Float64)

    def integrate(self, rgbd: o3d.geometry.RGBDImage, intrinsic: o3d.camera.PinholeCameraIntrinsic, world_to_camera: np.ndarray) -> None:
        color_legacy = np.asarray(rgbd.color)
        depth_legacy = np.asarray(rgbd.depth).astype(np.float32)
        color = o3d.t.geometry.Image(o3d.core.Tensor(color_legacy, o3d.core.Dtype.UInt8, self.device))
        depth = o3d.t.geometry.Image(o3d.core.Tensor(depth_legacy, o3d.core.Dtype.Float32, self.device))
        K = self._intrinsic_tensor(intrinsic)
        E = self._extrinsic_tensor(world_to_camera)
        block_coords = self.volume.compute_unique_block_coordinates(depth, K, E, depth_scale=1.0, depth_max=float(np.nanmax(depth_legacy) if np.any(depth_legacy > 0) else 1.0), trunc_voxel_multiplier=float(self.sdf_trunc / max(self.voxel_length, 1e-9)))
        self.volume.integrate(block_coords, depth, color, K, E, depth_scale=1.0, depth_max=float(np.nanmax(depth_legacy) if np.any(depth_legacy > 0) else 1.0), trunc_voxel_multiplier=float(self.sdf_trunc / max(self.voxel_length, 1e-9)))
        self.integrated_frames += 1

    def extract_triangle_mesh(self) -> o3d.geometry.TriangleMesh:
        mesh = self.volume.extract_triangle_mesh(weight_threshold=0.0)
        return mesh.to_legacy() if hasattr(mesh, "to_legacy") else mesh


def create_fusion_backend(name: str, voxel_length: float, sdf_trunc: float) -> FusionBackend:
    name = (name or "legacy_tsdf").lower()
    if name in {"legacy_tsdf", "scalable_tsdf", "open3d_legacy"}:
        return LegacyScalableTSDFBackend(voxel_length=voxel_length, sdf_trunc=sdf_trunc)
    if name in {"tensor_tsdf", "voxel_block_grid"}:
        return TensorVoxelBlockGridBackend(voxel_length=voxel_length, sdf_trunc=sdf_trunc)
    raise ValueError(f"Unknown fusion backend: {name}")


class ICPPoseRefiner:
    """Frame-to-model point-to-plane ICP pose refinement.

    The input pose is ARKit-derived world_to_camera. This refiner converts the
    current depth cloud to world coordinates, aligns it against a lightweight
    accumulated model cloud, and **returns an updated world_to_camera** when the
    correction passes fitness/RMSE and motion-sanity checks. Rejected updates are
    reported but not applied.
    """
    def __init__(self, enabled: bool = False, voxel_size: float = 0.02, max_correspondence_m: float = 0.04,
                 min_fitness: float = 0.25, max_rmse: float = 0.035, max_correction_m: float = 0.08,
                 max_correction_deg: float = 8.0, model_max_points: int = 250000):
        self.enabled = bool(enabled)
        self.voxel_size = float(voxel_size)
        self.max_correspondence_m = float(max_correspondence_m)
        self.min_fitness = float(min_fitness)
        self.max_rmse = float(max_rmse)
        self.max_correction_m = float(max_correction_m)
        self.max_correction_deg = float(max_correction_deg)
        self.model_max_points = int(model_max_points)
        self._model_pcd: Optional[o3d.geometry.PointCloud] = None
        self.accepted_updates = 0
        self.rejected_updates = 0
        self.correction_translation_m: List[float] = []
        self.correction_rotation_deg: List[float] = []

    @staticmethod
    def _rotation_angle_deg(R: np.ndarray) -> float:
        trace = float(np.trace(R))
        cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def _depth_to_world_cloud(self, depth: np.ndarray, intrinsic: o3d.camera.PinholeCameraIntrinsic, world_to_camera: np.ndarray) -> o3d.geometry.PointCloud:
        valid_max = float(np.nanmax(depth)) if np.any(depth > 0) else 1.0
        img = o3d.geometry.Image(depth.astype(np.float32))
        pcd = o3d.geometry.PointCloud.create_from_depth_image(img, intrinsic, depth_scale=1.0, depth_trunc=valid_max + 0.1)
        if len(pcd.points) == 0:
            return pcd
        pcd = pcd.voxel_down_sample(self.voxel_size)
        camera_to_world = np.linalg.inv(world_to_camera)
        pcd.transform(camera_to_world)
        return pcd

    def _prepare(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if len(pcd.points) == 0:
            return pcd
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(self.voxel_size * 4, 1e-3), max_nn=40))
        return pcd

    def _update_model(self, pcd_world: o3d.geometry.PointCloud) -> None:
        if len(pcd_world.points) == 0:
            return
        if self._model_pcd is None:
            self._model_pcd = pcd_world
        else:
            self._model_pcd += pcd_world
            self._model_pcd = self._model_pcd.voxel_down_sample(self.voxel_size)
            if len(self._model_pcd.points) > self.model_max_points:
                self._model_pcd = self._model_pcd.random_down_sample(self.model_max_points / len(self._model_pcd.points))
        self._model_pcd = self._prepare(self._model_pcd)

    def refine(self, depth: np.ndarray, intrinsic: o3d.camera.PinholeCameraIntrinsic, world_to_camera: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.enabled:
            return world_to_camera, {"enabled": False}
        try:
            source_world = self._prepare(self._depth_to_world_cloud(depth, intrinsic, world_to_camera))
            if len(source_world.points) < 150:
                self.rejected_updates += 1
                return world_to_camera, {"enabled": True, "accepted": False, "reason": "too_few_points"}
            if self._model_pcd is None or len(self._model_pcd.points) < 500:
                self._update_model(source_world)
                return world_to_camera, {"enabled": True, "accepted": False, "reason": "model_warmup"}

            result = o3d.pipelines.registration.registration_icp(
                source_world, self._model_pcd, self.max_correspondence_m, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
            )
            correction = np.asarray(result.transformation, dtype=np.float64)
            trans_m = float(np.linalg.norm(correction[:3, 3]))
            rot_deg = self._rotation_angle_deg(correction[:3, :3])
            accepted = bool(
                result.fitness >= self.min_fitness and
                result.inlier_rmse <= self.max_rmse and
                trans_m <= self.max_correction_m and
                rot_deg <= self.max_correction_deg and
                np.all(np.isfinite(correction))
            )
            if accepted:
                camera_to_world = np.linalg.inv(world_to_camera)
                refined_camera_to_world = correction @ camera_to_world
                refined_world_to_camera = np.linalg.inv(refined_camera_to_world)
                # Add corrected source to the model so the model stays globally consistent.
                corrected_source = source_world.transform(correction.copy())
                self._update_model(corrected_source)
                self.accepted_updates += 1
                self.correction_translation_m.append(trans_m)
                self.correction_rotation_deg.append(rot_deg)
                return refined_world_to_camera, {
                    "enabled": True, "accepted": True, "fitness": float(result.fitness), "rmse": float(result.inlier_rmse),
                    "correction_translation_m": trans_m, "correction_rotation_deg": rot_deg,
                    "accepted_updates": self.accepted_updates, "rejected_updates": self.rejected_updates,
                }
            self.rejected_updates += 1
            # Even rejected frames can improve the model slightly if ICP quality is close; otherwise keep model stable.
            if result.fitness >= max(0.1, self.min_fitness * 0.5):
                self._update_model(source_world)
            return world_to_camera, {
                "enabled": True, "accepted": False, "fitness": float(result.fitness), "rmse": float(result.inlier_rmse),
                "correction_translation_m": trans_m, "correction_rotation_deg": rot_deg,
                "reason": "quality_or_correction_gate", "accepted_updates": self.accepted_updates, "rejected_updates": self.rejected_updates,
            }
        except Exception as exc:
            self.rejected_updates += 1
            return world_to_camera, {"enabled": True, "accepted": False, "reason": str(exc), "rejected_updates": self.rejected_updates}
