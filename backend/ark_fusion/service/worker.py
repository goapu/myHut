from __future__ import annotations
import shutil
import types
from pathlib import Path
from typing import Any
from arkit_sensor_fusion.pipeline import ReconstructionPipeline
from service.job_store import JobStore


class ReconstructionWorker:
    def __init__(self, job_store: JobStore, workspace: Path):
        self.job_store = job_store
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run_job(self, job_id: str, dataset_zip: Path, profile: str, mode: str) -> dict[str, Any]:
        job = self.job_store.get_job(job_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")

        self.job_store.update_status(job_id, "validating")

        dataset_dir = self.workspace / job_id / "dataset"
        output_dir = self.workspace / job_id / "output"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            shutil.unpack_archive(str(dataset_zip), str(dataset_dir), format="zip")
        except Exception as exc:
            self.job_store.update_status(job_id, "failed")
            self.job_store.append_error(job_id, f"Dataset unzip failed: {exc}")
            return {"status": "failed", "errors": job.errors}

        self.job_store.update_status(job_id, "reconstructing")
        args = types.SimpleNamespace(
            dataset=str(dataset_dir),
            output=str(output_dir),
            profile=profile,
            mode=mode,
            visualize=False,
            json_logs=True,
            depth_trunc=None,
            min_depth=None,
            tsdf_voxel_length=None,
            tsdf_sdf_trunc=None,
            frame_step=1,
            disable_depth_filter=False,
            depth_filter="median_bilateral",
            min_confidence=1,
            depth_median_ksize=5,
            depth_bilateral_sigma_color=0.05,
            min_valid_depth_ratio=0.25,
            disable_pose_gate=False,
            max_pose_jump=0.12,
            max_rotation_jump_deg=15.0,
            disable_keyframes=False,
            keyframe_min_translation=0.015,
            keyframe_min_rotation_deg=2.0,
            use_confidence=False,
            crop_min=None,
            crop_max=None,
            object_plane_distance=0.012,
            object_cluster_eps=0.10,
            object_cluster_min_points=15,
            object_keep_cluster_diagonal=0.025,
            object_cluster_join_distance=0.45,
            object_support_distance=0.018,
            object_support_median_distance=0.030,
            object_below_padding=0.010,
            object_local_xy_radius=0.025,
            object_sample_points=300000,
            object_downsample_voxel=0.005,
            object_normal_radius=0.035,
            poisson_depth=8,
            poisson_scale=1.25,
            poisson_density_quantile=0.08,
            room_sample_points=1000000,
            room_plane_voxel=0.025,
            room_plane_distance=0.04,
            room_min_plane_points=3000,
            room_min_component_bbox_diagonal=0.12,
            room_min_component_bbox_extent=0.025,
            room_full_envelope_sample_points=700000,
            room_full_envelope_voxel=0.035,
            room_full_plane_distance=0.045,
            room_full_min_plane_points=2500,
            fusion_backend="legacy_tsdf",
            disable_prefusion_object_mask=False,
            object_mask_plane_distance=0.015,
            object_mask_depth_band_margin=0.15,
            object_mask_min_foreground_ratio=0.005,
            object_center_roi_fraction=0.72,
            object_center_depth_margin=0.18,
            object_max_depth_spread=0.55,
            disable_object_center_prior=False,
            disable_object_select_center_prior=False,
            pose_refinement="none",
            icp_voxel_size=0.02,
            icp_max_correspondence=0.04,
            icp_min_fitness=0.25,
            icp_max_rmse=0.035,
            duplicate_layer_voxel=0.008,
            duplicate_layer_gap=0.018,
            duplicate_layer_skip_poisson_score=0.08,
            disable_duplicate_layer_collapse=False,
            disable_poisson=False,
            ground_truth_object_mesh=None,
            ground_truth_room_mesh=None,
            ground_truth_measurements=None,
            benchmark_condition="service",
            object_mask_dir=None,
            semantic_mask_dir=None,
            save_preview_images=False,
        )

        try:
            pipeline = ReconstructionPipeline(args)
            metrics = pipeline.run()
            self.job_store.update_status(job_id, "completed")
            self.job_store.update_metrics(job_id, metrics)
            self.job_store.update_artifacts(job_id, pipeline.metrics.output_artifacts)
            return {"status": "completed", "metrics": metrics, "artifacts": pipeline.metrics.output_artifacts}
        except Exception as exc:
            self.job_store.update_status(job_id, "failed")
            self.job_store.append_error(job_id, str(exc))
            return {"status": "failed", "errors": job.errors}
