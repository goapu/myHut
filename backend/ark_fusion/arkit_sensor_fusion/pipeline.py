from __future__ import annotations
from .common import *
from .io import *
from .validation import *
from .pose_quality import *
from .preprocessing import *
from .fusion import *
from .artifacts import *
from .object_pipeline import *
from .room_pipeline import *
from .config import build_parser, apply_mode_defaults, apply_profile_defaults
from .scan_guidance import ScanGuidanceRecorder
try:
    from evaluation.accuracy import evaluate_geometry_accuracy, evaluate_room_measurements, write_accuracy_report
except Exception:  # optional in minimal installs
    evaluate_geometry_accuracy = None
    evaluate_room_measurements = None
    write_accuracy_report = None


class ReconstructionPipeline:
    """Composable sensor-fusion reconstruction pipeline.

    Stages: dataset validation -> decoding/preprocessing -> pose quality ->
    coverage-aware keyframes -> optional object masking/refinement -> TSDF fusion ->
    object/room postprocess -> metrics/artifacts.
    """
    def __init__(self, args: argparse.Namespace):
        if getattr(args, "disable_depth_filter", False):
            args.depth_filter = "none"
        args = apply_profile_defaults(args)
        if args.mode is None:
            raise ValueError("--mode is required unless --profile is provided with a preset that sets mode.")
        self.args = apply_mode_defaults(args)
        self.dataset = Path(self.args.dataset).expanduser().resolve()
        self.output = Path(self.args.output).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.logger = configure_logger(self.output, json_logs=self.args.json_logs)
        self.metrics = MetricsRecorder(mode=self.args.mode)
        self.writer = ArtifactWriter(self.output, self.metrics)
        self.guidance = ScanGuidanceRecorder()

    def run(self) -> Dict[str, Any]:
        args = self.args
        logger = self.logger
        metrics = self.metrics
        writer = self.writer

        logger.info("reconstruction_started", extra={"stage": "startup", "metrics": {"mode": args.mode, "dataset": str(self.dataset)}})
        write_json(self.output / "resolved_config.json", vars(args))

        frames_all = collect_complete_frames(self.dataset)
        validation = validate_dataset_schema(self.dataset, frames_all)
        write_json(self.output / "dataset_validation.json", asdict(validation))
        metrics.warnings.extend(validation.warnings + validation.calibration_warnings + validation.timestamp_warnings)
        if not validation.valid:
            metrics.skip(FailureCode.DATASET_SCHEMA_INVALID, "; ".join(validation.errors), severity="fatal")
            writer.write_metrics()
            raise RuntimeError("Dataset validation failed: " + "; ".join(validation.errors))

        frames = frames_all[:: max(1, int(args.frame_step))]
        metrics.input_frames_total = len(frames)
        if not frames:
            metrics.skip(FailureCode.DATASET_EMPTY, "No complete frames found after frame-step selection", severity="fatal")
            writer.write_metrics()
            raise RuntimeError("No complete frames found.")

        depth_filter = create_depth_filter(args.depth_filter, args.min_confidence, args.use_confidence, args.depth_median_ksize, args.depth_bilateral_sigma_color)
        pose_gate = PoseQualityGate(args.max_pose_jump, args.max_rotation_jump_deg, enabled=not args.disable_pose_gate)
        keyframes = CoverageAwareKeyframeSelector(args.keyframe_min_translation, args.keyframe_min_rotation_deg, args.keyframe_min_coverage_gain, enabled=not args.disable_keyframes)
        backend = create_fusion_backend(args.fusion_backend, args.tsdf_voxel_length, args.tsdf_sdf_trunc)
        masker = PreFusionObjectMasker(
            enabled=(args.mode == "object" and not args.disable_prefusion_object_mask),
            plane_distance_m=args.object_mask_plane_distance,
            depth_band_margin_m=args.object_mask_depth_band_margin,
            min_foreground_ratio=args.object_mask_min_foreground_ratio,
            center_prior=(not args.disable_object_center_prior),
            center_roi_fraction=args.object_center_roi_fraction,
            center_depth_margin_m=args.object_center_depth_margin,
            max_depth_spread_m=args.object_max_depth_spread,
        )
        refiner = ICPPoseRefiner(enabled=(args.pose_refinement == "icp"), voxel_size=args.icp_voxel_size, max_correspondence_m=args.icp_max_correspondence, min_fitness=args.icp_min_fitness, max_rmse=args.icp_max_rmse)

        logger.info("fusion_configured", extra={"stage": "startup", "metrics": {"frames": len(frames), "depth_filter": getattr(depth_filter, "name", str(depth_filter)), "object_mask": masker.enabled, "pose_refinement": args.pose_refinement}})

        prev_accepted_pose: Optional[np.ndarray] = None
        camera_trajectory: List[Dict[str, Any]] = []
        selected_keyframes: List[str] = []
        rejected_frames: List[Dict[str, Any]] = []
        for count, frame in enumerate(frames):
            metrics.frames_seen += 1
            try:
                rgb = read_rgb(frame.rgb)
                depth = read_depth_bin(frame.depth, frame.shape)
                depth = sanitize_depth(depth, depth_trunc=args.depth_trunc, min_depth=args.min_depth)

                confidence_norm: Optional[np.ndarray] = None
                last_high_conf: Optional[float] = None
                if args.use_confidence:
                    confidence_norm, confidence_report = read_confidence_normalized(frame, depth.shape)
                    metrics.confidence_reports.append({"frame": frame.stem, **asdict(confidence_report)})
                    if confidence_norm is not None:
                        last_high_conf = confidence_report.high_confidence_ratio
                        metrics.high_confidence_ratios.append(confidence_report.high_confidence_ratio)
                        if confidence_report.medium_or_high_ratio < args.min_valid_depth_ratio:
                            metrics.skip(FailureCode.LOW_CONFIDENCE_DEPTH, "low medium/high confidence depth coverage", frame=frame.stem, medium_or_high_ratio=confidence_report.medium_or_high_ratio)
                    depth = apply_confidence_mask(depth, confidence_norm, min_confidence=args.min_confidence)

                filter_result = depth_filter.apply(depth, confidence=confidence_norm, rgb=rgb)
                depth = filter_result.depth
                valid_ratio = depth_valid_ratio(depth)
                metrics.valid_depth_ratios.append(valid_ratio)
                valid_vals = depth[np.isfinite(depth) & (depth > 0)]
                self.guidance.observe_depth(frame.stem, valid_ratio, float(np.median(valid_vals)) if valid_vals.size else None, args.min_depth, args.depth_trunc)
                if valid_ratio < args.min_valid_depth_ratio:
                    metrics.skip(FailureCode.LOW_VALID_DEPTH, f"low valid depth ratio {valid_ratio:.3f}", frame=frame.stem, valid_ratio=valid_ratio)
                    rejected_frames.append({"frame": frame.stem, "reason": FailureCode.LOW_VALID_DEPTH.value, "valid_ratio": float(valid_ratio)})
                    logger.warning("frame_skipped", extra={"stage": "preprocess", "frame": frame.stem, "failure_code": FailureCode.LOW_VALID_DEPTH.value, "metrics": {"valid_ratio": valid_ratio}})
                    continue

                pose_matrix = load_matrix_txt(frame.pose, (4, 4))
                pose_quality = pose_gate.evaluate(pose_matrix, prev_accepted_pose)
                if pose_quality.translation_jump_m:
                    metrics.pose_translation_jumps_m.append(pose_quality.translation_jump_m)
                if pose_quality.rotation_jump_deg:
                    metrics.pose_rotation_jumps_deg.append(pose_quality.rotation_jump_deg)
                self.guidance.observe_motion(frame.stem, pose_quality.translation_jump_m, pose_quality.rotation_jump_deg)
                if not pose_quality.accepted:
                    metrics.skip(pose_quality.code or FailureCode.BAD_POSE_MATRIX, pose_quality.message, frame=frame.stem, translation_jump_m=pose_quality.translation_jump_m, rotation_jump_deg=pose_quality.rotation_jump_deg)
                    rejected_frames.append({"frame": frame.stem, "reason": (pose_quality.code or FailureCode.BAD_POSE_MATRIX).value, "message": pose_quality.message})
                    logger.warning("frame_skipped", extra={"stage": "pose_gate", "frame": frame.stem, "failure_code": (pose_quality.code or FailureCode.BAD_POSE_MATRIX).value, "details": pose_quality.message})
                    continue

                keep, keyframe_stats = keyframes.should_integrate_with_coverage(pose_matrix, valid_ratio, last_high_conf)
                if not keep:
                    metrics.skip(FailureCode.KEYFRAME_REJECTED, "coverage/motion keyframe rejected", frame=frame.stem, **keyframe_stats)
                    self.guidance.observe_keyframe(frame.stem, keyframe_stats)
                    rejected_frames.append({"frame": frame.stem, "reason": FailureCode.KEYFRAME_REJECTED.value, **keyframe_stats})
                    continue
                metrics.keyframes_selected += 1
                selected_keyframes.append(frame.stem)
                prev_accepted_pose = pose_matrix.copy()

                rgb_resized, scale_x, scale_y = resize_rgb_to_depth(rgb, depth)
                K_scaled = scale_intrinsics(load_matrix_txt(frame.intrinsics, (3, 3)), scale_x, scale_y)
                intr_err = validate_intrinsics(K_scaled, width=depth.shape[1], height=depth.shape[0])
                if intr_err:
                    metrics.skip(FailureCode.BAD_INTRINSICS, intr_err, frame=frame.stem)
                    continue
                intrinsic = make_open3d_intrinsic(K_scaled, width=depth.shape[1], height=depth.shape[0])

                if args.mode == "object":
                    mask_result = masker.apply(depth, intrinsic)
                    depth = mask_result.depth
                    # Optional manual/SAM/semantic masks override/strengthen heuristic object segmentation.
                    mask_dir = getattr(args, "object_mask_dir", None) or getattr(args, "semantic_mask_dir", None)
                    external_mask_used = False
                    if mask_dir:
                        mask_path = None
                        for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
                            candidate = Path(mask_dir) / f"{frame.stem}{suffix}"
                            if candidate.exists():
                                mask_path = candidate
                                break
                        if mask_path is not None:
                            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                            if mask_img is not None:
                                if mask_img.shape[:2] != depth.shape[:2]:
                                    mask_img = cv2.resize(mask_img, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
                                external = mask_img > 0
                                depth = np.where(external, depth, 0.0).astype(np.float32)
                                external_mask_used = True
                    self.guidance.observe_object_mask(frame.stem, mask_result.foreground_ratio, centered=True)
                    logger.info("object_mask", extra={"stage": "preprocess", "frame": frame.stem, "metrics": {"foreground_ratio": mask_result.foreground_ratio, "support_plane_found": mask_result.support_plane_found, "external_mask_used": external_mask_used, **mask_result.details}})
                    if mask_result.foreground_ratio < args.object_mask_min_foreground_ratio:
                        metrics.skip(FailureCode.LOW_VALID_DEPTH, "object foreground ratio too low after pre-fusion mask", frame=frame.stem, foreground_ratio=mask_result.foreground_ratio)
                        continue

                camera_to_world = arkit_camera_to_world_to_open3d_camera_to_world(pose_matrix)
                world_to_camera = np.linalg.inv(camera_to_world)
                world_to_camera, refine_stats = refiner.refine(depth, intrinsic, world_to_camera)
                if refine_stats.get("enabled"):
                    logger.info("pose_refinement", extra={"stage": "pose", "frame": frame.stem, "metrics": refine_stats})

                rgbd = make_rgbd(rgb_resized, depth, depth_trunc=args.depth_trunc, min_depth=args.min_depth, use_depth_filter=False, depth_filter=depth_filter, confidence=confidence_norm)
                backend.integrate(rgbd, intrinsic, world_to_camera)
                metrics.frames_integrated += 1
                camera_trajectory.append({"frame": frame.stem, "world_to_camera": np.asarray(world_to_camera, dtype=float).tolist(), "refinement": refine_stats})

                if count % 20 == 0:
                    logger.info("fusion_progress", extra={"stage": "fusion", "metrics": {"processed": count, "total": len(frames), "integrated": metrics.frames_integrated}})
            except Exception as exc:
                metrics.skip(FailureCode.FRAME_DECODE_ERROR, str(exc), frame=frame.stem)
                logger.exception("frame_error", extra={"stage": "frame", "frame": frame.stem, "failure_code": FailureCode.FRAME_DECODE_ERROR.value})

        if metrics.frames_integrated == 0:
            metrics.skip(FailureCode.FUSION_BACKEND_ERROR, "No frames were integrated. Check filters and dataset.", severity="fatal")
            writer.write_metrics()
            raise RuntimeError("No frames were integrated. Check filters and dataset.")

        raw_mesh = backend.extract_triangle_mesh()
        if len(raw_mesh.vertices) == 0:
            metrics.skip(FailureCode.EMPTY_MESH, "Mesh has no vertices. Check scale, poses, or depth limits.", severity="fatal")
            writer.write_metrics()
            return metrics.finalize()
        raw_mesh.compute_vertex_normals()
        metrics.mesh_vertices = len(raw_mesh.vertices)
        metrics.mesh_triangles = len(raw_mesh.triangles)
        writer.write_mesh("raw_tsdf_mesh.ply", raw_mesh)
        try:
            raw_cloud = raw_mesh.sample_points_uniformly(number_of_points=min(300000, max(10000, metrics.mesh_vertices * 4)))
            writer.write_point_cloud("raw_tsdf_cloud.ply", raw_cloud)
        except Exception as exc:
            logger.warning("raw_cloud_export_failed", extra={"stage": "artifacts", "details": str(exc)})
        write_json(self.output / "camera_trajectory.json", {"trajectory": camera_trajectory})
        write_json(self.output / "selected_keyframes.json", {"frames": selected_keyframes})
        write_json(self.output / "rejected_frames.json", {"frames": rejected_frames})

        try:
            if args.mode == "object":
                # Detect ghost layers before deciding whether Poisson should run.
                pcd_probe = raw_mesh.sample_points_uniformly(number_of_points=min(args.object_sample_points, 200000))
                dup = detect_duplicate_local_layers(pcd_probe, voxel_size=args.duplicate_layer_voxel, layer_gap_m=args.duplicate_layer_gap)
                logger.info("duplicate_layer_probe", extra={"stage": "quality", "metrics": dup})
                if dup["duplicate_layer_score"] > args.duplicate_layer_skip_poisson_score:
                    metrics.skip(FailureCode.OBJECT_GHOST_LAYER_RISK, "duplicate local layers detected; Poisson should be skipped or treated as low-confidence", duplicate_layer_score=dup["duplicate_layer_score"])
                skip_poisson = bool(args.disable_poisson or dup["duplicate_layer_score"] > args.duplicate_layer_skip_poisson_score)
                process_object_mode(
                    raw_mesh, self.output, args.visualize, args.crop_min, args.crop_max,
                    args.object_plane_distance, args.object_cluster_eps, args.object_cluster_min_points,
                    args.object_keep_cluster_diagonal, args.object_cluster_join_distance,
                    args.object_support_distance, args.object_support_median_distance, args.object_below_padding,
                    args.object_local_xy_radius, args.object_sample_points, args.object_downsample_voxel,
                    args.object_normal_radius, args.poisson_depth, args.poisson_scale,
                    args.poisson_density_quantile, logger,
                    disable_poisson=skip_poisson,
                    use_center_prior_selection=(not args.disable_object_select_center_prior),
                )
                metrics.output_artifacts["object_watertight_mesh.ply"] = str(self.output / "object_watertight_mesh.ply")
                metrics.output_artifacts["object_clean_cloud.ply"] = str(self.output / "object_clean_cloud.ply")
            elif args.mode == "measure_room":
                process_measure_room_mode(raw_mesh, self.output, args.visualize, args.room_sample_points, args.room_plane_voxel, args.room_plane_distance, args.room_min_plane_points, logger)
            elif args.mode == "room_full":
                process_room_full_mode(raw_mesh, self.output, args.visualize, args.room_min_component_bbox_diagonal, args.room_min_component_bbox_extent, args.room_full_envelope_sample_points, args.room_full_envelope_voxel, args.room_full_plane_distance, args.room_full_min_plane_points, logger)
        except Exception as exc:
            metrics.skip(FailureCode.POSTPROCESSING_ERROR, str(exc), severity="fatal")
            logger.exception("postprocessing_error", extra={"stage": "postprocess", "failure_code": FailureCode.POSTPROCESSING_ERROR.value})
            writer.write_metrics()
            raise

        write_json(self.output / "scan_guidance.json", self.guidance.summarize())

        benchmark_payload: Dict[str, Any] = {"condition": getattr(args, "benchmark_condition", "unspecified"), "metrics": {}}
        try:
            if evaluate_geometry_accuracy is not None:
                if args.mode == "object" and getattr(args, "ground_truth_object_mesh", None):
                    pred = self.output / "object_watertight_mesh.ply"
                    if not pred.exists():
                        pred = self.output / "object_clean_cloud.ply"
                    report = evaluate_geometry_accuracy(pred, Path(args.ground_truth_object_mesh))
                    benchmark_payload["metrics"]["object_reconstruction"] = asdict(report)
                    metrics.benchmark_accuracy_score = float(report.fscore_at_thresholds.get("10mm", report.fscore_at_thresholds.get("20mm", 0.0)))
                if args.mode in {"measure_room", "room_full"} and getattr(args, "ground_truth_room_mesh", None):
                    pred = self.output / ("room_full_mesh.ply" if args.mode == "room_full" else "raw_tsdf_mesh.ply")
                    report = evaluate_geometry_accuracy(pred, Path(args.ground_truth_room_mesh), thresholds_m=(0.02, 0.05, 0.10))
                    benchmark_payload["metrics"]["room_reconstruction"] = asdict(report)
                    metrics.benchmark_accuracy_score = float(report.fscore_at_thresholds.get("50mm", report.fscore_at_thresholds.get("100mm", 0.0)))
                if getattr(args, "ground_truth_measurements", None) and evaluate_room_measurements is not None:
                    pred_json = self.output / "room_dimensions.json"
                    if pred_json.exists():
                        benchmark_payload["metrics"]["room_measurements"] = evaluate_room_measurements(pred_json, Path(args.ground_truth_measurements))
            if benchmark_payload["metrics"] and write_accuracy_report is not None:
                write_accuracy_report(self.output / "benchmark_report.json", benchmark_payload)
                metrics.output_artifacts["benchmark_report.json"] = str(self.output / "benchmark_report.json")
        except Exception as exc:
            metrics.skip(FailureCode.POSTPROCESSING_ERROR, f"benchmark evaluation failed: {exc}", severity="warning")
            logger.warning("benchmark_failed", extra={"stage": "benchmark", "details": str(exc)})

        final_payload = metrics.finalize()
        write_json(self.output / "reconstruction_confidence.json", {
            "capture_quality_score": final_payload.get("capture_quality_score"),
            "fusion_health_score": final_payload.get("fusion_health_score"),
            "geometry_confidence_score": final_payload.get("geometry_confidence_score"),
            "measurement_confidence_score": final_payload.get("measurement_confidence_score"),
            "benchmark_accuracy_score": final_payload.get("benchmark_accuracy_score"),
            "quality_score_backward_compatible": final_payload.get("quality_score"),
            "note": "quality_score is an aggregate health/confidence score, not a direct accuracy claim unless benchmark_accuracy_score is present.",
        })
        writer.write_metrics()
        logger.info("reconstruction_finished", extra={"stage": "finish", "metrics": {"quality_score": metrics.quality_score}})
        return final_payload


def run_from_args(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    if args is None:
        parser = build_parser()
        args = parser.parse_args()
    return ReconstructionPipeline(args).run()


def main() -> None:
    run_from_args()
