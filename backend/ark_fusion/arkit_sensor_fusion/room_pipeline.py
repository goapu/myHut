from __future__ import annotations
from .common import *
from .geometry import *


def _plane_residual_mm(plane: Optional[PlaneInfo]) -> Optional[float]:
    if plane is None or len(plane.cloud.points) == 0:
        return None
    pts = np.asarray(plane.cloud.points, dtype=np.float64)
    n, d = normalize_plane_model(plane.model)
    residuals = np.abs(pts @ n + d)
    return float(np.median(residuals) * 1000.0)


def write_room_measurement_confidence(output_dir: Path, envelope: RoomEnvelope, planes: List[PlaneInfo]) -> Dict[str, Any]:
    dims = envelope.width_depth_m + [None, None]
    detected_floor = envelope.floor is not None
    detected_ceiling = envelope.ceiling is not None
    wall_pairs = len(envelope.wall_pairs)
    height_conf = 0.95 if detected_floor and detected_ceiling else 0.58 if detected_floor else 0.42
    footprint_conf = min(0.95, 0.40 + 0.25 * wall_pairs) if wall_pairs else 0.45
    if any("fallback" in n.lower() for n in envelope.notes):
        height_conf *= 0.85
        footprint_conf *= 0.80
    plane_residuals = {
        "floor": _plane_residual_mm(envelope.floor),
        "ceiling": _plane_residual_mm(envelope.ceiling),
    }
    report = {
        "height_m": float(envelope.height_m),
        "height_confidence": float(np.clip(height_conf, 0.0, 1.0)),
        "length_m": None if dims[0] is None else float(dims[0]),
        "length_confidence": float(np.clip(footprint_conf, 0.0, 1.0)),
        "width_m": None if dims[1] is None else float(dims[1]),
        "width_confidence": float(np.clip(footprint_conf if wall_pairs >= 2 else footprint_conf * 0.75, 0.0, 1.0)),
        "detected_floor": bool(detected_floor),
        "detected_ceiling": bool(detected_ceiling),
        "wall_planes": int(len(envelope.walls)),
        "wall_pairs": int(wall_pairs),
        "plane_residuals_mm": {k: v for k, v in plane_residuals.items() if v is not None},
        "warnings": list(envelope.notes),
        "planes": [
            {
                "normal": plane.normal.tolist(),
                "d": float(plane.d),
                "center": plane.center.tolist(),
                "area_hint": float(plane.area_hint),
                "median_residual_mm": _plane_residual_mm(plane),
            } for plane in planes
        ],
    }
    write_json(output_dir / "room_dimensions.json", report)
    write_json(output_dir / "room_planes.json", {"planes": report["planes"]})
    return report

def process_measure_room_mode(
    raw_mesh: o3d.geometry.TriangleMesh,
    output_dir: Path,
    visualize: bool,
    room_sample_points: int = 1000000,
    room_plane_voxel: float = 0.025,
    room_plane_distance: float = 0.04,
    room_min_plane_points: int = 3000,
    logger: Optional[logging.Logger] = None,
):
    print("\n--- MODE: MEASURE ROOM ---")

    raw_mesh = clean_mesh_basic(raw_mesh)
    raw_mesh = remove_small_mesh_components_by_physical_size(
        raw_mesh,
        min_bbox_diagonal=0.18,
        min_bbox_extent=0.03,
        keep_largest_only=False,
    )

    print("1. Sampling room point cloud...")
    if logger:
        logger.info("room_measurement_sampling_started", extra={"stage": "measure_room", "metrics": {"sample_points": room_sample_points}})
    pcd = raw_mesh.sample_points_uniformly(number_of_points=int(room_sample_points))

    print("2. Removing point outliers...")
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    pcd = pcd.select_by_index(ind)

    print("3. Downsampling for plane detection...")
    pcd_down = pcd.voxel_down_sample(voxel_size=float(room_plane_voxel))

    print("4. Extracting large room planes...")
    planes = extract_large_planes(
        pcd_down,
        max_planes=12,
        distance_threshold=float(room_plane_distance),
        min_plane_points=int(room_min_plane_points),
    )

    print(f"Detected {len(planes)} large planes.")

    envelope = estimate_room_envelope(pcd_down, planes)
    confidence_report = write_room_measurement_confidence(output_dir, envelope, planes)
    if logger:
        logger.info("room_measurement_confidence", extra={"stage": "measure_room", "metrics": {"height_confidence": confidence_report["height_confidence"], "length_confidence": confidence_report["length_confidence"], "width_confidence": confidence_report["width_confidence"]}})

    print("\n========================================")
    print("📐 ESTIMATED ROOM DIMENSIONS")
    print(f"   Height: {envelope.height_m:.2f} m")
    if envelope.width_depth_m:
        for idx, distance in enumerate(envelope.width_depth_m, start=1):
            label = "Length" if idx == 1 else "Width"
            print(f"   {label}: {distance:.2f} m")
    else:
        print("   Length/Width: unavailable")

    print("\nDetected room support:")
    print(f"   Floor plane:   {'yes' if envelope.floor is not None else 'no'}")
    print(f"   Ceiling plane: {'yes' if envelope.ceiling is not None else 'no'}")
    print(f"   Wall planes:   {len(envelope.walls)}")
    print(f"   Wall pairs:    {len(envelope.wall_pairs)}")
    for note in envelope.notes:
        print(f"   Note: {note}")
    print("========================================\n")

    pcd_path = output_dir / "room_measurement_cloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd_down)
    print(f"Saved measurement cloud: {pcd_path}")

    # Save detected planes together for visual inspection.
    plane_geometries = []
    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0.7, 0.4, 0.1],
        [0.4, 0.7, 0.1],
    ]

    for i, plane in enumerate(planes):
        cloud = plane.cloud
        cloud.paint_uniform_color(colors[i % len(colors)])
        plane_geometries.append(cloud)

    if visualize:
        o3d.visualization.draw_geometries(
            [pcd_down] + plane_geometries,
            window_name="Measure Room: Classified Room Planes",
        )


def process_room_full_mode(
    raw_mesh: o3d.geometry.TriangleMesh,
    output_dir: Path,
    visualize: bool,
    min_component_bbox_diagonal: float,
    min_component_bbox_extent: float,
    room_full_envelope_sample_points: int = 700000,
    room_full_envelope_voxel: float = 0.035,
    room_full_plane_distance: float = 0.045,
    room_full_min_plane_points: int = 2500,
    logger: Optional[logging.Logger] = None,
):
    print("\n--- MODE: FULL ROOM RECONSTRUCTION ---")

    print("1. Basic mesh cleanup...")
    mesh_out = clean_mesh_basic(raw_mesh)

    print("2. Detecting room envelope from classified planes...")
    if logger:
        logger.info("room_full_envelope_sampling_started", extra={"stage": "room_full", "metrics": {"sample_points": room_full_envelope_sample_points}})
    envelope_source = mesh_out.sample_points_uniformly(number_of_points=int(room_full_envelope_sample_points))
    _, ind = envelope_source.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    envelope_source = envelope_source.select_by_index(ind)
    envelope_cloud = envelope_source.voxel_down_sample(voxel_size=float(room_full_envelope_voxel))
    planes = extract_large_planes(
        envelope_cloud,
        max_planes=12,
        distance_threshold=float(room_full_plane_distance),
        min_plane_points=int(room_full_min_plane_points),
    )
    envelope = estimate_room_envelope(envelope_cloud, planes, margin_m=0.12)

    print("3. Clipping to room envelope...")
    mesh_out = crop_mesh_to_aabb(mesh_out, envelope.min_bound, envelope.max_bound)
    mesh_out = clean_mesh_basic(mesh_out)

    print("4. Removing tiny disconnected fragments by physical size...")
    mesh_out = remove_small_mesh_components_by_physical_size(
        mesh_out,
        min_bbox_diagonal=min_component_bbox_diagonal,
        min_bbox_extent=min_component_bbox_extent,
        keep_largest_only=False,
    )

    print("5. Light smoothing...")
    mesh_out = mesh_out.filter_smooth_simple(number_of_iterations=1)
    mesh_out.compute_vertex_normals()

    mesh_path = output_dir / "room_full_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh_out)
    print(f"Saved full room mesh: {mesh_path}")
    print("Room-envelope cleanup:")
    print(f"  Height:        {envelope.height_m:.2f} m")
    if envelope.width_depth_m:
        dims = ", ".join(f"{d:.2f} m" for d in envelope.width_depth_m)
        print(f"  Footprint dims: {dims}")
    print(f"  Wall planes:   {len(envelope.walls)}")
    print(f"  Wall pairs:    {len(envelope.wall_pairs)}")
    for note in envelope.notes:
        print(f"  Note: {note}")

    if visualize:
        o3d.visualization.draw_geometries(
            [mesh_out],
            window_name="Room Full: Envelope-Clipped Mesh",
        )
