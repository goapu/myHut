from __future__ import annotations
from .common import *
from .geometry import *


def detect_duplicate_local_layers(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.012, layer_gap_m: float = 0.018) -> Dict[str, Any]:
    """Detect likely ghost shells by checking whether local XY bins contain separated Z layers."""
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return {"duplicate_layer_score": 0.0, "multi_layer_bins": 0, "total_bins": 0}
    xy = np.floor(pts[:, :2] / max(voxel_size, 1e-6)).astype(np.int64)
    buckets: Dict[Tuple[int, int], List[float]] = {}
    for key, z in zip(map(tuple, xy), pts[:, 2]):
        buckets.setdefault(key, []).append(float(z))
    multi = 0
    for vals in buckets.values():
        if len(vals) < 4:
            continue
        v = np.sort(np.asarray(vals))
        if np.max(np.diff(v)) > layer_gap_m:
            multi += 1
    total = max(len(buckets), 1)
    return {"duplicate_layer_score": float(multi / total), "multi_layer_bins": int(multi), "total_bins": int(total)}


def collapse_duplicate_local_layers(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.008) -> o3d.geometry.PointCloud:
    """Conservative duplicate-layer collapse using voxel downsampling.

    A production version can replace this with view-support weighted layer selection.
    This conservative implementation reduces repeated local shells without inventing
    new geometry.
    """
    if len(pcd.points) == 0:
        return pcd
    return pcd.voxel_down_sample(voxel_size=max(voxel_size, 1e-6))


def keep_object_clusters_by_center_prior(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.10,
    min_points: int = 15,
    min_cluster_diagonal: float = 0.025,
    join_distance: float = 0.30,
) -> o3d.geometry.PointCloud:
    """Keep clusters near the robust center of the candidate object cloud.

    This avoids the common product-scan failure where a surrounding/background
    fragment is larger than the real object and the largest-cluster heuristic
    deletes the target shoe/headphone.
    """
    if len(pcd.points) == 0:
        return pcd
    pts = np.asarray(pcd.points)
    robust_center = np.median(pts, axis=0)
    labels = np.asarray(pcd.cluster_dbscan(eps=float(eps), min_points=int(min_points), print_progress=False))
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        return pcd

    # Score clusters by distance to robust cloud center, size, and physical plausibility.
    scored = []
    for label in np.unique(valid_labels):
        label = int(label)
        idx = np.where(labels == label)[0]
        cpts = pts[idx]
        if len(cpts) == 0:
            continue
        extent = cpts.max(axis=0) - cpts.min(axis=0)
        diagonal = float(np.linalg.norm(extent))
        center = cpts.mean(axis=0)
        dist = float(np.linalg.norm(center - robust_center))
        count = int(len(idx))
        if diagonal >= min_cluster_diagonal:
            # Lower score is better. Penalize far clusters, but do not let tiny dust win.
            score = dist - 0.0005 * min(count, 2000)
            scored.append((score, label, dist, diagonal, count))
    if not scored:
        return keep_object_clusters_near_largest(pcd, eps, min_points, min_cluster_diagonal, join_distance)

    scored.sort(key=lambda x: x[0])
    anchor_label = int(scored[0][1])
    anchor_pts = pts[labels == anchor_label]
    anchor_center = anchor_pts.mean(axis=0)

    keep_labels = set()
    for _, label, dist_to_global, diagonal, count in scored:
        cpts = pts[labels == label]
        center = cpts.mean(axis=0)
        dist_to_anchor = float(np.linalg.norm(center - anchor_center))
        if label == anchor_label or dist_to_anchor <= join_distance:
            keep_labels.add(int(label))

    keep_idx = np.where(np.asarray([label in keep_labels for label in labels], dtype=bool))[0]
    if len(keep_idx) == 0:
        return pcd
    return pcd.select_by_index(keep_idx)


def process_object_mode(
    raw_mesh: o3d.geometry.TriangleMesh,
    output_dir: Path,
    visualize: bool,
    crop_min: Optional[List[float]],
    crop_max: Optional[List[float]],
    object_plane_distance: float,
    object_cluster_eps: float,
    object_cluster_min_points: int,
    object_keep_cluster_diagonal: float,
    object_cluster_join_distance: float,
    object_support_distance: float,
    object_support_median_distance: float,
    object_below_padding: float,
    object_local_xy_radius: float,
    object_sample_points: int,
    object_downsample_voxel: float,
    object_normal_radius: float,
    poisson_depth: int,
    poisson_scale: float,
    poisson_density_quantile: float,
    logger: Optional[logging.Logger] = None,
    disable_poisson: bool = False,
    use_center_prior_selection: bool = True,
):
    print("\n--- MODE: OBJECT RECONSTRUCTION ---")

    raw_mesh = crop_geometry_if_requested(raw_mesh, crop_min, crop_max)
    raw_mesh = clean_mesh_basic(raw_mesh)

    if logger:
        logger.info("object_sampling_started", extra={"stage": "object", "metrics": {"sample_points": object_sample_points}})
    print("1. Sampling point cloud from TSDF mesh...")
    pcd = raw_mesh.sample_points_poisson_disk(number_of_points=int(object_sample_points))

    print("2. Isolating the main object and removing background...")
    
    # A. Slice away the dominant flat surface (the table or floor the shoe is on)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=object_plane_distance,
        ransac_n=3,
        num_iterations=1000,
    )
    pcd = pcd.select_by_index(inliers, invert=True)

    # B. Clean up floating LiDAR dust, but keep this gentle so sparse shoe
    # areas such as the heel rim, toe edge, sole, and laces survive.
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
    pcd = pcd.select_by_index(ind)

    # C. Keep the main object plus nearby meaningful fragments. Keeping only
    # the largest cluster often deletes real shoe parts because ARKit depth can
    # split a dark/thin object into several clusters.
    if use_center_prior_selection:
        pcd = keep_object_clusters_by_center_prior(
            pcd,
            eps=object_cluster_eps,
            min_points=object_cluster_min_points,
            min_cluster_diagonal=object_keep_cluster_diagonal,
            join_distance=min(object_cluster_join_distance, 0.30),
        )
    else:
        pcd = keep_object_clusters_near_largest(
            pcd,
            eps=object_cluster_eps,
            min_points=object_cluster_min_points,
            min_cluster_diagonal=object_keep_cluster_diagonal,
            join_distance=object_cluster_join_distance,
        )

    print("3. Downsampling...")
    pcd = pcd.voxel_down_sample(voxel_size=float(object_downsample_voxel))

    print("4. Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(object_normal_radius), max_nn=50)
    )
    pcd.orient_normals_consistent_tangent_plane(50)

    print("5. Poisson reconstruction...")
    # Aggressively clean points to prevent topological loops
    pcd = pcd.remove_duplicated_points()
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.6)
    pcd = pcd.select_by_index(ind)

    pcd_path = output_dir / "object_clean_cloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd)

    if disable_poisson:
        print("5. Poisson disabled; saving cleaned object cloud only.")
        print(f"Saved object point cloud: {pcd_path}")
        return
    
    # Run a highly stable version of Poisson
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(poisson_depth),
        scale=float(poisson_scale),
        linear_fit=False,
    )

    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, float(poisson_density_quantile))
    poisson_mesh.remove_vertices_by_mask(densities < density_threshold)

    print("6. Trimming unsupported Poisson extrapolation...")
    poisson_mesh = trim_mesh_vertices_far_from_points(
        poisson_mesh,
        support_pcd=pcd,
        max_distance=object_support_distance,
        median_distance=object_support_median_distance,
        knn=6,
    )
    poisson_mesh = trim_mesh_vertices_below_local_support(
        poisson_mesh,
        support_pcd=pcd,
        local_xy_radius=object_local_xy_radius,
        below_padding=object_below_padding,
    )

    poisson_mesh = clean_mesh_basic(poisson_mesh)
    poisson_mesh = remove_small_mesh_components_by_physical_size(
        poisson_mesh,
        min_bbox_diagonal=0.08,
        min_bbox_extent=0.015,
        keep_largest_only=True,
    )

    mesh_path = output_dir / "object_watertight_mesh.ply"

    o3d.io.write_triangle_mesh(str(mesh_path), poisson_mesh)
    o3d.io.write_point_cloud(str(pcd_path), pcd)

    print(f"Saved object mesh: {mesh_path}")
    print(f"Saved object point cloud: {pcd_path}")

    if visualize:
        o3d.visualization.draw_geometries(
            [poisson_mesh],
            window_name="Object Mode: Watertight Mesh",
        )
