from __future__ import annotations
from .common import *

def clean_mesh_basic(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def remove_small_mesh_components(
    mesh: o3d.geometry.TriangleMesh,
    min_triangles: int = 1000,
    keep_largest_only: bool = False,
) -> o3d.geometry.TriangleMesh:
    """
    Removes tiny disconnected fragments.

    For room reconstruction, this helps avoid very small objects and floating artifacts.
    """
    if len(mesh.triangles) == 0:
        return mesh

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if len(cluster_n_triangles) == 0:
        return mesh

    if keep_largest_only:
        largest_cluster = int(np.argmax(cluster_n_triangles))
        remove_mask = triangle_clusters != largest_cluster
    else:
        remove_mask = cluster_n_triangles[triangle_clusters] < min_triangles

    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def crop_geometry_if_requested(
    geometry,
    crop_min: Optional[List[float]],
    crop_max: Optional[List[float]],
):
    if crop_min is None or crop_max is None:
        return geometry

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array(crop_min, dtype=np.float64),
        max_bound=np.array(crop_max, dtype=np.float64),
    )
    return geometry.crop(bbox)


# ---------------------------------------------------------------------
# 4. ROOM PLANE DETECTION
# ---------------------------------------------------------------------

@dataclass
class PlaneInfo:
    model: np.ndarray
    cloud: o3d.geometry.PointCloud
    normal: np.ndarray
    d: float
    center: np.ndarray
    area_hint: float


def normalize_plane_model(model: np.ndarray) -> Tuple[np.ndarray, float]:
    model = np.asarray(model, dtype=np.float64)
    n = model[:3]
    d = float(model[3])
    norm = np.linalg.norm(n)

    if norm < 1e-9:
        return n, d

    n = n / norm
    d = d / norm

    # Canonical sign to make comparisons easier.
    axis = int(np.argmax(np.abs(n)))
    if n[axis] < 0:
        n = -n
        d = -d

    return n, d


def extract_large_planes(
    pcd: o3d.geometry.PointCloud,
    max_planes: int = 8,
    distance_threshold: float = 0.035,
    min_plane_points: int = 5000,
) -> List[PlaneInfo]:
    planes: List[PlaneInfo] = []
    remaining = pcd

    for _ in range(max_planes):
        if len(remaining.points) < min_plane_points:
            break

        model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1500,
        )

        if len(inliers) < min_plane_points:
            break

        plane_cloud = remaining.select_by_index(inliers)
        remaining = remaining.select_by_index(inliers, invert=True)

        normal, d = normalize_plane_model(np.asarray(model))
        points = np.asarray(plane_cloud.points)
        center = points.mean(axis=0)

        # Simple proxy for plane size.
        bbox = plane_cloud.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        area_hint = float(np.prod(sorted(extent)[-2:]))

        planes.append(
            PlaneInfo(
                model=np.asarray(model),
                cloud=plane_cloud,
                normal=normal,
                d=d,
                center=center,
                area_hint=area_hint,
            )
        )

    return planes


def angle_between_normals_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def estimate_room_dimensions_from_planes(
    planes: List[PlaneInfo],
    parallel_angle_deg: float = 12.0,
) -> List[Tuple[float, PlaneInfo, PlaneInfo]]:
    """
    Legacy generic parallel-plane pairing.

    Kept as a fallback/debug helper; room measurement now prefers
    gravity-aware floor/ceiling/wall classification below.
    """
    candidates = []

    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            p1 = planes[i]
            p2 = planes[j]

            angle = angle_between_normals_deg(p1.normal, p2.normal)
            if angle > parallel_angle_deg:
                continue

            distance = abs(float(np.dot(p2.center - p1.center, p1.normal)))

            # Ignore tiny separations; likely furniture planes, table faces, etc.
            if distance < 0.5:
                continue

            combined_area = p1.area_hint + p2.area_hint
            candidates.append((distance, combined_area, p1, p2))

    # Prefer large plane pairs.
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)

    selected = []
    used_dirs: List[np.ndarray] = []

    for distance, _, p1, p2 in candidates:
        direction = p1.normal

        duplicate_direction = False
        for existing in used_dirs:
            if angle_between_normals_deg(direction, existing) < parallel_angle_deg:
                duplicate_direction = True
                break

        if duplicate_direction:
            continue

        selected.append((distance, p1, p2))
        used_dirs.append(direction)

        if len(selected) == 3:
            break

    selected.sort(key=lambda x: x[0])
    return selected


@dataclass
class RoomEnvelope:
    floor: Optional[PlaneInfo]
    ceiling: Optional[PlaneInfo]
    walls: List[PlaneInfo]
    wall_pairs: List[Tuple[float, PlaneInfo, PlaneInfo]]
    height_m: float
    width_depth_m: List[float]
    min_bound: np.ndarray
    max_bound: np.ndarray
    notes: List[str]


UP_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _axis_value(points_or_center: np.ndarray, axis: np.ndarray = UP_AXIS) -> np.ndarray:
    return np.asarray(points_or_center) @ axis


def classify_room_planes(
    planes: List[PlaneInfo],
    up_axis: np.ndarray = UP_AXIS,
    horizontal_angle_deg: float = 18.0,
    wall_angle_deg: float = 18.0,
    min_wall_vertical_extent: float = 0.9,
) -> Tuple[Optional[PlaneInfo], Optional[PlaneInfo], List[PlaneInfo], List[str]]:
    """
    Classifies planes using the ARKit gravity-up axis.

    This avoids treating any two large parallel planes as room bounds. Horizontal
    planes must be aligned with gravity (floor/ceiling candidates), while walls
    must be close to vertical and have meaningful vertical support so table tops,
    beds, counters, and shelves are not used as walls/floors.
    """
    up = up_axis / max(np.linalg.norm(up_axis), 1e-9)
    horizontal: List[PlaneInfo] = []
    walls: List[PlaneInfo] = []
    notes: List[str] = []

    for plane in planes:
        normal = plane.normal / max(np.linalg.norm(plane.normal), 1e-9)
        up_dot = abs(float(np.dot(normal, up)))
        angle_to_up = float(np.degrees(np.arccos(np.clip(up_dot, -1.0, 1.0))))
        pts = np.asarray(plane.cloud.points)
        vertical_extent = float(np.ptp(_axis_value(pts, up))) if len(pts) else 0.0

        if angle_to_up <= horizontal_angle_deg:
            horizontal.append(plane)
        elif abs(angle_to_up - 90.0) <= wall_angle_deg and vertical_extent >= min_wall_vertical_extent:
            walls.append(plane)

    floor = None
    ceiling = None
    if horizontal:
        # Require broad support and use height order. This makes high furniture
        # surfaces much less likely to become the floor than the real low plane.
        horizontal.sort(key=lambda p: (_axis_value(p.center, up), -p.area_hint))
        floor = horizontal[0]

        floor_level = float(_axis_value(floor.center, up))
        ceiling_candidates = [
            p for p in horizontal[1:]
            if float(_axis_value(p.center, up)) - floor_level > 1.6
        ]
        if ceiling_candidates:
            ceiling = max(
                ceiling_candidates,
                key=lambda p: (float(_axis_value(p.center, up)), p.area_hint),
            )
        else:
            notes.append("No complete ceiling plane found; using point-height fallback.")
    else:
        notes.append("No gravity-aligned floor plane found; using point-height fallback.")

    if not walls:
        notes.append("No reliable vertical wall planes found; using point-percentile room footprint fallback.")

    return floor, ceiling, walls, notes


def find_opposite_wall_pairs(
    walls: List[PlaneInfo],
    parallel_angle_deg: float = 12.0,
    min_separation_m: float = 1.0,
) -> List[Tuple[float, PlaneInfo, PlaneInfo]]:
    """Finds opposite vertical wall pairs and suppresses duplicate directions."""
    candidates = []

    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            p1, p2 = walls[i], walls[j]
            if angle_between_normals_deg(p1.normal, p2.normal) > parallel_angle_deg:
                continue

            distance = abs(float(np.dot(p2.center - p1.center, p1.normal)))
            if distance < min_separation_m:
                continue

            candidates.append((distance, p1.area_hint + p2.area_hint, p1, p2))

    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)

    selected: List[Tuple[float, PlaneInfo, PlaneInfo]] = []
    used_dirs: List[np.ndarray] = []
    for distance, _, p1, p2 in candidates:
        direction = p1.normal.copy()
        direction[1] = 0.0
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-9:
            direction = p1.normal
        else:
            direction /= direction_norm

        if any(angle_between_normals_deg(direction, d) < parallel_angle_deg for d in used_dirs):
            continue

        selected.append((distance, p1, p2))
        used_dirs.append(direction)
        if len(selected) == 2:
            break

    selected.sort(key=lambda x: x[0], reverse=True)
    return selected


def estimate_room_envelope(
    pcd: o3d.geometry.PointCloud,
    planes: List[PlaneInfo],
    margin_m: float = 0.08,
) -> RoomEnvelope:
    """
    Returns room dimensions and a conservative clipping AABB.

    The clipping box is intentionally conservative. If wall planes are incomplete,
    percentiles of the actual scanned points are used so duplicate exterior shells
    and floating outside fragments are removed without over-cropping furniture.
    """
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        zeros = np.zeros(3, dtype=np.float64)
        return RoomEnvelope(None, None, [], [], 0.0, [], zeros, zeros, ["Empty point cloud."])

    floor, ceiling, walls, notes = classify_room_planes(planes)
    wall_pairs = find_opposite_wall_pairs(walls)

    up_values = _axis_value(pts)
    p02 = float(np.percentile(up_values, 2))
    p98 = float(np.percentile(up_values, 98))

    if floor is not None:
        floor_level = float(_axis_value(floor.center))
    else:
        floor_level = p02

    if ceiling is not None:
        ceiling_level = float(_axis_value(ceiling.center))
    else:
        ceiling_level = p98

    if ceiling_level <= floor_level:
        floor_level, ceiling_level = p02, p98
        notes.append("Floor/ceiling ordering was ambiguous; reverted to point percentiles.")

    height_m = max(0.0, ceiling_level - floor_level)
    if ceiling is None and floor is not None:
        height_m = max(0.0, p98 - floor_level)

    # Start from robust point percentiles for x/z. Plane detections can be sparse,
    # so this is safer than hard-clipping against every detected wall plane.
    min_bound = np.percentile(pts, 1, axis=0) - margin_m
    max_bound = np.percentile(pts, 99, axis=0) + margin_m
    min_bound[1] = floor_level - margin_m
    max_bound[1] = ceiling_level + margin_m

    width_depth = [float(distance) for distance, _, _ in wall_pairs]
    if len(width_depth) < 2:
        # Fallback footprint dimensions in the horizontal X/Z axes. ARKit world Y
        # is gravity-up, so this is stable for most captures.
        x_span = float(np.percentile(pts[:, 0], 99) - np.percentile(pts[:, 0], 1))
        z_span = float(np.percentile(pts[:, 2], 99) - np.percentile(pts[:, 2], 1))
        fallback_dims = sorted([x_span, z_span], reverse=True)
        while len(width_depth) < 2 and fallback_dims:
            width_depth.append(fallback_dims.pop(0))
        notes.append("One or more opposite wall pairs were missing; using footprint percentile fallback.")

    return RoomEnvelope(
        floor=floor,
        ceiling=ceiling,
        walls=walls,
        wall_pairs=wall_pairs,
        height_m=float(height_m),
        width_depth_m=width_depth[:2],
        min_bound=min_bound.astype(np.float64),
        max_bound=max_bound.astype(np.float64),
        notes=notes,
    )


def crop_mesh_to_aabb(
    mesh: o3d.geometry.TriangleMesh,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.asarray(min_bound, dtype=np.float64),
        max_bound=np.asarray(max_bound, dtype=np.float64),
    )
    return mesh.crop(bbox)


def remove_small_mesh_components_by_physical_size(
    mesh: o3d.geometry.TriangleMesh,
    min_bbox_diagonal: float = 0.12,
    min_bbox_extent: float = 0.025,
    keep_largest_only: bool = False,
) -> o3d.geometry.TriangleMesh:
    """
    Removes disconnected components by metric size, not triangle count.

    Triangle count varies with local TSDF/Poisson density, so it can delete real
    small furniture while preserving dense noisy patches. Physical extents are a
    more stable criterion for room+furniture reconstruction.
    """
    if len(mesh.triangles) == 0:
        return mesh

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(cluster_n_triangles) == 0:
        return mesh

    if keep_largest_only:
        largest_cluster = int(np.argmax(cluster_n_triangles))
        remove_mask = triangle_clusters != largest_cluster
    else:
        remove_mask = np.zeros(len(triangles), dtype=bool)
        for cluster_id in range(len(cluster_n_triangles)):
            tri_idx = np.where(triangle_clusters == cluster_id)[0]
            if len(tri_idx) == 0:
                continue
            vertex_idx = np.unique(triangles[tri_idx].reshape(-1))
            component_vertices = vertices[vertex_idx]
            extent = component_vertices.max(axis=0) - component_vertices.min(axis=0)
            diagonal = float(np.linalg.norm(extent))
            too_small = diagonal < min_bbox_diagonal or float(extent.max()) < min_bbox_extent
            if too_small:
                remove_mask[tri_idx] = True

    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def trim_mesh_vertices_far_from_points(
    mesh: o3d.geometry.TriangleMesh,
    support_pcd: o3d.geometry.PointCloud,
    max_distance: float = 0.018,
    median_distance: Optional[float] = 0.030,
    knn: int = 6,
) -> o3d.geometry.TriangleMesh:
    """
    Removes Poisson vertices unsupported by the actual scanned point cloud.

    Poisson creates a closed surface and can extrapolate a ghost shell where no
    depth samples exist. A nearest-neighbour support test cuts away vertices too
    far from the observed TSDF/sample points. The optional median kNN test makes
    the check less likely to preserve a hallucinated sheet because of one stray
    nearby point.
    """
    if len(mesh.vertices) == 0 or len(support_pcd.points) == 0:
        return mesh

    tree = o3d.geometry.KDTreeFlann(support_pcd)
    vertices = np.asarray(mesh.vertices)
    remove = np.zeros(len(vertices), dtype=bool)
    max_distance_sq = float(max_distance * max_distance)
    median_distance_sq = None if median_distance is None else float(median_distance * median_distance)
    k = max(1, int(knn))

    for idx, vertex in enumerate(vertices):
        _, _, dist2 = tree.search_knn_vector_3d(vertex, k)
        if not dist2:
            remove[idx] = True
            continue
        if float(dist2[0]) > max_distance_sq:
            remove[idx] = True
            continue
        if median_distance_sq is not None and len(dist2) >= 3:
            if float(np.median(np.asarray(dist2, dtype=np.float64))) > median_distance_sq:
                remove[idx] = True

    mesh.remove_vertices_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def trim_mesh_vertices_below_local_support(
    mesh: o3d.geometry.TriangleMesh,
    support_pcd: o3d.geometry.PointCloud,
    up_axis: np.ndarray = UP_AXIS,
    local_xy_radius: float = 0.025,
    below_padding: float = 0.010,
    min_neighbors: int = 3,
) -> o3d.geometry.TriangleMesh:
    """Remove downward Poisson sheets below the locally observed object surface.

    For each mesh vertex, this looks for scanned points in a local horizontal
    column. If the vertex is below all local scanned support by more than
    below_padding, it is likely a downward Poisson extrapolation and is removed.
    """
    if len(mesh.vertices) == 0 or len(support_pcd.points) == 0:
        return mesh

    pts = np.asarray(support_pcd.points, dtype=np.float64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    up = np.asarray(up_axis, dtype=np.float64)
    up = up / max(np.linalg.norm(up), 1e-9)

    # Build a horizontal basis perpendicular to gravity-up. For ARKit this is
    # normally close to X/Z, but this keeps the function generic.
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, up))) > 0.9:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    b1 = ref - np.dot(ref, up) * up
    b1 /= max(np.linalg.norm(b1), 1e-9)
    b2 = np.cross(up, b1)
    b2 /= max(np.linalg.norm(b2), 1e-9)

    pts_2d = np.column_stack((pts @ b1, pts @ b2))
    verts_2d = np.column_stack((verts @ b1, verts @ b2))
    pts_h = pts @ up
    verts_h = verts @ up

    # Open3D KDTree is 3D, so store horizontal coordinates with z=0.
    support_2d = o3d.geometry.PointCloud()
    support_2d.points = o3d.utility.Vector3dVector(
        np.column_stack((pts_2d[:, 0], pts_2d[:, 1], np.zeros(len(pts_2d))))
    )
    tree = o3d.geometry.KDTreeFlann(support_2d)

    remove = np.zeros(len(verts), dtype=bool)
    for i, xy in enumerate(verts_2d):
        query = np.array([xy[0], xy[1], 0.0], dtype=np.float64)
        _, idx, _ = tree.search_radius_vector_3d(query, float(local_xy_radius))
        if len(idx) < min_neighbors:
            continue
        local_min_h = float(np.percentile(pts_h[np.asarray(idx, dtype=np.int64)], 5))
        if float(verts_h[i]) < local_min_h - float(below_padding):
            remove[i] = True

    mesh.remove_vertices_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def keep_object_clusters_near_largest(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.10,
    min_points: int = 15,
    min_cluster_diagonal: float = 0.025,
    join_distance: float = 0.45,
) -> o3d.geometry.PointCloud:
    """Keep the main object plus nearby fragments instead of only one DBSCAN island.

    Shoes often split into toe/heel/laces/rim clusters because ARKit depth is
    sparse on dark, thin, or glossy areas. Keeping only the largest cluster can
    delete real parts of the object.
    """
    if len(pcd.points) == 0:
        return pcd

    labels = np.asarray(pcd.cluster_dbscan(eps=float(eps), min_points=int(min_points), print_progress=False))
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        return pcd

    pts = np.asarray(pcd.points)
    counts = np.bincount(valid_labels)
    largest_label = int(np.argmax(counts))
    largest_pts = pts[labels == largest_label]
    largest_center = largest_pts.mean(axis=0)

    keep_labels = set()
    for label in np.unique(valid_labels):
        label = int(label)
        cluster_pts = pts[labels == label]
        if len(cluster_pts) == 0:
            continue
        extent = cluster_pts.max(axis=0) - cluster_pts.min(axis=0)
        diagonal = float(np.linalg.norm(extent))
        center = cluster_pts.mean(axis=0)
        center_distance = float(np.linalg.norm(center - largest_center))
        if label == largest_label or (diagonal >= min_cluster_diagonal and center_distance <= join_distance):
            keep_labels.add(label)

    keep_mask = np.asarray([label in keep_labels for label in labels], dtype=bool)
    keep_idx = np.where(keep_mask)[0]
    if len(keep_idx) == 0:
        return pcd
    return pcd.select_by_index(keep_idx)
