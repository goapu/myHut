from __future__ import annotations
from .common import *
from .io import *

class ConfidenceCodec(str, Enum):
    MISSING = "missing"
    ARKIT_RAW_0_1_2 = "arkit_raw_0_1_2"
    UINT8_QUANTIZED = "uint8_quantized"
    BINARY_MASK = "binary_mask"
    UNKNOWN = "unknown"


@dataclass
class ConfidenceReport:
    codec: ConfidenceCodec
    histogram: Dict[str, int]
    min_value: float = 0.0
    max_value: float = 0.0
    high_confidence_ratio: float = 0.0
    medium_or_high_ratio: float = 0.0
    normalized_unique_values: List[int] = field(default_factory=list)


def sanitize_depth(depth: np.ndarray, depth_trunc: float, min_depth: float) -> np.ndarray:
    depth = np.ascontiguousarray(depth.astype(np.float32))
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < min_depth] = 0.0
    depth[depth > depth_trunc] = 0.0
    return depth


def _small_histogram(values: np.ndarray, max_items: int = 16) -> Dict[str, int]:
    finite = np.asarray(values[np.isfinite(values)]).reshape(-1)
    if finite.size == 0:
        return {}
    unique, counts = np.unique(finite, return_counts=True)
    if len(unique) <= max_items:
        return {str(float(k) if np.issubdtype(unique.dtype, np.floating) else int(k)): int(v) for k, v in zip(unique, counts)}
    hist, edges = np.histogram(finite, bins=max_items)
    return {f"{edges[i]:.2f}..{edges[i+1]:.2f}": int(hist[i]) for i in range(len(hist))}


def normalize_confidence_map(confidence: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], ConfidenceReport]:
    """Normalize ARKit confidence to explicit 0/1/2 raw confidence levels.

    Apple ARKit confidence maps contain ARConfidenceLevel raw values per depth
    component, but many capture/export pipelines save them as PNG/JPEG values
    such as 0/128/255. This function detects common encodings, records a
    histogram, and returns uint8 values in {0, 1, 2}.
    """
    if confidence is None:
        return None, ConfidenceReport(codec=ConfidenceCodec.MISSING, histogram={})

    conf = np.asarray(confidence)
    finite = conf[np.isfinite(conf)] if np.issubdtype(conf.dtype, np.floating) else conf.reshape(-1)
    if finite.size == 0:
        return np.zeros(conf.shape[:2], dtype=np.uint8), ConfidenceReport(codec=ConfidenceCodec.UNKNOWN, histogram={})

    min_v = float(np.min(finite))
    max_v = float(np.max(finite))
    unique = np.unique(finite)
    hist = _small_histogram(finite)

    normalized = np.zeros(conf.shape[:2], dtype=np.uint8)
    codec = ConfidenceCodec.UNKNOWN

    # ARKit raw export: 0 low, 1 medium, 2 high.
    if set(np.asarray(unique, dtype=np.int64).tolist()).issubset({0, 1, 2}):
        normalized = np.clip(conf, 0, 2).astype(np.uint8)
        codec = ConfidenceCodec.ARKIT_RAW_0_1_2
    # Common image export: 0, 128-ish, 255-ish.
    elif max_v > 2 and max_v <= 255:
        normalized[conf >= 200] = 2
        normalized[(conf >= 80) & (conf < 200)] = 1
        codec = ConfidenceCodec.UINT8_QUANTIZED
        if len(unique) <= 3 and set(np.asarray(unique, dtype=np.int64).tolist()).issubset({0, 255}):
            codec = ConfidenceCodec.BINARY_MASK
            normalized[conf > 0] = 2
    else:
        # Last-resort quantile mapping. Keep this explicit and visible in metrics.
        q1, q2 = np.quantile(finite.astype(np.float64), [0.33, 0.66])
        normalized[conf >= q2] = 2
        normalized[(conf >= q1) & (conf < q2)] = 1
        codec = ConfidenceCodec.UNKNOWN

    report = ConfidenceReport(
        codec=codec,
        histogram=hist,
        min_value=min_v,
        max_value=max_v,
        high_confidence_ratio=float(np.count_nonzero(normalized >= 2) / normalized.size),
        medium_or_high_ratio=float(np.count_nonzero(normalized >= 1) / normalized.size),
        normalized_unique_values=[int(x) for x in np.unique(normalized).tolist()],
    )
    return normalized, report


@dataclass
class DepthFilterResult:
    depth: np.ndarray
    valid_mask: np.ndarray
    name: str
    stats: Dict[str, float] = field(default_factory=dict)


class DepthFilter(Protocol):
    name: str
    def apply(self, depth: np.ndarray, confidence: Optional[np.ndarray] = None, rgb: Optional[np.ndarray] = None) -> DepthFilterResult:
        ...


@dataclass
class NoOpDepthFilter:
    name: str = "none"
    def apply(self, depth: np.ndarray, confidence: Optional[np.ndarray] = None, rgb: Optional[np.ndarray] = None) -> DepthFilterResult:
        valid = depth > 0
        return DepthFilterResult(depth=depth.astype(np.float32), valid_mask=valid, name=self.name, stats={"valid_ratio": depth_valid_ratio(depth)})


@dataclass
class MedianBilateralDepthFilter:
    median_ksize: int = 5
    bilateral_d: int = 5
    sigma_color_m: float = 0.05
    sigma_space_px: float = 5.0
    name: str = "median_bilateral"

    def apply(self, depth: np.ndarray, confidence: Optional[np.ndarray] = None, rgb: Optional[np.ndarray] = None) -> DepthFilterResult:
        valid_mask = depth > 0
        if np.count_nonzero(valid_mask) == 0:
            return DepthFilterResult(depth=depth.astype(np.float32), valid_mask=valid_mask, name=self.name, stats={"valid_ratio": 0.0})
        filtered = depth.copy().astype(np.float32)
        k = int(self.median_ksize)
        if k >= 3 and k % 2 == 1:
            filtered = cv2.medianBlur(filtered, k)
        filtered[~valid_mask] = 0.0
        filtered = cv2.bilateralFilter(filtered, d=int(self.bilateral_d), sigmaColor=float(self.sigma_color_m), sigmaSpace=float(self.sigma_space_px))
        filtered[~valid_mask] = 0.0
        return DepthFilterResult(depth=filtered.astype(np.float32), valid_mask=filtered > 0, name=self.name, stats={"valid_ratio": depth_valid_ratio(filtered)})


@dataclass
class ConfidenceAwareDepthFilter:
    base_filter: DepthFilter
    min_confidence: int = 1
    confidence_erosion_px: int = 0
    name: str = "confidence_aware"

    def apply(self, depth: np.ndarray, confidence: Optional[np.ndarray] = None, rgb: Optional[np.ndarray] = None) -> DepthFilterResult:
        d = depth.copy().astype(np.float32)
        if confidence is not None:
            mask = confidence >= int(self.min_confidence)
            if self.confidence_erosion_px > 0:
                kernel = np.ones((self.confidence_erosion_px, self.confidence_erosion_px), dtype=np.uint8)
                mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            d[~mask] = 0.0
        result = self.base_filter.apply(d, confidence=confidence, rgb=rgb)
        result.name = self.name + ":" + result.name
        if confidence is not None:
            result.stats["medium_or_high_confidence_ratio"] = float(np.count_nonzero(confidence >= 1) / confidence.size)
            result.stats["high_confidence_ratio"] = float(np.count_nonzero(confidence >= 2) / confidence.size)
        return result


def create_depth_filter(name: str, min_confidence: int, use_confidence: bool, median_ksize: int = 5, bilateral_sigma_color: float = 0.05) -> DepthFilter:
    name = (name or "median_bilateral").lower()
    if name == "none":
        base: DepthFilter = NoOpDepthFilter()
    elif name in {"median", "median_bilateral", "bilateral"}:
        base = MedianBilateralDepthFilter(median_ksize=median_ksize, sigma_color_m=bilateral_sigma_color)
    else:
        raise ValueError(f"Unknown depth filter '{name}'. Valid: none, median_bilateral")
    if use_confidence:
        return ConfidenceAwareDepthFilter(base_filter=base, min_confidence=min_confidence)
    return base


def filter_depth(depth: np.ndarray, median_ksize: int = 5) -> np.ndarray:
    return MedianBilateralDepthFilter(median_ksize=median_ksize).apply(depth).depth


def read_confidence(frame: FramePaths, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Read ARKit confidence maps without assuming float32 depth layout.

    ARKit confidence maps are normally ARConfidenceLevel raw values 0/1/2,
    commonly exported as uint8 one-byte-per-pixel buffers. Some exporters save
    PNG/TIFF images. This reader probes uint8 first, then float32 fallback.
    """
    if frame.confidence is None:
        return None
    path = frame.confidence
    suffix = path.suffix.lower()
    if suffix == ".bin":
        if frame.confidence_shape is None:
            return None
        h, w = frame.confidence_shape
        expected = int(h) * int(w)
        raw_u8 = np.fromfile(path, dtype=np.uint8)
        if raw_u8.size == expected:
            conf = raw_u8.reshape((h, w))
        else:
            raw_f32 = np.fromfile(path, dtype=np.float32)
            if raw_f32.size == expected:
                conf = raw_f32.reshape((h, w))
            else:
                raw_u16 = np.fromfile(path, dtype=np.uint16)
                if raw_u16.size == expected:
                    conf = raw_u16.reshape((h, w))
                else:
                    raise RuntimeError(
                        f"Confidence size mismatch in {path}: "
                        f"uint8={raw_u8.size}, uint16={raw_u16.size}, float32={raw_f32.size}, expected={expected}"
                    )
    else:
        conf = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if conf is None:
            return None
        if conf.ndim == 3:
            conf = conf[:, :, 0]
    if conf.shape[:2] != target_shape:
        conf = cv2.resize(conf, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return conf


def read_confidence_normalized(frame: FramePaths, target_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], ConfidenceReport]:
    raw = read_confidence(frame, target_shape)
    return normalize_confidence_map(raw)


def apply_confidence_mask(depth: np.ndarray, confidence: Optional[np.ndarray], min_confidence: int) -> np.ndarray:
    normalized, _ = normalize_confidence_map(confidence)
    if normalized is None:
        return depth
    depth = depth.copy()
    depth[normalized < int(min_confidence)] = 0.0
    return depth


def depth_valid_ratio(depth: np.ndarray) -> float:
    return float(np.count_nonzero(depth > 0)) / float(depth.size)


def make_rgbd(rgb: np.ndarray, depth: np.ndarray, depth_trunc: float, min_depth: float, use_depth_filter: bool, depth_filter: Optional[DepthFilter] = None, confidence: Optional[np.ndarray] = None) -> o3d.geometry.RGBDImage:
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    depth = sanitize_depth(depth, depth_trunc=depth_trunc, min_depth=min_depth)
    if use_depth_filter:
        if depth_filter is None:
            depth = filter_depth(depth)
        else:
            depth = depth_filter.apply(depth, confidence=confidence, rgb=rgb).depth
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(depth),
        depth_scale=1.0,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )


@dataclass
class PreFusionMaskResult:
    depth: np.ndarray
    mask: np.ndarray
    foreground_ratio: float
    support_plane_found: bool
    details: Dict[str, Any] = field(default_factory=dict)

class PreFusionObjectMasker:
    """Masks background/support-plane depth before TSDF fusion for object scans.

    This is deliberately conservative. It estimates a dominant support plane in
    camera space and keeps points above/near the object depth band, so table/floor
    geometry is not integrated and later turned into ghost shells.
    """
    def __init__(self, enabled: bool = True, plane_distance_m: float = 0.015,
                 depth_band_margin_m: float = 0.15, min_foreground_ratio: float = 0.005,
                 center_prior: bool = True, center_roi_fraction: float = 0.72,
                 center_depth_margin_m: float = 0.18, max_depth_spread_m: float = 0.55):
        self.enabled = bool(enabled)
        self.plane_distance_m = float(plane_distance_m)
        self.depth_band_margin_m = float(depth_band_margin_m)
        self.min_foreground_ratio = float(min_foreground_ratio)
        self.center_prior = bool(center_prior)
        self.center_roi_fraction = float(np.clip(center_roi_fraction, 0.2, 1.0))
        self.center_depth_margin_m = float(center_depth_margin_m)
        self.max_depth_spread_m = float(max_depth_spread_m)

    def apply(self, depth: np.ndarray, intrinsic: o3d.camera.PinholeCameraIntrinsic) -> PreFusionMaskResult:
        if not self.enabled:
            mask = depth > 0
            return PreFusionMaskResult(depth=depth, mask=mask, foreground_ratio=float(mask.mean()), support_plane_found=False)

        valid = np.isfinite(depth) & (depth > 0)
        if valid.mean() < self.min_foreground_ratio:
            return PreFusionMaskResult(depth=np.zeros_like(depth), mask=np.zeros_like(valid), foreground_ratio=float(valid.mean()), support_plane_found=False, details={"reason":"too_few_depth_pixels"})

        z = depth[valid]
        z_lo, z_hi = np.percentile(z, [5, 85])
        depth_band = valid & (depth >= max(0.0, z_lo - self.depth_band_margin_m)) & (depth <= z_hi + self.depth_band_margin_m)

        # Product object scans usually keep the target object near the image center.
        # Without this prior, shelves/table/background can become the largest cluster.
        center_mask = np.ones_like(valid, dtype=bool)
        center_z = None
        if self.center_prior:
            h, w = depth.shape[:2]
            frac = self.center_roi_fraction
            x0 = int(round((1.0 - frac) * 0.5 * w)); x1 = int(round((1.0 + frac) * 0.5 * w))
            y0 = int(round((1.0 - frac) * 0.5 * h)); y1 = int(round((1.0 + frac) * 0.5 * h))
            center_mask[:] = False
            center_mask[y0:y1, x0:x1] = True
            center_valid = valid & center_mask
            if np.count_nonzero(center_valid) > max(64, int(0.002 * depth.size)):
                center_vals = depth[center_valid]
                center_z = float(np.percentile(center_vals, 35))
                near = max(0.0, center_z - self.center_depth_margin_m)
                far = center_z + min(self.center_depth_margin_m, self.max_depth_spread_m)
                depth_band = depth_band & center_mask & (depth >= near) & (depth <= far)
            else:
                depth_band = depth_band & center_mask

        support_found = False
        plane_mask = np.zeros_like(valid)
        try:
            img = o3d.geometry.Image(depth.astype(np.float32))
            pcd = o3d.geometry.PointCloud.create_from_depth_image(img, intrinsic, depth_scale=1.0, depth_trunc=float(np.nanmax(depth[valid]) + 0.1))
            if len(pcd.points) > 250:
                pcd_small = pcd.voxel_down_sample(0.01)
                if len(pcd_small.points) > 250:
                    plane, inliers = pcd_small.segment_plane(distance_threshold=self.plane_distance_m, ransac_n=3, num_iterations=80)
                    if len(inliers) > 100:
                        support_found = True
                        # Project the detected support plane back to the full depth image and
                        # remove actual plane inliers, rather than only using depth-band heuristics.
                        # Plane equation is in camera coordinates: ax + by + cz + d = 0.
                        a, b, c, d = [float(v) for v in plane]
                        try:
                            K = np.asarray(intrinsic.intrinsic_matrix, dtype=np.float64)
                            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                        except Exception:
                            fx, fy = intrinsic.get_focal_length()
                            cx, cy = intrinsic.get_principal_point()
                        yy, xx = np.indices(depth.shape)
                        zz = depth.astype(np.float64)
                        x = (xx.astype(np.float64) - float(cx)) * zz / max(float(fx), 1e-9)
                        y = (yy.astype(np.float64) - float(cy)) * zz / max(float(fy), 1e-9)
                        norm = max(float(np.linalg.norm([a, b, c])), 1e-9)
                        signed = np.abs(a * x + b * y + c * zz + d) / norm
                        plane_mask = valid & (signed <= self.plane_distance_m * 1.5)
        except Exception:
            support_found = False

        mask = depth_band & ~plane_mask
        foreground_ratio = float(mask.mean())
        if foreground_ratio < self.min_foreground_ratio:
            # Fallback to depth band instead of producing empty object.
            mask = depth_band
            foreground_ratio = float(mask.mean())
        masked = np.where(mask, depth, 0.0).astype(np.float32)
        return PreFusionMaskResult(
            depth=masked,
            mask=mask,
            foreground_ratio=foreground_ratio,
            support_plane_found=support_found,
            details={
                "z_percentile_5": float(z_lo),
                "z_percentile_85": float(z_hi),
                "center_prior": bool(self.center_prior),
                "center_depth_m": None if center_z is None else float(center_z),
                "center_roi_fraction": float(self.center_roi_fraction),
            },
        )
