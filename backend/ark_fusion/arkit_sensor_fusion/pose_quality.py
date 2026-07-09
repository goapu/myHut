from __future__ import annotations
from .common import *
from .validation import validate_pose_matrix

def rotation_angle_deg(R_prev: np.ndarray, R_curr: np.ndarray) -> float:
    R_delta = R_prev.T @ R_curr
    value = float(np.clip((np.trace(R_delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))


@dataclass
class PoseQuality:
    accepted: bool
    code: Optional[FailureCode] = None
    message: str = ""
    translation_jump_m: float = 0.0
    rotation_jump_deg: float = 0.0


@dataclass
class PoseQualityGate:
    max_translation_jump_m: float = 0.12
    max_rotation_jump_deg: float = 15.0
    enabled: bool = True

    def evaluate(self, pose_matrix: np.ndarray, previous_pose: Optional[np.ndarray]) -> PoseQuality:
        err = validate_pose_matrix(pose_matrix)
        if err:
            return PoseQuality(False, FailureCode.BAD_POSE_MATRIX, err)
        if not self.enabled or previous_pose is None:
            return PoseQuality(True)
        t_jump = float(np.linalg.norm(pose_matrix[:3, 3] - previous_pose[:3, 3]))
        r_jump = rotation_angle_deg(previous_pose[:3, :3], pose_matrix[:3, :3])
        if t_jump > self.max_translation_jump_m:
            return PoseQuality(False, FailureCode.POSE_TRANSLATION_JUMP, f"translation jump {t_jump:.3f} m exceeds {self.max_translation_jump_m:.3f} m", t_jump, r_jump)
        if r_jump > self.max_rotation_jump_deg:
            return PoseQuality(False, FailureCode.POSE_ROTATION_JUMP, f"rotation jump {r_jump:.2f} deg exceeds {self.max_rotation_jump_deg:.2f} deg", t_jump, r_jump)
        return PoseQuality(True, translation_jump_m=t_jump, rotation_jump_deg=r_jump)


@dataclass
class KeyframeSelector:
    min_translation_delta_m: float = 0.015
    min_rotation_delta_deg: float = 2.0
    enabled: bool = True
    last_keyframe_pose: Optional[np.ndarray] = None

    def should_integrate(self, pose_matrix: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        if not self.enabled or self.last_keyframe_pose is None:
            self.last_keyframe_pose = pose_matrix.copy()
            return True, {"translation_delta_m": 0.0, "rotation_delta_deg": 0.0}
        t_delta = float(np.linalg.norm(pose_matrix[:3, 3] - self.last_keyframe_pose[:3, 3]))
        r_delta = rotation_angle_deg(self.last_keyframe_pose[:3, :3], pose_matrix[:3, :3])
        keep = t_delta >= self.min_translation_delta_m or r_delta >= self.min_rotation_delta_deg
        if keep:
            self.last_keyframe_pose = pose_matrix.copy()
        return keep, {"translation_delta_m": t_delta, "rotation_delta_deg": r_delta}


@dataclass
class CoverageDecision:
    keep: bool
    reason: str
    stats: Dict[str, Any] = field(default_factory=dict)

class CoverageAwareKeyframeSelector(KeyframeSelector):
    """Keyframe selector that combines motion gates with cheap depth coverage gain.

    It is intentionally lightweight: it avoids expensive global coverage maps while
    still rejecting frames that add little new valid/high-confidence depth.
    """
    def __init__(self, min_translation_delta_m: float, min_rotation_delta_deg: float,
                 min_coverage_gain: float = 0.015, enabled: bool = True):
        super().__init__(min_translation_delta_m, min_rotation_delta_deg, enabled)
        self.min_coverage_gain = float(min_coverage_gain)
        self._last_valid_ratio: Optional[float] = None

    def should_integrate_with_coverage(self, pose: np.ndarray, valid_ratio: float, high_confidence_ratio: Optional[float]) -> Tuple[bool, Dict[str, Any]]:
        keep, stats = super().should_integrate(pose)
        stats["valid_depth_ratio"] = float(valid_ratio)
        if high_confidence_ratio is not None:
            stats["high_confidence_ratio"] = float(high_confidence_ratio)
        if not self.enabled or self._last_valid_ratio is None:
            self._last_valid_ratio = float(valid_ratio)
            return keep, stats
        gain = max(0.0, float(valid_ratio) - self._last_valid_ratio)
        stats["coverage_gain_estimate"] = gain
        # Keep if motion is useful OR this frame adds coverage.
        keep = keep or gain >= self.min_coverage_gain
        if keep:
            self._last_valid_ratio = float(valid_ratio)
        return keep, stats
