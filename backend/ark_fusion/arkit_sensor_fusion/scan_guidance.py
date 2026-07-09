from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class ScanGuidanceEvent:
    frame: str
    severity: str
    message: str
    code: str
    details: Dict[str, Any] = field(default_factory=dict)


class ScanGuidanceRecorder:
    """Capture-time guidance derived from depth, confidence, motion, and coverage signals."""
    def __init__(self) -> None:
        self.events: List[ScanGuidanceEvent] = []

    def add(self, frame: str, code: str, message: str, severity: str = "warning", **details: Any) -> None:
        self.events.append(ScanGuidanceEvent(frame=frame, severity=severity, message=message, code=code, details=details))

    def observe_depth(self, frame: str, valid_ratio: float, median_depth: Optional[float], min_depth: float, depth_trunc: float) -> None:
        if valid_ratio < 0.35:
            self.add(frame, "low_confidence_depth", "Low usable depth coverage: rescan this area more slowly or improve lighting.", valid_ratio=float(valid_ratio))
        if median_depth is not None:
            if median_depth < min_depth + 0.10:
                self.add(frame, "too_close", "Camera is too close to the surface; back up slightly.", median_depth_m=float(median_depth))
            if median_depth > depth_trunc * 0.90:
                self.add(frame, "too_far", "Subject is near the depth truncation distance; move closer or raise depth_trunc.", median_depth_m=float(median_depth))

    def observe_motion(self, frame: str, translation_jump_m: Optional[float], rotation_jump_deg: Optional[float]) -> None:
        if translation_jump_m is not None and translation_jump_m > 0.08:
            self.add(frame, "move_slower", "Camera translation jump is large; move slower for better tracking/fusion.", translation_jump_m=float(translation_jump_m))
        if rotation_jump_deg is not None and rotation_jump_deg > 10.0:
            self.add(frame, "excessive_pose_jump", "Camera rotation jump is large; avoid fast turns and revisit the area.", rotation_jump_deg=float(rotation_jump_deg))

    def observe_keyframe(self, frame: str, stats: Dict[str, Any]) -> None:
        if stats.get("coverage_gain", 1.0) < 0.01 and stats.get("translation_m", 0.0) < 0.02:
            self.add(frame, "not_enough_parallax", "Not enough parallax/new surface coverage; move sideways or orbit the object.", **{k: float(v) for k, v in stats.items() if isinstance(v, (int, float))})

    def observe_object_mask(self, frame: str, foreground_ratio: float, centered: bool = True) -> None:
        if foreground_ratio < 0.02:
            self.add(frame, "object_not_centered", "Object foreground is very small; center the object or provide a manual/SAM mask.", foreground_ratio=float(foreground_ratio), centered=bool(centered))

    def summarize(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for e in self.events:
            counts[e.code] = counts.get(e.code, 0) + 1
        return {"events": [e.__dict__ for e in self.events], "counts": counts}
