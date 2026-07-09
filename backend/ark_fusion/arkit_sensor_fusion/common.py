#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

import cv2
import numpy as np
import open3d as o3d

OPEN3D_CAMERA_TO_ARKIT_CAMERA = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

class FailureCode(str, Enum):
    DATASET_EMPTY = "dataset_empty"
    DATASET_SCHEMA_INVALID = "dataset_schema_invalid"
    FRAME_DECODE_ERROR = "frame_decode_error"
    LOW_VALID_DEPTH = "low_valid_depth"
    LOW_CONFIDENCE_DEPTH = "low_confidence_depth"
    BAD_INTRINSICS = "bad_intrinsics"
    BAD_POSE_MATRIX = "bad_pose_matrix"
    POSE_TRANSLATION_JUMP = "pose_translation_jump"
    POSE_ROTATION_JUMP = "pose_rotation_jump"
    KEYFRAME_REJECTED = "keyframe_rejected"
    FUSION_BACKEND_ERROR = "fusion_backend_error"
    EMPTY_MESH = "empty_mesh"
    ROOM_GEOMETRY_LOW_CONFIDENCE = "room_geometry_low_confidence"
    OBJECT_GHOST_LAYER_RISK = "object_ghost_layer_risk"
    POSTPROCESSING_ERROR = "postprocessing_error"


@dataclass
class FailureEvent:
    code: FailureCode
    message: str
    frame: Optional[str] = None
    severity: str = "warning"
    details: Dict[str, Any] = field(default_factory=dict)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for key in ("stage", "frame", "metrics", "failure_code", "details"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, default=str)


def configure_logger(output_dir: Path, json_logs: bool = True) -> logging.Logger:
    logger = logging.getLogger("arkit_reconstruction")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    logger.addHandler(console)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "reconstruction.log.jsonl"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    return logger


@dataclass
class MetricsRecorder:
    started_at_s: float = field(default_factory=time.time)
    mode: str = "unknown"
    pipeline_version: str = "product-refactor-0.1"
    input_frames_total: int = 0
    frames_seen: int = 0
    frames_integrated: int = 0
    frames_skipped: Dict[str, int] = field(default_factory=dict)
    valid_depth_ratios: List[float] = field(default_factory=list)
    high_confidence_ratios: List[float] = field(default_factory=list)
    pose_translation_jumps_m: List[float] = field(default_factory=list)
    pose_rotation_jumps_deg: List[float] = field(default_factory=list)
    keyframes_selected: int = 0
    mesh_vertices: int = 0
    mesh_triangles: int = 0
    output_artifacts: Dict[str, str] = field(default_factory=dict)
    confidence_reports: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[FailureEvent] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    capture_quality_score: float = 0.0
    fusion_health_score: float = 0.0
    geometry_confidence_score: float = 0.0
    measurement_confidence_score: float = 0.0
    benchmark_accuracy_score: Optional[float] = None
    quality_score: float = 0.0  # backward-compatible aggregate, not an accuracy claim

    def skip(self, code: FailureCode, message: str, frame: Optional[str] = None, severity: str = "warning", **details: Any) -> None:
        self.frames_skipped[code.value] = self.frames_skipped.get(code.value, 0) + 1
        self.failures.append(FailureEvent(code=code, message=message, frame=frame, severity=severity, details=details))

    def finalize(self) -> Dict[str, Any]:
        elapsed = max(time.time() - self.started_at_s, 1e-9)
        integrated_ratio = self.frames_integrated / max(self.input_frames_total, 1)
        depth_median = float(np.median(self.valid_depth_ratios)) if self.valid_depth_ratios else 0.0
        confidence_median = float(np.median(self.high_confidence_ratios)) if self.high_confidence_ratios else None
        fatal_count = sum(1 for f in self.failures if f.severity == "fatal")
        warning_penalty = min(0.35, 0.02 * len([f for f in self.failures if f.severity != "fatal"]))
        confidence_component = confidence_median if confidence_median is not None else 0.7
        pose_penalty = min(0.25, 0.01 * len(self.pose_translation_jumps_m) + 0.01 * len(self.pose_rotation_jumps_deg))
        self.capture_quality_score = float(np.clip(0.45 * depth_median + 0.35 * confidence_component + 0.20 * integrated_ratio - pose_penalty - 0.15 * fatal_count, 0.0, 1.0))
        self.fusion_health_score = float(np.clip(0.70 * integrated_ratio + 0.30 * min(1.0, self.mesh_vertices / 50_000.0) - warning_penalty - 0.25 * fatal_count, 0.0, 1.0))
        self.geometry_confidence_score = float(np.clip(0.45 * self.fusion_health_score + 0.35 * self.capture_quality_score + 0.20 * min(1.0, self.mesh_triangles / 100_000.0), 0.0, 1.0))
        # Room/object postprocessors may overwrite this with confidence based on planes/completeness.
        if self.measurement_confidence_score == 0.0:
            self.measurement_confidence_score = float(np.clip(0.5 * self.geometry_confidence_score + 0.5 * self.capture_quality_score, 0.0, 1.0))
        components = [self.capture_quality_score, self.fusion_health_score, self.geometry_confidence_score, self.measurement_confidence_score]
        if self.benchmark_accuracy_score is not None:
            components.append(float(self.benchmark_accuracy_score))
        self.quality_score = float(np.clip(np.mean(components) - 0.25 * fatal_count, 0.0, 1.0))
        payload = asdict(self)
        payload["elapsed_s"] = elapsed
        payload["integrated_ratio"] = integrated_ratio
        payload["valid_depth_ratio_median"] = depth_median
        payload["high_confidence_ratio_median"] = confidence_median
        payload["status"] = "failed" if fatal_count else ("partial" if self.frames_skipped else "success")
        return payload


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
