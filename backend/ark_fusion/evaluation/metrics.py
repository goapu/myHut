from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class QualityReport:
    mode: str
    quality_score: float
    status: str
    frames_integrated: int
    frames_total: int
    mesh_vertices: int
    mesh_triangles: int
    integrated_ratio: float
    valid_depth_ratio_median: float
    high_confidence_ratio_median: float | None
    elapsed_s: float
    failures: List[Dict[str, Any]]

    def summary(self) -> str:
        lines = [
            f"Mode: {self.mode}",
            f"Status: {self.status}",
            f"Quality Score: {self.quality_score:.3f}",
            f"Integrated: {self.frames_integrated}/{self.frames_total} ({self.integrated_ratio*100:.1f}%)",
            f"Mesh: {self.mesh_vertices} verts, {self.mesh_triangles} tris",
            f"Depth Quality (median): {self.valid_depth_ratio_median:.3f}",
            f"Elapsed: {self.elapsed_s:.2f}s",
        ]
        if self.high_confidence_ratio_median is not None:
            lines.append(f"Confidence (median): {self.high_confidence_ratio_median:.3f}")
        return "\n".join(lines)
