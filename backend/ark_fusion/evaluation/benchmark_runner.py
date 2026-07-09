from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from evaluation.metrics import QualityReport

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Summarize reconstruction output folders and benchmark reports.

    This runner is intentionally file-based so it can aggregate CI/integration
    runs from many capture conditions: good light, low light, clutter, white
    walls, reflective/dark/thin objects, partial scans, fast motion, and loop
    closure scenarios.
    """

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_metrics(self, job_output_dir: Path) -> QualityReport:
        metrics_file = job_output_dir / "metrics.json"
        if not metrics_file.exists():
            raise FileNotFoundError(f"metrics.json not found in {job_output_dir}")

        data = json.loads(metrics_file.read_text(encoding="utf-8"))
        return QualityReport(
            mode=data["mode"],
            quality_score=data["quality_score"],
            status=data["status"],
            frames_integrated=data["frames_integrated"],
            frames_total=data["input_frames_total"],
            mesh_vertices=data["mesh_vertices"],
            mesh_triangles=data["mesh_triangles"],
            integrated_ratio=data["integrated_ratio"],
            valid_depth_ratio_median=data["valid_depth_ratio_median"],
            high_confidence_ratio_median=data.get("high_confidence_ratio_median"),
            elapsed_s=data["elapsed_s"],
            failures=[f for f in data["failures"] if f["severity"] == "fatal"],
        )

    def load_benchmark_report(self, job_output_dir: Path) -> Dict[str, Any]:
        path = job_output_dir / "benchmark_report.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def run_benchmark_suite(self, dataset_dirs: list[Path]) -> list[QualityReport]:
        reports = []
        for dataset_dir in dataset_dirs:
            try:
                reports.append(self.load_metrics(dataset_dir))
            except Exception as exc:
                logger.warning("Could not load metrics from %s: %s", dataset_dir, exc)
        return reports

    def summarize(self, reports: list[QualityReport]) -> dict:
        if not reports:
            return {}

        avg_quality = sum(r.quality_score for r in reports) / len(reports)
        modes: Dict[str, List[QualityReport]] = {}
        for r in reports:
            modes.setdefault(r.mode, []).append(r)

        return {
            "total_jobs": len(reports),
            "avg_quality_score": avg_quality,
            "by_mode": {mode: len(mode_reports) for mode, mode_reports in modes.items()},
            "failed_jobs": sum(1 for r in reports if r.status == "failed"),
        }
