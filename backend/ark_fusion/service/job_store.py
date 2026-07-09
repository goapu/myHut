from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class JobRecord:
    job_id: str
    status: str
    mode: str
    profile: Optional[str]
    dataset_path: Optional[Path]
    output_path: Optional[Path]
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class JobStore:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir = self.base_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.records: Dict[str, JobRecord] = {}

    def create_job(self, mode: str, profile: Optional[str], dataset_path: Optional[Path] = None) -> JobRecord:
        job_id = str(uuid.uuid4())
        output_path = self.jobs_dir / job_id
        record = JobRecord(job_id=job_id, status="pending", mode=mode, profile=profile, dataset_path=dataset_path, output_path=output_path)
        self.records[job_id] = record
        return record

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self.records.get(job_id)

    def update_status(self, job_id: str, status: str) -> None:
        record = self.get_job(job_id)
        if record:
            record.status = status

    def update_metrics(self, job_id: str, metrics: Dict[str, Any]) -> None:
        record = self.get_job(job_id)
        if record:
            record.metrics = metrics

    def update_artifacts(self, job_id: str, artifacts: Dict[str, str]) -> None:
        record = self.get_job(job_id)
        if record:
            record.artifacts = artifacts

    def append_error(self, job_id: str, error: str) -> None:
        record = self.get_job(job_id)
        if record:
            record.errors.append(error)
