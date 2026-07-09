from __future__ import annotations
from .common import *

@dataclass
class ArtifactWriter:
    output_dir: Path
    metrics: MetricsRecorder

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_mesh(self, name: str, mesh: o3d.geometry.TriangleMesh) -> Path:
        path = self.output_dir / name
        o3d.io.write_triangle_mesh(str(path), mesh)
        self.metrics.output_artifacts[name] = str(path)
        return path

    def write_point_cloud(self, name: str, pcd: o3d.geometry.PointCloud) -> Path:
        path = self.output_dir / name
        o3d.io.write_point_cloud(str(path), pcd)
        self.metrics.output_artifacts[name] = str(path)
        return path

    def write_metrics(self) -> Path:
        path = self.output_dir / "metrics.json"
        write_json(path, self.metrics.finalize())
        return path
