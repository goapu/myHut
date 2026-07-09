# ARKit Sensor Fusion Deployment Guide

## Quick Start

### Local CLI (Development)

```bash
cd arkit_sensor_fusion
pip install -e .

arkit-reconstruct --dataset /path/to/rgbd_capture --output out --profile product_object_on_support
```

### Service API (Production)

#### Docker Compose

```bash
cd arkit_sensor_fusion
docker-compose up -d

# Check health
curl http://localhost:8000/health

# Submit a job
curl -X POST http://localhost:8000/reconstructions \
  -F "dataset=@dataset.zip" \
  -F "profile=product_object_on_support"

# Poll status
curl http://localhost:8000/reconstructions/{job_id}

# Get metrics
curl http://localhost:8000/reconstructions/{job_id}/metrics

# Download artifact
curl http://localhost:8000/reconstructions/{job_id}/artifacts/object_clean_cloud.ply \
  -o cloud.ply
```

#### Manual Docker Build

```bash
docker build -t arkit-reconstruct:latest .
docker run -p 8000:8000 -v /data:/app/data arkit-reconstruct:latest
```

## Installation

### Package Installation

```bash
pip install arkit-sensor-fusion
```

Or with service dependencies:

```bash
pip install arkit-sensor-fusion[service]
```

Or development mode:

```bash
pip install -e .[dev,service]
```

## Profiles

Three product profiles are available via `--profile`:

- **`product_object_on_support`**: Object reconstruction (shoes, headphones on turntable)
  - Fine TSDF voxel size (6mm)
  - Object clustering with duplicate layer detection
  - Poisson mesh refinement

- **`product_room_measure`**: Room dimension measurement
  - Coarse TSDF voxel size (30mm)
  - Large plane extraction
  - Room envelope estimation

- **`product_room_full`**: Full room reconstruction with envelope clipping
  - Medium TSDF voxel size (25mm)
  - Envelope-based filtering
  - Mesh component filtering by physical size

## API Reference

### POST /reconstructions
Upload a dataset and start a reconstruction job.

**Request:**
```
multipart/form-data:
  - dataset: file (zip)
  - profile: string (optional; product_object_on_support, product_room_measure, product_room_full)
  - mode: string (optional; object, measure_room, room_full)
```

**Response (201 Created):**
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "profile": "product_object_on_support",
  "mode": "object"
}
```

### GET /reconstructions/{job_id}
Get job status and quality metrics.

**Response (200 OK):**
```json
{
  "job_id": "uuid-string",
  "status": "reconstructing|completed|failed",
  "mode": "object",
  "profile": "product_object_on_support",
  "quality_score": 0.75,
  "frames_integrated": 120,
  "errors": []
}
```

### GET /reconstructions/{job_id}/metrics
Get full metrics report.

**Response (200 OK):**
```json
{
  "status": "success",
  "quality_score": 0.75,
  "frames_integrated": 120,
  "frames_seen": 150,
  "mesh_vertices": 50000,
  "mesh_triangles": 100000,
  "integrated_ratio": 0.8,
  "valid_depth_ratio_median": 0.92,
  "high_confidence_ratio_median": null,
  "elapsed_s": 45.2,
  "output_artifacts": {...},
  "failures": [...]
}
```

### GET /reconstructions/{job_id}/artifacts/{artifact_name}
Download a reconstruction artifact.

**Supported artifact names:**
- `object_clean_cloud.ply` (point cloud)
- `object_watertight_mesh.ply` (mesh)
- `room_measurement_cloud.ply` (measurement cloud)
- `room_full_mesh.ply` (full room mesh)
- `raw_tsdf_mesh.ply` (raw TSDF output)
- `metrics.json` (metrics)

**Response (200 OK):**
Binary file stream

## Examples

### Object Reconstruction from CLI

```bash
arkit-reconstruct \
  --dataset /path/to/rgbd_capture \
  --output /path/to/output \
  --profile product_object_on_support \
  --use-confidence \
  --min-confidence 2 \
  --pose-refinement icp
```

### Service Job Submission (Python)

```python
import requests
import json

# Create job
with open("dataset.zip", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/reconstructions",
        files={"dataset": f},
        data={"profile": "product_object_on_support"}
    )

job_id = resp.json()["job_id"]
print(f"Job {job_id} submitted")

# Poll for completion
import time
while True:
    status = requests.get(f"http://localhost:8000/reconstructions/{job_id}").json()
    print(f"Status: {status['status']}")
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(5)

# Download metrics and artifacts
metrics = requests.get(f"http://localhost:8000/reconstructions/{job_id}/metrics").json()
print(json.dumps(metrics, indent=2))

# Download point cloud
cloud = requests.get(
    f"http://localhost:8000/reconstructions/{job_id}/artifacts/object_clean_cloud.ply"
)
with open("cloud.ply", "wb") as f:
    f.write(cloud.content)
```

## Configuration

All reconstruction parameters can be overridden via CLI or profile YAML:

```bash
# Override specific TSDF parameters
arkit-reconstruct \
  --dataset data \
  --output out \
  --profile product_object_on_support \
  --tsdf-voxel-length 0.005 \
  --object-cluster-eps 0.08
```

Profiles are loaded from:
1. `./profiles/` (repo root)
2. `./arkit_sensor_fusion/profiles/` (package)

## Testing

```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=arkit_sensor_fusion --cov-report=html

# Integration tests (requires open3d)
pytest tests/test_reconstruction_core.py -v
```

## Troubleshooting

### No frames integrated
- Check depth filter settings (`--depth-filter none` to disable)
- Verify min/max depth bounds (`--min-depth`, `--depth-trunc`)
- Check pose quality gate (`--disable-pose-gate` to skip)

### Low quality score
- Increase keyframe selection (`--keyframe-min-coverage-gain`)
- Adjust pose refinement (`--pose-refinement icp`)
- Check confidence map usage (`--use-confidence --min-confidence 2`)

### Docker memory issues
- Reduce TSDF voxel count: `--tsdf-voxel-length 0.05` (coarser)
- Reduce sample points: `--room-sample-points 500000`

## Support

For issues, documentation, and contributions:
- **Repository**: https://github.com/example/arkit-sensor-fusion
- **Issue Tracker**: https://github.com/example/arkit-sensor-fusion/issues
- **Documentation**: https://arkit-sensor-fusion.readthedocs.io
