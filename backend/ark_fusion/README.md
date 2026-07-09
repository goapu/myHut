# ARKit Sensor Fusion Reconstruction

Modular ARKit RGB-D reconstruction package for object and room scans.

## Architecture

```text
Upload / Dataset ID
  -> Dataset Validator
  -> Frame Preprocessor: RGB/depth/confidence, confidence normalization, depth filters, object masks
  -> PoseQualityGate
  -> Coverage-aware Keyframe Selector
  -> Fusion Backend: TSDF / future voxel blocks
  -> Object Pipeline or Room Pipeline
  -> Quality Evaluator: metrics + failure classification
  -> Artifact Writer: mesh, point cloud, JSON metrics, logs
```

## Run

```bash
python arkit_reconstruct.py --dataset /path/to/rgbd_capture --output /path/to/out --mode object \
  --use-confidence --min-confidence 2 --prefusion-object-mask --pose-refinement icp
```

## Tests

```bash
python -m pytest -q tests
```

## Accuracy, confidence, and benchmarking additions

This version separates capture/fusion health from true geometric accuracy. The old `quality_score` remains for backward compatibility, but production decisions should use the explicit fields in `metrics.json` and `reconstruction_confidence.json`:

- `capture_quality_score`: depth coverage, ARKit confidence, and camera-motion health.
- `fusion_health_score`: frame integration ratio and mesh density/health.
- `geometry_confidence_score`: confidence in reconstructed geometry when no ground truth exists.
- `measurement_confidence_score`: confidence in room/object measurements.
- `benchmark_accuracy_score`: only populated when ground-truth geometry or measurement JSON is provided.

Optional ground-truth flags:

```bash
arkit-reconstruct \
  --dataset /path/to/capture \
  --output /path/to/output \
  --mode object \
  --ground-truth-object-mesh /path/to/reference_mesh.ply \
  --benchmark-condition good_light
```

Object benchmark metrics include Chamfer-L1, precision/completeness, F-score at 5/10/20 mm, normal consistency, watertightness, and scale error. Room reconstruction benchmarks use 20/50/100 mm thresholds. Room measurement benchmarks compare `height_m`, `length_m`, and `width_m` against a JSON file.

Optional segmentation masks:

```bash
arkit-reconstruct --dataset capture --mode object --object-mask-dir masks_sam
```

Mask files should match frame stems, for example `rgb/000123.jpg` with `masks_sam/000123.png`. Nonzero mask pixels are fused; zero pixels are removed.

New run artifacts include:

- `raw_tsdf_mesh.ply`
- `raw_tsdf_cloud.ply`
- object or room cleaned outputs
- `camera_trajectory.json`
- `selected_keyframes.json`
- `rejected_frames.json`
- `room_dimensions.json`
- `room_planes.json`
- `scan_guidance.json`
- `reconstruction_confidence.json`
- `benchmark_report.json` when ground truth is supplied

The `--pose-refinement icp` mode now applies accepted frame-to-model point-to-plane ICP corrections to the camera pose instead of only logging ICP diagnostics. Updates are rejected when fitness/RMSE or correction magnitude is unsafe.

The `--fusion-backend tensor_tsdf` / `voxel_block_grid` option uses Open3D's sparse tensor `VoxelBlockGrid` backend when the installed Open3D build supports it, preserving TSDF/color/weight attributes for larger scenes.

---

## myHut iPhone upload server

This backend now includes `upload_server.py`, which is compatible with the Swift `ARDepthRecorder` used by the myHut iPhone app.

Start it on your Mac/PC:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python upload_server.py --host 0.0.0.0 --port 5001 --workspace .
```

Set the iOS app's local config to your Mac/PC IP and port `5001`:

```swift
enum LocalConfig {
    static let serverIP = "YOUR_MAC_LOCAL_IP"
    static let serverPort = "5001"
}
```

The phone streams files to:

```text
POST /upload/rgb
POST /upload/depth
POST /upload/confidence
POST /upload/pose
POST /upload/intrinsics
POST /upload/timestamp
POST /upload/frame_metadata
POST /upload/metadata
POST /command
```

Temporary captures are written to:

```text
incoming/<session_name>/
```

Saved captures are moved to:

```text
captures/<session_name>/
```

Run reconstruction manually:

```bash
python arkit_reconstruct.py \
  --dataset ./captures/SESSION_NAME \
  --output ./outputs/SESSION_NAME \
  --mode room_full \
  --use-confidence
```

Or trigger reconstruction through the upload server:

```bash
curl -X POST http://localhost:5001/reconstruct/SESSION_NAME \
  -H 'Content-Type: application/json' \
  -d '{"mode":"room_full","use_confidence":true}'
```

See `MYHUT_BACKEND_README.md` for the full myHut integration notes.
