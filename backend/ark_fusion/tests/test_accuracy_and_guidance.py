import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def install_fake_open3d():
    fake_o3d = types.SimpleNamespace(
        pipelines=types.SimpleNamespace(
            integration=types.SimpleNamespace(
                TSDFVolumeColorType=types.SimpleNamespace(RGB8="RGB8"),
                ScalableTSDFVolume=object,
            ),
            registration=types.SimpleNamespace(),
        ),
        geometry=types.SimpleNamespace(
            RGBDImage=types.SimpleNamespace(create_from_color_and_depth=lambda *a, **k: None),
            Image=lambda x: x,
            TriangleMesh=object,
            PointCloud=object,
            KDTreeFlann=object,
            AxisAlignedBoundingBox=object,
            KDTreeSearchParamHybrid=object,
        ),
        camera=types.SimpleNamespace(PinholeCameraIntrinsic=object),
        io=types.SimpleNamespace(write_triangle_mesh=lambda *a, **k: True, write_point_cloud=lambda *a, **k: True),
        utility=types.SimpleNamespace(Vector3dVector=lambda x: x),
    )
    sys.modules.setdefault("open3d", fake_o3d)


install_fake_open3d()


def test_metrics_split_scores_are_explicit():
    from arkit_sensor_fusion.common import MetricsRecorder

    m = MetricsRecorder(mode="object", input_frames_total=10, frames_integrated=8, mesh_vertices=5000, mesh_triangles=9000)
    m.valid_depth_ratios.extend([0.7, 0.8, 0.9])
    out = m.finalize()
    assert 0.0 <= out["capture_quality_score"] <= 1.0
    assert 0.0 <= out["fusion_health_score"] <= 1.0
    assert 0.0 <= out["geometry_confidence_score"] <= 1.0
    assert "quality_score" in out  # backward compatible aggregate only


def test_scan_guidance_emits_actionable_events():
    from arkit_sensor_fusion.scan_guidance import ScanGuidanceRecorder

    g = ScanGuidanceRecorder()
    g.observe_depth("0001", valid_ratio=0.1, median_depth=0.05, min_depth=0.1, depth_trunc=4.0)
    g.observe_motion("0001", translation_jump_m=0.2, rotation_jump_deg=20.0)
    summary = g.summarize()
    codes = {e["code"] for e in summary["events"]}
    assert {"low_confidence_depth", "too_close", "move_slower", "excessive_pose_jump"}.issubset(codes)
