"""Core tests for arkit_sensor_fusion.

These tests stub Open3D so CI can validate non-geometry business logic without
an Open3D wheel. Add integration tests with real Open3D in the reconstruction CI image.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


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


def load_modules():
    install_fake_open3d()
    from arkit_sensor_fusion import preprocessing as pre
    from arkit_sensor_fusion.pose_quality import PoseQualityGate
    from arkit_sensor_fusion.common import FailureCode
    return pre, PoseQualityGate, FailureCode


def test_confidence_normalizes_arkit_raw_values():
    pre, _, _ = load_modules()
    raw = np.array([[0, 1, 2], [2, 2, 1]], dtype=np.uint8)
    norm, report = pre.normalize_confidence_map(raw)
    assert report.codec == pre.ConfidenceCodec.ARKIT_RAW_0_1_2
    assert set(np.unique(norm).tolist()) == {0, 1, 2}
    assert report.high_confidence_ratio == 3 / 6


def test_confidence_normalizes_uint8_quantized_values():
    pre, _, _ = load_modules()
    raw = np.array([[0, 128, 255], [255, 128, 0]], dtype=np.uint8)
    norm, report = pre.normalize_confidence_map(raw)
    assert report.codec == pre.ConfidenceCodec.UINT8_QUANTIZED
    assert norm.tolist() == [[0, 1, 2], [2, 1, 0]]
    assert report.histogram


def test_pose_quality_gate_rejects_rotation_jump():
    _, PoseQualityGate, FailureCode = load_modules()
    prev = np.eye(4)
    curr = np.eye(4)
    theta = np.deg2rad(30.0)
    curr[:3, :3] = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    gate = PoseQualityGate(max_translation_jump_m=1.0, max_rotation_jump_deg=10.0)
    q = gate.evaluate(curr, prev)
    assert not q.accepted
    assert q.code == FailureCode.POSE_ROTATION_JUMP


def test_pose_quality_gate_rejects_bad_matrix():
    _, PoseQualityGate, FailureCode = load_modules()
    bad = np.eye(4)
    bad[3, 3] = 2.0
    gate = PoseQualityGate()
    q = gate.evaluate(bad, None)
    assert not q.accepted
    assert q.code == FailureCode.BAD_POSE_MATRIX
