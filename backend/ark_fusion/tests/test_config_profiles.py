"""Tests for reconstruction profile presets in arkit_sensor_fusion."""
from pathlib import Path
import sys
import types

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

from arkit_sensor_fusion.config import apply_profile_defaults, build_parser, PROFILE_PRESETS


def test_profile_object_on_support_sets_object_mode_and_defaults():
    parser = build_parser()
    args = parser.parse_args(["--dataset", "data", "--profile", "product_object_on_support"])
    args = apply_profile_defaults(args)

    preset = PROFILE_PRESETS["product_object_on_support"]
    assert args.mode == "object"
    assert args.depth_trunc == preset["depth_trunc"]
    assert args.object_plane_distance == preset["object_plane_distance"]


def test_profile_allows_mode_override_when_explicit():
    parser = build_parser()
    args = parser.parse_args(["--dataset", "data", "--profile", "product_object_on_support", "--mode", "measure_room"])
    args = apply_profile_defaults(args)

    assert args.mode == "measure_room"
    assert args.object_plane_distance == PROFILE_PRESETS["product_object_on_support"]["object_plane_distance"]
