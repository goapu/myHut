from __future__ import annotations
from .common import *
import yaml


def _load_profile_presets_at_import() -> Dict[str, Any]:
    """Best-effort profile map for tests, docs, and callers that need introspection."""
    presets: Dict[str, Any] = {}
    here = Path(__file__).resolve()
    candidate_dirs = [here.parents[2] / "profiles", here.parents[1] / "profiles"]
    for profile_dir in candidate_dirs:
        if not profile_dir.exists():
            continue
        for path in sorted(profile_dir.glob("product_*.yaml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    presets[path.stem] = data
            except Exception:
                continue
    return presets


PROFILE_PRESETS: Dict[str, Any] = _load_profile_presets_at_import()


def get_profile_directories() -> list[Path]:
    return [
        Path(__file__).resolve().parents[2] / "profiles",
        Path(__file__).resolve().parents[1] / "profiles",
    ]


def find_profile_files() -> list[Path]:
    profiles: list[Path] = []
    for profile_dir in get_profile_directories():
        if profile_dir.exists():
            profiles.extend(sorted(profile_dir.glob("product_*.yaml")))
    return profiles


def list_profile_names() -> list[str]:
    return [profile.stem for profile in find_profile_files()]


def load_profile_preset(profile_name: str) -> Dict[str, Any]:
    for profile_dir in get_profile_directories():
        preset_path = profile_dir / f"{profile_name}.yaml"
        if preset_path.exists():
            with preset_path.open("r", encoding="utf-8") as stream:
                preset = yaml.safe_load(stream)
            if not isinstance(preset, dict):
                raise ValueError(f"Invalid profile format in {preset_path}")
            return preset
    raise ValueError(f"Unknown profile: {profile_name}")


class ExplicitAction(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        super().__call__(parser, namespace, values, option_string)
        explicit = getattr(namespace, "_explicit_args", set())
        explicit.add(self.dest)
        setattr(namespace, "_explicit_args", explicit)


class ExplicitStoreTrueAction(argparse._StoreTrueAction):
    def __call__(self, parser, namespace, values, option_string=None):
        super().__call__(parser, namespace, values, option_string)
        explicit = getattr(namespace, "_explicit_args", set())
        explicit.add(self.dest)
        setattr(namespace, "_explicit_args", explicit)


class ExplicitStoreFalseAction(argparse._StoreFalseAction):
    def __call__(self, parser, namespace, values, option_string=None):
        super().__call__(parser, namespace, values, option_string)
        explicit = getattr(namespace, "_explicit_args", set())
        explicit.add(self.dest)
        setattr(namespace, "_explicit_args", explicit)


def apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    profile_name = getattr(args, "profile", None)
    if profile_name is None:
        return args

    preset = load_profile_preset(profile_name)
    explicit = getattr(args, "_explicit_args", set())
    if preset.get("mode") is not None and "mode" not in explicit:
        args.mode = preset["mode"]

    for key, value in preset.items():
        if key == "mode":
            continue
        if key in explicit:
            continue
        setattr(args, key, value)

    return args


def apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """
    Allows better default TSDF settings per mode.

    User-supplied values still override these defaults.
    """
    if args.mode == "object":
        if args.depth_trunc is None:
            args.depth_trunc = 2.0
        if args.min_depth is None:
            args.min_depth = 0.15
        if args.tsdf_voxel_length is None:
            args.tsdf_voxel_length = 0.006
        if args.tsdf_sdf_trunc is None:
            args.tsdf_sdf_trunc = 0.03

    elif args.mode == "measure_room":
        if args.depth_trunc is None:
            args.depth_trunc = 5.5
        if args.min_depth is None:
            args.min_depth = 0.2
        if args.tsdf_voxel_length is None:
            args.tsdf_voxel_length = 0.03
        if args.tsdf_sdf_trunc is None:
            args.tsdf_sdf_trunc = 0.15

    elif args.mode == "room_full":
        if args.depth_trunc is None:
            args.depth_trunc = 5.0
        if args.min_depth is None:
            args.min_depth = 0.2
        if args.tsdf_voxel_length is None:
            args.tsdf_voxel_length = 0.025
        if args.tsdf_sdf_trunc is None:
            args.tsdf_sdf_trunc = 0.12

    return args


# ---------------------------------------------------------------------
# 7. MAIN EXECUTION
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Product-ready ARKit RGB-D Fusion")
    def add(*args, **kwargs):
        action = kwargs.get("action")
        if action == "store_true":
            kwargs["action"] = ExplicitStoreTrueAction
        elif action == "store_false":
            kwargs["action"] = ExplicitStoreFalseAction
        elif action is None:
            kwargs["action"] = ExplicitAction
        parser.add_argument(*args, **kwargs)

    add("--dataset", required=True, type=str, help="Path to rgbd_capture folder")
    add("--output", default="arkit_output", type=str, help="Output folder")
    add("--mode", choices=["object", "measure_room", "room_full"], required=False, help="Processing mode")
    add("--profile", choices=list_profile_names(), help="Apply a named product profile")

    # TSDF params. Defaults are None so apply_mode_defaults can choose mode defaults.
    add("--depth-trunc", default=None, type=float)
    add("--min-depth", default=None, type=float)
    add("--tsdf-voxel-length", default=None, type=float)
    add("--tsdf-sdf-trunc", default=None, type=float)
    add("--fusion-backend", default="legacy_tsdf", choices=["legacy_tsdf", "scalable_tsdf", "open3d_legacy", "tensor_tsdf", "voxel_block_grid"])

    add("--frame-step", default=1, type=int, help="Use every Nth frame before quality/keyframe selection")
    add("--visualize", action="store_true", help="Open Open3D viewer")
    add("--json-logs", action="store_true", default=True, help="Write structured JSONL logs")

    # Depth quality and pluggable filtering.
    add("--depth-filter", default="median_bilateral", choices=["none", "median_bilateral", "median", "bilateral"])
    add("--disable-depth-filter", action="store_true", help="Deprecated alias for --depth-filter none")
    add("--depth-median-ksize", default=5, type=int)
    add("--depth-bilateral-sigma-color", default=0.05, type=float)
    add("--min-valid-depth-ratio", default=0.25, type=float, help="Skip frames with less valid depth than this ratio")

    # Market-ready pose validation.
    add("--disable-pose-gate", action="store_true")
    add("--max-pose-jump", default=0.12, type=float, help="Max camera translation jump in meters")
    add("--max-rotation-jump-deg", default=15.0, type=float, help="Max camera rotation jump in degrees")

    # Keyframe selection.
    add("--disable-keyframes", action="store_true")
    add("--keyframe-min-translation", default=0.015, type=float)
    add("--keyframe-min-rotation-deg", default=2.0, type=float)

    # Confidence maps.
    add("--use-confidence", action="store_true", help="Use optional confidence maps from dataset/confidence")
    add("--min-confidence", default=1, type=int, choices=[0, 1, 2], help="Minimum normalized confidence to keep: 0 low, 1 medium, 2 high")

    # Object crop.
    add("--crop-min", nargs=3, type=float, default=None, metavar=("XMIN", "YMIN", "ZMIN"))
    add("--crop-max", nargs=3, type=float, default=None, metavar=("XMAX", "YMAX", "ZMAX"))

    # Object extraction / Poisson cleanup. These used to be hard-coded inside the script.
    add("--object-plane-distance", default=0.012, type=float)
    add("--object-cluster-eps", default=0.10, type=float)
    add("--object-cluster-min-points", default=15, type=int)
    add("--object-keep-cluster-diagonal", default=0.025, type=float)
    add("--object-cluster-join-distance", default=0.45, type=float)
    add("--object-support-distance", default=0.018, type=float)
    add("--object-support-median-distance", default=0.030, type=float)
    add("--object-below-padding", default=0.010, type=float)
    add("--object-local-xy-radius", default=0.025, type=float)
    add("--object-sample-points", default=300000, type=int)
    add("--object-downsample-voxel", default=0.005, type=float)
    add("--object-normal-radius", default=0.035, type=float)
    add("--poisson-depth", default=8, type=int)
    add("--poisson-scale", default=1.25, type=float)
    add("--poisson-density-quantile", default=0.08, type=float)

    # Room cleanup / measurement budgets. These make post-processing bounded.
    add("--room-min-component-bbox-diagonal", default=0.12, type=float)
    add("--room-min-component-bbox-extent", default=0.025, type=float)
    add("--room-sample-points", default=1000000, type=int)
    add("--room-plane-voxel", default=0.025, type=float)
    add("--room-plane-distance", default=0.04, type=float)
    add("--room-min-plane-points", default=3000, type=int)
    add("--room-full-envelope-sample-points", default=700000, type=int)
    add("--room-full-envelope-voxel", default=0.035, type=float)
    add("--room-full-plane-distance", default=0.045, type=float)
    add("--room-full-min-plane-points", default=2500, type=int)

    # Ground-truth / benchmarking inputs. These are optional and only run when supplied.
    add("--ground-truth-object-mesh", default=None, type=str, help="Reference object mesh/point cloud for Chamfer/F-score/normal/scale metrics")
    add("--ground-truth-room-mesh", default=None, type=str, help="Reference room mesh/point cloud for mesh distance/completeness metrics")
    add("--ground-truth-measurements", default=None, type=str, help="JSON with expected room height/length/width and optional plane models")
    add("--benchmark-condition", default="unspecified", type=str, help="Capture condition label, e.g. good_light, low_light, clutter, loop_closure")
    add("--object-mask-dir", default=None, type=str, help="Optional directory of per-frame manual/SAM/semantic object masks")
    add("--semantic-mask-dir", default=None, type=str, help="Optional directory of per-frame semantic masks for fusion/benchmarking")
    add("--save-preview-images", action="store_true", help="Save lightweight frame previews for QA artifacts")
    return parser


def add_sensor_fusion_options(parser: argparse.ArgumentParser) -> None:
    # Safe to add only when absent, for compatibility with existing parser.
    existing = {a.dest for a in parser._actions}
    def add(*args, **kwargs):
        dest = kwargs.get("dest")
        if dest is None:
            for a in args:
                if a.startswith("--"):
                    dest = a.lstrip("-").replace("-", "_")
                    break
        if dest not in existing:
            action = kwargs.get("action")
            if action == "store_true":
                kwargs["action"] = ExplicitStoreTrueAction
            elif action == "store_false":
                kwargs["action"] = ExplicitStoreFalseAction
            elif action is None:
                kwargs["action"] = ExplicitAction
            parser.add_argument(*args, **kwargs)
            existing.add(dest)
    add("--prefusion-object-mask", action="store_true", default=True, help="Mask object/background depth before object TSDF fusion.")
    add("--disable-prefusion-object-mask", action="store_true", help="Disable pre-fusion object/background masking.")
    add("--object-mask-plane-distance", type=float, default=0.015)
    add("--object-mask-depth-band-margin", type=float, default=0.15)
    add("--object-mask-min-foreground-ratio", type=float, default=0.005)
    add("--keyframe-min-coverage-gain", type=float, default=0.015)
    add("--pose-refinement", choices=["none", "icp"], default="none")
    add("--icp-voxel-size", type=float, default=0.02)
    add("--icp-max-correspondence", type=float, default=0.04)
    add("--icp-min-fitness", type=float, default=0.25)
    add("--icp-max-rmse", type=float, default=0.035)
    add("--duplicate-layer-voxel", type=float, default=0.008)
    add("--duplicate-layer-gap", type=float, default=0.018)
    add("--duplicate-layer-skip-poisson-score", type=float, default=0.08)
    add("--disable-duplicate-layer-collapse", action="store_true")
    add("--disable-poisson", action="store_true")

    add("--object-center-prior", action="store_true", default=True, help="Use centered-object prior for object scans; recommended for shoes/headphones on turntable/table.")
    add("--disable-object-center-prior", action="store_true", help="Disable centered-object prior.")
    add("--object-center-roi-fraction", type=float, default=0.72, help="Fraction of image width/height used as target-object ROI before fusion.")
    add("--object-center-depth-margin", type=float, default=0.18, help="Depth margin around central object depth before fusion, in meters.")
    add("--object-max-depth-spread", type=float, default=0.55, help="Maximum kept object depth spread after center-depth estimation, in meters.")
    add("--object-select-center-prior", action="store_true", default=True, help="After fusion, prefer clusters near the object cloud center instead of the largest surrounding cluster.")
    add("--disable-object-select-center-prior", action="store_true", help="Disable center-prior cluster selection after fusion.")

_original_build_parser = build_parser
def build_parser() -> argparse.ArgumentParser:  # type: ignore[override]
    parser = _original_build_parser()
    add_sensor_fusion_options(parser)
    return parser
