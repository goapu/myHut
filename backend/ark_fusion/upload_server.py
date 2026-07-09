#!/usr/bin/env python3
"""myHut iPhone RGB-D upload server.

This server matches the Swift ARDepthRecorder streaming API:

    POST /upload/<type>
      Headers:
        Session-Name: object_2026-07-09_13-00-00
        Frame-Index: 000001 or 000001_shape
        File-Extension: .jpg, .bin, .txt, .json
        Capture-Mode: object | measure_room | room_full

    POST /command
      JSON: {"session_name": "...", "action": "save" | "discard"}

The server first writes incoming captures to ./incoming/<session_name>/.
When the phone sends action=save, the session is moved to ./captures/<session_name>/.
When action=discard, the temporary incoming session is deleted.

Optional:
    POST /reconstruct/<session_name>
      JSON: {"mode": "room_full", "use_confidence": true}

This launches the CLI pipeline in the background and writes output to
./outputs/<session_name>/.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request


BASE_DIR = Path(os.environ.get("MYHUT_WORKSPACE", ".")).expanduser().resolve()
INCOMING_DIR = BASE_DIR / "incoming"
CAPTURES_DIR = BASE_DIR / "captures"
OUTPUTS_DIR = BASE_DIR / "outputs"

ALLOWED_TYPES = {
    "rgb",
    "depth",
    "confidence",
    "pose",
    "intrinsics",
    "timestamp",
    "frame_metadata",
    "metadata",
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bin", ".txt", ".json"}
VALID_MODES = {"object", "measure_room", "room_full"}

app = Flask(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str, fallback: str = "unnamed") -> str:
    value = (value or "").strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._")
    return value or fallback


def safe_extension(value: str) -> str:
    value = (value or "").strip().lower()
    if not value.startswith("."):
        value = f".{value}"
    if value not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {value}")
    return value


def session_paths(session_name: str) -> tuple[Path, Path]:
    safe_session = safe_name(session_name, fallback="session")
    return INCOMING_DIR / safe_session, CAPTURES_DIR / safe_session


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def resolve_saved_dataset(session_name: str) -> Path | None:
    _, saved_dir = session_paths(session_name)
    if saved_dir.exists():
        return saved_dir
    incoming_dir, _ = session_paths(session_name)
    if incoming_dir.exists():
        return incoming_dir
    return None


def get_mode_for_session(dataset_dir: Path, requested_mode: str | None = None) -> str:
    if requested_mode in VALID_MODES:
        return requested_mode

    metadata = read_json_file(dataset_dir / "metadata.json")
    mode = metadata.get("capture_mode") or metadata.get("recommended_python_mode")
    if mode in VALID_MODES:
        return str(mode)

    # Fallback: infer from session name prefix.
    name = dataset_dir.name
    for candidate in VALID_MODES:
        if name.startswith(candidate):
            return candidate

    return "room_full"


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "myhut-upload-server",
            "workspace": str(BASE_DIR),
            "incoming_dir": str(INCOMING_DIR),
            "captures_dir": str(CAPTURES_DIR),
            "outputs_dir": str(OUTPUTS_DIR),
        }
    )


@app.post("/upload/<data_type>")
def upload(data_type: str):
    data_type = safe_name(data_type)
    if data_type not in ALLOWED_TYPES:
        return jsonify({"error": f"Unsupported upload type: {data_type}"}), 400

    session_name = request.headers.get("Session-Name") or request.form.get("session_name")
    frame_index = request.headers.get("Frame-Index") or request.form.get("frame_index")
    file_extension = request.headers.get("File-Extension") or request.form.get("file_extension") or ".bin"
    capture_mode = request.headers.get("Capture-Mode") or request.form.get("capture_mode") or "unknown"

    if not session_name:
        return jsonify({"error": "Missing Session-Name header"}), 400
    if not frame_index:
        return jsonify({"error": "Missing Frame-Index header"}), 400

    try:
        ext = safe_extension(file_extension)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    safe_session = safe_name(session_name)
    safe_index = safe_name(frame_index)
    safe_mode = safe_name(capture_mode, fallback="unknown")

    incoming_dir, _ = session_paths(safe_session)
    target_dir = incoming_dir / data_type
    target_dir.mkdir(parents=True, exist_ok=True)

    body = request.get_data()
    if not body:
        return jsonify({"error": "Empty request body"}), 400

    target_path = target_dir / f"{safe_index}{ext}"
    target_path.write_bytes(body)

    # The reconstruction pipeline ignores metadata/, but a root metadata.json is useful.
    if data_type == "metadata" and ext == ".json":
        try:
            metadata = json.loads(body.decode("utf-8"))
        except Exception:
            metadata = {}
        metadata.setdefault("capture_mode", safe_mode)
        metadata.setdefault("received_at", utc_now())
        metadata.setdefault("session_name", safe_session)
        write_json(incoming_dir / "metadata.json", metadata)

    # Keep a lightweight server-side session manifest.
    manifest_path = incoming_dir / "server_manifest.json"
    manifest = read_json_file(manifest_path)
    manifest.setdefault("session_name", safe_session)
    manifest.setdefault("capture_mode", safe_mode)
    manifest.setdefault("created_at", utc_now())
    manifest["updated_at"] = utc_now()
    counts = manifest.setdefault("received_counts", {})
    counts[data_type] = int(counts.get(data_type, 0)) + 1
    write_json(manifest_path, manifest)

    return jsonify(
        {
            "status": "ok",
            "session_name": safe_session,
            "type": data_type,
            "file": str(target_path.relative_to(BASE_DIR)),
        }
    )


@app.post("/command")
def command():
    payload = request.get_json(silent=True) or {}
    session_name = payload.get("session_name")
    action = payload.get("action")

    if not session_name:
        return jsonify({"error": "Missing session_name"}), 400
    if action not in {"save", "discard"}:
        return jsonify({"error": "action must be 'save' or 'discard'"}), 400

    incoming_dir, saved_dir = session_paths(session_name)

    if action == "discard":
        if incoming_dir.exists():
            shutil.rmtree(incoming_dir)
        return jsonify({"status": "discarded", "session_name": safe_name(session_name)})

    # action == save
    if not incoming_dir.exists() and saved_dir.exists():
        return jsonify(
            {
                "status": "already_saved",
                "session_name": safe_name(session_name),
                "dataset_path": str(saved_dir),
            }
        )

    if not incoming_dir.exists():
        return jsonify({"error": f"Session not found: {session_name}"}), 404

    saved_dir.parent.mkdir(parents=True, exist_ok=True)
    if saved_dir.exists():
        shutil.rmtree(saved_dir)
    shutil.move(str(incoming_dir), str(saved_dir))

    decision = {
        "session_name": safe_name(session_name),
        "action": "save",
        "saved_at": utc_now(),
        "capture_mode": payload.get("capture_mode"),
        "recommended_python_mode": payload.get("recommended_python_mode"),
    }
    write_json(saved_dir / "capture_decision.json", decision)

    mode = get_mode_for_session(saved_dir, payload.get("recommended_python_mode") or payload.get("capture_mode"))
    return jsonify(
        {
            "status": "saved",
            "session_name": saved_dir.name,
            "dataset_path": str(saved_dir),
            "recommended_command": (
                f"python arkit_reconstruct.py --dataset {saved_dir} "
                f"--output {OUTPUTS_DIR / saved_dir.name} --mode {mode} --use-confidence"
            ),
        }
    )


@app.get("/sessions")
def sessions():
    def describe(path: Path, status: str) -> dict[str, Any]:
        manifest = read_json_file(path / "server_manifest.json")
        metadata = read_json_file(path / "metadata.json")
        return {
            "session_name": path.name,
            "status": status,
            "path": str(path),
            "capture_mode": metadata.get("capture_mode") or manifest.get("capture_mode"),
            "received_counts": manifest.get("received_counts", {}),
            "updated_at": manifest.get("updated_at"),
        }

    incoming = [describe(p, "incoming") for p in sorted(INCOMING_DIR.glob("*")) if p.is_dir()]
    saved = [describe(p, "saved") for p in sorted(CAPTURES_DIR.glob("*")) if p.is_dir()]
    return jsonify({"incoming": incoming, "saved": saved})


@app.post("/reconstruct/<session_name>")
def reconstruct(session_name: str):
    dataset_dir = resolve_saved_dataset(session_name)
    if dataset_dir is None:
        return jsonify({"error": f"Session not found: {session_name}"}), 404

    payload = request.get_json(silent=True) or {}
    mode = get_mode_for_session(dataset_dir, payload.get("mode"))
    use_confidence = bool(payload.get("use_confidence", True))
    output_dir = OUTPUTS_DIR / dataset_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "arkit_reconstruct.py"),
        "--dataset",
        str(dataset_dir),
        "--output",
        str(output_dir),
        "--mode",
        mode,
    ]
    if use_confidence:
        cmd.append("--use-confidence")

    log_path = output_dir / "reconstruction_server.log"

    def run() -> None:
        with log_path.open("ab") as log:
            log.write(("\n=== reconstruction started " + utc_now() + " ===\n").encode("utf-8"))
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(Path(__file__).resolve().parent))
            log.write(("\n=== reconstruction finished " + utc_now() + " ===\n").encode("utf-8"))

    threading.Thread(target=run, daemon=True).start()

    return jsonify(
        {
            "status": "started",
            "session_name": dataset_dir.name,
            "mode": mode,
            "dataset_path": str(dataset_dir),
            "output_path": str(output_dir),
            "log_path": str(log_path),
            "command": " ".join(cmd),
        }
    ), 202


@app.get("/outputs/<session_name>")
def output_status(session_name: str):
    output_dir = OUTPUTS_DIR / safe_name(session_name)
    if not output_dir.exists():
        return jsonify({"error": "No output found for this session"}), 404
    files = [str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file()]
    return jsonify({"session_name": output_dir.name, "output_path": str(output_dir), "files": sorted(files)})


def main() -> None:
    global BASE_DIR, INCOMING_DIR, CAPTURES_DIR, OUTPUTS_DIR
    import argparse

    parser = argparse.ArgumentParser(description="myHut iPhone RGB-D upload server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=5001, type=int)
    parser.add_argument("--workspace", default=str(BASE_DIR), help="Where incoming/captures/outputs folders are stored")
    args = parser.parse_args()

    BASE_DIR = Path(args.workspace).expanduser().resolve()
    INCOMING_DIR = BASE_DIR / "incoming"
    CAPTURES_DIR = BASE_DIR / "captures"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    for d in (INCOMING_DIR, CAPTURES_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"myHut upload server listening on http://{args.host}:{args.port}")
    print(f"Workspace: {BASE_DIR}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
