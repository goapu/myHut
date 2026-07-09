from __future__ import annotations
import io
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from service.job_store import JobStore
from service.worker import ReconstructionWorker


def create_app(workspace_dir: Path) -> Flask:
    app = Flask(__name__)
    job_store = JobStore(workspace_dir / "store")
    worker = ReconstructionWorker(job_store, workspace_dir / "data")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "service": "arkit-reconstruct"})

    @app.route("/reconstructions", methods=["POST"])
    def create_reconstruction():
        """POST /reconstructions
        
        Request:
          - multipart form with file 'dataset' (zip)
          - form data: profile (optional), mode (required if profile not given)
        
        Returns:
          - job_id, status, estimated processing time
        """
        if "dataset" not in request.files:
            return jsonify({"error": "No dataset file provided"}), 400

        profile = request.form.get("profile")
        mode = request.form.get("mode")

        if profile is None and mode is None:
            return jsonify({"error": "Either profile or mode is required"}), 400

        file = request.files["dataset"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        temp_zip = job_store.base_dir / "temp" / filename
        temp_zip.parent.mkdir(parents=True, exist_ok=True)
        file.save(str(temp_zip))

        # Create and start the reconstruction job.
        # The previous service implementation only created a pending job; this
        # version launches the worker in a background thread so API clients can
        # poll /reconstructions/<job_id> for progress/status.
        job = job_store.create_job(mode=mode, profile=profile, dataset_path=temp_zip)

        resolved_mode = mode or "room_full"
        resolved_profile = profile

        def run_background_job() -> None:
            try:
                worker.run_job(job.job_id, temp_zip, resolved_profile, resolved_mode)
            except Exception as exc:  # defensive: worker already records most errors
                job_store.update_status(job.job_id, "failed")
                job_store.append_error(job.job_id, str(exc))

        threading.Thread(target=run_background_job, daemon=True).start()

        return jsonify(
            {
                "job_id": job.job_id,
                "status": "started",
                "profile": job.profile,
                "mode": job.mode,
            }
        ), 201

    @app.route("/reconstructions/<job_id>", methods=["GET"])
    def get_reconstruction_status(job_id: str):
        """GET /reconstructions/{job_id}
        
        Returns:
          - job status, progress, quality score, errors
        """
        job = job_store.get_job(job_id)
        if job is None:
            return jsonify({"error": "Job not found"}), 404

        return jsonify(
            {
                "job_id": job.job_id,
                "status": job.status,
                "mode": job.mode,
                "profile": job.profile,
                "quality_score": job.metrics.get("quality_score"),
                "frames_integrated": job.metrics.get("frames_integrated"),
                "errors": job.errors,
            }
        ), 200

    @app.route("/reconstructions/<job_id>/metrics", methods=["GET"])
    def get_reconstruction_metrics(job_id: str):
        """GET /reconstructions/{job_id}/metrics
        
        Returns:
          - full metrics dictionary (status, quality, frames, timestamps, etc.)
        """
        job = job_store.get_job(job_id)
        if job is None:
            return jsonify({"error": "Job not found"}), 404

        if not job.metrics:
            return jsonify({"error": "Metrics not available"}), 204

        return jsonify(job.metrics), 200

    @app.route("/reconstructions/<job_id>/artifacts/<artifact_name>", methods=["GET"])
    def get_artifact(job_id: str, artifact_name: str):
        """GET /reconstructions/{job_id}/artifacts/{artifact_name}
        
        Artifact names: object_clean_cloud.ply, room_dimensions (JSON), etc.
        Returns:
          - file stream or error
        """
        job = job_store.get_job(job_id)
        if job is None:
            return jsonify({"error": "Job not found"}), 404

        artifact_path = job.artifacts.get(artifact_name)
        if artifact_path is None:
            return jsonify({"error": f"Artifact not found: {artifact_name}"}), 404

        path = Path(artifact_path)
        if not path.exists():
            return jsonify({"error": f"Artifact file missing: {artifact_name}"}), 404

        return send_file(str(path), as_attachment=True, download_name=artifact_name)

    return app


if __name__ == "__main__":
    import sys

    workspace = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/arkit_workspace")
    app = create_app(workspace)
    app.run(host="0.0.0.0", port=8000, debug=False)
