"""Local browser interface for the existing Deep-Live-Cam video pipeline."""

from __future__ import annotations

import secrets
import shutil
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge


ROOT = Path(__file__).resolve().parent
JOBS_DIR = ROOT / "web_jobs"
VIDEO_EXTENSIONS = {".mp4", ".mkv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_processing_lock = threading.Lock()


def _same_origin() -> bool:
    """Block cross-site form submissions to this local processing service."""
    origin = request.headers.get("Origin")
    return (
        request.headers.get("X-Requested-With") == "Deep-Live-Cam"
        and (not origin or origin == request.host_url.rstrip("/"))
    )


def _valid_image(path: Path) -> bool:
    from PIL import Image

    try:
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG", "BMP", "WEBP"}:
                return False
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombError):
        return False


def _valid_video(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return result.returncode == 0 and "video" in result.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def _snapshot(job: dict) -> dict:
    return {
        "id": job["id"],
        "status": job["status"],
        "message": job["message"],
        "logs": list(job["logs"]),
        "preview_url": f"/api/jobs/{job['id']}/preview/result" if job.get("preview_ready") else None,
        "original_url": f"/api/jobs/{job['id']}/preview/original" if job.get("preview_ready") else None,
        "preview_strength": job.get("preview_strength"),
        "preview_seconds": job.get("preview_seconds"),
        "download_url": f"/api/jobs/{job['id']}/download" if job["status"] == "complete" else None,
        "result_url": f"/api/jobs/{job['id']}/result" if job["status"] == "complete" else None,
    }


def _parse_strength(value: str | None) -> int | None:
    try:
        strength = int(value)
    except (TypeError, ValueError):
        return None
    return strength if 0 <= strength <= 100 else None


def _run_preview(job_id: str, video: Path, face: Path, strength: int, seconds: float) -> None:
    job_dir = video.parent
    original = job_dir / "preview-original.png"
    result = job_dir / "preview-result.png"
    with _processing_lock:
        try:
            original.unlink(missing_ok=True)
            result.unlink(missing_ok=True)
            extracted = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(seconds),
                 "-i", str(video), "-frames:v", "1", "-y", str(original)],
                cwd=ROOT, capture_output=True, text=True, timeout=90, check=False,
            )
            if extracted.returncode or not original.is_file():
                raise RuntimeError("Could not extract that frame. Choose an earlier time in the video.")
            if strength == 0:
                shutil.copyfile(original, result)
            else:
                command = [
                    sys.executable, "-u", str(ROOT / "run.py"),
                    "--source", str(face), "--target", str(original),
                    "--output", str(result), "--frame-processor", "face_swapper",
                    "--opacity", str(strength / 100),
                ]
                rendered = subprocess.run(
                    command, cwd=ROOT, capture_output=True, text=True,
                    errors="replace", check=False,
                )
                if rendered.returncode or not result.is_file() or not result.stat().st_size:
                    details = (rendered.stdout + rendered.stderr).strip().splitlines()
                    with _jobs_lock:
                        _jobs[job_id]["logs"].extend(details[-20:])
                    raise RuntimeError("Could not render a preview. See the processing details.")
            with _jobs_lock:
                job = _jobs[job_id]
                job["status"] = "ready"
                job["message"] = "Preview ready. Compare the frame, then create your video."
                job["preview_ready"] = True
                job["preview_strength"] = strength
                job["preview_seconds"] = seconds
        except Exception as exc:
            with _jobs_lock:
                job = _jobs[job_id]
                job["status"] = "ready"
                job["message"] = f"Preview failed: {exc} You can still create the video."
                job["preview_ready"] = False


def _run_job(job_id: str, video: Path, face: Path, output: Path, strength: int) -> None:
    with _processing_lock:
        with _jobs_lock:
            job = _jobs[job_id]
            job["status"] = "processing"
            job["message"] = "Processing video…"

        command = [
            sys.executable, "-u", str(ROOT / "run.py"),
            "--source", str(face), "--target", str(video),
            "--output", str(output), "--keep-fps",
            "--frame-processor", "face_swapper",
            "--opacity", str(strength / 100),
        ]
        try:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                line = line.strip()
                if line:
                    with _jobs_lock:
                        job["logs"].append(line)
                        if line.startswith("[DLC.CORE]"):
                            job["message"] = line.removeprefix("[DLC.CORE]").strip()
            return_code = process.wait()
            with _jobs_lock:
                if return_code == 0 and output.is_file() and output.stat().st_size:
                    job["status"] = "complete"
                    job["message"] = "Your video is ready."
                else:
                    job["status"] = "failed"
                    job["message"] = "Processing failed. See the details below."
                    if return_code != 0:
                        job["logs"].append(f"Process exited with code {return_code}.")
        except Exception as exc:
            with _jobs_lock:
                job["status"] = "failed"
                job["message"] = f"Could not start processing: {exc}"
        finally:
            # Keep only the finished output and preview frames.
            video.unlink(missing_ok=True)
            face.unlink(missing_ok=True)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/jobs")
def create_job():
    if not _same_origin():
        abort(403)
    video_upload = request.files.get("video")
    face_upload = request.files.get("face")
    if not video_upload or not face_upload:
        return jsonify(error="Choose both a video and a face image."), 400

    video_ext = Path(video_upload.filename or "").suffix.lower()
    face_ext = Path(face_upload.filename or "").suffix.lower()
    if video_ext not in VIDEO_EXTENSIONS or face_ext not in IMAGE_EXTENSIONS:
        return jsonify(error="Use an MP4 or MKV video and a JPG, PNG, BMP, or WebP image."), 400

    job_id = secrets.token_urlsafe(18)
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    video = job_dir / f"video{video_ext}"
    face = job_dir / f"face{face_ext}"
    output = job_dir / "result.mp4"
    try:
        video_upload.save(video)
        face_upload.save(face)
        if not video.stat().st_size or not _valid_video(video):
            return jsonify(error="The video could not be read. Try an MP4 or MKV with a video track."), 400
        if not face.stat().st_size or not _valid_image(face):
            return jsonify(error="The reference image could not be read."), 400
        with _jobs_lock:
            _jobs[job_id] = {
                "id": job_id, "status": "ready", "message": "Files ready. Preview a frame or create your video.",
                "logs": deque(maxlen=80), "preview_ready": False,
            }
        with _jobs_lock:
            return jsonify(_snapshot(_jobs[job_id])), 202
    finally:
        if job_id not in _jobs:
            shutil.rmtree(job_dir, ignore_errors=True)


@app.post("/api/jobs/<job_id>/preview")
def preview_job(job_id: str):
    if not _same_origin():
        abort(403)
    strength = _parse_strength(request.form.get("strength"))
    try:
        seconds = float(request.form.get("seconds", "0"))
    except ValueError:
        seconds = -1
    if strength is None or not 0 <= seconds <= 36000:
        return jsonify(error="Choose a blend from 0 to 100 and a valid frame time."), 400
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            abort(404)
        if job["status"] != "ready":
            return jsonify(error="This job is already processing or finished."), 409
        job["status"] = "previewing"
        job["message"] = "Rendering your preview frame…"
        job["preview_ready"] = False
    job_dir = JOBS_DIR / job_id
    video = next((job_dir / f"video{ext}" for ext in VIDEO_EXTENSIONS if (job_dir / f"video{ext}").is_file()), None)
    face = next((job_dir / f"face{ext}" for ext in IMAGE_EXTENSIONS if (job_dir / f"face{ext}").is_file()), None)
    if video is None or face is None:
        with _jobs_lock:
            job["status"] = "failed"
        return jsonify(error="Uploaded files are no longer available."), 409
    threading.Thread(target=_run_preview, args=(job_id, video, face, strength, seconds), daemon=True).start()
    with _jobs_lock:
        return jsonify(_snapshot(job)), 202


@app.post("/api/jobs/<job_id>/render")
def render_job(job_id: str):
    if not _same_origin():
        abort(403)
    strength = _parse_strength(request.form.get("strength"))
    if strength is None:
        return jsonify(error="Choose an effect blend from 0 to 100."), 400
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            abort(404)
        if job["status"] != "ready":
            return jsonify(error="This job is already processing or finished."), 409
        job["status"] = "queued"
        job["message"] = "Waiting to process…"
    job_dir = JOBS_DIR / job_id
    video = next((job_dir / f"video{ext}" for ext in VIDEO_EXTENSIONS if (job_dir / f"video{ext}").is_file()), None)
    face = next((job_dir / f"face{ext}" for ext in IMAGE_EXTENSIONS if (job_dir / f"face{ext}").is_file()), None)
    if video is None or face is None:
        with _jobs_lock:
            job["status"] = "failed"
        return jsonify(error="Uploaded files are no longer available."), 409
    threading.Thread(target=_run_job, args=(job_id, video, face, job_dir / "result.mp4", strength), daemon=True).start()
    with _jobs_lock:
        return jsonify(_snapshot(job)), 202


@app.get("/api/jobs/<job_id>/preview/<kind>")
def preview_image(job_id: str, kind: str):
    if kind not in {"original", "result"}:
        abort(404)
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None or not job.get("preview_ready"):
            abort(404)
    return send_file(JOBS_DIR / job_id / f"preview-{kind}.png", mimetype="image/png", max_age=0)


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            abort(404)
        return jsonify(_snapshot(job))


@app.get("/api/jobs/<job_id>/download")
def download(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None or job["status"] != "complete":
            abort(404)
    return send_file(JOBS_DIR / job_id / "result.mp4", as_attachment=True,
                     download_name="face-swapped-video.mp4")


@app.get("/api/jobs/<job_id>/result")
def result_video(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None or job["status"] != "complete":
            abort(404)
    return send_file(JOBS_DIR / job_id / "result.mp4", mimetype="video/mp4", conditional=True)


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify(error="Files are too large. The combined upload limit is 2 GB."), 413


if __name__ == "__main__":
    import argparse
    import webbrowser

    parser = argparse.ArgumentParser(description="Local Deep-Live-Cam web interface")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True)
