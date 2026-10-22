"""Local browser interface for the existing Deep-Live-Cam video pipeline."""

from __future__ import annotations

import json
import os
import re
import secrets
import signal
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge


ROOT = Path(__file__).resolve().parent
JOBS_DIR = ROOT / "web_jobs"
VIDEO_EXTENSIONS = {".mp4", ".mkv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
PREVIEW_TIMEOUT_SECONDS = 15 * 60
RENDER_TIMEOUT_SECONDS = 4 * 60 * 60
STAGED_JOB_TTL_SECONDS = 7 * 24 * 60 * 60
JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{24}\Z")
# A GPU-enabled runtime can advertise CUDA on machines without NVIDIA hardware.
WEB_EXECUTION_PROVIDER = os.environ.get("DLC_WEB_EXECUTION_PROVIDER", "cpu").strip().lower()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_processing_lock = threading.Lock()
_processes: dict[str, subprocess.Popen] = {}


def _save_job(job: dict) -> None:
    """Atomically save the state needed to reopen a job after a restart."""
    job_dir = JOBS_DIR / job["id"]
    state = {key: value for key, value in job.items()
             if key not in {"cancel_requested", "source_face_missing"}}
    state["logs"] = list(job["logs"])
    temporary = job_dir / "job.json.tmp"
    temporary.write_text(json.dumps(state), encoding="utf-8")
    temporary.replace(job_dir / "job.json")


def _get_job(job_id: str) -> dict | None:
    """Call while holding _jobs_lock."""
    if not JOB_ID_PATTERN.fullmatch(job_id):
        return None
    if job_id in _jobs:
        return _jobs[job_id]
    job_dir = JOBS_DIR / job_id
    try:
        state = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        if state.get("id") != job_id or state.get("status") not in {
            "ready", "queued", "previewing", "processing", "complete", "failed"
        }:
            return None
        state["logs"] = deque(state.get("logs", []), maxlen=80)
        state["cancel_requested"] = False
        if state["status"] in {"queued", "previewing", "processing"}:
            state["status"] = "failed"
            state["message"] = "Processing was interrupted when the server stopped. Start a new job."
            _save_job(state)
        if state["status"] == "complete" and not (job_dir / "result.mp4").is_file():
            state["status"] = "failed"
            state["message"] = "The result file is no longer available."
            _save_job(state)
        if state.get("preview_ready") and not all(
            (job_dir / f"preview-{kind}.png").is_file() for kind in ("original", "result")
        ):
            state["preview_ready"] = False
            _save_job(state)
        _jobs[job_id] = state
        return state
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _prune_staged_jobs() -> None:
    """Remove abandoned uploads while retaining completed results."""
    if not JOBS_DIR.is_dir():
        return
    cutoff = time.time() - STAGED_JOB_TTL_SECONDS
    for job_dir in JOBS_DIR.iterdir():
        if not job_dir.is_dir() or not JOB_ID_PATTERN.fullmatch(job_dir.name):
            continue
        state_file = job_dir / "job.json"
        try:
            if (state_file if state_file.exists() else job_dir).stat().st_mtime >= cutoff:
                continue
            with _jobs_lock:
                job = _get_job(job_dir.name)
                if job and job["status"] in {"queued", "previewing", "processing", "complete"}:
                    continue
                _jobs.pop(job_dir.name, None)
            shutil.rmtree(job_dir)
        except OSError:
            continue


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=10, check=False)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.TimeoutExpired):
        pass
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass


def _run_process(job_id: str, command: list[str], timeout: int) -> tuple[int, list[str], bool]:
    """Run one model command with a killable process tree and an absolute limit."""
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, errors="replace",
                               bufsize=1, **options)
    timed_out = threading.Event()

    def expire() -> None:
        timed_out.set()
        _terminate_process(process)

    timer = threading.Timer(timeout, expire)
    with _jobs_lock:
        _processes[job_id] = process
        job = _jobs[job_id]
        job["source_face_missing"] = False
        cancelled = job.get("cancel_requested", False)
    timer.start()
    if cancelled:
        _terminate_process(process)
    lines: deque[str] = deque(maxlen=20)
    try:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if line:
                lines.append(line)
                with _jobs_lock:
                    job = _jobs[job_id]
                    job["logs"].append(line)
                    if "No face detected in source image" in line:
                        job["source_face_missing"] = True
                    if line.startswith("[DLC.CORE]"):
                        job["message"] = line.removeprefix("[DLC.CORE]").strip()
        return process.wait(), list(lines), timed_out.is_set()
    finally:
        timer.cancel()
        with _jobs_lock:
            _processes.pop(job_id, None)
        if process.stdout is not None:
            process.stdout.close()


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
            with _jobs_lock:
                if _jobs[job_id].get("cancel_requested"):
                    raise InterruptedError("Preview cancelled.")
            original.unlink(missing_ok=True)
            result.unlink(missing_ok=True)
            return_code, _lines, timed_out = _run_process(job_id, [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(seconds),
                 "-i", str(video), "-frames:v", "1", "-y", str(original)], 90)
            with _jobs_lock:
                cancelled = _jobs[job_id].get("cancel_requested", False)
            if cancelled:
                raise InterruptedError("Preview cancelled.")
            if timed_out:
                raise TimeoutError("Frame extraction exceeded the 90 second limit.")
            if return_code or not original.is_file():
                raise RuntimeError("Could not extract that frame. Choose an earlier time in the video.")
            if strength == 0:
                shutil.copyfile(original, result)
            else:
                command = [
                    sys.executable, "-u", str(ROOT / "run.py"),
                    "--source", str(face), "--target", str(original),
                    "--output", str(result), "--frame-processor", "face_swapper",
                    "--opacity", str(strength / 100),
                    "--execution-provider", WEB_EXECUTION_PROVIDER,
                ]
                return_code, _lines, timed_out = _run_process(job_id, command, PREVIEW_TIMEOUT_SECONDS)
                with _jobs_lock:
                    cancelled = _jobs[job_id].get("cancel_requested", False)
                if cancelled:
                    raise InterruptedError("Preview cancelled.")
                if timed_out:
                    raise TimeoutError("Preview exceeded the 15 minute limit.")
                with _jobs_lock:
                    source_face_missing = _jobs[job_id].get("source_face_missing", False)
                if source_face_missing:
                    raise ValueError("No face detected in the reference image. Choose a clear face photo.")
                if return_code or not result.is_file() or not result.stat().st_size:
                    raise RuntimeError("Could not render a preview. See the processing details.")
            with _jobs_lock:
                job = _jobs[job_id]
                if job.get("cancel_requested"):
                    raise InterruptedError("Preview cancelled.")
                job["status"] = "ready"
                job["message"] = "Preview ready. Compare the frame, then create your video."
                job["preview_ready"] = True
                job["preview_strength"] = strength
                job["preview_seconds"] = seconds
                _save_job(job)
        except Exception as exc:
            with _jobs_lock:
                job = _jobs[job_id]
                job["status"] = "ready"
                job["message"] = (
                    "Preview cancelled. You can try again."
                    if isinstance(exc, InterruptedError)
                    else str(exc) if job.get("source_face_missing")
                    else f"Preview failed: {exc} You can still create the video."
                )
                job["preview_ready"] = False
                job["cancel_requested"] = False
                _save_job(job)


def _run_job(job_id: str, video: Path, face: Path, output: Path, strength: int) -> None:
    with _processing_lock:
        with _jobs_lock:
            job = _jobs[job_id]
            if job.get("cancel_requested"):
                job["status"] = "failed"
                job["message"] = "Processing cancelled."
                _save_job(job)
                return
            job["status"] = "processing"
            job["message"] = "Processing video…"
            _save_job(job)

        command = [
            sys.executable, "-u", str(ROOT / "run.py"),
            "--source", str(face), "--target", str(video),
            "--output", str(output), "--keep-fps",
            "--frame-processor", "face_swapper",
            "--opacity", str(strength / 100),
            "--execution-provider", WEB_EXECUTION_PROVIDER,
        ]
        try:
            return_code, _lines, timed_out = _run_process(job_id, command, RENDER_TIMEOUT_SECONDS)
            with _jobs_lock:
                if job.get("cancel_requested"):
                    job["status"] = "failed"
                    job["message"] = "Processing cancelled."
                elif timed_out:
                    job["status"] = "failed"
                    job["message"] = "Processing exceeded the 4 hour limit."
                elif strength > 0 and job.get("source_face_missing"):
                    job["status"] = "failed"
                    job["message"] = "No face detected in the reference image. Choose a clear face photo."
                    output.unlink(missing_ok=True)
                elif return_code == 0 and output.is_file() and output.stat().st_size:
                    job["status"] = "complete"
                    job["message"] = "Your video is ready."
                else:
                    job["status"] = "failed"
                    job["message"] = "Processing failed. See the details below."
                    if return_code != 0:
                        job["logs"].append(f"Process exited with code {return_code}.")
                _save_job(job)
        except Exception as exc:
            with _jobs_lock:
                job["status"] = "failed"
                job["message"] = f"Could not start processing: {exc}"
                _save_job(job)
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
    _prune_staged_jobs()
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
            job = {
                "id": job_id, "status": "ready", "message": "Files ready. Preview a frame or create your video.",
                "logs": deque(maxlen=80), "preview_ready": False, "cancel_requested": False,
                "created_at": time.time(),
            }
            _save_job(job)
            _jobs[job_id] = job
            return jsonify(_snapshot(job)), 202
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
        job = _get_job(job_id)
        if job is None:
            abort(404)
        if job["status"] != "ready":
            return jsonify(error="This job is already processing or finished."), 409
        job["status"] = "previewing"
        job["message"] = "Rendering your preview frame…"
        job["preview_ready"] = False
        job["cancel_requested"] = False
        _save_job(job)
    job_dir = JOBS_DIR / job_id
    video = next((job_dir / f"video{ext}" for ext in VIDEO_EXTENSIONS if (job_dir / f"video{ext}").is_file()), None)
    face = next((job_dir / f"face{ext}" for ext in IMAGE_EXTENSIONS if (job_dir / f"face{ext}").is_file()), None)
    if video is None or face is None:
        with _jobs_lock:
            job["status"] = "failed"
            _save_job(job)
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
        job = _get_job(job_id)
        if job is None:
            abort(404)
        if job["status"] != "ready":
            return jsonify(error="This job is already processing or finished."), 409
        job["status"] = "queued"
        job["message"] = "Waiting to process…"
        job["cancel_requested"] = False
        _save_job(job)
    job_dir = JOBS_DIR / job_id
    video = next((job_dir / f"video{ext}" for ext in VIDEO_EXTENSIONS if (job_dir / f"video{ext}").is_file()), None)
    face = next((job_dir / f"face{ext}" for ext in IMAGE_EXTENSIONS if (job_dir / f"face{ext}").is_file()), None)
    if video is None or face is None:
        with _jobs_lock:
            job["status"] = "failed"
            _save_job(job)
        return jsonify(error="Uploaded files are no longer available."), 409
    threading.Thread(target=_run_job, args=(job_id, video, face, job_dir / "result.mp4", strength), daemon=True).start()
    with _jobs_lock:
        return jsonify(_snapshot(job)), 202


@app.get("/api/jobs/<job_id>/preview/<kind>")
def preview_image(job_id: str, kind: str):
    if kind not in {"original", "result"}:
        abort(404)
    with _jobs_lock:
        job = _get_job(job_id)
        if job is None or not job.get("preview_ready"):
            abort(404)
    return send_file(JOBS_DIR / job_id / f"preview-{kind}.png", mimetype="image/png", max_age=0)


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with _jobs_lock:
        job = _get_job(job_id)
        if job is None:
            abort(404)
        return jsonify(_snapshot(job))


@app.get("/api/jobs")
def list_results():
    _prune_staged_jobs()
    results = []
    with _jobs_lock:
        for job_dir in JOBS_DIR.iterdir() if JOBS_DIR.is_dir() else ():
            if not job_dir.is_dir():
                continue
            job = _get_job(job_dir.name)
            if job and job["status"] == "complete":
                results.append({"id": job["id"], "created_at": job.get("created_at", 0),
                                "download_url": _snapshot(job)["download_url"]})
    return jsonify(sorted(results, key=lambda item: item["created_at"], reverse=True))


@app.get("/api/jobs/<job_id>/download")
def download(job_id: str):
    with _jobs_lock:
        job = _get_job(job_id)
        if job is None or job["status"] != "complete":
            abort(404)
    return send_file(JOBS_DIR / job_id / "result.mp4", as_attachment=True,
                     download_name="face-swapped-video.mp4")


@app.get("/api/jobs/<job_id>/result")
def result_video(job_id: str):
    with _jobs_lock:
        job = _get_job(job_id)
        if job is None or job["status"] != "complete":
            abort(404)
    return send_file(JOBS_DIR / job_id / "result.mp4", mimetype="video/mp4", conditional=True)


@app.post("/api/jobs/<job_id>/cancel")
def cancel_job(job_id: str):
    if not _same_origin():
        abort(403)
    with _jobs_lock:
        job = _get_job(job_id)
        if job is None:
            abort(404)
        if job["status"] not in {"queued", "previewing", "processing"}:
            return jsonify(error="This job is not processing."), 409
        job["cancel_requested"] = True
        job["message"] = "Cancelling processing…"
        _save_job(job)
        process = _processes.get(job_id)
        snapshot = _snapshot(job)
    if process is not None:
        _terminate_process(process)
    return jsonify(snapshot), 202


@app.delete("/api/jobs/<job_id>")
def delete_job(job_id: str):
    if not _same_origin():
        abort(403)
    with _jobs_lock:
        job = _get_job(job_id)
        if job is None:
            abort(404)
        if job["status"] in {"queued", "previewing", "processing"}:
            return jsonify(error="Cancel processing before removing this job."), 409
        shutil.rmtree(JOBS_DIR / job_id)
        _jobs.pop(job_id, None)
    return "", 204


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
    _prune_staged_jobs()
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True)
