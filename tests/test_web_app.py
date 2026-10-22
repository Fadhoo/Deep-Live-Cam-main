"""HTTP and subprocess wiring checks for the local video interface."""

import io
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import web_app


HEADERS = {"X-Requested-With": "Deep-Live-Cam"}


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.jobs_patch = patch.object(web_app, "JOBS_DIR", Path(self.temp_dir.name))
        self.jobs_patch.start()
        web_app._jobs.clear()
        self.client = web_app.app.test_client()

    def tearDown(self):
        self.jobs_patch.stop()
        self.temp_dir.cleanup()

    def _post(self, face=None, header=True):
        if face is None:
            image_data = io.BytesIO()
            Image.new("RGB", (2, 2), "white").save(image_data, format="JPEG")
            face = image_data.getvalue()
        return self.client.post(
            "/api/jobs",
            data={
                "video": (io.BytesIO(b"video data"), "clip.mp4"),
                "face": (io.BytesIO(face), "face.jpg"),
            },
            headers=HEADERS if header else {},
        )

    def _create(self):
        with patch.object(web_app, "_valid_video", return_value=True):
            response = self._post()
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json["status"], "ready")
        return response.json["id"]

    def test_preview_uses_selected_frame_and_blend(self):
        job_id = self._create()
        with patch.object(web_app.threading, "Thread") as thread:
            response = self.client.post(
                f"/api/jobs/{job_id}/preview",
                data={"strength": "35", "seconds": "1.5"}, headers=HEADERS,
            )
            self.assertEqual(response.status_code, 202)
            thread.return_value.start.assert_called_once()

        calls = []

        def fake_process(_job_id, command, _timeout):
            calls.append(command)
            if command[0] == "ffmpeg":
                Path(command[-1]).write_bytes(b"original")
            else:
                Path(command[command.index("--output") + 1]).write_bytes(b"swapped")
            return 0, [], False

        job_dir = web_app.JOBS_DIR / job_id
        with patch.object(web_app, "_run_process", side_effect=fake_process):
            web_app._run_preview(job_id, job_dir / "video.mp4", job_dir / "face.jpg", 35, 1.5)
        self.assertEqual(calls[0][calls[0].index("-ss") + 1], "1.5")
        self.assertEqual(calls[1][calls[1].index("--opacity") + 1], "0.35")
        self.assertEqual(calls[1][calls[1].index("--execution-provider") + 1],
                         web_app.WEB_EXECUTION_PROVIDER)
        status = self.client.get(f"/api/jobs/{job_id}").json
        self.assertEqual(status["preview_strength"], 35)
        self.assertEqual(status["preview_seconds"], 1.5)
        image = self.client.get(status["preview_url"])
        self.assertEqual(image.data, b"swapped")
        image.close()

    def test_upload_validation_and_origin(self):
        with patch.object(web_app, "_valid_video", return_value=True):
            self.assertEqual(self._post(header=False).status_code, 403)
            response = self._post(face=b"\xff\xd8\xff\x00")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(list(web_app.JOBS_DIR.iterdir()), [])

    def test_render_wires_opacity_and_serves_result(self):
        job_id = self._create()
        with patch.object(web_app.threading, "Thread") as thread:
            response = self.client.post(
                f"/api/jobs/{job_id}/render", data={"strength": "35"}, headers=HEADERS,
            )
            self.assertEqual(response.status_code, 202)
            thread.return_value.start.assert_called_once()
            self.assertEqual(thread.call_args.kwargs["args"][-1], 35)

        job_dir = web_app.JOBS_DIR / job_id
        output = job_dir / "result.mp4"
        output.write_bytes(b"result")
        with patch.object(web_app, "_run_process", return_value=(0, [], False)) as process:
            web_app._run_job(job_id, job_dir / "video.mp4", job_dir / "face.jpg", output, 35)
            command = process.call_args.args[1]
            self.assertEqual(command[command.index("--opacity") + 1], "0.35")
            self.assertEqual(command[command.index("--execution-provider") + 1],
                             web_app.WEB_EXECUTION_PROVIDER)

        self.assertEqual(self.client.get(f"/api/jobs/{job_id}").json["status"], "complete")
        result = self.client.get(f"/api/jobs/{job_id}/result")
        self.assertEqual(result.data, b"result")
        result.close()
        download = self.client.get(f"/api/jobs/{job_id}/download")
        self.assertEqual(download.data, b"result")
        download.close()

    def test_strength_range_is_checked(self):
        job_id = self._create()
        response = self.client.post(
            f"/api/jobs/{job_id}/render", data={"strength": "101"}, headers=HEADERS,
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}").json["status"], "ready")

    def test_staged_job_recovers_after_restart_and_can_be_deleted(self):
        job_id = self._create()
        web_app._jobs.clear()
        status = self.client.get(f"/api/jobs/{job_id}")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json["status"], "ready")
        deleted = self.client.delete(f"/api/jobs/{job_id}", headers=HEADERS)
        self.assertEqual(deleted.status_code, 204)
        self.assertFalse((web_app.JOBS_DIR / job_id).exists())
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}").status_code, 404)

    def test_completed_result_remains_listed_and_downloadable_after_restart(self):
        job_id = self._create()
        output = web_app.JOBS_DIR / job_id / "result.mp4"
        output.write_bytes(b"result")
        with web_app._jobs_lock:
            job = web_app._jobs[job_id]
            job["status"] = "complete"
            web_app._save_job(job)
        web_app._jobs.clear()
        listed = self.client.get("/api/jobs").json
        self.assertEqual([item["id"] for item in listed], [job_id])
        download = self.client.get(f"/api/jobs/{job_id}/download")
        self.assertEqual(download.data, b"result")
        download.close()

    def test_abandoned_staged_job_expires_but_completed_result_stays(self):
        staged_id = self._create()
        completed_id = self._create()
        with web_app._jobs_lock:
            completed = web_app._jobs[completed_id]
            completed["status"] = "complete"
            web_app._save_job(completed)
        for job_id in (staged_id, completed_id):
            state_file = web_app.JOBS_DIR / job_id / "job.json"
            old = time.time() - web_app.STAGED_JOB_TTL_SECONDS - 5
            os.utime(state_file, (old, old))
        web_app._prune_staged_jobs()
        self.assertFalse((web_app.JOBS_DIR / staged_id).exists())
        self.assertTrue((web_app.JOBS_DIR / completed_id).exists())

    def test_cancel_queued_job_and_timeout_running_job(self):
        job_id = self._create()
        job_dir = web_app.JOBS_DIR / job_id
        with patch.object(web_app.threading, "Thread"):
            response = self.client.post(f"/api/jobs/{job_id}/render",
                                        data={"strength": "50"}, headers=HEADERS)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.client.post(f"/api/jobs/{job_id}/cancel",
                                          headers=HEADERS).status_code, 202)
        with patch.object(web_app, "_run_process") as process:
            web_app._run_job(job_id, job_dir / "video.mp4", job_dir / "face.jpg",
                             job_dir / "result.mp4", 50)
            process.assert_not_called()
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}").json["status"], "failed")

        next_id = self._create()
        next_dir = web_app.JOBS_DIR / next_id
        with patch.object(web_app, "_run_process", return_value=(1, [], True)):
            web_app._run_job(next_id, next_dir / "video.mp4", next_dir / "face.jpg",
                             next_dir / "result.mp4", 50)
        status = self.client.get(f"/api/jobs/{next_id}").json
        self.assertEqual(status["status"], "failed")
        self.assertIn("4 hour limit", status["message"])

    def test_running_process_can_be_cancelled_and_timeout_kills_it(self):
        job_id = self._create()
        with web_app._jobs_lock:
            job = web_app._jobs[job_id]
            job["status"] = "processing"
            web_app._save_job(job)
        result = []
        worker = threading.Thread(target=lambda: result.append(web_app._run_process(
            job_id, [sys.executable, "-c", "import time; time.sleep(30)"], 10)))
        worker.start()
        deadline = time.monotonic() + 5
        while job_id not in web_app._processes and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIn(job_id, web_app._processes)
        response = self.client.post(f"/api/jobs/{job_id}/cancel", headers=HEADERS)
        self.assertEqual(response.status_code, 202)
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertNotEqual(result[0][0], 0)
        self.assertNotIn(job_id, web_app._processes)

        with web_app._jobs_lock:
            web_app._jobs[job_id]["cancel_requested"] = False
        return_code, _lines, timed_out = web_app._run_process(
            job_id, [sys.executable, "-c", "import time; time.sleep(30)"], 1)
        self.assertNotEqual(return_code, 0)
        self.assertTrue(timed_out)
        self.assertNotIn(job_id, web_app._processes)

    def test_missing_reference_face_is_not_reported_as_a_finished_swap(self):
        job_id = self._create()
        job_dir = web_app.JOBS_DIR / job_id
        output = job_dir / "result.mp4"
        output.write_bytes(b"unchanged video")

        def no_face(_job_id, _command, _timeout):
            web_app._jobs[job_id]["source_face_missing"] = True
            return 0, [], False

        with patch.object(web_app, "_run_process", side_effect=no_face):
            web_app._run_job(job_id, job_dir / "video.mp4", job_dir / "face.jpg", output, 50)
        status = self.client.get(f"/api/jobs/{job_id}").json
        self.assertEqual(status["status"], "failed")
        self.assertIn("No face detected", status["message"])
        self.assertFalse(output.exists())
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}/download").status_code, 404)


if __name__ == "__main__":
    unittest.main()
