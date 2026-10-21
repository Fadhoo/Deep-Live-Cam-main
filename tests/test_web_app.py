"""HTTP and subprocess wiring checks for the local video interface."""

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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

        def fake_run(command, **_kwargs):
            calls.append(command)
            if command[0] == "ffmpeg":
                Path(command[-1]).write_bytes(b"original")
            else:
                Path(command[command.index("--output") + 1]).write_bytes(b"swapped")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        job_dir = web_app.JOBS_DIR / job_id
        with patch.object(web_app.subprocess, "run", side_effect=fake_run):
            web_app._run_preview(job_id, job_dir / "video.mp4", job_dir / "face.jpg", 35, 1.5)
        self.assertEqual(calls[0][calls[0].index("-ss") + 1], "1.5")
        self.assertEqual(calls[1][calls[1].index("--opacity") + 1], "0.35")
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
        fake_process = SimpleNamespace(stdout=iter(["[DLC.CORE] Processing...\n"]), wait=lambda: 0)
        with patch.object(web_app.subprocess, "Popen", return_value=fake_process) as popen:
            web_app._run_job(job_id, job_dir / "video.mp4", job_dir / "face.jpg", output, 35)
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--opacity") + 1], "0.35")

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


if __name__ == "__main__":
    unittest.main()
