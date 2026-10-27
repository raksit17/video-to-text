"""Local Web API smoke tests with mocked inference (no model download)."""

import json
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import web_app


def fake_transcribe_file(video, **options):
    options["on_progress"]({
        "stage": "transcribing", "progress": 50, "device": "cpu",
        "duration_seconds": 4, "processed_seconds": 2,
        "segment_count": 1, "last_text": "hello",
    })
    destination = options["output_dir"]
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for kind in ("txt", "json", "srt"):
        path = destination / ("sample.transcript." + kind)
        path.write_text("hello", encoding="utf-8")
        outputs[kind] = str(path)
    return {"device": "cpu", "segments": 1, "files": outputs}


class WebApiTests(unittest.TestCase):
    def test_upload_poll_and_download(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(web_app, "WORK", Path(folder)), \
                 patch.object(web_app, "transcribe_file", fake_transcribe_file), \
                 patch.dict(web_app.JOBS, {}, clear=True):
                server = web_app.ThreadingHTTPServer(("127.0.0.1", 0), web_app.Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base = "http://127.0.0.1:" + str(server.server_port)
                try:
                    request = urllib.request.Request(
                        base + "/api/upload?model=small&device=cpu&language=th",
                        data=b"fake video",
                        headers={
                            "Content-Type": "application/octet-stream",
                            "X-File-Name": "sample.mp4",
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=5) as response:
                        self.assertEqual(response.status, 202)
                        task = json.load(response)
                    job_id = task["id"]
                    self.assertEqual(len(job_id), 32)

                    for _ in range(80):
                        with urllib.request.urlopen(
                            base + "/api/jobs/" + job_id, timeout=5
                        ) as response:
                            job = json.load(response)
                        if job["status"] == "completed":
                            break
                        time.sleep(0.025)
                    self.assertEqual(job["status"], "completed")
                    self.assertEqual(job["progress"], 100)
                    self.assertEqual(job["device"], "cpu")
                    self.assertEqual(set(job["downloads"]), {"txt", "json", "srt"})
                    with urllib.request.urlopen(
                        base + job["downloads"]["txt"], timeout=5
                    ) as response:
                        self.assertEqual(response.read().decode(), "hello")

                    self.assertFalse((Path(folder) / job_id / "sample.mp4").exists())
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
