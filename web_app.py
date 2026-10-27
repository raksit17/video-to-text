#!/usr/bin/env python3
"""Local-only Web UI and API for large media uploads and transcription jobs."""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from video_transcriber import MODELS, TranscriptionCancelled, gpu_status, transcribe_file

ROOT = Path(__file__).resolve().parent
WORK = ROOT / ".work"
MAX_BYTES = 10 * 1024**3  # 10 GiB; request body is streamed to disk, not RAM.
EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".mpg", ".mpeg", ".m4v",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
}
JOBS: dict[str, dict] = {}
LOCK = threading.RLock()
ACTIVE = {"uploading", "queued", "running", "cancelling"}


def public_job(job: dict) -> dict:
    with LOCK:
        fields = (
            "id", "filename", "status", "stage", "progress", "message",
            "device", "duration_seconds", "processed_seconds", "eta_seconds",
            "segment_count", "last_text", "error",
        )
        response = {field: job.get(field) for field in fields}
        if job.get("started_at") is not None:
            response["elapsed_seconds"] = round(time.monotonic() - job["started_at"], 1)
        else:
            response["elapsed_seconds"] = 0
        if job["status"] == "completed":
            response["downloads"] = {
                kind: f"/api/jobs/{job['id']}/download/{kind}"
                for kind in job.get("files", {})
            }
        return response


def worker(job: dict) -> None:
    with LOCK:
        if job["cancel"].is_set():
            job.update(status="cancelled", stage="cancelled", message="ยกเลิกแล้ว")
            job["upload_path"].unlink(missing_ok=True)
            return
        job.update(status="running", stage="loading_model",
                   started_at=time.monotonic(), message="กำลังเริ่มต้น")

    def update(event: dict) -> None:
        with LOCK:
            job.update(event)
            # 'cancelling' is preserved until the worker actually stops.

    try:
        result = transcribe_file(
            job["upload_path"], model_name=job["options"]["model"],
            language=job["options"]["language"], device=job["options"]["device"],
            beam_size=job["options"]["beam_size"],
            vad_filter=job["options"]["vad_filter"],
            cpu_threads=job["options"]["cpu_threads"],
            word_timestamps=job["options"]["word_timestamps"],
            output_dir=job["directory"] / "outputs", output_format="all",
            on_progress=update, cancel_event=job["cancel"],
        )
        with LOCK:
            job.update(status="completed", stage="completed", progress=100,
                       eta_seconds=0, device=result["device"],
                       segment_count=result["segments"], files=result["files"],
                       message="ถอดเสียงสำเร็จ ดาวน์โหลดผลลัพธ์ได้แล้ว")
    except TranscriptionCancelled:
        with LOCK:
            job.update(status="cancelled", stage="cancelled", eta_seconds=None,
                       message="ยกเลิกแล้ว")
    except Exception as exc:
        with LOCK:
            job.update(status="failed", stage="failed", error=str(exc),
                       eta_seconds=None, message="ถอดเสียงไม่สำเร็จ")
    finally:
        # Retain TXT/JSON/SRT downloads but discard the uploaded multi-GB video.
        job["upload_path"].unlink(missing_ok=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "VideoToText/2.0"

    def json_response(self, status: int, data: dict) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        route = urlsplit(self.path).path
        if route in ("/", "/index.html"):
            content = (ROOT / "web" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        if route == "/api/system":
            self.json_response(200, {
                "gpu": gpu_status(),
                "python": sys.version.split()[0],
                "max_upload_gb": 10,
            })
            return

        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})", route)
        if match:
            with LOCK:
                job = JOBS.get(match.group(1))
                data = public_job(job) if job else None
            self.json_response(200, data) if data else self.json_response(404, {"error": "ไม่พบงาน"})
            return

        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/download/(txt|json|srt)", route)
        if match:
            with LOCK:
                job = JOBS.get(match.group(1))
                path = (
                    Path(job["files"][match.group(2)])
                    if job and job["status"] == "completed"
                    and match.group(2) in job.get("files", {}) else None
                )
            if path is None or not path.is_file():
                self.json_response(404, {"error": "ไม่พบไฟล์ผลลัพธ์"})
                return
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json; charset=utf-8" if path.suffix == ".json"
                else "text/plain; charset=utf-8",
            )
            self.send_header("Content-Disposition",
                             f'attachment; filename="transcript.{match.group(2)}"')
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            with path.open("rb") as source:
                while block := source.read(1024 * 1024):
                    self.wfile.write(block)
            return

        self.json_response(404, {"error": "ไม่พบ API"})

    def do_POST(self) -> None:
        route = urlsplit(self.path)
        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/cancel", route.path)
        if match:
            with LOCK:
                job = JOBS.get(match.group(1))
                if not job:
                    self.json_response(404, {"error": "ไม่พบงาน"})
                    return
                if job["status"] in ACTIVE:
                    job["cancel"].set()
                    job.update(status="cancelling", message="กำลังยกเลิก...")
                data = public_job(job)
            self.json_response(200, data)
            return

        if route.path != "/api/upload":
            self.json_response(404, {"error": "ไม่พบ API"})
            return

        with LOCK:
            if any(job["status"] in ACTIVE for job in JOBS.values()):
                self.json_response(409, {"error": "มีงานอื่นกำลังทำงานอยู่ กรุณารอหรือยกเลิกก่อน"})
                return

        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if size <= 0 or size > MAX_BYTES:
            self.json_response(413, {"error": "ไฟล์ต้องมีขนาดมากกว่า 0 และไม่เกิน 10 GiB"})
            return

        raw_name = unquote(self.headers.get("X-File-Name", "media.mp4"))
        filename = raw_name.replace("\\", "/").split("/")[-1][:200]
        if Path(filename).suffix.lower() not in EXTENSIONS:
            self.json_response(400, {"error": "ชนิดไฟล์ไม่รองรับ"})
            return

        query = parse_qs(route.query)
        value = lambda key, default: query.get(key, [default])[0]
        try:
            model = value("model", "small")
            language = value("language", "th").lower()
            device = value("device", "auto")
            beam = int(value("beam_size", "1"))
            threads = int(value("cpu_threads", "0"))
            if model not in MODELS or device not in ("auto", "cpu", "cuda"):
                raise ValueError("โมเดลหรืออุปกรณ์ไม่ถูกต้อง")
            if not (1 <= beam <= 10 and 0 <= threads <= 128):
                raise ValueError("beam_size หรือ cpu_threads ไม่ถูกต้อง")
            if not re.fullmatch(r"auto|[a-z]{2,3}(?:-[a-z]{2,3})?", language):
                raise ValueError("รหัสภาษาไม่ถูกต้อง")
        except ValueError as exc:
            self.json_response(400, {"error": str(exc)})
            return

        identifier = uuid.uuid4().hex
        directory = WORK / identifier
        directory.mkdir(parents=True, exist_ok=True)
        video = directory / filename
        job = {
            "id": identifier, "filename": filename,
            "status": "uploading", "stage": "uploading", "progress": 0,
            "message": "กำลังอัปโหลดไฟล์", "device": None,
            "duration_seconds": None, "processed_seconds": None,
            "eta_seconds": None, "segment_count": 0, "last_text": "",
            "error": None, "started_at": None, "files": {},
            "cancel": threading.Event(), "directory": directory,
            "upload_path": video,
            "options": {
                "model": model, "language": language, "device": device,
                "beam_size": beam, "cpu_threads": threads,
                "vad_filter": value("vad", "true") == "true",
                "word_timestamps": value("words", "false") == "true",
            },
        }
        with LOCK:
            JOBS[identifier] = job

        try:
            remaining = size
            with video.open("wb") as output:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ConnectionError("การอัปโหลดถูกตัดกลางทาง")
                    output.write(chunk)
                    remaining -= len(chunk)
            with LOCK:
                job.update(status="queued", stage="queued", message="เตรียมถอดเสียง")
            threading.Thread(target=worker, args=(job,), daemon=True).start()
            self.json_response(202, {"id": identifier, "job": public_job(job)})
        except (ConnectionError, OSError) as exc:
            with LOCK:
                job.update(status="failed", stage="failed", error=str(exc))
            video.unlink(missing_ok=True)
            try:
                self.json_response(500, {"error": "อัปโหลดไม่สำเร็จ: " + str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass

    def log_message(self, format: str, *args: object) -> None:
        print("[%s] %s" % (self.log_date_time_string(), format % args))


def main() -> None:
    parser = argparse.ArgumentParser(description="Local transcription web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"เปิดเว็บ: http://{args.host}:{args.port}")
    print("รองรับไฟล์ใหญ่, CPU, GPU, progress และ ETA | หยุดด้วย Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nปิดเซิร์ฟเวอร์แล้ว")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
