"""Unit tests using a fake Whisper model (no GPU/model download required)."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

from video_transcriber import TranscriptionCancelled, timestamp, transcribe_file


class Segment:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text
        self.avg_logprob, self.no_speech_prob = -0.1, 0.01
        self.words = None


class FakeWhisper:
    calls = []

    def __init__(self, model, **kwargs):
        self.calls.append((model, kwargs))

    def transcribe(self, path, **kwargs):
        info = types.SimpleNamespace(duration=12.0, language="th", language_probability=1.0)
        return iter([Segment(0, 4, "สวัสดี"), Segment(4, 8, "ทดสอบ"), Segment(8, 12, "เสร็จ")]), info


class CoreTests(unittest.TestCase):
    def test_timestamp_and_srt(self):
        self.assertEqual(timestamp(3661.234), "01:01:01.234")
        self.assertEqual(timestamp(1.2, subtitle=True), "00:00:01,200")

    def test_transcribe_cpu_and_files(self):
        events = []
        with tempfile.TemporaryDirectory() as root:
            video = Path(root) / "venus.mp4"
            video.write_bytes(b"fake media bytes")
            fake_module = types.SimpleNamespace(WhisperModel=FakeWhisper)
            with patch.dict(sys.modules, {"faster_whisper": fake_module}):
                result = transcribe_file(
                    video, model_name="small", language="th", device="cpu",
                    output_dir=Path(root) / "output", output_format="all",
                    on_progress=events.append,
                )
            self.assertEqual(result["device"], "cpu")
            self.assertEqual(result["segments"], 3)
            self.assertEqual(set(result["files"]), {"txt", "json", "srt"})
            self.assertEqual(events[-1]["progress"], 100)
            content = Path(result["files"]["txt"]).read_text(encoding="utf-8")
            data = json.loads(Path(result["files"]["json"]).read_text(encoding="utf-8"))
            srt = Path(result["files"]["srt"]).read_text(encoding="utf-8")
            self.assertIn("สวัสดี", content)
            self.assertIn("ทดสอบ", data["full_text"])
            self.assertIn("00:00:00,000 --> 00:00:04,000", srt)

    def test_cancel_between_segments(self):
        event = Event()

        def cancel_after_first(data):
            if data.get("segment_count") == 1:
                event.set()

        with tempfile.TemporaryDirectory() as root:
            video = Path(root) / "video.mp4"
            video.write_bytes(b"fake media bytes")
            with patch.dict(sys.modules, {"faster_whisper": types.SimpleNamespace(WhisperModel=FakeWhisper)}):
                with self.assertRaises(TranscriptionCancelled):
                    transcribe_file(video, device="cpu", cancel_event=event,
                                    on_progress=cancel_after_first)

    def test_reject_bad_options(self):
        with tempfile.TemporaryDirectory() as root:
            video = Path(root) / "video.mp4"
            video.touch()
            with self.assertRaises(ValueError):
                transcribe_file(video, beam_size=0)
            with self.assertRaises(ValueError):
                transcribe_file(video, model_name="not-real")


if __name__ == "__main__":
    unittest.main()
