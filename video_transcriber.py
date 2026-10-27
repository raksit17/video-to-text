#!/usr/bin/env python3
"""Local video/audio transcription with progress, ETA and safe CPU/GPU selection."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable

MODELS = ("tiny", "base", "small", "medium", "large-v3", "turbo")
FORMATS = ("both", "txt", "json", "srt", "all")
ProgressCallback = Callable[[dict[str, Any]], None]


class TranscriptionCancelled(Exception):
    """Raised when the user cancels a transcription."""


def timestamp(seconds: float, *, subtitle: bool = False) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    separator = "," if subtitle else "."
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def gpu_status() -> dict[str, Any]:
    """Best-effort CUDA diagnostics; no PyTorch installation required."""
    result: dict[str, Any] = {
        "available": False,
        "name": None,
        "device_count": 0,
        "reason": "",
    }
    try:
        command = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=4, check=False,
        )
        if command.returncode == 0 and command.stdout.strip():
            result["name"] = command.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass

    try:
        import ctranslate2
        count = int(ctranslate2.get_cuda_device_count())
        result["device_count"] = count
        if count == 0:
            result["reason"] = "ไม่พบ NVIDIA GPU ที่ CUDA ใช้งานได้ หรือไดรเวอร์ไม่พร้อม"
            return result
    except Exception as exc:
        result["reason"] = f"ตรวจสอบ CTranslate2/CUDA ไม่สำเร็จ: {exc}"
        return result

    # CUDA can report a device even though cuBLAS/cuDNN DLLs are absent.
    required = (
        ("cublas64_12.dll", "cudnn64_9.dll")
        if os.name == "nt" else ("libcublas.so.12", "libcudnn.so.9")
    )
    for library in required:
        try:
            if os.name == "nt":
                ctypes.WinDLL(library)
            else:
                ctypes.CDLL(library)
        except OSError:
            result["reason"] = (
                f"โหลด {library} ไม่ได้; ติดตั้ง CUDA 12 cuBLAS และ cuDNN 9 "
                "ให้ตรงกับ CTranslate2 และตรวจสอบ PATH"
            )
            return result
    result["available"] = True
    result["reason"] = "CUDA พร้อมใช้งาน (ผ่านการตรวจสอบเบื้องต้น)"
    return result


def _segment_data(segment: Any, word_timestamps: bool) -> dict[str, Any]:
    item: dict[str, Any] = {
        "start": round(segment.start, 3),
        "end": round(segment.end, 3),
        "text": segment.text.strip(),
        "avg_logprob": round(segment.avg_logprob, 4),
        "no_speech_prob": round(segment.no_speech_prob, 4),
    }
    if word_timestamps and segment.words:
        item["words"] = [
            {
                "start": round(word.start, 3),
                "end": round(word.end, 3),
                "text": word.word,
                "probability": round(word.probability, 4),
            }
            for word in segment.words
        ]
    return item


def transcribe_file(
    video: Path,
    *,
    model_name: str = "small",
    language: str = "th",
    device: str = "auto",
    compute_type: str | None = None,
    beam_size: int = 1,
    vad_filter: bool = True,
    word_timestamps: bool = False,
    cpu_threads: int = 0,
    output_dir: Path | None = None,
    output_format: str = "both",
    on_progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> dict[str, Any]:
    """Transcribe locally, report per-segment progress, and return saved paths.

    ETA is derived from audio time processed / elapsed processing time. It is a
    rough estimate only, especially during silence or the first few segments.
    """
    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise ValueError(f"ไม่พบไฟล์: {video}")
    if model_name not in MODELS:
        raise ValueError(f"โมเดลไม่ถูกต้อง: {model_name}")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device ต้องเป็น auto, cpu หรือ cuda")
    if output_format not in FORMATS:
        raise ValueError("รูปแบบไฟล์ไม่ถูกต้อง")
    if not (1 <= beam_size <= 10) or not (0 <= cpu_threads <= 128):
        raise ValueError("beam_size ต้องเป็น 1-10 และ cpu_threads ต้องเป็น 0-128")
    if not re.fullmatch(r"auto|[a-z]{2,3}(?:-[a-z]{2,3})?", language):
        raise ValueError("รหัสภาษาไม่ถูกต้อง")

    cancelled = cancel_event or Event()

    def emit(**values: Any) -> None:
        if on_progress:
            on_progress(values)

    def check_cancel() -> None:
        if cancelled.is_set():
            raise TranscriptionCancelled("ยกเลิกโดยผู้ใช้")

    check_cancel()
    diagnostics = gpu_status() if device != "cpu" else {"available": False, "reason": "ใช้ CPU"}
    selected = "cuda" if device != "cpu" and diagnostics["available"] else "cpu"
    if device == "cuda" and selected != "cuda":
        raise RuntimeError("ไม่สามารถใช้ GPU: " + diagnostics["reason"])

    def load_model(target: str):
        from faster_whisper import WhisperModel
        precision = compute_type or ("float16" if target == "cuda" else "int8")
        return WhisperModel(
            model_name, device=target, compute_type=precision,
            cpu_threads=cpu_threads,
        )

    emit(stage="loading_model", device=selected, progress=0,
         message="กำลังโหลดโมเดล (ครั้งแรกอาจต้องดาวน์โหลด)")
    check_cancel()
    try:
        model = load_model(selected)
    except Exception as exc:
        if device == "auto" and selected == "cuda":
            emit(stage="loading_model", device="cpu", progress=0,
                 message=f"GPU เริ่มทำงานไม่ได้ ({exc}); กำลังเปลี่ยนเป็น CPU")
            selected = "cpu"
            model = load_model("cpu")
        else:
            raise RuntimeError(
                f"โหลดโมเดลไม่สำเร็จ ({selected}): {exc}"
                + ("; ตรวจสอบ CUDA 12/cuDNN 9 หรือใช้ --device cpu" if selected == "cuda" else "")
            ) from exc

    check_cancel()
    decode_language = None if language == "auto" else language
    segments: list[dict[str, Any]] = []
    processing_started = time.monotonic()

    def process(use_vad: bool) -> tuple[Any, list[dict[str, Any]]]:
        nonlocal processing_started
        iterator, info = model.transcribe(
            str(video), language=decode_language,
            beam_size=beam_size, vad_filter=use_vad,
            word_timestamps=word_timestamps,
        )
        duration = max(0.0, float(info.duration or 0))
        processing_started = time.monotonic()
        emit(stage="transcribing", device=selected, progress=0,
             duration_seconds=duration, eta_seconds=None,
             message="กำลังถอดเสียง...")
        output: list[dict[str, Any]] = []
        for segment in iterator:
            check_cancel()
            item = _segment_data(segment, word_timestamps)
            output.append(item)
            elapsed = time.monotonic() - processing_started
            fraction = min(0.995, max(0.0, item["end"] / duration)) if duration else 0
            # Avoid showing a misleading ETA during model warmup or very early segments.
            eta = (
                round(elapsed * (1 - fraction) / fraction)
                if len(output) >= 3 and elapsed >= 5 and fraction >= 0.02 else None
            )
            emit(
                stage="transcribing", device=selected,
                progress=round(fraction * 100, 1),
                duration_seconds=duration, processed_seconds=item["end"],
                eta_seconds=eta, segment_count=len(output),
                last_text=item["text"], message="กำลังถอดเสียง...",
            )
        return info, output

    info, segments = process(vad_filter)
    if not segments and vad_filter:
        emit(stage="transcribing", device=selected, progress=0,
             message="ไม่พบคำพูดจาก VAD กำลังลองใหม่โดยไม่กรองเสียงเงียบ")
        info, segments = process(False)
    check_cancel()
    if not segments:
        raise RuntimeError("ไม่พบเสียงพูดในไฟล์ กรุณาตรวจสอบแทร็กเสียง")

    emit(stage="saving", device=selected, progress=99.5,
         message="กำลังบันทึกผลลัพธ์")
    target = Path(output_dir or video.parent).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    stem = video.stem
    paths: dict[str, str] = {}
    if output_format in ("both", "txt", "all"):
        lines = [
            f"[{timestamp(s['start'])} - {timestamp(s['end'])}] {s['text']}"
            for s in segments
        ]
        path = target / f"{stem}.transcript.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths["txt"] = str(path)
    if output_format in ("both", "json", "all"):
        path = target / f"{stem}.transcript.json"
        data = {
            "source_file": video.name, "model": model_name,
            "device": selected, "language": info.language,
            "language_probability": round(float(info.language_probability or 0), 4),
            "duration_seconds": round(float(info.duration or 0), 3),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "full_text": " ".join(s["text"] for s in segments),
            "segments": segments,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths["json"] = str(path)
    if output_format in ("srt", "all"):
        path = target / f"{stem}.transcript.srt"
        path.write_text(
            "\n".join(
                f"{index}\n{timestamp(s['start'], subtitle=True)} --> "
                f"{timestamp(s['end'], subtitle=True)}\n{s['text']}\n"
                for index, s in enumerate(segments, 1)
            ),
            encoding="utf-8",
        )
        paths["srt"] = str(path)
    emit(stage="completed", device=selected, progress=100,
         eta_seconds=0, message="เสร็จเรียบร้อย")
    return {
        "files": paths, "device": selected, "language": info.language,
        "segments": len(segments), "duration_seconds": float(info.duration or 0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ถอดเสียงจากวิดีโอ/เสียงเป็น TXT, JSON หรือ SRT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", nargs="?", type=Path, help="ไฟล์วิดีโอหรือไฟล์เสียง")
    parser.add_argument("--model", choices=MODELS, default="small")
    parser.add_argument("--language", default="auto", help="th, en หรือ auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--format", choices=FORMATS, default="both")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--compute-type", default=None, help="int8 / float16 / int8_float16")
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--no-vad-filter", action="store_true")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=0, help="0 = ให้ระบบเลือก")
    parser.add_argument("--check-gpu", action="store_true", help="ตรวจสอบ GPU โดยไม่ถอดเสียง")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_gpu:
        print(json.dumps(gpu_status(), ensure_ascii=False, indent=2))
        return 0

    if args.video is None:
        print("กรุณาระบุพาธไฟล์วิดีโอ หรือใช้ --check-gpu", file=sys.stderr)
        return 2

    last_progress = -1.0

    def display(event: dict[str, Any]) -> None:
        nonlocal last_progress
        stage = event.get("stage")
        progress = float(event.get("progress", 0))
        if stage != "transcribing" or progress - last_progress >= 1:
            print(
                f"\r{event.get('message', stage)} "
                f"{progress:.1f}% | {event.get('device', '?')} | "
                f"ETA {event.get('eta_seconds', '?')}s      ",
                end="\n" if stage in ("completed", "loading_model", "saving") else "\r",
                flush=True,
            )
            last_progress = progress

    try:
        result = transcribe_file(
            args.video, model_name=args.model, language=args.language,
            output_dir=args.output_dir, output_format=args.format,
            device=args.device, compute_type=args.compute_type,
            beam_size=args.beam_size, vad_filter=not args.no_vad_filter,
            word_timestamps=args.word_timestamps, cpu_threads=args.cpu_threads,
            on_progress=display,
        )
        print()
        for kind, path in result["files"].items():
            print(f"{kind.upper()}: {path}")
        print(f"เสร็จแล้ว {result['segments']} ช่วงเสียง ({result['device']})")
        return 0
    except (ValueError, ImportError, RuntimeError, OSError) as exc:
        print(f"\nถอดเสียงไม่สำเร็จ: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nยกเลิกแล้ว", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
