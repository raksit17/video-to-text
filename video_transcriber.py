#!/usr/bin/env python3
"""ถอดเสียงจากไฟล์วิดีโอเป็นข้อความ พร้อมเวลาและข้อมูล JSON สำหรับวิเคราะห์ต่อ."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ถอดเสียงจากวิดีโอและบันทึก transcript เป็น TXT/JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="พาธไฟล์วิดีโอ เช่น interview.mp4")
    parser.add_argument(
        "--model", default="small",
        help="โมเดล Whisper เช่น tiny, base, small, medium, large-v3",
    )
    parser.add_argument(
        "--language", default="auto",
        help="ภาษาของเสียง เช่น th, en; ใช้ auto ให้ตรวจภาษาอัตโนมัติ",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="โฟลเดอร์เก็บผลลัพธ์ (ค่าเริ่มต้นคือโฟลเดอร์เดียวกับวิดีโอ)",
    )
    parser.add_argument(
        "--format", choices=("both", "txt", "json"), default="both",
        help="ชนิดไฟล์ผลลัพธ์",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="cpu",
        help="อุปกรณ์ประมวลผล",
    )
    parser.add_argument(
        "--compute-type", default=None,
        help="ชนิดการคำนวณ เช่น int8 (CPU) หรือ float16 (GPU); ค่าเริ่มต้นเลือกอัตโนมัติ",
    )
    parser.add_argument(
        "--word-timestamps", action="store_true",
        help="เก็บเวลาเริ่ม/จบของแต่ละคำเพิ่มใน JSON (อาจใช้เวลาประมวลผลเพิ่ม)",
    )
    parser.add_argument(
        "--no-vad-filter", action="store_true",
        help="ปิดตัวกรองเสียงเงียบ เหมาะเมื่อคลิปมีเสียงพูดแต่ได้ผลลัพธ์ 0 ช่วงเสียง",
    )
    parser.add_argument(
        "--beam-size", type=int, default=5,
        help="ค่า beam size: เพิ่มเพื่อความแม่นยำ แต่อาจทำงานช้าลง",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"ไม่พบไฟล์วิดีโอ: {video}", file=sys.stderr)
        return 2
    if args.beam_size < 1:
        print("--beam-size ต้องมีค่าอย่างน้อย 1", file=sys.stderr)
        return 2

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "ยังไม่ได้ติดตั้งแพ็กเกจที่ต้องใช้\n"
            "ติดตั้งด้วยคำสั่ง: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    device = args.device
    compute_type = args.compute_type or ("int8" if device == "cpu" else "default")
    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
        language = None if args.language.lower() == "auto" else args.language.lower()
        segments_iter, info = model.transcribe(
            str(video),
            language=language,
            beam_size=args.beam_size,
            vad_filter=not args.no_vad_filter,
            word_timestamps=args.word_timestamps,
        )

        segments: list[dict[str, Any]] = []
        for segment in segments_iter:
            item: dict[str, Any] = {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
                "avg_logprob": round(segment.avg_logprob, 4),
                "no_speech_prob": round(segment.no_speech_prob, 4),
            }
            if args.word_timestamps and segment.words:
                item["words"] = [
                    {
                        "start": round(word.start, 3),
                        "end": round(word.end, 3),
                        "text": word.word,
                        "probability": round(word.probability, 4),
                    }
                    for word in segment.words
                ]
            segments.append(item)

        # VAD can occasionally remove all speech in noisy, quiet, or unusual media.
        # Retry once without it before treating the file as having no usable speech.
        if not segments and not args.no_vad_filter:
            print("ไม่พบช่วงเสียงเมื่อเปิด VAD; กำลังลองถอดใหม่โดยปิดตัวกรองเสียงเงียบ...", file=sys.stderr)
            segments_iter, info = model.transcribe(
                str(video),
                language=language,
                beam_size=args.beam_size,
                vad_filter=False,
                word_timestamps=args.word_timestamps,
            )
            for segment in segments_iter:
                item = {
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": segment.text.strip(),
                    "avg_logprob": round(segment.avg_logprob, 4),
                    "no_speech_prob": round(segment.no_speech_prob, 4),
                }
                if args.word_timestamps and segment.words:
                    item["words"] = [
                        {
                            "start": round(word.start, 3),
                            "end": round(word.end, 3),
                            "text": word.word,
                            "probability": round(word.probability, 4),
                        }
                        for word in segment.words
                    ]
                segments.append(item)

        if not segments:
            print(
                "ไม่พบเสียงพูดในไฟล์ จึงไม่ได้สร้างไฟล์ผลลัพธ์ใหม่\n"
                "ตรวจว่าไฟล์มีแทร็กเสียงและเสียงพูดดังพอ แล้วลองใช้ --no-vad-filter",
                file=sys.stderr,
            )
            return 1
    except Exception as exc:
        print(f"ถอดเสียงไม่สำเร็จ: {exc}", file=sys.stderr)
        if isinstance(exc, TypeError) and "metadata_errors" in str(exc):
            print(
                "เวอร์ชัน PyAV ไม่เข้ากับ faster-whisper; แก้ใน virtual environment นี้ด้วย:\n"
                "python -m pip install \"av<19\"",
                file=sys.stderr,
            )
        else:
            print(
                "ตรวจสอบว่าไฟล์วิดีโอเปิดได้ และลองติดตั้งแพ็กเกจใหม่ด้วย "
                "python -m pip install -r requirements.txt",
                file=sys.stderr,
            )
        return 1

    output_dir = (args.output_dir or video.parent).expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = video.stem
        transcript = "\n".join(
            f"[{timestamp(segment['start'])} - {timestamp(segment['end'])}] {segment['text']}"
            for segment in segments
        )
        if args.format in ("both", "txt"):
            txt_path = output_dir / f"{stem}.transcript.txt"
            txt_path.write_text(transcript + ("\n" if transcript else ""), encoding="utf-8")
            print(f"บันทึกข้อความ: {txt_path}")
        if args.format in ("both", "json"):
            json_path = output_dir / f"{stem}.transcript.json"
            result = {
                "source_file": video.name,
                "model": args.model,
                "language": info.language,
                "language_probability": round(info.language_probability, 4),
                "duration_seconds": round(info.duration, 3),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "full_text": " ".join(segment["text"] for segment in segments),
                "segments": segments,
            }
            json_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"บันทึกข้อมูลวิเคราะห์: {json_path}")
    except OSError as exc:
        print(f"บันทึกไฟล์ผลลัพธ์ไม่สำเร็จ: {exc}", file=sys.stderr)
        return 1

    print(f"ถอดเสียงเสร็จแล้ว ({len(segments)} ช่วงเสียง, ภาษา {info.language})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
