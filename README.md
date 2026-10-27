# Video to Text — Web UI + CLI

Local video/audio transcription using faster-whisper. Supports a browser interface, real-time segment progress, estimated remaining time (ETA), CPU processing, and optional NVIDIA GPU acceleration. Thai transcription is supported.

## Features

- Upload video/audio from a browser; files are streamed to disk so a 1.6 GB video is not kept in Python RAM.
- Live upload progress, transcription percent, elapsed time, rough ETA, last recognized text, and segment count.
- Download timestamped TXT, complete JSON, or subtitles (SRT).
- Auto mode checks CUDA dependencies and uses CPU if unavailable; manual CPU or CUDA modes are also available.
- Choose tiny/base/small/medium/large-v3/turbo, language, VAD, beam size, and CPU threads.
- Cancel a running job (cooperative cancellation, after the current segment).
- Existing command-line usage continues to work.
- No Node.js or separate FFmpeg executable required (PyAV handles decoding).

## Windows quick start (PowerShell)

Open PowerShell in the repository folder:

    cd D:\GitPoom\video-to-text

If you already have a working .venv, keep it. Otherwise create one with Python 3.12:

    py -3.12 -m venv .venv

Install packages into the same environment:

    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Start the local website:

    .\.venv\Scripts\python.exe web_app.py

Open http://127.0.0.1:8765 in your browser. You do not need to activate .venv, and you do not need the system-wide python command to work.

The first time you use a Whisper model, it is downloaded to the local Hugging Face cache. Download time is not included in transcription ETA.

## GPU setup / diagnosis

NVIDIA GPU inference with recent faster-whisper/CTranslate2 requires CUDA 12 cuBLAS and cuDNN 9 for CUDA 12. An NVIDIA driver by itself does not supply all the required DLLs. The error "cublas64_12.dll is not found" means a runtime library is not loadable.

1. Run nvidia-smi and confirm Windows can see your NVIDIA GPU.
2. Install matching CUDA 12 libraries and cuDNN 9 from NVIDIA, and add directories containing their DLLs (typically bin directories) to PATH.
3. Open a fresh PowerShell window.
4. Check diagnostics:

       .\.venv\Scripts\python.exe video_transcriber.py --check-gpu

5. Start the Web UI and check Hardware status. Choose Auto, CUDA, or CPU.

The GPU detector performs an initial dependency check; it cannot guarantee enough VRAM, driver compatibility or successful inference. If forced CUDA fails, select CPU or fix your GPU installation. Auto uses CPU when libraries are missing and attempts CPU if loading a GPU model raises an exception. A native CUDA library crash during later inference cannot always be recovered automatically.

More details: https://github.com/SYSTRAN/faster-whisper#gpu

## CLI examples

Fast Thai transcription on CPU:

    .\.venv\Scripts\python.exe video_transcriber.py "venus.mp4" --model small --language th --device cpu --compute-type int8 --beam-size 1

Prefer GPU when available and fall back to CPU when CUDA libraries are missing:

    .\.venv\Scripts\python.exe video_transcriber.py "venus.mp4" --model medium --language th --device auto

Force GPU (requires CUDA libraries):

    .\.venv\Scripts\python.exe video_transcriber.py "venus.mp4" --model medium --language th --device cuda --compute-type float16

Get every format, including SRT:

    .\.venv\Scripts\python.exe video_transcriber.py "venus.mp4" --format all --language th

The CLI reports progress from timestamped segments; ETA becomes available after enough segments have been processed. CPU threads can be set with --cpu-threads 8 (0 means library default). --word-timestamps provides word-level timings at additional computational cost.

## Files, privacy and limitations

- The web server binds to 127.0.0.1 by default and uses local inference, with no upload to a hosted transcription API. Model download needs an internet connection on first use.
- The Web UI holds one active job at a time to avoid overloading CPU/GPU. Outputs are retained in .work/<job-id>/outputs, while the uploaded video is removed after completion/cancellation/failure. Your original media is not deleted.
- .work and media files are ignored by Git. To reclaim disk space, delete old .work job folders after downloading outputs.
- Upload limit: 10 GiB; disk space should accommodate a temporary upload and outputs.
- Progress is based on timestamps from faster-whisper, not video file bytes. It can jump during silence/VAD. ETA is approximate; unknown during model loading/download.
- Currently no speaker diarization or live streaming transcription. Cancellation may wait for a segment to finish.
- Intended for trusted local use. Do not publicly expose --host 0.0.0.0 without authentication, rate limiting and storage controls.

## Tests

No GPU or model download is required for unit tests:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -v

## Project files

- video_transcriber.py: inference engine, GPU detection, progress callbacks and CLI
- web_app.py: local Python HTTP server, streamed upload, jobs and downloads
- web/index.html: responsive frontend; no Node build needed
- tests/: mocked inference tests
