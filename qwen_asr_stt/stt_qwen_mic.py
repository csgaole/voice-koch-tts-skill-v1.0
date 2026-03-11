#!/usr/bin/env python3
import argparse
import atexit
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from stt_qwen import (
    FALLBACK_MODEL_ID,
    QWEN_MODEL_ID,
    configure_runtime_logging,
    fail,
    transcribe_audio_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record from the microphone and immediately transcribe with Qwen ASR."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Recording duration in seconds. Default: 5",
    )
    parser.add_argument(
        "--device",
        default="default",
        help="ALSA capture device passed to arecord. Default: default",
    )
    parser.add_argument(
        "--model",
        default=QWEN_MODEL_ID,
        help=f"Primary ASR model. Default: {QWEN_MODEL_ID}",
    )
    parser.add_argument(
        "--fallback-model",
        default=FALLBACK_MODEL_ID,
        help=f"Fallback model used if Qwen ASR is unavailable. Default: {FALLBACK_MODEL_ID}",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Use only locally cached models and never try to download from the network.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the recorded wav file instead of deleting it on exit.",
    )
    parser.add_argument(
        "--output",
        help="Optional output wav path. If omitted, a temp file is used.",
    )
    return parser


def check_arecord() -> None:
    if shutil.which("arecord") is None:
        fail("arecord is not installed or not in PATH.")


def prepare_output_path(output: str | None, keep_audio: bool) -> Path:
    if output:
        return Path(output).expanduser().resolve()

    handle = tempfile.NamedTemporaryFile(prefix="qwen_mic_", suffix=".wav", delete=False)
    handle.close()
    output_path = Path(handle.name)
    if not keep_audio:
        atexit.register(lambda: output_path.exists() and output_path.unlink())
    return output_path


def record_audio(output_path: Path, duration: int, device: str) -> None:
    command = [
        "arecord",
        "-q",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-c",
        "1",
        "-r",
        "16000",
        "-d",
        str(duration),
        str(output_path),
    ]
    print(f"[INFO] Recording {duration}s from ALSA device '{device}'...", file=sys.stderr)
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        fail(
            "Microphone recording failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Details: {completed.stderr.strip()}"
        )


def main() -> int:
    configure_runtime_logging()
    parser = build_parser()
    args = parser.parse_args()
    if args.duration <= 0:
        fail("--duration must be greater than 0.")

    check_arecord()
    output_path = prepare_output_path(args.output, args.keep_audio)
    record_audio(output_path, args.duration, args.device)
    print(f"[INFO] Saved recording to {output_path}", file=sys.stderr)

    try:
        text = transcribe_audio_file(
            str(output_path),
            model=args.model,
            fallback_model=args.fallback_model,
            local_only=args.local_only,
        )
        print(text)
        return 0
    except KeyboardInterrupt:
        print("\n[WARN] Transcription interrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
