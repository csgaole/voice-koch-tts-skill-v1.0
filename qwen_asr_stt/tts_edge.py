#!/usr/bin/env python3
import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_RATE = "+18%"
DEFAULT_PITCH = "+8Hz"


def fail(message: str, exit_code: int = 1) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def check_audio_player() -> str:
    if shutil.which("ffplay"):
        return "ffplay"
    if shutil.which("aplay"):
        return "aplay"
    fail("Neither ffplay nor aplay is installed.")


async def synthesize_to_file(text: str, output_path: Path, voice: str, rate: str, pitch: str) -> None:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("edge-tts is not installed in the current environment.") from exc

    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    await communicate.save(str(output_path))


def speak_text(text: str, voice: str = DEFAULT_VOICE, rate: str = DEFAULT_RATE, pitch: str = DEFAULT_PITCH) -> None:
    cleaned = text.strip()
    if not cleaned:
        return

    player = check_audio_player()
    with tempfile.NamedTemporaryFile(prefix="voice_koch_tts_", suffix=".mp3", delete=False) as tmp:
        output_path = Path(tmp.name)

    try:
        asyncio.run(synthesize_to_file(cleaned, output_path, voice=voice, rate=rate, pitch=pitch))
        if player == "ffplay":
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(output_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            wav_path = output_path.with_suffix(".wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(output_path), str(wav_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["aplay", str(wav_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            wav_path.unlink(missing_ok=True)
    finally:
        output_path.unlink(missing_ok=True)
