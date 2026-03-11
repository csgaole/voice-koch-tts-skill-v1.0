#!/usr/bin/env python3
import argparse
import atexit
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path


QWEN_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
FALLBACK_MODEL_ID = "openai/whisper-small"


def configure_runtime_logging() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"The following generation flags are not valid and may be ignored: \['temperature'\].*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`return_token_timestamps` is deprecated for WhisperFeatureExtractor.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Using custom `forced_decoder_ids` from the \(generation\) config\..*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Transcription using a multilingual Whisper will default to language detection.*",
    )

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Qwen/Qwen3-ASR-0.6B and fall back to Whisper if needed."
    )
    parser.add_argument("--audio", required=True, help="Path to the input audio file.")
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
    return parser


def fail(message: str, exit_code: int = 1) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        fail("ffmpeg is not installed or not in PATH.")


def normalize_audio(input_path: Path) -> Path:
    suffix = ".wav"
    temp_file = tempfile.NamedTemporaryFile(prefix="qwen_asr_", suffix=suffix, delete=False)
    temp_file.close()
    output_path = Path(temp_file.name)
    atexit.register(lambda: output_path.exists() and output_path.unlink())

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        fail(f"ffmpeg preprocessing failed:\n{completed.stderr.strip()}")
    return output_path


def resolve_device(torch_module):
    if torch_module.cuda.is_available():
        if hasattr(torch_module.cuda, "is_bf16_supported") and torch_module.cuda.is_bf16_supported():
            dtype = torch_module.bfloat16
        else:
            dtype = torch_module.float16
        return "cuda:0", dtype, "GPU"
    return "cpu", torch_module.float32, "CPU"


def extract_text(result) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if hasattr(result, "text"):
        return str(result.text).strip()
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(result).strip()


def resolve_model_source(model_name: str, local_only: bool) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path.resolve())

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is not installed.") from exc

    try:
        cached_path = snapshot_download(model_name, local_files_only=True)
        print(f"[INFO] Using cached model: {model_name}", file=sys.stderr)
        return cached_path
    except LocalEntryNotFoundError:
        if local_only:
            raise RuntimeError(f"Model not cached locally: {model_name}")

    print(f"[INFO] Model not found in local cache. Downloading: {model_name}", file=sys.stderr)
    try:
        return snapshot_download(model_name, local_files_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"Model not cached locally and download failed: {model_name}. "
            "Check network access or pre-download the model."
        ) from exc


def transcribe_with_qwen(audio_path: Path, model_name: str, local_only: bool = False) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed.") from exc

    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError("qwen-asr is not installed.") from exc

    model_source = resolve_model_source(model_name, local_only=local_only)
    device_map, dtype, device_label = resolve_device(torch)
    print(f"[INFO] Loading {model_name} on {device_label} ({device_map})...", file=sys.stderr)
    load_variants = [
        {
            "dtype": dtype,
            "device_map": device_map,
            "max_inference_batch_size": 1,
            "max_new_tokens": 256,
        },
        {"dtype": dtype, "device_map": device_map},
        {},
    ]

    last_error = None
    model = None
    for kwargs in load_variants:
        try:
            model = Qwen3ASRModel.from_pretrained(model_source, **kwargs)
            break
        except TypeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
            break

    if model is None:
        raise RuntimeError(f"Unable to initialize Qwen ASR model: {last_error}")

    results = model.transcribe(audio=str(audio_path), language=None)
    if not results:
        raise RuntimeError("Qwen ASR returned no results.")
    text = extract_text(results[0])
    if not text:
        raise RuntimeError("Qwen ASR returned empty text.")
    return text


def transcribe_with_whisper(audio_path: Path, model_name: str, local_only: bool = False) -> str:
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Fallback requires torch and transformers.") from exc

    model_source = resolve_model_source(model_name, local_only=local_only)
    device = 0 if torch.cuda.is_available() else -1
    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(
        f"[WARN] Falling back to {model_name} because Qwen ASR is unavailable in this environment.",
        file=sys.stderr,
    )
    recognizer = pipeline(
        task="automatic-speech-recognition",
        model=model_source,
        device=device,
        dtype=model_dtype,
    )
    result = recognizer(str(audio_path))
    text = extract_text(result)
    if not text:
        raise RuntimeError("Fallback ASR returned empty text.")
    return text


def transcribe_audio_file(
    audio: str,
    model: str = QWEN_MODEL_ID,
    fallback_model: str = FALLBACK_MODEL_ID,
    local_only: bool = False,
) -> str:
    configure_runtime_logging()
    audio_path = Path(audio).expanduser().resolve()
    if not audio_path.exists():
        fail(f"Audio file not found: {audio_path}")

    check_ffmpeg()
    prepared_audio = normalize_audio(audio_path)
    print(f"[INFO] Using normalized audio: {prepared_audio}", file=sys.stderr)

    try:
        text = transcribe_with_qwen(prepared_audio, model, local_only=local_only)
    except Exception as qwen_error:
        print(f"[WARN] Qwen transcription failed: {qwen_error}", file=sys.stderr)
        try:
            text = transcribe_with_whisper(prepared_audio, fallback_model, local_only=local_only)
        except Exception as fallback_error:
            fail(
                "Both Qwen and fallback ASR failed.\n"
                f"Qwen error: {qwen_error}\n"
                f"Fallback error: {fallback_error}"
            )

    return text


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        text = transcribe_audio_file(
            args.audio,
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
