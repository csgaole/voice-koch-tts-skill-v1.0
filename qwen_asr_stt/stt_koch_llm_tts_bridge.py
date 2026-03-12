#!/usr/bin/env python3
import argparse
import json
import shlex
import sys
from pathlib import Path

from assistant_reply import fallback_general_reply, get_weather_reply, is_weather_query, post_chat_reply
from stt_koch_bridge import execute_koch_command, execute_koch_payload
from stt_koch_llm_bridge import (
    ACTION_TO_KOCH_COMMAND,
    build_parser as build_llm_parser,
    call_llm,
    normalize_base_url,
)
from stt_qwen import fail, transcribe_audio_file
from tts_edge import DEFAULT_PITCH, DEFAULT_RATE, DEFAULT_VOICE, speak_text
from stt_qwen_mic import check_arecord, prepare_output_path, record_audio


def build_parser() -> argparse.ArgumentParser:
    parser = build_llm_parser()
    parser.description = "Record speech, transcribe it, infer robot intent, execute koch-skill, then speak a TTS reply."
    parser.add_argument("--tts-voice", default=DEFAULT_VOICE, help=f"TTS voice. Default: {DEFAULT_VOICE}")
    parser.add_argument("--tts-rate", default=DEFAULT_RATE, help=f"TTS speaking rate. Default: {DEFAULT_RATE.replace('%', '%%')}")
    parser.add_argument("--tts-pitch", default=DEFAULT_PITCH, help=f"TTS pitch. Default: {DEFAULT_PITCH}")
    parser.add_argument("--mute-tts", action="store_true", help="Disable TTS playback.")
    parser.add_argument("--assistant-location", default="Shanghai", help="Default location for weather replies. Default: Shanghai")
    return parser


def acquire_audio(args: argparse.Namespace) -> Path:
    if args.audio:
        audio_path = Path(args.audio).expanduser().resolve()
        if not audio_path.exists():
            fail(f"Audio file not found: {audio_path}")
        return audio_path

    if args.duration <= 0:
        fail("--duration must be greater than 0.")

    check_arecord()
    output_path = prepare_output_path(args.output, args.keep_audio)
    record_audio(output_path, args.duration, args.device)
    print(f"[INFO] Saved recording to {output_path}", file=sys.stderr)
    return output_path


def maybe_speak(text: str, args: argparse.Namespace) -> None:
    if args.mute_tts or not text.strip():
        return
    print(f"[INFO] Speaking reply via TTS: {text}", file=sys.stderr)
    speak_text(text, voice=args.tts_voice, rate=args.tts_rate, pitch=args.tts_pitch)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        audio_path = acquire_audio(args)
        text = transcribe_audio_file(
            str(audio_path),
            model=args.model,
            fallback_model=args.fallback_model,
            local_only=args.local_only,
        )
        print(f"[STT] {text}")

        if not text.strip():
            fail("Recognized text is empty, nothing to send to the LLM.")

        if is_weather_query(text):
            print(f"[INFO] Handling weather query without robot execution.", file=sys.stderr)
            weather_reply = get_weather_reply(text, default_location=args.assistant_location, timeout=args.llm_timeout)
            print(f"[ASSISTANT] {weather_reply}")
            maybe_speak(weather_reply, args)
            return 0

        print(
            f"[INFO] Calling LLM {args.llm_model} via {normalize_base_url(args.llm_base_url or '')}",
            file=sys.stderr,
        )
        llm_result = call_llm(
            user_text=text,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            model=args.llm_model,
            timeout=args.llm_timeout,
        )
        print(f"[LLM] {json.dumps(llm_result, ensure_ascii=False)}")

        if not llm_result["execute"]:
            print(f"[INFO] LLM declined execution: {llm_result.get('reason', '')}", file=sys.stderr)
            print("[INFO] Generating spoken assistant reply without robot execution...", file=sys.stderr)
            try:
                assistant_reply = post_chat_reply(
                    user_text=text,
                    base_url=args.llm_base_url,
                    api_key=args.llm_api_key,
                    model=args.llm_model,
                    timeout=min(args.llm_timeout, 20),
                )
            except Exception as exc:
                print(f"[WARN] Assistant reply fallback triggered: {exc}", file=sys.stderr)
                assistant_reply = fallback_general_reply(text)
            print(f"[ASSISTANT] {assistant_reply}")
            maybe_speak(assistant_reply, args)
            return 0
        if llm_result["confidence"] < args.min_confidence:
            fail(
                f"LLM confidence too low: {llm_result['confidence']:.2f} < {args.min_confidence:.2f}",
                exit_code=2,
            )

        if args.dry_run:
            maybe_speak(llm_result.get("reply", ""), args)
            return 0

        maybe_speak(llm_result.get("reply", ""), args)

        if llm_result["action"] == "custom_sequence":
            payload = {
                "action": "custom_sequence",
                "sequence": llm_result["sequence"],
                "hold_position": False,
                "return_to_home": True,
                "power_down_after": True,
            }
            print(
                f"[INFO] Sending custom sequence to koch-skill: {json.dumps(payload, ensure_ascii=False)}",
                file=sys.stderr,
            )
            ok = execute_koch_payload(payload)
        else:
            koch_command = ACTION_TO_KOCH_COMMAND[llm_result["action"]]
            print(f"[INFO] Sending to koch-skill: {shlex.quote(koch_command)}", file=sys.stderr)
            ok = execute_koch_command(koch_command)

        if not ok:
            fail("koch-skill rejected or failed to execute the LLM action.")

        print(f"[KOCH] command executed: {llm_result['action']}")
        maybe_speak("动作完成，已回到 home 并卸掉力矩。", args)
        return 0
    except KeyboardInterrupt:
        print("\n[WARN] LLM TTS bridge interrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
