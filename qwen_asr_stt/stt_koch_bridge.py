#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from stt_qwen import FALLBACK_MODEL_ID, QWEN_MODEL_ID, fail
from stt_qwen_mic import check_arecord, prepare_output_path, record_audio
from stt_qwen import transcribe_audio_file


LEROBOT_PYTHON = "/home/legao/miniforge3/envs/lerobot/bin/python"
KOCH_SKILL_ROOT = "/home/legao/.openclaw/skills/koch-skill"
KOCH_CONFIG_PATH = "/home/legao/.openclaw/skills/koch-skill/config/robot_config.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record speech, transcribe it locally, then execute the command via koch-skill."
    )
    parser.add_argument("--duration", type=int, default=5, help="Recording duration in seconds. Default: 5")
    parser.add_argument("--device", default="default", help="ALSA capture device. Default: default")
    parser.add_argument("--model", default=QWEN_MODEL_ID, help=f"Primary ASR model. Default: {QWEN_MODEL_ID}")
    parser.add_argument(
        "--fallback-model",
        default=FALLBACK_MODEL_ID,
        help=f"Fallback ASR model. Default: {FALLBACK_MODEL_ID}",
    )
    parser.add_argument("--local-only", action="store_true", help="Use only locally cached ASR models.")
    parser.add_argument("--keep-audio", action="store_true", help="Keep the recorded wav file.")
    parser.add_argument("--output", help="Optional output wav path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the recognized text. Do not send it to koch-skill.",
    )
    return parser


def execute_koch_command(command_text: str) -> bool:
    script = (
        "import sys; "
        f"sys.path.insert(0, {KOCH_SKILL_ROOT!r}); "
        "from koch_skill import KochRobotSkill; "
        f"skill = KochRobotSkill(config_path={KOCH_CONFIG_PATH!r}); "
        "ok = skill.connect(); "
        "print(f'[INFO] koch connect={ok}', file=sys.stderr); "
        "result = False; "
        "result = skill.execute_command(sys.argv[1]) if ok else False; "
        "print(result); "
        "skill.disconnect(power_down=False) if ok else None; "
        "raise SystemExit(0 if result else 2)"
    )
    completed = subprocess.run(
        [LEROBOT_PYTHON, "-c", script, command_text],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    if completed.returncode != 0:
        return False
    return completed.stdout.strip().endswith("True")


def execute_koch_payload(payload: dict) -> bool:
    script = f"""
import json
import sys

sys.path.insert(0, {KOCH_SKILL_ROOT!r})

from koch_skill import KochRobotSkill

skill = KochRobotSkill(config_path={KOCH_CONFIG_PATH!r})
ok = skill.connect()
print(f'[INFO] koch connect={{ok}}', file=sys.stderr)
result = False
payload = json.loads(sys.argv[1])
action = payload.get('action')

if ok and action == 'custom_sequence':
    result = skill.execute_joint_sequence(
        payload.get('sequence', []),
        hold_position=bool(payload.get('hold_position', True)),
        return_to_home=bool(payload.get('return_to_home', False)),
    )
elif ok and action == 'command':
    result = skill.execute_command(str(payload.get('command', '')))

print(result)
if ok:
    skill.disconnect(power_down=bool(payload.get('power_down_after', False)))
raise SystemExit(0 if result else 2)
"""
    completed = subprocess.run(
        [LEROBOT_PYTHON, "-c", script, json.dumps(payload, ensure_ascii=False)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    if completed.returncode != 0:
        return False
    return completed.stdout.strip().endswith("True")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.duration <= 0:
        fail("--duration must be greater than 0.")

    check_arecord()
    output_path = prepare_output_path(args.output, args.keep_audio)
    record_audio(output_path, args.duration, args.device)
    print(f"[INFO] Saved recording to {output_path}", file=sys.stderr)

    text = transcribe_audio_file(
        str(Path(output_path)),
        model=args.model,
        fallback_model=args.fallback_model,
        local_only=args.local_only,
    )
    print(f"[STT] {text}")

    if args.dry_run:
        return 0

    if not text.strip():
        fail("Recognized text is empty, nothing to send to koch-skill.")

    print(f"[INFO] Sending to koch-skill: {shlex.quote(text)}", file=sys.stderr)
    ok = execute_koch_command(text)
    if not ok:
        fail("koch-skill rejected or failed to execute the recognized command.")

    print("[KOCH] command executed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
