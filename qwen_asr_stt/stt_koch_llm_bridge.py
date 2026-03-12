#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import time
import sys
import urllib.error
import urllib.request
from pathlib import Path

from stt_koch_bridge import execute_koch_command, execute_koch_payload
from stt_qwen import FALLBACK_MODEL_ID, QWEN_MODEL_ID, fail, transcribe_audio_file
from stt_qwen_mic import check_arecord, prepare_output_path, record_audio


OPENCLAW_CONFIG = Path(__file__).resolve().parents[1] / ".openclaw" / "openclaw0309.json"
DEFAULT_KIMI_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_KIMI_MODEL = "moonshot/kimi-k2.5"
SUPPORTED_ACTIONS = {"home", "ready", "dance", "status", "power_down", "custom_sequence"}
ACTION_TO_KOCH_COMMAND = {
    "home": "home",
    "ready": "ready",
    "dance": "dance",
    "status": "状态",
    "power_down": "下电",
}
SYSTEM_PROMPT = """你是 Koch 机械臂的意图解析器。

你只能从以下动作中选择一个：
- home
- ready
- dance
- status
- power_down
- custom_sequence

如果用户意图不明确、不是在控制机械臂、或你没有把握，请返回 execute=false。
优先输出结构化动作，不要输出代码。
当固定动作不足以表达用户需求时，使用 custom_sequence。
custom_sequence 只允许使用以下关节名：
- shoulder_pan
- shoulder_lift
- elbow_flex
- wrist_flex
- wrist_roll
- gripper

关节位置建议范围：
- shoulder_pan: 1600-2500
- shoulder_lift: 1500-2100
- elbow_flex: 2100-2600
- wrist_flex: 2200-2900
- wrist_roll: 1700-2400
- gripper: 2100-2600

每一步只输出必要关节，桥接层会基于当前姿态补全其它关节。
每一步 duration 取 0.2 到 3.0 秒，pause 取 0 到 2.0 秒。
尽量把 custom_sequence 控制在 2 到 4 步，优先给出最短可执行序列。
只输出一个 JSON 对象，不要输出任何解释、markdown 或多余文字。

输出格式：
{
  "execute": true,
  "action": "dance",
  "confidence": 0.93,
  "reason": "用户明确要求机械臂跳舞。",
  "reply": "执行跳舞动作。"
}

自定义动作格式：
{
  "execute": true,
  "action": "custom_sequence",
  "confidence": 0.90,
  "reason": "用户要求摆手，固定动作不足以精确表达。",
  "reply": "执行摆手动作。",
  "return_to_home": false,
  "sequence": [
    {
      "targets": {
        "shoulder_pan": 2280,
        "elbow_flex": 2140,
        "wrist_flex": 2700,
        "gripper": 2380
      },
      "duration": 0.8,
      "pause": 0.1
    },
    {
      "targets": {
        "shoulder_pan": 1820,
        "gripper": 2520
      },
      "duration": 0.8
    }
  ]
}

few-shot 示例 1：摆摆手
{
  "execute": true,
  "action": "custom_sequence",
  "confidence": 0.94,
  "reason": "用户要求机械臂摆手，适合先抬手，再通过 shoulder_pan 左右摆动形成挥手效果。",
  "reply": "执行摆手动作。",
  "return_to_home": false,
  "sequence": [
    {
      "targets": {
        "shoulder_lift": 1980,
        "elbow_flex": 2220,
        "wrist_flex": 2660,
        "gripper": 2380
      },
      "duration": 0.9,
      "pause": 0.1
    },
    {
      "targets": {
        "shoulder_pan": 2260,
        "gripper": 2480
      },
      "duration": 0.6,
      "pause": 0.1
    },
    {
      "targets": {
        "shoulder_pan": 1820,
        "gripper": 2340
      },
      "duration": 0.6,
      "pause": 0.1
    },
    {
      "targets": {
        "shoulder_pan": 2260,
        "gripper": 2480
      },
      "duration": 0.6
    }
  ]
}

few-shot 示例 2：把手抬起来，挥挥手
{
  "execute": true,
  "action": "custom_sequence",
  "confidence": 0.95,
  "reason": "用户要求先抬手，再挥手，需要 shoulder_lift、elbow_flex、shoulder_pan 联动。",
  "reply": "执行抬手挥手动作。",
  "return_to_home": false,
  "sequence": [
    {
      "targets": {
        "shoulder_lift": 1940,
        "elbow_flex": 2180,
        "wrist_flex": 2720,
        "gripper": 2400
      },
      "duration": 1.0,
      "pause": 0.15
    },
    {
      "targets": {
        "shoulder_pan": 2300
      },
      "duration": 0.65,
      "pause": 0.1
    },
    {
      "targets": {
        "shoulder_pan": 1780
      },
      "duration": 0.65,
      "pause": 0.1
    },
    {
      "targets": {
        "shoulder_pan": 2300
      },
      "duration": 0.65
    }
  ]
}

few-shot 示例 3：夹爪张开再合上
{
  "execute": true,
  "action": "custom_sequence",
  "confidence": 0.93,
  "reason": "用户要求夹爪开合，适合固定手臂姿态，只控制 gripper。",
  "reply": "执行夹爪开合动作。",
  "return_to_home": false,
  "sequence": [
    {
      "targets": {
        "shoulder_lift": 2000,
        "elbow_flex": 2240,
        "wrist_flex": 2620,
        "gripper": 2550
      },
      "duration": 0.8,
      "pause": 0.15
    },
    {
      "targets": {
        "gripper": 2200
      },
      "duration": 0.6,
      "pause": 0.15
    },
    {
      "targets": {
        "gripper": 2550
      },
      "duration": 0.6
    }
  ]
}

few-shot 示例 4：升到最高
{
  "execute": true,
  "action": "custom_sequence",
  "confidence": 0.94,
  "reason": "用户要求机械臂尽量升高，适合通过 shoulder_lift、elbow_flex、wrist_flex 联动抬高手臂。",
  "reply": "执行升高动作。",
  "return_to_home": false,
  "sequence": [
    {
      "targets": {
        "shoulder_pan": 2050,
        "shoulder_lift": 1980,
        "elbow_flex": 2250,
        "wrist_flex": 2550,
        "gripper": 2350
      },
      "duration": 1.0,
      "pause": 0.15
    },
    {
      "targets": {
        "shoulder_lift": 2100,
        "elbow_flex": 2100,
        "wrist_flex": 2620
      },
      "duration": 0.9
    }
  ]
}

few-shot 示例 5：抬手
{
  "execute": true,
  "action": "custom_sequence",
  "confidence": 0.93,
  "reason": "用户要求机械臂抬手，适合抬升 shoulder_lift 并打开 elbow_flex。",
  "reply": "执行抬手动作。",
  "return_to_home": false,
  "sequence": [
    {
      "targets": {
        "shoulder_lift": 1960,
        "elbow_flex": 2200,
        "wrist_flex": 2680,
        "gripper": 2380
      },
      "duration": 1.0
    }
  ]
}

few-shot 示例 6：摆手
{
  "execute": true,
  "action": "custom_sequence",
  "confidence": 0.94,
  "reason": "用户要求摆手，适合先抬手，再通过 shoulder_pan 左右摆动形成摆手动作。",
  "reply": "执行摆手动作。",
  "return_to_home": false,
  "sequence": [
    {
      "targets": {
        "shoulder_lift": 1960,
        "elbow_flex": 2200,
        "wrist_flex": 2680,
        "gripper": 2400
      },
      "duration": 0.9,
      "pause": 0.1
    },
    {
      "targets": {
        "shoulder_pan": 2280
      },
      "duration": 0.6,
      "pause": 0.1
    },
    {
      "targets": {
        "shoulder_pan": 1800
      },
      "duration": 0.6,
      "pause": 0.1
    },
    {
      "targets": {
        "shoulder_pan": 2280
      },
      "duration": 0.6
    }
  ]
}

few-shot 示例 7：点点头
{
  "execute": true,
  "action": "custom_sequence",
  "confidence": 0.92,
  "reason": "用户要求点头，适合保持手臂基本稳定，通过 wrist_flex 上下摆动模拟点头。",
  "reply": "执行点头动作。",
  "return_to_home": false,
  "sequence": [
    {
      "targets": {
        "shoulder_lift": 2020,
        "elbow_flex": 2260,
        "wrist_flex": 2460
      },
      "duration": 0.8,
      "pause": 0.1
    },
    {
      "targets": {
        "wrist_flex": 2780
      },
      "duration": 0.45,
      "pause": 0.1
    },
    {
      "targets": {
        "wrist_flex": 2380
      },
      "duration": 0.45,
      "pause": 0.1
    },
    {
      "targets": {
        "wrist_flex": 2780
      },
      "duration": 0.45
    }
  ]
}

few-shot 示例 8：抓一下再松开
{
  "execute": true,
  "action": "custom_sequence",
  "confidence": 0.94,
  "reason": "用户要求抓一下再松开，适合固定手臂姿态并让 gripper 闭合再张开。",
  "reply": "执行抓取再松开动作。",
  "return_to_home": false,
  "sequence": [
    {
      "targets": {
        "shoulder_lift": 2000,
        "elbow_flex": 2240,
        "wrist_flex": 2620,
        "gripper": 2520
      },
      "duration": 0.8,
      "pause": 0.15
    },
    {
      "targets": {
        "gripper": 2180
      },
      "duration": 0.55,
      "pause": 0.15
    },
    {
      "targets": {
        "gripper": 2520
      },
      "duration": 0.55
    }
  ]
}
"""


def load_openclaw_defaults() -> dict[str, str | None]:
    defaults: dict[str, str | None] = {
        "base_url": (
            os.getenv("KIMI_LLM_BASE_URL")
            or os.getenv("MOONSHOT_BASE_URL")
            or os.getenv("QWEN_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or DEFAULT_KIMI_BASE_URL
        ),
        "api_key": (
            os.getenv("KIMI_API_KEY")
            or os.getenv("MOONSHOT_API_KEY")
            or os.getenv("QWEN_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        ),
        "model": (
            os.getenv("KIMI_LLM_MODEL")
            or os.getenv("MOONSHOT_MODEL")
            or os.getenv("QWEN_LLM_MODEL")
            or os.getenv("OPENAI_MODEL")
            or DEFAULT_KIMI_MODEL
        ),
    }

    if not OPENCLAW_CONFIG.exists():
        return defaults

    try:
        config = json.loads(OPENCLAW_CONFIG.read_text(encoding="utf-8"))
        provider = config.get("models", {}).get("providers", {}).get("vllm", {})

        if not defaults["base_url"]:
            defaults["base_url"] = provider.get("baseUrl")

        models = provider.get("models", [])
        if not defaults["model"] and models:
            defaults["model"] = models[0].get("id")

        api_key_ref = provider.get("apiKey")
        if not defaults["api_key"] and isinstance(api_key_ref, str):
            defaults["api_key"] = os.getenv(api_key_ref)
    except Exception:
        return defaults

    return defaults


def build_parser() -> argparse.ArgumentParser:
    llm_defaults = load_openclaw_defaults()
    parser = argparse.ArgumentParser(
        description="Record speech, transcribe it, infer robot intent with an LLM, then execute koch-skill."
    )
    parser.add_argument("--audio", help="Optional input audio path. If omitted, the microphone is used.")
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
    parser.add_argument("--output", help="Optional recorded wav output path.")
    parser.add_argument("--dry-run", action="store_true", help="Do not execute koch-skill.")
    parser.add_argument(
        "--llm-base-url",
        default=llm_defaults["base_url"],
        help="OpenAI-compatible LLM base URL. Defaults to OpenClaw config or env.",
    )
    parser.add_argument(
        "--llm-model",
        default=llm_defaults["model"] or DEFAULT_KIMI_MODEL,
        help=f"LLM model id. Default: {DEFAULT_KIMI_MODEL}",
    )
    parser.add_argument(
        "--llm-api-key",
        default=llm_defaults["api_key"],
        help="LLM API key. Defaults to env vars referenced by OpenClaw config.",
    )
    parser.add_argument("--llm-timeout", type=int, default=90, help="LLM request timeout in seconds.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        help="Reject LLM decisions below this confidence. Default: 0.60",
    )
    return parser


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise RuntimeError(f"LLM did not return JSON: {text}")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise RuntimeError(f"LLM returned an incomplete JSON object: {text}")


def repair_common_json_issues(text: str) -> str:
    repaired = text.strip()
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
    repaired = repaired.replace("\uff0c", ",").replace("\uff1a", ":")
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


def parse_json_object(text: str) -> dict:
    stripped = strip_markdown_fences(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        candidate = extract_first_json_object(stripped)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = repair_common_json_issues(candidate)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Unable to parse LLM JSON: {candidate}") from exc


def repair_json_with_llm(
    raw_text: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
) -> dict:
    endpoint = f"{normalize_base_url(base_url)}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 600,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是 JSON 修复器。"
                    "把用户提供的内容修复成一个合法 JSON 对象。"
                    "不要改语义，不要解释，只输出 JSON。"
                ),
            },
            {"role": "user", "content": raw_text},
        ],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    raw = post_json_request(request, timeout=timeout, retries=1)

    body = json.loads(raw)
    content = body["choices"][0]["message"]["content"]
    return parse_json_object(content)


def post_json_request(request: urllib.request.Request, timeout: int, retries: int = 1) -> str:
    last_error = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {details}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            wait_seconds = min(2 * (attempt + 1), 4)
            print(
                f"[WARN] LLM request failed ({exc}). Retrying in {wait_seconds}s "
                f"[attempt {attempt + 1}/{retries}]...",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"LLM network error: {last_error}") from last_error


def validate_llm_result(result: dict) -> dict:
    if not isinstance(result, dict):
        raise RuntimeError("LLM result must be a JSON object.")

    execute = bool(result.get("execute", False))
    raw_action = result.get("action")
    action = "" if raw_action is None else str(raw_action).strip()
    reason = str(result.get("reason", "")).strip()
    reply = str(result.get("reply", "")).strip()
    sequence = result.get("sequence", [])
    return_to_home = bool(result.get("return_to_home", False))

    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("LLM confidence must be numeric.") from exc

    if not execute:
        action = action if action in SUPPORTED_ACTIONS else ""
    elif action not in SUPPORTED_ACTIONS:
        raise RuntimeError(f"Unsupported LLM action: {action}")

    normalized_sequence = []
    if action == "custom_sequence":
        if not isinstance(sequence, list) or not sequence:
            raise RuntimeError("custom_sequence requires a non-empty sequence list.")
        if len(sequence) > 16:
            raise RuntimeError("custom_sequence supports at most 16 steps.")

        for idx, step in enumerate(sequence, start=1):
            if not isinstance(step, dict):
                raise RuntimeError(f"Sequence step {idx} must be an object.")
            targets = step.get("targets")
            if not isinstance(targets, dict) or not targets:
                raise RuntimeError(f"Sequence step {idx} requires non-empty targets.")
            normalized_targets = {}
            for joint_name, raw_position in targets.items():
                joint = str(joint_name).strip()
                if not joint:
                    raise RuntimeError(f"Sequence step {idx} has empty joint name.")
                try:
                    normalized_targets[joint] = int(raw_position)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"Sequence step {idx} has invalid position for {joint}.") from exc

            try:
                duration = float(step.get("duration", 0.8))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Sequence step {idx} has invalid duration.") from exc
            try:
                pause = float(step.get("pause", 0.0))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Sequence step {idx} has invalid pause.") from exc

            normalized_sequence.append(
                {
                    "targets": normalized_targets,
                    "duration": max(0.2, min(3.0, duration)),
                    "pause": max(0.0, min(2.0, pause)),
                }
            )

    return {
        "execute": execute,
        "action": action,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": reason,
        "reply": reply,
        "sequence": normalized_sequence,
        "return_to_home": return_to_home,
    }


def call_llm(user_text: str, base_url: str, api_key: str, model: str, timeout: int) -> dict:
    if not base_url:
        raise RuntimeError("LLM base URL is not configured.")
    if not api_key:
        raise RuntimeError("LLM API key is not configured.")
    if not model:
        raise RuntimeError("LLM model is not configured.")

    endpoint = f"{normalize_base_url(base_url)}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 600,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"ASR 转写文本：{user_text}"},
        ],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    raw = post_json_request(request, timeout=timeout, retries=1)

    try:
        body = json.loads(raw)
        content = body["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected LLM response body: {raw}") from exc

    try:
        return validate_llm_result(parse_json_object(content))
    except Exception as first_error:
        print(f"[WARN] Primary LLM JSON parsing failed, retrying repair: {first_error}", file=sys.stderr)
        try:
            repaired = repair_json_with_llm(
                raw_text=content,
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout=timeout,
            )
            return validate_llm_result(repaired)
        except Exception as second_error:
            raise RuntimeError(
                "LLM returned malformed JSON and repair also failed.\n"
                f"Original content:\n{content}"
            ) from second_error


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
            fail("LLM decided not to execute any robot action.", exit_code=2)
        if llm_result["confidence"] < args.min_confidence:
            fail(
                f"LLM confidence too low: {llm_result['confidence']:.2f} < {args.min_confidence:.2f}",
                exit_code=2,
            )

        if args.dry_run:
            return 0

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
        return 0
    except KeyboardInterrupt:
        print("\n[WARN] LLM bridge interrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
