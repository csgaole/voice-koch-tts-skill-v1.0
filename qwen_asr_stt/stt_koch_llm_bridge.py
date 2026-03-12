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
DEFAULT_KIMI_MODEL = "kimi-k2.5"
DEFAULT_KIMI_FALLBACK_MODEL = "moonshot-v1-8k"
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

KIMI_SEMANTIC_PROMPT = """你是 Koch 机械臂的语义动作解析器。

你的任务是把用户中文指令解析成一个简短 JSON 对象。
不要生成关节值，不要生成代码，不要解释。

如果用户不是在控制机械臂，输出：
{"execute": false, "reason": "...", "reply": "...", "actions": []}

如果用户在控制机械臂，输出：
{"execute": true, "reason": "...", "reply": "...", "actions": [...]}

actions 只允许这些动作名：
- preset: 预设动作，preset 只能是 home / ready / dance / status / power_down
- raise: 抬手
- wave: 摆手，允许 count
- nod: 点头，允许 count
- grip: 夹爪动作，mode 只能是 open_close
- highest: 升到最高

示例 1：
{"execute": true, "reason": "用户要求抬手并摆手。", "reply": "执行抬手摆手动作。", "actions": [{"name": "raise"}, {"name": "wave", "count": 5}]}

示例 2：
{"execute": true, "reason": "用户要求抬手并点头。", "reply": "执行抬手点头动作。", "actions": [{"name": "raise"}, {"name": "nod", "count": 3}]}

示例 3：
{"execute": true, "reason": "用户要求回 home。", "reply": "回到 home 位置。", "actions": [{"name": "preset", "preset": "home"}]}

示例 4：
{"execute": false, "reason": "用户在聊汽车，不是机械臂控制。", "reply": "这个问题和机械臂无关。", "actions": []}
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
        "fallback_model": (
            os.getenv("KIMI_FALLBACK_MODEL")
            or os.getenv("MOONSHOT_FALLBACK_MODEL")
            or DEFAULT_KIMI_FALLBACK_MODEL
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
        "--llm-fallback-model",
        default=llm_defaults["fallback_model"] or DEFAULT_KIMI_FALLBACK_MODEL,
        help=f"Fallback LLM model if the default model is unavailable. Default: {DEFAULT_KIMI_FALLBACK_MODEL}",
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


def temperature_for_model(model: str) -> float | int:
    normalized = (model or "").strip().lower()
    if normalized == "kimi-k2.5":
        return 1
    return 0


def system_prompt_for_model(model: str) -> str:
    normalized = (model or "").strip().lower()
    if normalized == "kimi-k2.5":
        return KIMI_SEMANTIC_PROMPT
    return SYSTEM_PROMPT


def should_retry_with_fallback_model(error: Exception, model: str, fallback_model: str | None) -> bool:
    if not fallback_model or fallback_model == model:
        return False
    message = str(error)
    return (
        "resource_not_found_error" in message
        or "Permission denied" in message
        or "Not found the model" in message
        or "LLM returned empty content" in message
    )


def should_retry_parse_failure_with_fallback(error: Exception, model: str, fallback_model: str | None) -> bool:
    if not fallback_model or fallback_model == model:
        return False
    message = str(error)
    return (
        "LLM returned an incomplete JSON object" in message
        or "LLM returned malformed JSON and repair also failed" in message
        or "LLM returned empty content during JSON repair" in message
        or "Unable to parse LLM JSON" in message
    )


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


def extract_message_content(body: dict) -> str:
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LLM response body: {body}") from exc

    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def repair_json_with_llm(
    raw_text: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
) -> dict:
    if not raw_text.strip():
        raise RuntimeError("LLM returned empty content; nothing to repair.")

    endpoint = f"{normalize_base_url(base_url)}/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature_for_model(model),
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
    content = extract_message_content(body)
    if not content:
        raise RuntimeError("LLM returned empty content during JSON repair.")
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

    if "actions" in result and "action" not in result:
        return validate_semantic_action_result(result)

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


def _prep_raise_step() -> dict:
    return {
        "targets": {
            "shoulder_lift": 1960,
            "elbow_flex": 2200,
            "wrist_flex": 2680,
            "gripper": 2380,
        },
        "duration": 1.0,
        "pause": 0.15,
    }


def _highest_step_sequence() -> list[dict]:
    return [
        {
            "targets": {
                "shoulder_pan": 2050,
                "shoulder_lift": 1980,
                "elbow_flex": 2250,
                "wrist_flex": 2550,
                "gripper": 2350,
            },
            "duration": 1.0,
            "pause": 0.15,
        },
        {
            "targets": {
                "shoulder_lift": 2100,
                "elbow_flex": 2100,
                "wrist_flex": 2620,
            },
            "duration": 0.9,
            "pause": 0.0,
        },
    ]


def _wave_steps(count: int) -> list[dict]:
    steps: list[dict] = []
    for index in range(count):
        steps.append({"targets": {"shoulder_pan": 2280}, "duration": 0.6, "pause": 0.1})
        steps.append(
            {
                "targets": {"shoulder_pan": 1800},
                "duration": 0.6,
                "pause": 0.0 if index == count - 1 else 0.1,
            }
        )
    return steps


def _nod_steps(count: int) -> list[dict]:
    steps: list[dict] = []
    for index in range(count):
        steps.append({"targets": {"wrist_flex": 2400}, "duration": 0.45, "pause": 0.1})
        steps.append(
            {
                "targets": {"wrist_flex": 2800},
                "duration": 0.45,
                "pause": 0.0 if index == count - 1 else 0.1,
            }
        )
    return steps


def _grip_open_close_steps() -> list[dict]:
    return [
        {
            "targets": {
                "shoulder_lift": 2000,
                "elbow_flex": 2240,
                "wrist_flex": 2620,
                "gripper": 2550,
            },
            "duration": 0.8,
            "pause": 0.15,
        },
        {"targets": {"gripper": 2200}, "duration": 0.6, "pause": 0.15},
        {"targets": {"gripper": 2550}, "duration": 0.6, "pause": 0.0},
    ]


def validate_semantic_action_result(result: dict) -> dict:
    execute = bool(result.get("execute", False))
    reason = str(result.get("reason", "")).strip()
    reply = str(result.get("reply", "")).strip()
    actions = result.get("actions", [])

    try:
        confidence = float(result.get("confidence", 0.9 if execute else 0.0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("LLM confidence must be numeric.") from exc

    if not execute:
        return {
            "execute": False,
            "action": "",
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": reason,
            "reply": reply,
            "sequence": [],
            "return_to_home": False,
        }

    if not isinstance(actions, list) or not actions:
        raise RuntimeError("Semantic actions result requires a non-empty actions list.")

    if len(actions) == 1 and isinstance(actions[0], dict) and str(actions[0].get("name", "")).strip() == "preset":
        preset = str(actions[0].get("preset", "")).strip()
        if preset not in SUPPORTED_ACTIONS - {"custom_sequence"}:
            raise RuntimeError(f"Unsupported semantic preset: {preset}")
        return {
            "execute": True,
            "action": preset,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": reason,
            "reply": reply,
            "sequence": [],
            "return_to_home": False,
        }

    sequence: list[dict] = []
    raised = False
    for raw_action in actions:
        if not isinstance(raw_action, dict):
            raise RuntimeError("Each semantic action must be an object.")
        name = str(raw_action.get("name", "")).strip()
        if name == "raise":
            if not raised:
                sequence.append(_prep_raise_step())
                raised = True
        elif name == "highest":
            sequence.extend(_highest_step_sequence())
            raised = True
        elif name == "wave":
            count = max(1, min(int(raw_action.get("count", 2)), 6))
            if not raised:
                sequence.append(_prep_raise_step())
                raised = True
            sequence.extend(_wave_steps(count))
        elif name == "nod":
            count = max(1, min(int(raw_action.get("count", 2)), 6))
            if not raised:
                sequence.append(_prep_raise_step())
                raised = True
            sequence.extend(_nod_steps(count))
        elif name == "grip":
            mode = str(raw_action.get("mode", "open_close")).strip()
            if mode != "open_close":
                raise RuntimeError(f"Unsupported grip mode: {mode}")
            sequence.extend(_grip_open_close_steps())
        elif name == "preset":
            raise RuntimeError("preset must be the only action when using semantic actions.")
        else:
            raise RuntimeError(f"Unsupported semantic action: {name}")

    return {
        "execute": True,
        "action": "custom_sequence",
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": reason,
        "reply": reply,
        "sequence": sequence,
        "return_to_home": False,
    }


CHINESE_NUMBER_MAP = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def interpret_robot_intent(
    user_text: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    fallback_model: str | None = None,
) -> dict:
    """Layer 1: ask the LLM for semantic robot intent only."""
    return call_llm(
        user_text=user_text,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        fallback_model=fallback_model,
    )


def parse_count_token(token: str) -> int | None:
    token = token.strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    if token == "十":
        return 10
    if len(token) == 2 and token[0] == "十" and token[1] in CHINESE_NUMBER_MAP:
        return 10 + CHINESE_NUMBER_MAP[token[1]]
    if len(token) == 2 and token[1] == "十" and token[0] in CHINESE_NUMBER_MAP:
        return CHINESE_NUMBER_MAP[token[0]] * 10
    if len(token) == 3 and token[1] == "十" and token[0] in CHINESE_NUMBER_MAP and token[2] in CHINESE_NUMBER_MAP:
        return CHINESE_NUMBER_MAP[token[0]] * 10 + CHINESE_NUMBER_MAP[token[2]]
    if token in CHINESE_NUMBER_MAP:
        return CHINESE_NUMBER_MAP[token]
    return None


def extract_requested_swing_count(user_text: str) -> int | None:
    match = re.search(r"(?:摆动|摆摆手|摆手|摆|摇动|摇摆|挥动|挥手|挥)([零一二两三四五六七八九十\d]+)下", user_text)
    if not match:
        return None
    count = parse_count_token(match.group(1))
    if count is None:
        return None
    return max(1, min(count, 6))


def extract_requested_nod_count(user_text: str) -> int | None:
    match = re.search(r"(?:点头|点)([零一二两三四五六七八九十\d]+)下", user_text)
    if not match:
        return None
    count = parse_count_token(match.group(1))
    if count is None:
        return None
    return max(1, min(count, 6))


def enforce_swing_count_if_needed(user_text: str, llm_result: dict) -> dict:
    if llm_result.get("action") != "custom_sequence":
        return llm_result

    requested_count = extract_requested_swing_count(user_text)
    if not requested_count:
        return llm_result

    sequence = llm_result.get("sequence", [])
    if len(sequence) < 2:
        return llm_result

    pan_steps = [step for step in sequence if "shoulder_pan" in step.get("targets", {})]
    if len(pan_steps) < 2:
        return llm_result

    pan_values = [step["targets"]["shoulder_pan"] for step in pan_steps]
    low = min(pan_values)
    high = max(pan_values)
    if low == high:
        return llm_result

    prep_step = dict(sequence[0])
    prep_targets = dict(prep_step.get("targets", {}))
    prep_targets.pop("shoulder_pan", None)
    prep_step["targets"] = prep_targets

    base_duration = pan_steps[0].get("duration", 0.5)
    base_pause = pan_steps[0].get("pause", 0.1)

    normalized_sequence = [prep_step]
    for index in range(requested_count):
        normalized_sequence.append(
            {
                "targets": {"shoulder_pan": high},
                "duration": base_duration,
                "pause": base_pause,
            }
        )
        normalized_sequence.append(
            {
                "targets": {"shoulder_pan": low},
                "duration": base_duration,
                "pause": 0.0 if index == requested_count - 1 else base_pause,
            }
        )

    updated = dict(llm_result)
    updated["sequence"] = normalized_sequence
    return updated


def enforce_nod_count_if_needed(user_text: str, llm_result: dict) -> dict:
    if llm_result.get("action") != "custom_sequence":
        return llm_result

    requested_count = extract_requested_nod_count(user_text)
    if not requested_count:
        return llm_result

    sequence = llm_result.get("sequence", [])
    if len(sequence) < 2:
        return llm_result

    nod_steps = [step for step in sequence if "wrist_flex" in step.get("targets", {})]
    if len(nod_steps) < 2:
        return llm_result

    nod_values = [step["targets"]["wrist_flex"] for step in nod_steps]
    low = min(nod_values)
    high = max(nod_values)
    if low == high:
        return llm_result

    prep_step = dict(sequence[0])
    prep_targets = dict(prep_step.get("targets", {}))
    prep_targets.pop("wrist_flex", None)
    prep_step["targets"] = prep_targets

    base_duration = nod_steps[0].get("duration", 0.45)
    base_pause = nod_steps[0].get("pause", 0.1)

    normalized_sequence = [prep_step]
    for index in range(requested_count):
        normalized_sequence.append(
            {
                "targets": {"wrist_flex": low},
                "duration": base_duration,
                "pause": base_pause,
            }
        )
        normalized_sequence.append(
            {
                "targets": {"wrist_flex": high},
                "duration": base_duration,
                "pause": 0.0 if index == requested_count - 1 else base_pause,
            }
        )

    updated = dict(llm_result)
    updated["sequence"] = normalized_sequence
    return updated


def apply_robot_rules(user_text: str, llm_result: dict) -> dict:
    """Layer 2: apply deterministic local rules to the semantic plan."""
    adjusted = dict(llm_result)
    adjusted = enforce_swing_count_if_needed(user_text, adjusted)
    adjusted = enforce_nod_count_if_needed(user_text, adjusted)
    return adjusted


def build_robot_execution_payload(llm_result: dict) -> tuple[str, str | dict]:
    """Layer 3 input: translate the rule-adjusted plan into a koch executor payload."""
    if llm_result["action"] == "custom_sequence":
        return (
            "payload",
            {
                "action": "custom_sequence",
                "sequence": llm_result["sequence"],
                "hold_position": False,
                "return_to_home": True,
                "power_down_after": True,
            },
        )
    return ("command", ACTION_TO_KOCH_COMMAND[llm_result["action"]])


def execute_robot_plan(llm_result: dict) -> bool:
    """Layer 3: execute the safe plan through koch-skill only."""
    execution_kind, execution_value = build_robot_execution_payload(llm_result)
    if execution_kind == "payload":
        assert isinstance(execution_value, dict)
        print(
            f"[INFO] Sending custom sequence to koch-skill: {json.dumps(execution_value, ensure_ascii=False)}",
            file=sys.stderr,
        )
        return execute_koch_payload(execution_value)

    assert isinstance(execution_value, str)
    print(f"[INFO] Sending to koch-skill: {shlex.quote(execution_value)}", file=sys.stderr)
    return execute_koch_command(execution_value)


def call_llm(
    user_text: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    fallback_model: str | None = None,
) -> dict:
    if not base_url:
        raise RuntimeError("LLM base URL is not configured.")
    if not api_key:
        raise RuntimeError("LLM API key is not configured.")
    if not model:
        raise RuntimeError("LLM model is not configured.")

    endpoint = f"{normalize_base_url(base_url)}/chat/completions"
    def request_once(model_name: str) -> str:
        payload = {
            "model": model_name,
            "temperature": temperature_for_model(model_name),
            "max_tokens": 1200 if (model_name or "").strip().lower() == "kimi-k2.5" else 600,
            "messages": [
                {"role": "system", "content": system_prompt_for_model(model_name)},
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
        raw_response = post_json_request(request, timeout=timeout, retries=1)
        body = json.loads(raw_response)
        content = extract_message_content(body)
        if not content:
            raise RuntimeError(f"LLM returned empty content for model {model_name}")
        return content

    fallback_model = (
        fallback_model
        or os.getenv("KIMI_FALLBACK_MODEL")
        or os.getenv("MOONSHOT_FALLBACK_MODEL")
        or DEFAULT_KIMI_FALLBACK_MODEL
    )
    active_model = model
    try:
        content = request_once(active_model)
    except Exception as exc:
        if should_retry_with_fallback_model(exc, model=active_model, fallback_model=fallback_model):
            print(
                f"[WARN] LLM model {active_model} unavailable; retrying with fallback model {fallback_model}",
                file=sys.stderr,
            )
            active_model = fallback_model
            content = request_once(active_model)
        else:
            raise

    try:
        return validate_llm_result(parse_json_object(content))
    except Exception as first_error:
        print(f"[WARN] Primary LLM JSON parsing failed, retrying repair: {first_error}", file=sys.stderr)
        try:
            repaired = repair_json_with_llm(
                raw_text=content,
                base_url=base_url,
                api_key=api_key,
                model=active_model,
                timeout=timeout,
            )
            return validate_llm_result(repaired)
        except Exception as second_error:
            if should_retry_parse_failure_with_fallback(second_error, model=active_model, fallback_model=fallback_model):
                print(
                    f"[WARN] LLM model {active_model} produced unusable JSON; retrying with fallback model {fallback_model}",
                    file=sys.stderr,
                )
                fallback_content = request_once(fallback_model)
                return validate_llm_result(parse_json_object(fallback_content))
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
        semantic_result = interpret_robot_intent(
            user_text=text,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            model=args.llm_model,
            timeout=args.llm_timeout,
            fallback_model=args.llm_fallback_model,
        )
        llm_result = apply_robot_rules(text, semantic_result)
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

        ok = execute_robot_plan(llm_result)

        if not ok:
            fail("koch-skill rejected or failed to execute the LLM action.")

        print(f"[KOCH] command executed: {llm_result['action']}")
        return 0
    except KeyboardInterrupt:
        print("\n[WARN] LLM bridge interrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
