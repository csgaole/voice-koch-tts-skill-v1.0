#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/legao/openclaw_workspace/qwen_asr_stt"
VENV_PY="$PROJECT_DIR/.venv/bin/python"
SCRIPT="$PROJECT_DIR/stt_koch_llm_tts_bridge.py"
ENV_FILE="$PROJECT_DIR/.env"

if [[ ! -x "$VENV_PY" ]]; then
  echo "[ERROR] Python venv not found: $VENV_PY" >&2
  exit 1
fi

if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERROR] Bridge script not found: $SCRIPT" >&2
  exit 1
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

export KIMI_LLM_BASE_URL="${KIMI_LLM_BASE_URL:-${MOONSHOT_BASE_URL:-https://api.moonshot.cn/v1}}"
export KIMI_LLM_MODEL="${KIMI_LLM_MODEL:-${MOONSHOT_MODEL:-kimi-k2.5}}"

exec "$VENV_PY" "$SCRIPT" "$@"
