---
name: voice-koch-tts-skill
description: 语音控制 Koch 机械臂并播放 TTS 回复的组合技能。适用于 OpenClaw 需要通过麦克风录音做本地 ASR、LLM 意图理解、Koch 机械臂控制，并在执行前后播放活泼童声风格语音反馈的场景。
---

# voice-koch-tts-skill

把“麦克风录音 + 本地 ASR + LLM 意图理解 + Koch 机械臂控制 + TTS 语音播放”收口成一个稳定入口。

## 默认入口

```bash
bash /home/legao/.openclaw/skills/voice-koch-tts-skill/run_voice_control_tts.sh --local-only
```

只验证识别和意图理解，不执行机械臂：

```bash
bash /home/legao/.openclaw/skills/voice-koch-tts-skill/run_voice_control_tts.sh --local-only --dry-run
```

关闭 TTS 播放：

```bash
bash /home/legao/.openclaw/skills/voice-koch-tts-skill/run_voice_control_tts.sh --local-only --mute-tts
```

## 行为约定

- 调用工作区项目 `/home/legao/openclaw_workspace/qwen_asr_stt`
- 入口脚本实际执行 `stt_koch_llm_tts_bridge.py`
- TTS 使用 `edge-tts` + `ffplay`
- 语音风格为“活泼、明快、童声感”，不做版权角色精确模仿

## 关键文件

- `run_voice_control_tts.sh`: 语音控制机械臂并播放语音反馈主入口
