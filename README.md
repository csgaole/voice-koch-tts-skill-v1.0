# voice-koch-tts-skill

OpenClaw skill for voice-driven Koch robot control with TTS playback.

This package wires together:

- microphone recording
- local ASR
- LLM intent understanding
- Koch robot execution
- TTS voice playback with an energetic childlike style

## Repository Layout

```text
voice-koch-tts-skill/
├── openclaw_skills/
│   └── voice-koch-tts-skill/
│       ├── SKILL.md
│       └── run_voice_control_tts.sh
└── qwen_asr_stt/
    ├── stt_koch_llm_tts_bridge.py
    ├── tts_edge.py
    └── ...
```

## Notes

- TTS uses `edge-tts` and local playback with `ffplay`
- Voice style is configured as lively and childlike, but not a direct copyrighted character imitation
- Free-form custom robot actions return to `home` and power down after execution
