# qwen_asr_stt

本项目在 Ubuntu 上优先使用 `Qwen/Qwen3-ASR-0.6B` 做本地语音识别（STT），并在当前环境不满足 Qwen 运行条件时自动降级到 `openai/whisper-small`，目标是优先保证能跑通。

## 环境检查结果

在当前机器上已确认：

- `python3`: `3.12.3`
- `pip`: `24.0`
- `ffmpeg`: 已安装，可用
- `nvidia-smi`: 当前返回 `Failed to initialize NVML: Unknown Error`
- `torch cuda`: 当前系统未安装 `torch`，因此还无法直接确认 CUDA 推理可用性

这意味着：当前环境还没有装好 Qwen ASR 推理依赖，所以脚本会先尝试 Qwen，若依赖缺失或模型加载失败，再降级到 Whisper。

## 文件说明

- `requirements.txt`: Python 依赖
- `stt_qwen.py`: STT 主程序
- `stt_qwen_mic.py`: 麦克风录音后立即转写
- `stt_koch_bridge.py`: 录音转写后直接调用 `koch-skill`
- `stt_koch_llm_bridge.py`: 录音转写后先交给 `qwen3.5-plus` 做意图理解，再调用 `koch-skill`

## 安装

建议先创建独立虚拟环境：

```bash
cd qwen_asr_stt
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

如果你只是想先在当前机器上把 STT 跑通，优先安装 CPU 版 PyTorch 更稳：

```bash
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python3 -m pip install qwen-asr transformers soundfile
```

如果你需要 GPU 版 PyTorch，优先按你的 CUDA 版本安装官方轮子，再补装其余依赖。例如 CUDA 12.1：

```bash
python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch
python3 -m pip install qwen-asr soundfile ffmpeg-python
```

## 运行

```bash
cd qwen_asr_stt
python3 stt_qwen.py --audio test.wav
```

也可以指定任意音频文件：

```bash
python3 stt_qwen.py --audio /path/to/your_audio.mp3
```

只使用本地缓存模型，不联网：

```bash
python3 stt_qwen.py --audio test.wav --local-only
```

## 麦克风直接测试

默认录 5 秒，然后立即转写：

```bash
cd qwen_asr_stt
source .venv/bin/activate
python3 stt_qwen_mic.py
```

录更久一些：

```bash
python3 stt_qwen_mic.py --duration 8
```

指定 ALSA 设备：

```bash
python3 stt_qwen_mic.py --device hw:0,0 --duration 5
```

只使用本地缓存模型，不联网：

```bash
python3 stt_qwen_mic.py --local-only
```

保留录下来的 wav 文件：

```bash
python3 stt_qwen_mic.py --keep-audio --output mic_test.wav
```

## 接到 koch-skill

标准启动命令：

```bash
bash /home/legao/.openclaw/skills/qwen-asr-stt-skill/run_koch.sh --local-only
```

项目内直接运行：

```bash
cd qwen_asr_stt
source .venv/bin/activate
python3 stt_koch_bridge.py --local-only
```

当前状态：

- 语音识别 ASR 链路已完成
- `koch-skill` 机械臂控制链路已完成
- `ASR -> koch-skill -> 从臂动作` 已在你的本机终端环境实测成功
- 当前 `koch-skill` 已收口为“纯机械臂控制”，STT 只在桥接层负责录音与转写

只看识别结果，不真的控制机械臂：

```bash
python3 stt_koch_bridge.py --local-only --dry-run
```

## 接到 LLM 意图理解

项目内直接运行：

```bash
cd qwen_asr_stt
source .venv/bin/activate
python3 stt_koch_llm_bridge.py --local-only
```

只做 `ASR -> LLM`，不真的控制机械臂：

```bash
python3 stt_koch_llm_bridge.py --local-only --dry-run
```

直接用已有音频文件测试：

```bash
python3 stt_koch_llm_bridge.py --audio /path/to/test.wav --local-only --dry-run
```

默认 LLM：

- 模型：`qwen3.5-plus`
- 接口：OpenAI-compatible `/chat/completions`
- 默认 `base_url`：优先读取 `.openclaw/openclaw0309.json` 里的 `vllm.baseUrl`
- 默认 `api_key`：优先读取 OpenClaw 配置里 `vllm.apiKey` 指向的环境变量

也可以手动指定：

```bash
python3 stt_koch_llm_bridge.py \
  --local-only \
  --llm-base-url https://coding.dashscope.aliyuncs.com/v1 \
  --llm-model qwen3.5-plus \
  --llm-api-key "$VLLM_API_KEY"
```

### LLM 输出格式

桥接层要求 LLM 严格输出一个 JSON 对象：

```json
{
  "execute": true,
  "action": "dance",
  "confidence": 0.93,
  "reason": "用户明确要求机械臂跳舞。",
  "reply": "执行跳舞动作。"
}
```

约束：

- `action` 只允许：`home` / `ready` / `dance` / `status` / `power_down`
- `execute=false` 时，桥接层不会调用 `koch-skill`
- `confidence` 低于 `--min-confidence` 时，桥接层会拒绝执行
- `koch-skill` 保持“纯机械臂控制”，LLM 只存在于桥接层

适合的口令示例：

```text
跳舞
回到 home 位置
准备抓取
状态
下电
```

## 工作方式

脚本会按以下顺序执行：

1. 检查 `ffmpeg` 是否存在
2. 把输入音频转成 `16kHz / 单声道 / WAV`
3. 自动检测 GPU / CPU
4. 优先加载 `Qwen/Qwen3-ASR-0.6B`
5. 若 Qwen 在当前环境不可用，则自动回退到 `openai/whisper-small`
6. 在终端直接输出最终识别文本

模型加载策略：

1. 先检查你传入的是不是本地目录
2. 不是本地目录时，优先查 Hugging Face 本地缓存
3. 本地缓存存在则直接加载，不访问网络
4. 仅当缓存不存在且没加 `--local-only` 时，才尝试联网下载
5. 若网络失败，会直接提示“模型未缓存”或“下载失败”，不再依赖 `from_pretrained()` 自己刷重试日志

## 模型说明

### 主模型

- 模型名：`Qwen/Qwen3-ASR-0.6B`
- 用途：多语言自动语音识别
- 优先级：默认优先使用

官方推荐方式是安装 `qwen-asr` 包后，通过 `Qwen3ASRModel.from_pretrained(...)` 加载模型。

### 替代方案

当前环境下，`Qwen/Qwen3-ASR-0.6B` 不可直接运行的原因是：

- 机器里还没有安装 `torch`
- 也没有安装 `qwen-asr`
- `nvidia-smi` 当前异常，GPU 状态不可靠
- 默认 `pip install torch` 会倾向下载体积很大的 CUDA 轮子，不适合作为当前机器上的首选验证路径

因此脚本内置了替代实现：

- 备用模型：`openai/whisper-small`
- 触发条件：Qwen 依赖缺失、模型下载失败、模型初始化失败、推理时报错

这样可以先把本地 STT 跑起来，再回头补齐 Qwen 环境。

麦克风版本使用系统自带的 `arecord` 做录音，不额外引入 Python 录音库，目的是减少环境问题。

`stt_koch_bridge.py` 不直接依赖 `qwen_asr_stt` 虚拟环境里的机器人库，而是调用：

```bash
/home/legao/miniforge3/envs/lerobot/bin/python
```

去执行本机现有的 `koch-skill` 实现。

## 常见报错排查

### 1. `Audio file not found`

说明 `--audio` 指向的文件不存在。确认路径正确。

### 2. `ffmpeg is not installed or not in PATH`

安装 ffmpeg：

```bash
sudo apt update
sudo apt install -y ffmpeg
```

### 3. `Microphone recording failed`

先检查麦克风设备列表：

```bash
arecord -l
```

如果默认设备不对，显式指定：

```bash
python3 stt_qwen_mic.py --device hw:0,0
```

### 4. `PyTorch is not installed`

先安装 PyTorch，再安装其他依赖：

```bash
python3 -m pip install torch
python3 -m pip install -r requirements.txt
```

如果你要用 GPU，请改为安装与你 CUDA 匹配的 PyTorch 版本。

### 5. `qwen-asr is not installed`

```bash
python3 -m pip install qwen-asr
```

### 6. `Both Qwen and fallback ASR failed`

通常是以下几类原因：

- 没有网络，模型无法从 Hugging Face 下载
- `torch` / `transformers` / `qwen-asr` 缺失
- 音频文件损坏
- 机器内存或显存不足

建议按下面顺序排查：

1. 确认 `python3 -m pip install -r requirements.txt` 成功
2. 确认能访问 Hugging Face
3. 先用一个很短的 `wav` 文件测试
4. 如果 GPU 不稳定，先走 CPU

如果你不想依赖网络，可以先手动缓存模型，然后使用 `--local-only`：

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-ASR-0.6B")
snapshot_download("openai/whisper-small")
PY
python3 stt_qwen.py --audio test.wav --local-only
```

### 7. `Model not cached locally`

这表示你启用了 `--local-only`，但本地缓存里还没有对应模型。

先手动下载模型：

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-ASR-0.6B")
PY
```

如果还需要 Whisper 回退，也建议一起缓存：

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("openai/whisper-small")
PY
```

### 8. `Failed to initialize NVML: Unknown Error`

这表示 `nvidia-smi` 当前拿不到有效 GPU 状态。脚本会自动回退到 CPU，不会因为这个报错直接退出。

## 备注

如果你后面确定要完全锁定为 Qwen 方案，可以把 `stt_qwen.py` 里的 Whisper 回退逻辑去掉，只保留 `Qwen/Qwen3-ASR-0.6B`。
