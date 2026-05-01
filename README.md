# Qwen3-TTS Server

A production-ready FastAPI server for Text-to-Speech synthesis using Qwen3-TTS models, optimized for LXC containers with GPU passthrough and support for the latest NVIDIA GPU architectures.

## 🚀 Quick Start

```bash
sudo bash system_install.sh
```

See [QUICK_START.md](QUICK_START.md) for detailed instructions.

## ✨ Features

- 🎯 **Three TTS Modes**: Voice Design, Custom Voice, and Voice Clone
- 🧠 **Multiple Model Sizes**: 0.6B and 1.7B parameter models
- 🔄 **Smart Memory Management**: Automatic model loading/unloading
- 🐳 **LXC Optimized**: Intelligent CUDA detection for GPU passthrough
- ⚡ **Modern GPU Support**: Auto-detects RTX 50/40/30-series (Blackwell/Ada/Ampere)
- 📊 **Production Ready**: Systemd service, logging, and monitoring

## 🎮 Supported GPU Architectures

| GPU Series | Architecture | Compute Cap | PyTorch CUDA |
|------------|--------------|-------------|--------------|
| RTX 50-series | Blackwell | sm_120 | 12.4 ✅ |
| RTX 40-series | Ada Lovelace | sm_89 | 12.4 ✅ |
| H100 | Hopper | sm_90 | 12.4 ✅ |
| RTX 30-series | Ampere | sm_86 | 11.8 ✅ |
| RTX 20-series | Turing | sm_75 | 11.8 ✅ |
| GTX 10-series | Pascal | sm_61 | 11.8 ✅ |

## 🔧 System Requirements

- **OS**: Debian 11+ (Trixie/Bookworm)
- **Container**: LXC with GPU passthrough (or bare-metal)
- **GPU**: NVIDIA GPU with CUDA support
- **RAM**: 8GB minimum (16GB recommended for 1.7B models)
- **Disk**: 10GB free space for models
- **Driver**: NVIDIA driver installed (for LXC: use `--no-kernel-module`)

## 📦 What's Included

- `server.py` - FastAPI application with TTS endpoints
- `config.yaml` - Server and model configuration
- `requirements.txt` - Python dependencies
- `system_install.sh` - Automated installation script
- `LXC_GPU_INSTALLATION.md` - Detailed installation guide
- `QUICK_START.md` - Quick reference guide

## 🎯 API Endpoints

All endpoints listen on `http://localhost:8000` (or your configured host/port).

---

### POST /preload

**Pre-load models and cache voice prompts** to eliminate loading latency on the actual TTS request. Recommended for production use.

**Parameters:**

| Parameter | Type | Required | Description | Valid Values |
|-----------|------|----------|-------------|--------------|
| `model_type` | string | Yes | Type of TTS model | `"design"`, `"custom"`, `"clone"` |
| `model_size` | string | No | Model size (ignored for design) | `"0.6b"`, `"1.7b"` (default: `"1.7b"`) |
| `reference_text` | string | No | Transcript of reference audio | Any text matching the audio |
| `reference_audio` | file | No | Reference WAV file for cloning | WAV audio file |
| `warmup` | boolean | No | Run GPU kernel warm-up | `true` (default), `false` |
| `warmup_language` | string | No | Language for warm-up pass | `"en"` (default), `"zh"`, etc. |

**Example Request:**

```bash
# Preload voice-clone model with reference audio
curl -X POST http://localhost:8000/preload \
  -F "model_type=clone" \
  -F "model_size=1.7b" \
  -F "reference_text=This is my reference voice sample" \
  -F "reference_audio=@reference.wav" \
  -F "warmup=true"
```

**Response (202 Accepted):**
```json
{
  "status": "loading",
  "model": "1.7b-clone",
  "warmup": true,
  "voice_prompt_preload": true
}
```

---

### GET /status

**Check server status**, model state, cache information, and default reference details.

**Example Request:**

```bash
curl http://localhost:8000/status
```

**Response:**
```json
{
  "device": "cuda",
  "model_loaded": true,
  "model_name": "1.7b-clone",
  "preload_state": "ready",
  "requests_in_flight": 0,
  "voice_prompts_cached": 2,
  "default_references": {
    "1.7b-clone": {
      "id": "a3f7b2c1",
      "reference_text_preview": "This is my voice speaking clearly"
    }
  }
}
```

---

### POST /voice-design

**Generate speech from a text description** of desired voice characteristics. Only the 1.7B model is available for voice design.

**Parameters:**

| Parameter | Type | Required | Description | Valid Values |
|-----------|------|----------|-------------|--------------|
| `target_text` | string | Yes | Text to synthesize | Any text (max ~1000 chars recommended) |
| `language` | string | Yes | Language code | `"en"`, `"zh"`, `"ja"`, `"fr"`, `"de"`, `"es"`, `"pt"`, `"ar"`, `"hi"` |
| `instruct` | string | Yes | Voice characteristics description | e.g., `"cheerful young female voice"`, `"deep professional male voice"` |

**Example Request:**

```bash
curl -X POST http://localhost:8000/voice-design \
  -F "target_text=Hello! Welcome to our text-to-speech demonstration." \
  -F "language=en" \
  -F "instruct=cheerful young female voice with clear pronunciation and friendly tone" \
  -o output_design.wav
```

**Response:** WAV audio file

**Instruct Examples:**
- `"cheerful young female voice with energetic tone"`
- `"deep authoritative male voice, professional broadcaster"`
- `"soft gentle female voice, calm and soothing"`
- `"enthusiastic male voice with upbeat personality"`

---

### POST /custom-voice

**Generate speech with predefined voice characteristics** from the model's built-in speaker set.

**Parameters:**

| Parameter | Type | Required | Description | Valid Values |
|-----------|------|----------|-------------|--------------|
| `model_size` | string | Yes | Model size | `"0.6b"`, `"1.7b"` |
| `language` | string | Yes | Language code | `"en"`, `"zh"`, `"ja"`, `"fr"`, `"de"`, `"es"`, `"pt"`, `"ar"`, `"hi"` |
| `speaker` | string | Yes | Speaker identifier | `"male_1"`, `"female_1"`, etc. (model-dependent) |
| `instruct` | string | Yes | Voice modification instructions | e.g., `"speak slowly and clearly"`, `"excited and energetic"` |
| `target_text` | string | Yes | Text to synthesize | Any text (max ~1000 chars recommended) |

**Example Request:**

```bash
curl -X POST http://localhost:8000/custom-voice \
  -F "model_size=1.7b" \
  -F "language=en" \
  -F "speaker=female_1" \
  -F "instruct=speak with clear pronunciation and moderate pace" \
  -F "target_text=This is a demonstration of custom voice synthesis." \
  -o output_custom.wav
```

**Response:** WAV audio file

**Instruct Examples:**
- `"speak slowly with emphasis on clarity"`
- `"energetic and enthusiastic delivery"`
- `"calm and professional tone"`
- `"dramatic reading with emotion"`

---

### POST /voice-clone/set-reference

**Set a default reference voice** that persists across multiple voice-clone requests. This eliminates the need to upload reference audio on every request.

**Parameters:**

| Parameter | Type | Required | Description | Valid Values |
|-----------|------|----------|-------------|--------------|
| `model_size` | string | Yes | Model size | `"0.6b"`, `"1.7b"` |
| `reference_text` | string | Yes | Transcript of the reference audio | Exact words spoken in reference audio |
| `reference_audio` | file | Yes | Reference audio file | WAV format, 3-10 seconds, clean audio recommended |

**Example Request:**

```bash
# Set default reference once
curl -X POST http://localhost:8000/voice-clone/set-reference \
  -F "model_size=1.7b" \
  -F "reference_text=This is my voice speaking clearly" \
  -F "reference_audio=@/path/to/reference.wav"
```

**Response:**
```json
{
  "status": "success",
  "model": "1.7b-clone",
  "reference_id": "a3f7b2c1",
  "message": "Default reference set for 1.7b-clone model"
}
```

**Benefits:**
- Set once, use many times without re-uploading
- Per-model defaults (0.6b and 1.7b can have different defaults)
- Persists until server restart or manually cleared
- Ideal for production with consistent voice identity

---

### POST /voice-clone

**Clone a voice and generate speech.** Reference audio/text can be provided inline or omitted to use the default reference set via `/voice-clone/set-reference`.

**Parameters:**

| Parameter | Type | Required | Description | Valid Values |
|-----------|------|----------|-------------|--------------|
| `model_size` | string | Yes | Model size | `"0.6b"`, `"1.7b"` |
| `target_text` | string | Yes | Text to synthesize in cloned voice | Any text (max ~1000 chars recommended) |
| `language` | string | Yes | Language code | `"en"`, `"zh"`, `"ja"`, `"fr"`, `"de"`, `"es"`, `"pt"`, `"ar"`, `"hi"` |
| `reference_text` | string | **Optional** | Transcript of the reference audio | Exact words spoken in reference audio |
| `reference_audio` | file | **Optional** | Reference audio file | WAV format, 3-10 seconds, clean audio recommended |

**Behavior:**
- **Both** `reference_audio` and `reference_text` provided → uses them (overrides default)
- **Neither** provided → uses default reference (must be set first via `/voice-clone/set-reference`)
- **Only one** provided → returns error (must provide both or neither)

**Example Request (with inline reference):**

```bash
curl -X POST http://localhost:8000/voice-clone \
  -F "model_size=1.7b" \
  -F "target_text=Hello, this is a test of the voice cloning system." \
  -F "language=en" \
  -F "reference_text=This is my voice speaking clearly" \
  -F "reference_audio=@/path/to/reference.wav" \
  -o output_clone.wav
```

**Example Request (using default reference):**

```bash
# First, set the default reference (once)
curl -X POST http://localhost:8000/voice-clone/set-reference \
  -F "model_size=1.7b" \
  -F "reference_text=This is my voice speaking clearly" \
  -F "reference_audio=@/path/to/reference.wav"

# Then use voice-clone WITHOUT reference parameters (many times)
curl -X POST http://localhost:8000/voice-clone \
  -F "model_size=1.7b" \
  -F "target_text=Hello, this is message one." \
  -F "language=en" \
  -o message1.wav

curl -X POST http://localhost:8000/voice-clone \
  -F "model_size=1.7b" \
  -F "target_text=Hello, this is message two." \
  -F "language=en" \
  -o message2.wav
```

**Response:** WAV audio file

**Tips:**
- Use 3-10 seconds of clean reference audio
- Reference text must match what's spoken in the audio
- Higher quality reference audio = better cloning results
- Use `/voice-clone/set-reference` for production workflows with consistent voices
- Use inline reference for one-off cloning or testing different voices

---

### DELETE /voice-clone/cache

**Clear the voice-clone prompt cache**. Optionally clear default references as well.

**Parameters:**

| Parameter | Type | Required | Description | Valid Values |
|-----------|------|----------|-------------|--------------|
| `clear_defaults` | boolean | No | Also clear default references | `true`, `false` (default) |

**Example Request (clear cache only):**

```bash
curl -X DELETE http://localhost:8000/voice-clone/cache
```

**Response:**
```json
{
  "cache_cleared": 3,
  "defaults_cleared": 0
}
```

**Example Request (clear cache and defaults):**

```bash
curl -X DELETE "http://localhost:8000/voice-clone/cache?clear_defaults=true"
```

**Response:**
```json
{
  "cache_cleared": 3,
  "defaults_cleared": 2
}
```

**Use Cases:**
- Clear cache only: When you want to force re-encoding but keep default references
- Clear both: When changing reference audio and want to reset everything

---

## 📝 API Usage Notes

### Language Codes
Supported languages include:
- `en` - English
- `zh` - Chinese (Mandarin)
- `ja` - Japanese
- `fr` - French
- `de` - German
- `es` - Spanish
- `pt` - Portuguese
- `ar` - Arabic
- `hi` - Hindi

### Audio Requirements
- **Format**: WAV (recommended)
- **Sample Rate**: 12kHz output (automatically handled)
- **Reference Audio**: 3-10 seconds of clean speech for best cloning results

### Performance Optimization
1. Use `/preload` before making actual TTS requests
2. Keep model sizes consistent (avoid switching between 0.6b/1.7b)
3. Voice-clone prompts are automatically cached (keyed by audio+text hash)
4. Models auto-unload after 60 seconds of inactivity (configurable)

### Error Handling
All endpoints return:
- `200 OK` - Success (with audio file or JSON response)
- `202 Accepted` - Preload accepted (processing in background)
- `400 Bad Request` - Invalid parameters
- `500 Internal Server Error` - Processing error (check logs)

## 🐛 Troubleshooting

### LXC Container + GPU Passthrough Issues

**Fixed Issues:**
- ✅ SHA1 signature errors on Debian Trixie
- ✅ `libtinfo5` dependency conflicts
- ✅ Redundant CUDA toolkit installation
- ✅ RTX 50-series (Blackwell) compatibility
- ✅ `sm_120 is not compatible` errors

**See**: [LXC_GPU_INSTALLATION.md](LXC_GPU_INSTALLATION.md) for complete troubleshooting guide.

### Common Issues

**GPU not detected:**
```bash
nvidia-smi
ldconfig -p | grep libcuda
```

**Server won't start:**
```bash
systemctl status qwen-tts
journalctl -u qwen-tts -n 50
```

**Out of memory:**
- Use 0.6B models instead of 1.7B
- Check VRAM usage: `nvidia-smi`
- Increase container RAM allocation

## 📊 Memory Management

The server implements intelligent memory management:

- **Lazy Loading**: Models load on first request
- **Auto-Unload**: Models unload after 180 seconds of inactivity
- **Model Switching**: Seamlessly switches between model sizes
- **VRAM Usage**: ~8GB for 1.7B models, ~3GB for 0.6B models

## 🔍 Monitoring

```bash
# Service status
systemctl status qwen-tts

# Real-time logs
journalctl -u qwen-tts -f

# Application logs
tail -f /opt/qwen-tts-server/logs/server.log

# GPU monitoring
nvidia-smi -l 1
```

## ⚙️ Configuration

Edit `/opt/qwen-tts-server/config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  unload_timeout: 180
  request_timeout: 7200

models:
  "1.7b-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
  "1.7b-custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
  "1.7b-clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
  "0.6b-custom": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
  "0.6b-clone": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

logging:
  level: "INFO"
  rotation: "10 MB"
  retention: "10 days"
  file: "/opt/qwen-tts-server/logs/server.log"
```

After changes: `systemctl restart qwen-tts`

## 🌟 What Makes This Special?

### Intelligent Installation
- Auto-detects GPU compute capability
- Installs correct PyTorch version (CUDA 11.8 or 12.4)
- Skips CUDA toolkit on LXC containers with GPU passthrough
- Verifies GPU accessibility post-installation

### Production Features
- Systemd service with auto-restart
- Comprehensive logging with rotation
- Request timeout handling
- Clean error handling and reporting

### LXC Optimization
- No SHA1 signature issues
- No dependency conflicts on Debian Trixie
- 5-10 minute installation (vs 30+ minutes)
- Smaller disk footprint

## 📝 License

This project uses the Qwen3-TTS models. Please refer to the [Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS) for model licensing information.

## 🤝 Contributing

Contributions are welcome! This project is specifically optimized for:
- LXC containers with GPU passthrough
- Latest NVIDIA GPU architectures (Blackwell, Ada, Hopper)
- Debian-based systems (Trixie, Bookworm)

## 📚 Documentation

- [Quick Start Guide](QUICK_START.md) - Get started in 5 minutes
- [LXC GPU Installation](LXC_GPU_INSTALLATION.md) - Detailed installation guide for LXC
- [Qwen3-TTS Official Docs](https://github.com/QwenLM/Qwen3-TTS) - Model documentation

## 🔗 Links

- **GitHub**: [derLars/Qwen-TTS-LXC-deploy](https://github.com/derLars/Qwen-TTS-LXC-deploy)
- **Qwen3-TTS**: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

---

**Status**: Production Ready  
**Tested On**: Debian Trixie LXC with RTX 5060 Ti (Blackwell)  
**Last Updated**: February 3, 2026
