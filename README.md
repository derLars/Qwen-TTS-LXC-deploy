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

### 1. Voice Design
Create voices from text descriptions.

```bash
POST /voice-design
```

### 2. Custom Voice
Generate speech with predefined voice characteristics.

```bash
POST /custom-voice
```

### 3. Voice Clone
Clone voices from reference audio samples.

```bash
POST /voice-clone
```

See [QUICK_START.md](QUICK_START.md) for API usage examples.

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
