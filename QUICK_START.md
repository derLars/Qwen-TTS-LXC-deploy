# Qwen3-TTS Server - Quick Start Guide

## For LXC Container Installation (RTX 5060 Ti / GPU Passthrough)

### Prerequisites
✅ Debian-based LXC container (Trixie/Bookworm)  
✅ GPU passthrough configured in LXC  
✅ NVIDIA driver installed in container (using `--no-kernel-module`)  
✅ `nvidia-smi` command working  

### Installation (One Command!)

```bash
sudo bash system_install.sh
```

**That's it!** The script will:
1. Detect your RTX 5060 Ti and compute capability (sm_120)
2. Skip CUDA toolkit installation (uses host's CUDA via passthrough)
3. Install PyTorch with CUDA 12.4 support automatically
4. Set up the TTS server as a systemd service
5. Verify GPU accessibility

### Expected Installation Time
- **5-10 minutes** (with GPU passthrough, no CUDA toolkit download)
- **30+ minutes** (bare-metal with full CUDA installation)

### Verify Installation

```bash
# Check service status
systemctl status qwen-tts

# View GPU verification
journalctl -u qwen-tts | grep "GPU Verification" -A 10

# Test PyTorch CUDA access
/opt/qwen-tts-server/venv/bin/python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Check Server Logs

```bash
# Real-time logs
journalctl -u qwen-tts -f

# Last 50 lines
journalctl -u qwen-tts -n 50

# Application logs
tail -f /opt/qwen-tts-server/logs/server.log
```

### Test API (Example - Voice Clone)

```bash
curl -X POST http://localhost:8000/voice-clone \
  -F "model_size=1.7b" \
  -F "target_text=Hello, this is a test of the voice cloning system." \
  -F "language=en" \
  -F "reference_text=Sample reference text" \
  -F "reference_audio=@/path/to/reference.wav" \
  -o output.wav
```

### Common Commands

```bash
# Start service
systemctl start qwen-tts

# Stop service
systemctl stop qwen-tts

# Restart service
systemctl restart qwen-tts

# View status
systemctl status qwen-tts

# Disable auto-start
systemctl disable qwen-tts

# Enable auto-start
systemctl enable qwen-tts
```

## API Endpoints

All endpoints listen on `http://0.0.0.0:8000`

### 1. Voice Design
**Endpoint:** `POST /voice-design`

**Parameters:**
- `target_text` (string): Text to synthesize
- `language` (string): Language code (e.g., "en", "zh")
- `instruct` (string): Voice characteristics description

### 2. Custom Voice
**Endpoint:** `POST /custom-voice`

**Parameters:**
- `model_size` (string): "1.7b" or "0.6b"
- `language` (string): Language code
- `speaker` (string): Speaker identifier
- `instruct` (string): Voice characteristics
- `target_text` (string): Text to synthesize

### 3. Voice Clone
**Endpoint:** `POST /voice-clone`

**Parameters:**
- `model_size` (string): "1.7b" or "0.6b"
- `target_text` (string): Text to synthesize
- `language` (string): Language code
- `reference_text` (string): Text in reference audio
- `reference_audio` (file): Audio file for voice cloning

## What's Special About This Installation?

### Intelligent GPU Detection
The script automatically detects:
- **RTX 50-series (Blackwell - sm_120)** → Installs PyTorch CUDA 12.4
- **RTX 40-series (Ada - sm_89)** → Installs PyTorch CUDA 12.4
- **RTX 30-series (Ampere - sm_86)** → Installs PyTorch CUDA 11.8
- **Older GPUs** → Installs appropriate PyTorch version

### LXC Container Optimization
- Skips CUDA toolkit when libraries are already available via passthrough
- Much faster installation (no 3GB CUDA download)
- No SHA1 signature or dependency issues on Debian Trixie
- Smaller disk footprint

### Memory Management
- Models load on-demand (lazy loading)
- Automatic unload after 180 seconds of inactivity
- Efficient model switching
- ~8GB VRAM for 1.7B models, ~3GB for 0.6B models

## Troubleshooting

### "CUDA error: no kernel image is available"
**Solution:** This is fixed! Re-run `system_install.sh` to install PyTorch CUDA 12.4

### Server won't start
```bash
journalctl -u qwen-tts -n 100
```
Check for port conflicts or permission issues.

### GPU not detected
```bash
nvidia-smi
ldconfig -p | grep libcuda
```
Verify GPU passthrough is working.

### Out of memory
- Use 0.6B models instead of 1.7B models
- Increase container RAM allocation
- Check for other GPU processes: `nvidia-smi`

## Configuration

Edit `/opt/qwen-tts-server/config.yaml` to customize:
- Server port (default: 8000)
- Model unload timeout (default: 180s)
- Logging levels
- Available models

After changes:
```bash
systemctl restart qwen-tts
```

## Performance Tips

1. **Use smaller models** (0.6B) if VRAM is limited
2. **Keep requests frequent** to avoid model reload delays
3. **Monitor VRAM usage**: `nvidia-smi -l 1`
4. **Check logs** for memory manager activity

## Support

- **Installation Issues**: See `LXC_GPU_INSTALLATION.md`
- **GPU Compatibility**: All NVIDIA GPUs with CUDA support
- **Tested On**: Debian Trixie LXC with RTX 5060 Ti (Blackwell)

---

**Last Updated**: February 3, 2026  
**Installation Time**: ~5-10 minutes  
**Disk Space Required**: ~10GB
