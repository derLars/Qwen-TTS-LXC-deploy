#!/bin/bash

# Exit on error
set -e

export DEBIAN_FRONTEND=noninteractive

# --- Configuration ---
APP_DIR="/opt/qwen-tts-server"
GITHUB_REPO="https://github.com/derlars/Qwen-TTS-LXC-deploy/raw/main"
SERVICE_FILE="/etc/systemd/system/qwen-tts.service"

# --- Logging ---
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] - $1"
}

log "=== Starting Qwen3-TTS Server Installation/Update ==="

# --- System Dependencies ---
log "[1/6] Installing system dependencies..."
apt-get update
apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    build-essential ffmpeg libsndfile1 git wget curl

# --- GPU Detection and PyTorch Installation ---
if command -v nvidia-smi &> /dev/null; then
    log "[2/6] NVIDIA GPU detected. Installing CUDA and GPU-enabled PyTorch."
    
    # Enable contrib, non-free and non-free-firmware repositories
    sed -i -e's/ main/ main contrib non-free non-free-firmware/g' /etc/apt/sources.list.d/debian.sources
    
    apt-get update
    if ! dpkg -l | grep -q nvidia-driver; then
        log "No existing NVIDIA driver found. Installing Debian package."
        apt-get install -y nvidia-driver firmware-misc-nonfree
    else
        log "Existing NVIDIA driver found. Skipping installation."
    fi
    
    # Add NVIDIA CUDA repository
    # The debian11 repo is the latest one provided by NVIDIA that works for debian12+
    wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb
    rm cuda-keyring_1.0-1_all.deb
    apt-get update
    apt-get -y install cuda-toolkit-11-8
    
    export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6"
    PIP_INSTALL_TORCH="pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118"
else
    log "[2/6] No NVIDIA GPU detected. Using CPU-only PyTorch."
    PIP_INSTALL_TORCH="pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
fi

# --- Application Directory and Update Logic ---
if [ -d "$APP_DIR" ]; then
    log "[3/6] Existing installation found. Stopping server and updating files."
    systemctl stop qwen-tts || true
    # Backup old files
    mv $APP_DIR/server.py $APP_DIR/server.py.bak || true
    mv $APP_DIR/requirements.txt $APP_DIR/requirements.txt.bak || true
    mv $APP_DIR/config.yaml $APP_DIR/config.yaml.bak || true
else
    log "[3/6] No existing installation found. Creating application directory."
    mkdir -p $APP_DIR
fi

cd $APP_DIR
mkdir -p logs

# --- Download Application Files ---
log "[4/6] Downloading application files from GitHub..."
wget -q -O $APP_DIR/server.py $GITHUB_REPO/server.py
wget -q -O $APP_DIR/requirements.txt $GITHUB_REPO/requirements.txt
wget -q -O $APP_DIR/config.yaml $GITHUB_REPO/config.yaml

# --- Python Virtual Environment and Dependencies ---
if [ ! -d "$APP_DIR/venv" ]; then
    log "[5/6] Creating Python virtual environment."
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
log "[5/6] Installing Python requirements..."
$PIP_INSTALL_TORCH
pip install -r requirements.txt --no-cache-dir

# --- Systemd Service ---
log "[6/6] Configuring and starting Systemd service..."
cat <<EOF > $SERVICE_FILE
[Unit]
Description=Qwen3-TTS API Server
After=network.target

[Service]
User=root
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 7200
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable qwen-tts
systemctl restart qwen-tts

log "=== Installation/Update Complete ==="
log "The service is now running and will start automatically on boot."
log "Check status with: systemctl status qwen-tts"
log "Logs are available at: $APP_DIR/logs/server.log and via journalctl -u qwen-tts"
