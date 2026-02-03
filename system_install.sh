#!/bin/bash

# Exit on error
set -e

echo "=== Starting Qwen3-TTS Server Installation (LXC/Debian) ==="

# 1. Update and install system dependencies
echo "[1/5] Installing system dependencies..."
apt-get update
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    wget

# 2. Create Application Directory
APP_DIR="/opt/qwen-tts-server"
echo "[2/5] Creating application directory at $APP_DIR..."
mkdir -p $APP_DIR
# Assuming server.py and requirements.txt are in the current directory, move them
# If running this script from the same folder as the source files:
if [ -f "server.py" ]; then
    cp server.py $APP_DIR/
fi
if [ -f "requirements.txt" ]; then
    cp requirements.txt $APP_DIR/
fi

cd $APP_DIR

# 3. Set up Python Virtual Environment
echo "[3/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# 4. Install Python Libraries
echo "[4/5] Installing Python requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found in $APP_DIR."
    exit 1
fi

# 5. Create Systemd Service for Auto-Start
echo "[5/5] Configuring Systemd Service..."

SERVICE_FILE="/etc/systemd/system/qwen-tts.service"

cat <<EOF > $SERVICE_FILE
[Unit]
Description=Qwen3-TTS API Server
After=network.target

[Service]
User=root
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$APP_DIR/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd to recognize the new service
systemctl daemon-reload

# Enable the service to start on boot
systemctl enable qwen-tts

# Start the service immediately
systemctl start qwen-tts

echo "=== Installation Complete ==="
echo "The service is now running and will start automatically on boot."
echo "Check status with: systemctl status qwen-tts"