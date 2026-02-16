#!/bin/bash
# deploy.sh - Deploy Face Validation API ke server
# Usage: ./deploy.sh [target_dir] [port]

set -e

# Konfigurasi default
TARGET_DIR="${1:-/opt/face-api}"
PORT="${2:-8000}"
APP_USER="${APP_USER:-$USER}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "  Deploy Face Validation API"
echo "=========================================="
echo "Target dir: $TARGET_DIR"
echo "Port: $PORT"
echo "Script dir: $SCRIPT_DIR"
echo "=========================================="

# 1. Cek root/sudo untuk install sistem
need_sudo=""
if [ "$(id -u)" -ne 0 ]; then
  need_sudo="sudo"
fi

# 2. Install dependensi sistem (Ubuntu/Debian)
echo "[1/6] Memeriksa dependensi sistem..."
if command -v apt-get &>/dev/null; then
  $need_sudo apt-get update -qq
  $need_sudo apt-get install -y -qq \
    python3 \
    python3-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-ind \
    || true
  echo "    Dependensi sistem OK."
else
  echo "    Skip apt (bukan Debian/Ubuntu). Pastikan Python3, OpenCV libs, dan Tesseract terpasang."
fi

# 3. Buat direktori target dan salin file
echo "[2/6] Menyiapkan direktori $TARGET_DIR..."
$need_sudo mkdir -p "$TARGET_DIR"
$need_sudo chown "$(whoami):$(id -gn)" "$TARGET_DIR" 2>/dev/null || true

cp -r "$SCRIPT_DIR"/* "$TARGET_DIR/" 2>/dev/null || true
# Pastikan file penting ada
for f in validate.py requirements.txt; do
  if [ ! -f "$TARGET_DIR/$f" ]; then
    cp "$SCRIPT_DIR/$f" "$TARGET_DIR/" 2>/dev/null || true
  fi
done

# File model & cascade (jika ada di source)
[ -f "$SCRIPT_DIR/model.keras" ] && cp "$SCRIPT_DIR/model.keras" "$TARGET_DIR/" || true
[ -f "$SCRIPT_DIR/haarcascade_frontalface_default.xml" ] && cp "$SCRIPT_DIR/haarcascade_frontalface_default.xml" "$TARGET_DIR/" || true

cd "$TARGET_DIR"

# 4. Virtual environment dan dependensi Python
echo "[3/6] Membuat virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "[4/6] Menginstall dependensi Python..."
pip install --upgrade pip -q
pip install -r requirements.txt

# 5. Cek file wajib
echo "[5/6] Memeriksa file wajib..."
missing=""
[ ! -f "validate.py" ] && missing="validate.py"
[ ! -f "model.keras" ] && missing="${missing:+$missing }model.keras"
if [ -n "$missing" ]; then
  echo "    Peringatan: File berikut tidak ditemukan: $missing"
  echo "    Pastikan model.keras ada di $TARGET_DIR sebelum menjalankan."
fi

# 6. Systemd service (opsional)
echo "[6/6] Membuat unit systemd (opsional)..."
SERVICE_FILE="/etc/systemd/system/face-api.service"
if [ -w "/etc/systemd/system" ] 2>/dev/null || $need_sudo test -w "/etc/systemd/system" 2>/dev/null; then
  cat << EOF | $need_sudo tee "$SERVICE_FILE" > /dev/null
[Unit]
Description=Face Validation API
After=network.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$TARGET_DIR
Environment="PATH=$TARGET_DIR/.venv/bin"
ExecStart=$TARGET_DIR/.venv/bin/uvicorn validate:app --host 0.0.0.0 --port $PORT
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
  echo "    Service dibuat: $SERVICE_FILE"
  echo "    Aktifkan dengan: sudo systemctl daemon-reload && sudo systemctl enable --now face-api"
else
  echo "    Skip systemd (no write access). Jalankan manual:"
  echo "    cd $TARGET_DIR && .venv/bin/uvicorn validate:app --host 0.0.0.0 --port $PORT"
fi

echo ""
echo "=========================================="
echo "  Deploy selesai."
echo "=========================================="
echo "Jalankan aplikasi:"
echo "  cd $TARGET_DIR && source .venv/bin/activate"
echo "  uvicorn validate:app --host 0.0.0.0 --port $PORT"
echo ""
echo "Atau dengan systemd:"
echo "  sudo systemctl start face-api"
echo "  sudo systemctl status face-api"
echo "=========================================="
