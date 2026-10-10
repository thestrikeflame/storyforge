#!/usr/bin/env bash
# Creates a self-signed certificate so StoryForge can also be opened over https.
# Browsers only allow the microphone on https pages (or on the Jetson itself via localhost).
# Run once, then restart StoryForge. The first visit shows a browser warning: choose
# "Advanced" and "Proceed". That is expected for a certificate you made yourself.
set -euo pipefail
source "$(dirname "$0")/_settings.sh"
CERT_DIR="${CERT_DIR:-$PROJECT_DIR/certs}"
mkdir -p "$CERT_DIR"
ADDRESS="${1:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
ADDRESS="${ADDRESS:-127.0.0.1}"
openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
  -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
  -subj "/CN=StoryForge" \
  -addext "subjectAltName=IP:$ADDRESS,IP:127.0.0.1,DNS:localhost,DNS:$(hostname)" 2>/dev/null
chmod 600 "$CERT_DIR/key.pem"
echo "Certificate created for $ADDRESS in $CERT_DIR"
echo "Restart StoryForge, then open:  https://$ADDRESS:${HTTPS_PORT:-8443}"
