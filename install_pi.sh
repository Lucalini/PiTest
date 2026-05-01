#!/usr/bin/env bash
# Raspberry Pi OS — system packages for USB audio + Python bindings.
set -euo pipefail
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv portaudio19-dev libasound2-plugins
