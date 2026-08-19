#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
