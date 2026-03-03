#!/bin/bash
set -e

echo "Installing PyTorch CPU-only from official PyTorch index..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Installing remaining dependencies..."
pip install -r requirements.txt

echo "Downloading ML model for offline inference..."
python download_model.py || echo "Model download skipped or failed, will download on first use"

echo "Build complete!"
