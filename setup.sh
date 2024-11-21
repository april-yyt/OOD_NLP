#!/bin/bash

# Check if Python 3.9 is available
if ! command -v python3.9 &> /dev/null; then
    echo "Python 3.9 is required but not found. Please install Python 3.9 first."
    exit 1
fi

# Create and activate virtual environment
echo "Creating virtual environment with Python 3.9..."
python3.9 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch first (CPU version for compatibility)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt

# Install Ollama if not already installed (for macOS and Linux)
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing Ollama..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        curl https://ollama.ai/install.sh | sh
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl https://ollama.ai/install.sh | sh
    else
        echo "Please install Ollama manually from https://ollama.ai"
    fi
fi

echo "Setup completed! Activate the environment with: source .venv/bin/activate"