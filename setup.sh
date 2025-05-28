#!/bin/bash

# Deep Q-Learning Setup Script
# This script sets up the environment and installs all dependencies

set -e  # Exit on any error

echo "🚀 Deep Q-Learning (DQN) Setup Script"
echo "======================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install requirements
echo "📥 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models plots logs training_videos

# Test installation
echo "🧪 Testing installation..."
python -c "
import tensorflow as tf
import gym
import numpy as np
import matplotlib
print('✅ All core dependencies imported successfully')
print(f'TensorFlow: {tf.__version__}')
print(f'Gym: {gym.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
"

# Test environment creation
python -c "
import gym
env = gym.make('CartPole-v1')
print('✅ CartPole environment created successfully')
env.close()
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the training: python main.py"
echo ""
echo "For more information, see README_modular.md and INSTALL.md"