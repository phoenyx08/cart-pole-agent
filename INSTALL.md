# Installation Guide

This guide provides step-by-step instructions for setting up the Deep Q-Learning environment.

## 🔧 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment support (recommended)

## 📦 Quick Installation

### 1. Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd reinforce

# Or download and extract the files to a directory called 'reinforce'
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
# Test the installation
python main.py
```

## 🔍 Manual Installation

If you prefer to install packages individually:

```bash
# Core dependencies
pip install tensorflow==2.16.2
pip install gym==0.26.2
pip install numpy==1.26.4
pip install matplotlib==3.10.3
pip install moviepy==2.2.1

# Optional utilities
pip install tqdm==4.67.1
pip install rich==14.0.0
```

## 🐋 Docker Installation (Optional)

Create a `Dockerfile` for containerized deployment:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t dqn-cartpole .
docker run dqn-cartpole
```

## 🔧 System-Specific Instructions

### Ubuntu/Debian
```bash
# Update system packages
sudo apt update

# Install Python and pip if not already installed
sudo apt install python3 python3-pip python3-venv

# May need additional libraries for video processing
sudo apt install ffmpeg
```

### macOS
```bash
# Install Python using Homebrew (if not already installed)
brew install python

# Install ffmpeg for video processing
brew install ffmpeg
```

### Windows
1. Download Python from python.org
2. Ensure pip is installed with Python
3. Install ffmpeg from https://ffmpeg.org/ (for video processing)

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. **TensorFlow Installation Issues**
```bash
# If TensorFlow fails to install, try:
pip install --upgrade pip
pip install tensorflow==2.16.2 --no-cache-dir
```

#### 2. **Gym Environment Issues**
```bash
# If gym environments don't work:
pip install gym[classic_control]
```

#### 3. **Video Recording Issues**
```bash
# Install video codecs
pip install imageio-ffmpeg
```

#### 4. **Display Issues (Linux)**
```bash
# For headless servers, you might need:
export DISPLAY=:0
# Or disable real-time visualization in config.py
```

#### 5. **Memory Issues**
```bash
# If you encounter memory issues, reduce batch size in config.py:
AGENT_CONFIG = {
    'batch_size': 16,  # Reduce from 32
    'memory_size': 1000,  # Reduce from 2000
}
```

## ✅ Verification Steps

After installation, verify everything works:

### 1. **Check Python Version**
```bash
python --version  # Should be 3.8+
```

### 2. **Test Imports**
```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import gym; print(f'Gym: {gym.__version__}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
```

### 3. **Test Environment**
```bash
python -c "
import gym
env = gym.make('CartPole-v1')
print('CartPole environment created successfully')
env.close()
"
```

### 4. **Run Quick Test**
```bash
# Run with minimal configuration for quick test
python main.py
```

## 🎯 Performance Optimization

### CPU Optimization
Add to your environment or config.py:
```python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable CPU optimizations
```

### GPU Support (Optional)
For GPU acceleration (requires CUDA-compatible GPU):
```bash
pip install tensorflow[and-cuda]==2.16.2
```

## 📁 Directory Structure After Installation

```
reinforce/
├── venv/                     # Virtual environment
├── requirements.txt          # Package dependencies
├── main.py                   # Main script
├── config.py                 # Configuration
├── dqn_agent.py             # Agent implementation
├── environment_wrapper.py   # Environment wrappers
├── trainer.py               # Training logic
├── visualizer.py            # Visualization tools
├── models/                  # Generated model files
├── plots/                   # Generated plots
├── training_videos/         # Generated videos
└── logs/                    # Generated logs
```

## 🎮 Next Steps

After successful installation:

1. **Review Configuration**: Check `config.py` for parameters
2. **Run Training**: Execute `python main.py`
3. **Explore Results**: Check `models/`, `plots/`, and `training_videos/`
4. **Customize**: Modify hyperparameters in `config.py`

## 📞 Support

If you encounter issues:

1. Check this troubleshooting guide
2. Verify all dependencies are correctly installed
3. Ensure Python version compatibility
4. Check system-specific requirements

## 🔄 Updates

To update dependencies:
```bash
pip install -r requirements.txt --upgrade
```