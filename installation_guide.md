# Audio Feedback Analyzer - Installation Guide

## Prerequisites

### 1. Python Installation
- **Python 3.8+** is required
- Check your Python version: `python --version` or `python3 --version`

### 2. System Dependencies

#### Windows
```bash
# Install ffmpeg (required for pydub)
# Download from https://ffmpeg.org/download.html
# Or use chocolatey:
choco install ffmpeg

# Optional: Install Microsoft Visual C++ 14.0+ for some audio libraries
# Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3-pip python3-dev python3-venv
sudo apt install ffmpeg libsndfile1 libasound2-dev portaudio19-dev
sudo apt install build-essential cmake pkg-config
```

#### Linux (CentOS/RHEL/Fedora)
```bash
# Fedora
sudo dnf install python3-pip python3-devel python3-venv
sudo dnf install ffmpeg-free libsndfile-devel alsa-lib-devel portaudio-devel
sudo dnf install gcc-c++ cmake pkgconfig

# CentOS/RHEL (enable EPEL repository first)
sudo yum install epel-release
sudo yum install python3-pip python3-devel python3-venv
sudo yum install ffmpeg libsndfile-devel alsa-lib-devel portaudio-devel
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python3 ffmpeg libsndfile portaudio
```

### 3. Ollama Installation

#### All Platforms
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai/download

# Pull the Gemma2 27B model (this will take some time - ~16GB download)
ollama pull gemma2:27b

# Start Ollama service
ollama serve
```

## Quick Installation

### Method 1: Using pip (Recommended)
```bash
# Create virtual environment
python -m venv audiofeedback_env

# Activate virtual environment
# Windows:
audiofeedback_env\Scripts\activate
# Linux/macOS:
source audiofeedback_env/bin/activate

# Install the package
pip install -r requirements.txt
```

### Method 2: Development Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/audiofeedback.git
cd audiofeedback

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install in development mode
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt
```

## Verification

### Test Your Installation
```bash
# Test Python imports
python -c "import librosa, pydub, numpy, matplotlib; print('Audio libraries OK')"

# Test Ollama connection
python -c "import ollama; client = ollama.Client(); print('Models:', [m['name'] for m in client.list()['models']])"

# Run the analyzer on a test file
python audiofeedback.py --help
```

### Create Test Audio File
```python
# Create a simple test audio file
import numpy as np
import librosa
import soundfile as sf

# Generate a 10-second test tone
duration = 10.0
sample_rate = 44100
t = np.linspace(0, duration, int(sample_rate * duration))
frequency = 440  # A4 note
audio = 0.5 * np.sin(2 * np.pi * frequency * t)

# Add some harmonics for more interesting analysis
audio += 0.25 * np.sin(2 * np.pi * frequency * 2 * t)  # Octave
audio += 0.125 * np.sin(2 * np.pi * frequency * 3 * t)  # Fifth

# Save test file
sf.write('test_audio.wav', audio, sample_rate)
print("Test audio file created: test_audio.wav")
```

## Usage Examples

### Basic Analysis
```bash
# Analyze an audio file
python audiofeedback.py input_song.wav

# Save results to directory
python audiofeedback.py input_song.wav -o ./results/

# Create visualization
python audiofeedback.py input_song.wav -v -o ./results/

# Use different model
python audiofeedback.py input_song.wav -m gemma2:9b
```

### Advanced Usage
```python
# Python API usage
from audiofeedback import AudioFeedbackApp

app = AudioFeedbackApp()
result = app.process_audio_file("my_song.wav", output_dir="./analysis/")
print(result['feedback'])
```

## Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'librosa'"
```bash
# Ensure you're in the correct virtual environment
pip install librosa
```

#### 2. "ffmpeg not found" error
```bash
# Windows: Install ffmpeg and add to PATH
# Linux: sudo apt install ffmpeg
# macOS: brew install ffmpeg
```

#### 3. "Cannot connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve

# Check if model is installed
ollama list

# Pull model if missing
ollama pull gemma2:27b
```

#### 4. "Permission denied" on Linux/macOS
```bash
# Make script executable
chmod +x audiofeedback.py

# Or run with python explicitly
python audiofeedback.py input.wav
```

#### 5. Low memory issues with large models
```bash
# Use smaller model
ollama pull gemma2:9b
python audiofeedback.py input.wav -m gemma2:9b

# Or adjust Ollama memory settings
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_KEEP_ALIVE=5m
```

### Performance Optimization

#### For Large Audio Files
```python
# Reduce sample rate for analysis (faster processing)
analyzer = AudioAnalyzer(sample_rate=16000)  # Default is 22050

# Process in chunks for very long files
# The analyzer automatically handles this
```

#### For Faster LLM Response
```bash
# Use quantized models
ollama pull gemma2:27b-q4_0  # 4-bit quantized version

# Reduce context length
# Modify the prompt in LLMFeedbackGenerator.create_analysis_prompt()
```

## System Requirements

### Minimum Requirements
- **RAM**: 8GB (16GB recommended for 27B model)
- **Storage**: 20GB free space
- **CPU**: Multi-core processor (4+ cores recommended)
- **GPU**: Not required but helps with Ollama performance

### Recommended Requirements
- **RAM**: 32GB+ for smooth operation with large models
- **Storage**: SSD with 50GB+ free space
- **CPU**: 8+ cores, 3GHz+
- **GPU**: NVIDIA GPU with 8GB+ VRAM for GPU acceleration

### GPU Acceleration (Optional)
```bash
# For NVIDIA GPUs, install CUDA support for Ollama
# Follow instructions at: https://ollama.ai/blog/nvidia-gpu-support

# Verify GPU is being used
nvidia-smi
```

## Directory Structure
```
audiofeedback/
├── audiofeedback.py          # Main application
├── requirements.txt          # Python dependencies
├── setup.py                 # Package setup
├── INSTALLATION.md          # This file
├── README.md               # Project documentation
├── examples/               # Example audio files and scripts
│   ├── test_audio.py      # Generate test files
│   └── batch_process.py   # Batch processing script
├── docs/                  # Additional documentation
└── tests/                 # Unit tests
```

## Next Steps

1. **Test the installation** with the verification steps above
2. **Run your first analysis** on a sample audio file
3. **Explore the visualization options** to understand your audio better
4. **Experiment with different LLM models** to find the best balance of speed/quality
5. **Check the documentation** for advanced features and customization options

## Support

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check the docs/ directory for detailed guides
- **Community**: Join discussions about audio analysis and AI feedback

## Updates

To update the application:
```bash
# Activate your virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Update dependencies
pip install -r requirements.txt --upgrade

# Update Ollama models
ollama pull gemma2:27b
```