# 🎵 Audio Feedback Analyzer

A professional-grade audio analysis tool that provides comprehensive technical feedback using advanced signal processing and AI-powered evaluation. Get detailed insights into your audio productions as if reviewed by an experienced audio engineer with genre-specific expertise.

**Made with ❤️ for the audio production community**

## ✨ Features

### 🔬 Comprehensive Audio Analysis
- **Dynamic Range Analysis**: RMS, peak levels, crest factor, loudness range
- **Frequency Spectrum Analysis**: Spectral centroid, bandwidth, contrast, and 7-band frequency analysis (sub-bass to brilliance)
- **Temporal Characteristics**: Tempo detection, beat tracking, onset detection, transient analysis
- **Stereo Imaging**: Correlation, stereo width, left-right balance, mid-side analysis
- **Harmonic Content**: Harmonic vs percussive separation, pitch stability, fundamental frequency analysis

### 🎯 Genre-Specific Analysis
- **Automatic Genre Detection**: Analyzes audio characteristics to identify likely genre
- **Supported Genres**: Rock, Metal, Electronic, Drum & Bass, Classical, Jazz, Podcast
- **Genre-Aware Feedback**: Compares your audio against professional standards for the detected genre
- **Custom Genre Profiles**: Each genre has specific dynamic range, frequency emphasis, and stereo width expectations

### 🤖 AI-Powered Professional Feedback
- Uses **Gemma 3 27B** (or other Ollama models) for intelligent analysis
- Provides feedback like a professional audio engineer with 20+ years experience
- Specific recommendations with frequencies, dB values, and technical terminology
- Genre-contextual analysis and professional reference comparisons

### 📊 Visual Analysis & Reporting
- Frequency band energy distribution charts
- Dynamic characteristics visualization
- Stereo properties analysis (for stereo files)
- Harmonic vs percussive content breakdown
- JSON technical data export
- Markdown feedback reports

### 💻 Cross-Platform Support
- **Windows** (10/11) - Automated setup with `setup.bat`
- **Linux** (Ubuntu, Debian, CentOS, Fedora) - Automated setup with `setup.sh`
- **macOS** (Intel and Apple Silicon) - Automated setup with `setup.sh`

## 🚀 Quick Start

### Automated Setup (Recommended)

**No manual installation required!** Our setup scripts handle everything for you in an isolated environment.

#### Windows
```cmd
# Download the project
git clone https://github.com/yourusername/audiofeedback.git
cd audiofeedback

# Run automated setup (handles everything)
setup.bat
```

#### Linux/macOS
```bash
# Download the project
git clone https://github.com/yourusername/audiofeedback.git
cd audiofeedback

# Make setup script executable and run
chmod +x setup.sh
./setup.sh
```

### What the Setup Does
1. **Checks system dependencies** (Python 3.8+, FFmpeg)
2. **Creates isolated virtual environment** (`audiofeedback_env`)
3. **Installs all Python packages** in the virtual environment (no system pollution)
4. **Downloads and configures Ollama** with AI models
5. **Creates convenience scripts** for easy usage
6. **Generates test audio files** for testing
7. **Runs system validation tests**

### First Usage

After setup completes, you have several options:

#### Option 1: Convenience Scripts (Windows)
```cmd
# Analyze a single file
run_analyzer.bat your_song.wav

# With visualization and output folder
run_analyzer.bat your_song.wav -o results -v

# Process multiple files
run_batch_process.bat ./audio_folder/ -o ./batch_results/

# Open activated environment for manual use
activate_env.bat
```

#### Option 2: Manual Activation (All Platforms)
```bash
# Activate the isolated environment
source audiofeedback_env/bin/activate  # Linux/macOS
# OR
audiofeedback_env\Scripts\activate.bat  # Windows

# Now use the analyzer
python audiofeedback.py your_song.wav
python audiofeedback.py your_song.wav -o results -v -g metal
python batch_process.py ./audio_folder/ -o ./results/
```

## 📋 Command Line Usage

### Single File Analysis
```bash
# Basic analysis
python audiofeedback.py input_song.wav

# With output directory and visualization
python audiofeedback.py input_song.wav -o ./results/ -v

# Force specific genre
python audiofeedback.py input_song.wav -g metal

# Use different AI model
python audiofeedback.py input_song.wav -m gemma2:9b

# Complete example
python audiofeedback.py "my_track.wav" -o "./analysis/" -v -g "drum_and_bass" -m "gemma3:27b"
```

### Batch Processing
```bash
# Process all audio files in a directory
python batch_process.py ./audio_folder/ -o ./batch_results/

# With visualizations and recursive search
python batch_process.py ./music_library/ -o ./analysis/ -v -r

# Parallel processing (memory intensive)
python batch_process.py ./audio_folder/ -o ./results/ -p -w 4
```

### Available Options

#### Single File (`audiofeedback.py`)
- `input_file` - Path to audio file (WAV, MP3, FLAC, M4A, AAC, OGG)
- `-o, --output` - Output directory for results
- `-v, --visualize` - Create analysis visualization charts
- `-g, --genre` - Force specific genre (rock, metal, electronic, drum_and_bass, classical, jazz, podcast)
- `-m, --model` - AI model to use (default: gemma3:27b)

#### Batch Processing (`batch_process.py`)
- `input_dir` - Directory containing audio files
- `-o, --output` - Output directory (required)
- `-r, --recursive` - Search subdirectories
- `-v, --visualize` - Create visualizations for all files
- `-p, --parallel` - Process files in parallel (use with caution)
- `-w, --workers` - Number of parallel workers (default: 2)
- `-m, --model` - AI model to use (default: gemma3:27b)

## 🎵 Genre-Specific Analysis

The system automatically detects the genre of your audio and provides targeted feedback:

### Supported Genres
- **Rock**: Balanced dynamics, emphasis on low-mids and presence
- **Metal**: Tight compression, emphasis on bass, low-mids, presence, and brilliance
- **Electronic**: Wide stereo field, sub-bass and brilliance emphasis
- **Drum & Bass**: High energy, sub-bass focus, wide stereo imaging
- **Classical**: Excellent dynamic range, natural frequency balance
- **Jazz**: Good dynamics, balanced frequency response
- **Podcast**: Speech-optimized, mid-range focus, narrow stereo

### How It Works
1. **Analysis**: Extracts technical parameters (dynamic range, frequency content, tempo, stereo width)
2. **Genre Detection**: Compares parameters against genre profiles to identify the most likely style
3. **Contextual Feedback**: Provides recommendations specific to the detected genre's professional standards

Example output:
```
🎯 Detected genre: Metal
📊 Dynamic Range: 8.2 dB (Expected: 6-12 dB) ✅
📊 RMS Level: -7.8 dB (Expected: -10 to -6 dB) ✅
📊 Frequency Emphasis: Strong bass and presence ✅
```

## 📊 Analysis Output

### Technical Data (JSON)
```json
{
  "file_info": {
    "duration_seconds": 245.6,
    "sample_rate": 44100,
    "channels": 2
  },
  "genre_detection": {
    "detected_genre": "metal",
    "confidence": "high"
  },
  "dynamics": {
    "rms_db": -7.8,
    "peak_db": -0.2,
    "dynamic_range_db": 8.2,
    "crest_factor_db": 12.4
  },
  "frequency_spectrum": {
    "frequency_bands": {
      "sub_bass_energy": 0.15,
      "bass_energy": 0.28,
      "brilliance_energy": 0.18
    }
  }
}
```

### AI Feedback Report (Markdown)
```markdown
# Audio Analysis Feedback - my_track.wav

**Detected Genre:** Metal

## OVERALL ASSESSMENT
**Technical Quality: 8.5/10**

This metal track demonstrates excellent technical execution with appropriate compression for the genre while maintaining good dynamics. The frequency balance aligns well with modern metal production standards.

## GENRE ANALYSIS
Based on the detected genre (Metal), this track fits well within the typical characteristics of modern metal productions. The dynamic range of 8.2 dB is optimal for metal (expected: 6-12 dB), and the frequency emphasis on bass, low-mids, and presence creates the characteristic metal sound signature.

## DYNAMICS ANALYSIS
- **RMS Level**: -7.8 dB is perfect for metal productions (expected: -10 to -6 dB)
- **Peak Level**: -0.2 dB shows appropriate limiting without over-compression
- **Dynamic Range**: 8.2 dB is excellent for metal genre standards
- **Crest Factor**: 12.4 dB indicates good transient preservation despite compression

## FREQUENCY BALANCE
- **Low End**: Sub-bass (15%) and bass (28%) provide solid foundation
- **Midrange**: Low-mids (22%) and mids (18%) well-balanced for guitar clarity
- **High End**: Presence (19%) and brilliance (18%) deliver characteristic metal aggression
- **Analysis**: Frequency distribution aligns perfectly with metal genre emphasis

## TECHNICAL RECOMMENDATIONS
1. **Maintain current balance** - frequency distribution is genre-appropriate
2. **Consider slight boost at 3-5 kHz** to enhance guitar presence (+0.5-1 dB)
3. **Stereo width is optimal** at 1.2 for metal productions

## PRIORITY FIXES
1. No critical issues detected - track meets professional metal standards
2. Optional: Fine-tune presence frequencies for extra clarity
3. Consider A/B testing against reference metal tracks
```

## 🧪 Testing Your Installation

### Test with Generated Audio
```bash
# Generate test files (done automatically during setup)
python generate_test_audio.py -o ./test_audio/

# Test different audio characteristics
python audiofeedback.py ./test_audio/01_sine_440hz.wav -v
python audiofeedback.py ./test_audio/05_sweep_20hz_20khz.wav -v
python audiofeedback.py ./test_audio/07_dynamic_natural.wav -v
```

### System Validation
```bash
# Run complete system test
python test_complete_system.py

# Check if all components work
python -c "import librosa, pydub, numpy, matplotlib, ollama; print('✅ All libraries OK')"
```

## 🔧 Advanced Features

### Python API
```python
from audiofeedback import AudioFeedbackApp

# Initialize the analyzer
app = AudioFeedbackApp()

# Process audio file with specific genre
result = app.process_audio_file(
    file_path="song.wav", 
    output_dir="./analysis/",
    genre="metal"  # Optional: force specific genre
)

# Access results
analysis = result['analysis']
feedback = result['feedback']
detected_genre = analysis['genre_detection']['detected_genre']

# Create visualization
app.create_visualization(analysis, "chart.png")
```

### Custom Analysis Parameters
```python
from audiofeedback import AudioAnalyzer
from config_example import Config

# Use custom sample rate for faster processing
analyzer = AudioAnalyzer(sample_rate=Config.AUDIO_SAMPLE_RATE)

# Load and analyze
success = analyzer.load_audio("song.wav")
if success:
    analysis = analyzer.get_complete_analysis()
    detected_genre = analyzer.detect_genre(analysis)
```

### Different AI Models
```python
from audiofeedback import LLMFeedbackGenerator

# Use smaller/faster model
llm = LLMFeedbackGenerator(model_name="gemma2:9b")
feedback = llm.generate_feedback(analysis_data)

# Available models:
# - gemma3:27b (best quality, high RAM usage)
# - gemma2:27b (excellent quality, high RAM usage)  
# - gemma2:9b (good quality, moderate RAM usage)
# - llama2:13b (alternative architecture)
# - mistral:7b (fastest, lowest quality)
```

## 📊 Technical Specifications

### Supported Audio Formats
- **WAV** - All sample rates and bit depths (recommended)
- **MP3** - All bitrates
- **FLAC** - Lossless compression
- **M4A/AAC** - Apple formats
- **OGG/Vorbis** - Open source format

### Analysis Capabilities
- **Sample Rates**: Up to 192 kHz (processed at 44.1 kHz for optimal analysis)
- **Bit Depths**: 16, 24, 32-bit integer and floating-point
- **Channels**: Mono and stereo (multichannel reduced to stereo)
- **File Size**: No practical limit (tested up to 2GB files)
- **Processing Speed**: ~2-5x real-time depending on model and hardware

### System Requirements

#### Minimum
- **RAM**: 8GB
- **CPU**: 4-core processor
- **Storage**: 20GB free space
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+

#### Recommended
- **RAM**: 16GB+ (32GB for gemma3:27b)
- **CPU**: 8+ cores, 3GHz+
- **Storage**: SSD with 50GB+ free space
- **GPU**: Optional - NVIDIA GPU for Ollama acceleration

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### "Cannot connect to Ollama"
```bash
# Ensure Ollama is running
ollama serve

# Check available models
ollama list

# Pull missing model
ollama pull gemma3:27b
```

#### "Python module not found"
```bash
# Activate virtual environment first
source audiofeedback_env/bin/activate  # Linux/macOS
audiofeedback_env\Scripts\activate.bat  # Windows

# Then run your command
python audiofeedback.py your_file.wav
```

#### "FFmpeg not found"
- **Windows**: Download from https://ffmpeg.org or install via chocolatey: `choco install ffmpeg`
- **Linux**: `sudo apt install ffmpeg` (Ubuntu) or `sudo dnf install ffmpeg-free` (Fedora)
- **macOS**: `brew install ffmpeg`

#### Memory Issues with Large Models
```bash
# Use smaller model
python audiofeedback.py song.wav -m gemma2:9b

# Or force garbage collection between files
python batch_process.py folder/ -o results/ -w 1  # Single worker
```

#### Permission Errors (Linux/macOS)
```bash
# Make scripts executable
chmod +x setup.sh
chmod +x audiofeedback.py

# Or run with python explicitly
python audiofeedback.py input.wav
```

### Performance Tips
- **Use WAV files** for fastest analysis
- **Batch process** multiple files for efficiency
- **Use SSD storage** for better I/O performance
- **Close other applications** when using large models
- **Use smaller models** (gemma2:9b) for routine analysis

## 📦 Installation Details

### Prerequisites
- **Python 3.8+** (checked automatically by setup scripts)
- **FFmpeg** (for audio format support)
- **Git** (for downloading the project)
- **Internet connection** (for downloading AI models)

### Virtual Environment Benefits
The setup creates an isolated environment that:
- ✅ **Doesn't pollute your system Python**
- ✅ **Prevents version conflicts**
- ✅ **Can be easily removed** (just delete `audiofeedback_env` folder)
- ✅ **Includes all required packages**
- ✅ **Works independently** of other Python projects

### Disk Space Requirements
- **Python packages**: ~2GB
- **Ollama models**: 
  - gemma2:9b: ~5GB
  - gemma3:27b: ~16GB
- **Generated test files**: ~50MB
- **Total**: 7-18GB depending on models

## 🔄 Updates & Maintenance

### Updating the System
```bash
# Activate environment
source audiofeedback_env/bin/activate

# Update Python packages
pip install -r requirements.txt --upgrade

# Update Ollama models
ollama pull gemma3:27b
ollama pull gemma2:9b
```

### Adding New Models
```bash
# Pull new model
ollama pull llama2:13b

# Use in analysis
python audiofeedback.py song.wav -m llama2:13b
```

### Uninstalling
```bash
# Remove virtual environment
rm -rf audiofeedback_env  # Linux/macOS
rmdir /s audiofeedback_env  # Windows

# Remove Ollama models (optional)
ollama rm gemma3:27b
ollama rm gemma2:9b

# Remove project folder
cd ..
rm -rf audiofeedback  # Linux/macOS
rmdir /s audiofeedback  # Windows
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** in the isolated virtual environment
4. **Test thoroughly** with `python test_complete_system.py`
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/audiofeedback.git
cd audiofeedback

# Run setup as normal
./setup.sh  # or setup.bat on Windows

# Activate environment for development
source audiofeedback_env/bin/activate

# Install development dependencies
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/
```

### Project Structure
```
audiofeedback/
├── audiofeedback.py          # Main application
├── batch_process.py          # Batch processing script
├── config_example.py         # Configuration and genre profiles
├── generate_test_audio.py    # Test audio file generator
├── test_complete_system.py   # System validation tests
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── setup.bat                 # Windows setup script
├── setup.sh                  # Linux/macOS setup script
├── README.md                 # This file
├── run_analyzer.bat          # Windows convenience script
├── run_batch_process.bat     # Windows batch convenience script
├── activate_env.bat          # Windows environment activation
├── audiofeedback_env/        # Virtual environment (created by setup)
└── test_audio/              # Generated test files (created by setup)
```

## 📜 License

This project is licensed under... nothing, we just want you to be free and happy.

## 🙏 Acknowledgments

- **Librosa** - Comprehensive audio analysis library
- **Ollama** - Local LLM inference platform
- **Google Gemma** - State-of-the-art AI model for feedback generation
- **Audio production community** - For inspiration, feedback, and real-world testing

## 📞 Support

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check the comments in `config_example.py` for customization options
- **Community**: Join discussions in the project's GitHub repository
- **Wiki**: Detailed guides and examples (coming soon)

---

**Made with ❤️ for the audio production community**

Transform your audio analysis workflow with AI-powered professional feedback that understands your genre and provides actionable insights!
