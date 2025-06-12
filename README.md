# 🎵 Audio Feedback Analyzer

A professional-grade audio analysis tool that provides comprehensive technical feedback using advanced signal processing and AI-powered evaluation. Get detailed insights into your audio productions as if reviewed by an experienced audio engineer.

## ✨ Features

### 🔬 Comprehensive Audio Analysis
- **Dynamic Range Analysis**: RMS, peak levels, crest factor, loudness range
- **Frequency Spectrum Analysis**: Spectral centroid, bandwidth, contrast, and 7-band frequency analysis
- **Temporal Characteristics**: Tempo detection, beat tracking, onset detection, transient analysis
- **Stereo Imaging**: Correlation, stereo width, left-right balance, mid-side analysis
- **Harmonic Content**: Harmonic vs percussive separation, pitch stability, fundamental frequency analysis

### 🤖 AI-Powered Professional Feedback
- Uses **Gemma 2 27B** (or other Ollama models) for intelligent analysis
- Provides feedback like a professional audio engineer with 20+ years experience
- Specific recommendations with frequencies, dB values, and technical terminology
- Genre-aware analysis and professional reference comparisons

### 📊 Visual Analysis
- Frequency band energy distribution
- Dynamic characteristics visualization
- Stereo properties analysis (for stereo files)
- Harmonic vs percussive content breakdown

### 💻 Cross-Platform Support
- **Windows** (10/11)
- **Linux** (Ubuntu, Debian, CentOS, Fedora)
- **macOS** (Intel and Apple Silicon)

## 🚀 Quick Start

### One-Click Setup

#### Linux/macOS
```bash
# Download and run setup script
curl -fsSL https://raw.githubusercontent.com/yourusername/audiofeedback/main/setup.sh | bash

# Or manually:
git clone https://github.com/yourusername/audiofeedback.git
cd audiofeedback
chmod +x setup.sh
./setup.sh
```

#### Windows
```batch
# Download and run setup script
# Or manually:
git clone https://github.com/yourusername/audiofeedback.git
cd audiofeedback
setup.bat
```

### Manual Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/audiofeedback.git
cd audiofeedback

# Create virtual environment
python -m venv audiofeedback_env
source audiofeedback_env/bin/activate  # Windows: audiofeedback_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install and setup Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull gemma2:9b
ollama serve
```

### Basic Usage
```bash
# Analyze an audio file
python audiofeedback.py your_song.wav

# Save detailed results and create visualization
python audiofeedback.py your_song.wav -o ./results/ -v

# Use a different model
python audiofeedback.py your_song.wav -m gemma2:27b

# Batch process multiple files
python batch_process.py ./audio_folder/ -o ./batch_results/ -v
```

## 🧪 Testing Your Installation

### Quick Test
```bash
# Run complete system test
python test_complete_system.py

# Generate and analyze test audio
python generate_test_audio.py
python audiofeedback.py ./test_audio/01_sine_440hz.wav -v
```

### Generated Test Files
The system includes a comprehensive test suite:
- **01_sine_440hz.wav** - Pure 440Hz tone
- **02_complex_tone.wav** - Harmonic content test
- **03_white_noise.wav** - Noise analysis test
- **04_pink_noise.wav** - Frequency response test
- **05_sweep_20hz_20khz.wav** - Full spectrum sweep
- **06_drum_hits.wav** - Transient analysis test
- **07_dynamic_*.wav** - Compression comparison
- **08_stereo_*.wav** - Stereo imaging tests
- **09_musical_mix.wav** - Complex musical content
- **10_problematic_*.wav** - Edge case testing

### Command Line Interface
```bash
python audiofeedback.py [OPTIONS] INPUT_FILE

Options:
  -o, --output DIR     Output directory for results (JSON + Markdown reports)
  -v, --visualize      Create analysis visualization charts
  -m, --model MODEL    LLM model name (default: gemma2:27b)
  -h, --help          Show help message
```

### Python API
```python
from audiofeedback import AudioFeedbackApp

# Initialize the analyzer
app = AudioFeedbackApp()

# Process audio file
result = app.process_audio_file("song.wav", output_dir="./analysis/")

# Access the analysis data
analysis = result['analysis']
feedback = result['feedback']

# Create visualization
app.create_visualization(analysis, "analysis_chart.png")
```

## 📋 Analysis Output

### Technical Analysis (JSON)
```json
{
  "file_info": {
    "duration_seconds": 245.3,
    "sample_rate": 44100,
    "is_stereo": true,
    "channels": 2
  },
  "dynamics": {
    "rms_db": -18.5,
    "peak_db": -0.1,
    "crest_factor_db": 18.4,
    "dynamic_range_db": 24.8,
    "loudness_range_db": 12.3
  },
  "frequency_spectrum": {
    "spectral_centroid_mean": 2156.7,
    "frequency_bands": {
      "sub_bass_energy": 0.045,
      "bass_energy": 0.123,
      "low_mids_energy": 0.098,
      "mids_energy": 0.234,
      "high_mids_energy": 0.187,
      "presence_energy": 0.145,
      "brilliance_energy": 0.089
    }
  },
  "temporal_features": {
    "tempo": 128.5,
    "onset_rate": 2.3,
    "tempo_stability": 0.89
  },
  "stereo_properties": {
    "correlation": 0.76,
    "stereo_width": 0.42,
    "balance": -0.02
  },
  "harmonic_content": {
    "harmonic_ratio": 0.67,
    "percussive_ratio": 0.33,
    "pitch_stability": 0.84
  }
}
```

### AI Feedback Example
```markdown
## OVERALL ASSESSMENT
**Technical Quality: 7.5/10**

This track shows solid technical competency with good frequency balance and appropriate stereo imaging. The dynamic range is well-preserved, though there are opportunities for improvement in the low-end management and stereo field optimization.

## DYNAMICS ANALYSIS
- **RMS Level**: -18.5 dB indicates healthy headroom with good loudness
- **Peak Level**: -0.1 dB shows appropriate peak limiting without over-compression
- **Crest Factor**: 18.4 dB suggests excellent dynamic preservation
- **Dynamic Range**: 24.8 dB is excellent for modern production standards

## FREQUENCY BALANCE
- **Low End**: Sub-bass (4.5%) and bass (12.3%) are well-controlled
- **Midrange**: Strong presence in mids (23.4%) with good clarity
- **High End**: Balanced high-mids (18.7%) and presence (14.5%)
- **Air**: Brilliance region (8.9%) could use slight enhancement for more sparkle

## TECHNICAL RECOMMENDATIONS
1. **Boost 8-12 kHz by 1-2 dB** to enhance air and brilliance
2. **Apply gentle high-pass filter at 30 Hz** to clean up sub-bass rumble
3. **Consider M/S processing** to widen stereo field slightly (current width: 0.42)

## PRIORITY FIXES
1. Enhance high-frequency air and sparkle
2. Optimize stereo width for better spatial presentation
3. Fine-tune sub-bass management for tighter low end
```

## 🔧 Advanced Features

### Batch Processing
```python
import os
from audiofeedback import AudioFeedbackApp

app = AudioFeedbackApp()
audio_dir = "./audio_files/"
output_dir = "./batch_analysis/"

for filename in os.listdir(audio_dir):
    if filename.endswith(('.wav', '.mp3', '.flac')):
        filepath = os.path.join(audio_dir, filename)
        result = app.process_audio_file(filepath, output_dir)
        print(f"Processed: {filename}")
```

### Custom Analysis Parameters
```python
from audiofeedback import AudioAnalyzer

# Custom sample rate for faster processing
analyzer = AudioAnalyzer(sample_rate=16000)

# Load and analyze
analyzer.load_audio("song.wav")
analysis = analyzer.get_complete_analysis()
```

### Custom LLM Models
```python
from audiofeedback import LLMFeedbackGenerator

# Use different model
llm = LLMFeedbackGenerator(model_name="llama2:13b")
feedback = llm.generate_feedback(analysis_data)
```

## 📊 Technical Specifications

### Supported Audio Formats
- **WAV** (recommended for best analysis)
- **MP3** (all bitrates)
- **FLAC** (lossless)
- **M4A/AAC**
- **OGG/Vorbis**

### Analysis Capabilities
- **Sample Rates**: Up to 192 kHz (auto-resampled to 22.05 kHz for analysis)
- **Bit Depths**: 16, 24, 32-bit integer and floating-point
- **Channels**: Mono and stereo (5.1+ reduced to stereo)
- **File Size**: No practical limit (tested up to 2GB files)

### System Requirements
- **Minimum**: 8GB RAM, 4-core CPU, 20GB storage
- **Recommended**: 16GB+ RAM, 8-core CPU, SSD storage
- **For 27B model**: 32GB RAM recommended

## 🛠️ Development

### Project Structure
```
audiofeedback/
├── audiofeedback.py      # Main application
├── requirements.txt      # Dependencies
├── setup.py             # Package configuration
├── README.md           # This file
├── INSTALLATION.md     # Detailed setup guide
├── examples/           # Example scripts and audio
├── tests/             # Unit tests
└── docs/             # Additional documentation
```

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest --cov=audiofeedback tests/
```

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🔍 Troubleshooting

### Common Issues
- **"Cannot connect to Ollama"**: Ensure `ollama serve` is running
- **"Model not found"**: Run `ollama pull gemma2:27b`
- **"ffmpeg not found"**: Install ffmpeg for your platform
- **Memory issues**: Use smaller model like `gemma2:9b`

### Performance Tips
- Use WAV files for fastest analysis
- Lower sample rate (16kHz) for speed vs quality trade-off
- Use SSD storage for large model performance
- Close other applications when using 27B model

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Librosa](https://librosa.org/)** for comprehensive audio analysis
- **[Ollama](https://ollama.ai/)** for local LLM inference
- **[Google](https://ai.google.dev/gemma)** for the Gemma 2 model
- **Audio engineering community** for inspiration and domain knowledge

## 🚀 Future Roadmap

- [ ] **Real-time analysis** for live monitoring
- [ ] **Plugin integration** (VST/AU) for DAWs
- [ ] **Cloud deployment** options
- [ ] **Advanced ML models** for specific genre analysis
- [ ] **LUFS/EBU R128** loudness analysis
- [ ] **Automatic reference track comparison**
- [ ] **Multi-language support** for feedback
- [ ] **Web interface** for browser-based analysis

---

**Made with ❤️ for the audio production community**