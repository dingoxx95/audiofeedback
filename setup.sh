#!/bin/bash
# Audio Feedback Analyzer - Setup Script
# Cross-platform setup script for Linux/macOS

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo
    echo "=================================================="
    echo "  $1"
    echo "=================================================="
    echo
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if command -v apt-get &> /dev/null; then
            DISTRO="debian"
        elif command -v yum &> /dev/null; then
            DISTRO="rhel"
        elif command -v dnf &> /dev/null; then
            DISTRO="fedora"
        else
            DISTRO="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        DISTRO="macos"
    else
        OS="unknown"
        DISTRO="unknown"
    fi
    
    print_status "Detected OS: $OS ($DISTRO)"
}

# Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Install system dependencies
install_system_deps() {
    print_header "Installing System Dependencies"
    
    case $DISTRO in
        debian)
            print_status "Installing packages for Debian/Ubuntu..."
            sudo apt update
            sudo apt install -y python3 python3-pip python3-venv python3-dev
            sudo apt install -y ffmpeg libsndfile1 libasound2-dev portaudio19-dev
            sudo apt install -y build-essential cmake pkg-config
            ;;
        rhel)
            print_status "Installing packages for RHEL/CentOS..."
            sudo yum install -y epel-release
            sudo yum install -y python3 python3-pip python3-devel
            sudo yum install -y ffmpeg libsndfile-devel alsa-lib-devel portaudio-devel
            sudo yum install -y gcc-c++ cmake pkgconfig
            ;;
        fedora)
            print_status "Installing packages for Fedora..."
            sudo dnf install -y python3 python3-pip python3-devel python3-venv
            sudo dnf install -y ffmpeg-free libsndfile-devel alsa-lib-devel portaudio-devel
            sudo dnf install -y gcc-c++ cmake pkgconfig
            ;;
        macos)
            print_status "Installing packages for macOS..."
            if ! command_exists brew; then
                print_status "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install python3 ffmpeg libsndfile portaudio
            ;;
        *)
            print_warning "Unknown distribution. Please install dependencies manually:"
            print_warning "- Python 3.8+"
            print_warning "- FFmpeg"
            print_warning "- libsndfile"
            print_warning "- portaudio"
            ;;
    esac
    
    print_success "System dependencies installed"
}

# Setup Python virtual environment
setup_python_env() {
    print_header "Setting Up Python Environment"
    
    # Check Python version
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    print_status "Python version: $PYTHON_VERSION"
    
    # Create virtual environment
    print_status "Creating virtual environment..."
    $PYTHON_CMD -m venv audiofeedback_env
    
    # Activate virtual environment
    source audiofeedback_env/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install Python dependencies
    print_status "Installing Python packages..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        print_status "requirements.txt not found, installing core packages..."
        pip install librosa pydub numpy scipy matplotlib seaborn requests ollama soundfile
    fi
    
    print_success "Python environment setup complete"
}

# Install and setup Ollama
setup_ollama() {
    print_header "Setting Up Ollama"
    
    if command_exists ollama; then
        print_status "Ollama already installed"
    else
        print_status "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
    
    # Start Ollama service
    print_status "Starting Ollama service..."
    if [[ "$OS" == "linux" ]]; then
        # On Linux, start as service
        if command_exists systemctl; then
            sudo systemctl start ollama
            sudo systemctl enable ollama
        else
            # Fallback: start in background
            nohup ollama serve > /dev/null 2>&1 &
        fi
    else
        # On macOS, start in background
        nohup ollama serve > /dev/null 2>&1 &
    fi
    
    # Wait for Ollama to start
    sleep 3
    
    # Download recommended model
    print_status "Downloading Gemma 2 9B model (this may take a while)..."
    ollama pull gemma2:9b
    
    print_success "Ollama setup complete"
}

# Test installation
test_installation() {
    print_header "Testing Installation"
    
    # Activate virtual environment
    source audiofeedback_env/bin/activate
    
    # Run system test
    if [ -f "test_complete_system.py" ]; then
        print_status "Running comprehensive system test..."
        python test_complete_system.py
    else
        print_status "Running basic tests..."
        
        # Test Python imports
        python -c "import librosa, pydub, numpy, matplotlib, ollama; print('✅ All packages imported successfully')"
        
        # Test Ollama connection
        python -c "import ollama; client = ollama.Client(); models = client.list(); print('✅ Ollama connection successful')"
        
        print_success "Basic tests passed"
    fi
}

# Create test audio files
create_test_files() {
    print_header "Creating Test Audio Files"
    
    source audiofeedback_env/bin/activate
    
    if [ -f "generate_test_audio.py" ]; then
        print_status "Generating test audio files..."
        python generate_test_audio.py -o ./test_audio/
        print_success "Test audio files created in ./test_audio/"
    else
        print_warning "Test audio generator not found, skipping..."
    fi
}

# Display usage instructions
show_usage() {
    print_header "Setup Complete!"
    
    echo "Your Audio Feedback Analyzer is ready to use!"
    echo
    echo "To get started:"
    echo "1. Activate the virtual environment:"
    echo "   source audiofeedback_env/bin/activate"
    echo
    echo "2. Analyze an audio file:"
    echo "   python audiofeedback.py your_song.wav"
    echo
    echo "3. Generate detailed report with visualization:"
    echo "   python audiofeedback.py your_song.wav -o ./results/ -v"
    echo
    echo "4. Process multiple files:"
    echo "   python batch_process.py ./audio_folder/ -o ./batch_results/"
    echo
    echo "5. Test with generated files:"
    echo "   python audiofeedback.py ./test_audio/01_sine_440hz.wav -v"
    echo
    echo "Available models:"
    echo "- gemma2:9b (recommended for most users)"
    echo "- gemma2:27b (higher quality, requires more RAM)"
    echo
    echo "For more information, see README.md and INSTALLATION.md"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up temporary files..."
    # Add any cleanup tasks here
}

# Error handler
error_handler() {
    print_error "An error occurred during setup. Please check the output above."
    cleanup
    exit 1
}

# Set up error handling
trap error_handler ERR

# Main setup function
main() {
    print_header "Audio Feedback Analyzer Setup"
    
    echo "This script will install and configure the Audio Feedback Analyzer"
    echo "on your system. It will:"
    echo "- Install system dependencies"
    echo "- Set up Python virtual environment"
    echo "- Install Python packages"
    echo "- Install and configure Ollama"
    echo "- Download AI model"
    echo "- Test the installation"
    echo
    
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Setup cancelled."
        exit 0
    fi
    
    # Detect operating system
    detect_os
    
    # Install system dependencies
    install_system_deps
    
    # Setup Python environment
    setup_python_env
    
    # Setup Ollama
    setup_ollama
    
    # Create test files
    create_test_files
    
    # Test installation
    test_installation
    
    # Show usage instructions
    show_usage
    
    print_success "Setup completed successfully!"
}

# Check if script is run with bash
if [ -z "$BASH_VERSION" ]; then
    echo "This script requires bash. Please run with: bash setup.sh"
    exit 1
fi

# Check for required tools
if ! command_exists curl; then
    print_error "curl is required but not installed. Please install curl first."
    exit 1
fi

# Run main function
main "$@"
