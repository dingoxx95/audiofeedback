#!/usr/bin/env python3
"""
Complete System Test for Audio Feedback Analyzer
Tests all components of the system and validates functionality.
"""

import sys
import os
import time
import subprocess
from pathlib import Path
import json
import traceback

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print formatted section"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")

def test_python_environment():
    """Test Python environment and version"""
    print_section("Testing Python Environment")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def test_required_packages():
    """Test if all required packages are available"""
    print_section("Testing Required Packages")
    
    required_packages = [
        'numpy', 'scipy', 'librosa', 'pydub', 'matplotlib', 
        'seaborn', 'requests', 'ollama', 'soundfile'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All required packages are available")
        return True

def test_ollama_connection():
    """Test Ollama connection and model availability"""
    print_section("Testing Ollama Connection")
    
    try:
        import ollama
        client = ollama.Client()
        
        # Test connection
        models = client.list()
        print("✅ Ollama connection successful")
        
        # List available models
        model_names = [model['name'] for model in models['models']]
        print(f"Available models: {model_names}")
        
        # Check for recommended models
        recommended = ['gemma2:27b', 'gemma2:9b', 'llama2:13b']
        found_models = [m for m in recommended if m in model_names]
        
        if found_models:
            print(f"✅ Found recommended models: {found_models}")
            return True, found_models[0]
        else:
            print(f"⚠️  No recommended models found. Available: {model_names}")
            if model_names:
                return True, model_names[0]  # Use first available
            else:
                print("❌ No models available")
                return False, None
                
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("Make sure Ollama is running: ollama serve")
        return False, None

def test_ffmpeg():
    """Test FFmpeg availability (required for pydub)"""
    print_section("Testing FFmpeg")
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg found: {version_line}")
            return True
        else:
            print("❌ FFmpeg not working properly")
            return False
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg command timed out")
        return False
    except FileNotFoundError:
        print("❌ FFmpeg not found in PATH")
        print("Install FFmpeg:")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        print("  Linux: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        return False
    except Exception as e:
        print(f"❌ Error testing FFmpeg: {e}")
        return False

def test_audio_analysis():
    """Test audio analysis functionality"""
    print_section("Testing Audio Analysis")
    
    try:
        # Generate test audio
        print("Generating test audio...")
        import numpy as np
        import soundfile as sf
        
        # Create simple test audio
        duration = 5.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Mix of frequencies
        audio = (0.5 * np.sin(2 * np.pi * 440 * t) +  # A4
                0.3 * np.sin(2 * np.pi * 880 * t) +   # A5
                0.1 * np.random.normal(0, 1, len(t)))  # Noise
        
        test_file = Path("test_audio_temp.wav")
        sf.write(test_file, audio, sample_rate)
        print(f"✅ Test audio created: {test_file}")
        
        # Test our analyzer
        print("Testing AudioAnalyzer...")
        
        # Import our module
        try:
            from audiofeedback import AudioAnalyzer
        except ImportError:
            # Try importing from local file
            import importlib.util
            spec = importlib.util.spec_from_file_location("audiofeedback", "audiofeedback.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            AudioAnalyzer = module.AudioAnalyzer
        
        analyzer = AudioAnalyzer()
        
        # Load and analyze
        if analyzer.load_audio(str(test_file)):
            print("✅ Audio loading successful")
            
            analysis = analyzer.get_complete_analysis()
            print("✅ Audio analysis successful")
            
            # Check analysis results
            expected_keys = ['file_info', 'dynamics', 'frequency_spectrum', 
                           'temporal_features', 'stereo_properties', 'harmonic_content']
            
            for key in expected_keys:
                if key in analysis:
                    print(f"✅ Analysis section: {key}")
                else:
                    print(f"❌ Missing analysis section: {key}")
            
            # Print some results
            print(f"Duration: {analysis['file_info']['duration_seconds']:.1f}s")
            print(f"RMS Level: {analysis['dynamics']['rms_db']:.1f} dB")
            print(f"Peak Level: {analysis['dynamics']['peak_db']:.1f} dB")
            
        else:
            print("❌ Audio loading failed")
            return False
        
        # Cleanup
        test_file.unlink()
        print("✅ Test audio cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio analysis test failed: {e}")
        traceback.print_exc()
        return False

def test_llm_feedback(model_name):
    """Test LLM feedback generation"""
    print_section("Testing LLM Feedback Generation")
    
    try:
        # Import LLM generator
        try:
            from audiofeedback import LLMFeedbackGenerator
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location("audiofeedback", "audiofeedback.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            LLMFeedbackGenerator = module.LLMFeedbackGenerator
        
        llm = LLMFeedbackGenerator(model_name)
        
        # Test connection
        if not llm.test_connection():
            print("❌ LLM connection test failed")
            return False
        
        print("✅ LLM connection successful")
        
        # Create sample analysis data
        sample_analysis = {
            'file_info': {
                'duration_seconds': 5.0,
                'sample_rate': 44100,
                'is_stereo': False,
                'channels': 1
            },
            'dynamics': {
                'rms_db': -18.5,
                'peak_db': -3.2,
                'crest_factor_db': 15.3,
                'dynamic_range_db': 12.8
            },
            'frequency_spectrum': {
                'spectral_centroid_mean': 2156.7,
                'frequency_bands': {
                    'bass_energy': 0.123,
                    'mids_energy': 0.234,
                    'high_mids_energy': 0.187
                }
            }
        }
        
        print("Generating feedback (this may take a moment)...")
        start_time = time.time()
        
        feedback = llm.generate_feedback(sample_analysis)
        
        generation_time = time.time() - start_time
        
        if feedback and not feedback.startswith("Error"):
            print(f"✅ Feedback generated successfully in {generation_time:.1f}s")
            print(f"Feedback length: {len(feedback)} characters")
            print("Sample feedback:")
            print("-" * 40)
            print(feedback[:500] + "..." if len(feedback) > 500 else feedback)
            print("-" * 40)
            return True
        else:
            print(f"❌ Feedback generation failed: {feedback}")
            return False
            
    except Exception as e:
        print(f"❌ LLM feedback test failed: {e}")
        traceback.print_exc()
        return False

def test_complete_pipeline(model_name):
    """Test the complete analysis pipeline"""
    print_section("Testing Complete Pipeline")
    
    try:
        # Generate test audio
        print("Creating test audio file...")
        
        import numpy as np
        import soundfile as sf
        
        # Create more complex test audio
        duration = 10.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Musical content
        audio = np.zeros_like(t)
        
        # Add bass line
        for i, freq in enumerate([110, 146.83, 164.81, 123.47]):  # A2, D3, E3, B2
            start = int(i * 2.5 * sample_rate)
            end = int((i + 2.5) * sample_rate)
            if end <= len(audio):
                audio[start:end] += 0.3 * np.sin(2 * np.pi * freq * t[start:end])
        
        # Add melody
        for i, freq in enumerate([440, 493.88, 523.25, 466.16]):  # A4, B4, C5, A#4
            start = int(i * 2.5 * sample_rate)
            end = int((i + 2.5) * sample_rate)
            if end <= len(audio):
                audio[start:end] += 0.2 * np.sin(2 * np.pi * freq * t[start:end])
        
        # Add some noise for realism
        audio += 0.05 * np.random.normal(0, 1, len(audio))
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        test_file = Path("test_complete_pipeline.wav")
        sf.write(test_file, audio, sample_rate)
        print(f"✅ Complex test audio created: {test_file}")
        
        # Test complete pipeline
        print("Running complete analysis pipeline...")
        
        try:
            from audiofeedback import AudioFeedbackApp
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location("audiofeedback", "audiofeedback.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            AudioFeedbackApp = module.AudioFeedbackApp
        
        app = AudioFeedbackApp()
        app.llm_generator.model_name = model_name
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        start_time = time.time()
        result = app.process_audio_file(str(test_file), str(output_dir))
        total_time = time.time() - start_time
        
        if "error" not in result:
            print(f"✅ Complete pipeline successful in {total_time:.1f}s")
            
            # Check output files
            analysis_file = output_dir / f"{test_file.stem}_analysis.json"
            feedback_file = output_dir / f"{test_file.stem}_feedback.md"
            
            if analysis_file.exists():
                print(f"✅ Analysis file created: {analysis_file}")
                
                # Validate JSON
                with open(analysis_file) as f:
                    analysis_data = json.load(f)
                print(f"✅ Analysis JSON is valid ({len(analysis_data)} sections)")
            
            if feedback_file.exists():
                print(f"✅ Feedback file created: {feedback_file}")
                
                with open(feedback_file) as f:
                    feedback_content = f.read()
                print(f"✅ Feedback content length: {len(feedback_content)} characters")
            
            # Test visualization
            print("Testing visualization...")
            try:
                viz_file = output_dir / f"{test_file.stem}_visualization.png"
                app.create_visualization(result['analysis'], str(viz_file))
                
                if viz_file.exists():
                    print(f"✅ Visualization created: {viz_file}")
                else:
                    print("⚠️  Visualization file not found")
            except Exception as e:
                print(f"⚠️  Visualization failed: {e}")
            
        else:
            print(f"❌ Pipeline failed: {result['error']}")
            return False
        
        # Cleanup
        test_file.unlink()
        print("✅ Test files cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete pipeline test failed: {e}")
        traceback.print_exc()
        return False

def run_performance_benchmark(model_name):
    """Run performance benchmark"""
    print_section("Performance Benchmark")
    
    try:
        import numpy as np
        import soundfile as sf
        
        # Create test files of different sizes
        test_cases = [
            ("short", 5.0),   # 5 seconds
            ("medium", 30.0), # 30 seconds
            ("long", 120.0),  # 2 minutes
        ]
        
        try:
            from audiofeedback import AudioFeedbackApp
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location("audiofeedback", "audiofeedback.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            AudioFeedbackApp = module.AudioFeedbackApp
        
        app = AudioFeedbackApp()
        app.llm_generator.model_name = model_name
        
        results = []
        
        for name, duration in test_cases:
            print(f"Testing {name} file ({duration}s)...")
            
            # Generate test audio
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.normal(0, 1, len(t))
            
            test_file = Path(f"benchmark_{name}.wav")
            sf.write(test_file, audio, sample_rate)
            
            # Time the analysis
            start_time = time.time()
            result = app.process_audio_file(str(test_file), None)  # No output saving
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            if "error" not in result:
                print(f"  ✅ {name}: {processing_time:.1f}s")
                results.append((name, duration, processing_time))
            else:
                print(f"  ❌ {name}: Failed")
            
            # Cleanup
            test_file.unlink()
        
        print("\nPerformance Summary:")
        for name, duration, proc_time in results:
            ratio = proc_time / duration
            print(f"  {name:6}: {proc_time:5.1f}s for {duration:5.1f}s audio (ratio: {ratio:.2f}x)")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

def main():
    """Run complete system test"""
    print_header("Audio Feedback Analyzer - Complete System Test")
    
    # Track test results
    tests = []
    
    # Test 1: Python environment
    tests.append(("Python Environment", test_python_environment()))
    
    # Test 2: Required packages
    tests.append(("Required Packages", test_required_packages()))
    
    # Test 3: FFmpeg
    tests.append(("FFmpeg", test_ffmpeg()))
    
    # Test 4: Ollama connection
    ollama_ok, model_name = test_ollama_connection()
    tests.append(("Ollama Connection", ollama_ok))
    
    if not ollama_ok:
        print("\n❌ Cannot proceed without Ollama connection")
        model_name = "gemma2:27b"  # Default for remaining tests
    
    # Test 5: Audio analysis
    tests.append(("Audio Analysis", test_audio_analysis()))
    
    # Test 6: LLM feedback (only if Ollama works)
    if ollama_ok:
        tests.append(("LLM Feedback", test_llm_feedback(model_name)))
        
        # Test 7: Complete pipeline
        tests.append(("Complete Pipeline", test_complete_pipeline(model_name)))
        
        # Test 8: Performance benchmark
        if input("\nRun performance benchmark? (y/N): ").lower() == 'y':
            tests.append(("Performance Benchmark", run_performance_benchmark(model_name)))
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your system is ready to use.")
        print("\nNext steps:")
        print("1. Try analyzing your own audio files")
        print("2. Experiment with different models")
        print("3. Check the documentation for advanced features")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Install FFmpeg for your platform")
        print("- Start Ollama: ollama serve")
        print("- Pull a model: ollama pull gemma2:27b")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
