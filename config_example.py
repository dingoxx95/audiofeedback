#!/usr/bin/env python3
"""
Configuration file for Audio Feedback Analyzer
Customize analysis parameters and LLM settings here.
"""

import os
from pathlib import Path

class Config:
    """Configuration class for audio analysis"""
    
    # Audio Analysis Settings
    AUDIO_SAMPLE_RATE = 44100  # Hz - CD quality standard, good balance between quality and performance
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    
    # Analysis Parameters
    ANALYSIS_WINDOW_SIZE = 0.1  # seconds - for windowed analysis
    SPECTRAL_BANDS = {
        'sub_bass': (20, 60),
        'bass': (60, 250),
        'low_mids': (250, 500),
        'mids': (500, 2000),
        'high_mids': (2000, 4000),
        'presence': (4000, 6000),
        'brilliance': (6000, 20000)
    }
    
    # LLM Settings
    DEFAULT_MODEL = "gemma3:27b"
    ALTERNATIVE_MODELS = [
        "gemma2:27b",    # Previous version
        "gemma2:9b",     # Faster, less detailed
        "llama2:13b",    # Alternative architecture
        "mistral:7b",    # Even faster option
    ]
    
    # Ollama Connection
    OLLAMA_HOST = "http://localhost:11434"
    OLLAMA_TIMEOUT = 300  # seconds
    
    # LLM Generation Parameters
    LLM_TEMPERATURE = 0.3  # Lower = more consistent, higher = more creative
    LLM_TOP_P = 0.9
    LLM_MAX_TOKENS = 4096
    
    # Professional Standards (for comparison)
    PROFESSIONAL_STANDARDS = {
        'streaming': {
            'lufs_target': -14.0,
            'peak_max': -1.0,
            'dynamic_range_min': 6.0
        },
        'cd_mastering': {
            'lufs_target': -9.0,
            'peak_max': -0.1,
            'dynamic_range_min': 8.0
        },
        'broadcast': {
            'lufs_target': -23.0,
            'peak_max': -3.0,
            'dynamic_range_min': 12.0
        }
    }
    
    # Quality Thresholds
    QUALITY_THRESHOLDS = {
        'excellent': {
            'dynamic_range_min': 15.0,
            'crest_factor_min': 10.0,
            'peak_max': -3.0
        },
        'good': {
            'dynamic_range_min': 10.0,
            'crest_factor_min': 6.0,
            'peak_max': -1.0
        },
        'acceptable': {
            'dynamic_range_min': 6.0,
            'crest_factor_min': 3.0,
            'peak_max': -0.1
        },
        'poor': {
            'dynamic_range_min': 0.0,
            'crest_factor_min': 0.0,
            'peak_max': 0.0
        }
    }
    
    # Output Settings
    OUTPUT_FORMATS = {
        'json': True,      # Technical analysis data
        'markdown': True,  # Human-readable feedback
        'visualization': True,  # Charts and graphs
        'csv': False,      # Tabular data export
    }
    
    # Visualization Settings
    VISUALIZATION_CONFIG = {
        'figure_size': (15, 10),
        'dpi': 300,
        'style': 'seaborn-v0_8',  # matplotlib style
        'color_palette': 'viridis',
        'font_size': 12
    }
    
    # Performance Settings
    PERFORMANCE = {
        'max_file_size_mb': 500,  # Maximum file size to process
        'chunk_size_seconds': 30,  # For very long files
        'parallel_workers': 2,     # For batch processing
        'memory_limit_gb': 8,      # Memory limit for processing
    }
    
    # Paths
    BASE_DIR = Path(__file__).parent
    TEMP_DIR = BASE_DIR / "temp"
    CACHE_DIR = BASE_DIR / "cache"
    
    # Advanced Analysis Features
    ADVANCED_FEATURES = {
        'pitch_detection': True,
        'tempo_analysis': True,
        'key_detection': False,  # Requires additional libraries
        'mood_analysis': False,  # Experimental feature
        'genre_classification': False,  # Requires training data
    }
    
    # Frequency Analysis Settings
    FFT_SETTINGS = {
        'window_type': 'hann',
        'overlap': 0.5,
        'nperseg': 2048,
    }
    
    # Custom Prompts for Different Analysis Types
    PROMPT_TEMPLATES = {
        'professional': """You are a professional audio engineer with 20+ years of experience in mixing and mastering. 
Analyze this audio data and provide detailed technical feedback suitable for professional audio production.""",
        
        'educational': """You are an audio engineering instructor. 
Analyze this audio data and provide educational feedback that explains concepts clearly for students learning audio production.""",
        
        'creative': """You are a creative audio producer focused on artistic expression. 
Analyze this audio data and provide feedback that balances technical quality with creative considerations.""",
        
        'broadcast': """You are a broadcast engineer ensuring content meets broadcasting standards. 
Analyze this audio data for compliance with broadcast specifications and loudness standards."""
    }
    
    # Genre-Specific Analysis Parameters
    GENRE_PROFILES = {
        'rock': {
            'expected_dynamic_range': (8, 15),
            'expected_rms': (-12, -8),
            'frequency_emphasis': ['low_mids', 'presence'],
            'stereo_width_target': (0.6, 1.2)
        },
        'metal': {
            'expected_dynamic_range': (6, 12),
            'expected_rms': (-10, -6),
            'frequency_emphasis': ['bass', 'low_mids', 'presence', 'brilliance'],
            'stereo_width_target': (0.8, 1.4)
        },
        'electronic': {
            'expected_dynamic_range': (6, 12),
            'expected_rms': (-10, -6),
            'frequency_emphasis': ['sub_bass', 'brilliance'],
            'stereo_width_target': (0.8, 1.5)
        },
        'drum_and_bass': {
            'expected_dynamic_range': (8, 14),
            'expected_rms': (-8, -4),
            'frequency_emphasis': ['sub_bass', 'bass', 'brilliance'],
            'stereo_width_target': (1.0, 1.6)
        },
        'classical': {
            'expected_dynamic_range': (15, 25),
            'expected_rms': (-20, -14),
            'frequency_emphasis': ['mids', 'high_mids'],
            'stereo_width_target': (0.4, 0.8)
        },
        'jazz': {
            'expected_dynamic_range': (12, 20),
            'expected_rms': (-16, -10),
            'frequency_emphasis': ['bass', 'mids', 'presence'],
            'stereo_width_target': (0.5, 1.0)
        },
        'podcast': {
            'expected_dynamic_range': (6, 12),
            'expected_rms': (-18, -14),
            'frequency_emphasis': ['mids', 'high_mids'],
            'stereo_width_target': (0.0, 0.3)  # Usually mono or narrow
        }
    }
    
    # Warning Thresholds
    WARNING_THRESHOLDS = {
        'clipping_detected': 0.95,  # Peak amplitude threshold
        'excessive_loudness': -6.0,  # RMS dB threshold
        'insufficient_headroom': -0.5,  # Peak dB threshold
        'poor_dynamics': 3.0,  # Dynamic range dB threshold
        'phase_correlation_issue': 0.3,  # Stereo correlation threshold
    }
    
    # File Processing Options
    FILE_PROCESSING = {
        'auto_normalize': False,  # Normalize input before analysis
        'remove_dc_offset': True,  # Remove DC bias
        'apply_fade_in': False,   # Apply fade to avoid clicks
        'apply_fade_out': False,  # Apply fade to avoid clicks
        'mono_conversion': False,  # Force mono conversion
    }
    
    # Caching Settings
    CACHE_SETTINGS = {
        'enabled': True,
        'max_size_mb': 1000,  # Maximum cache size
        'ttl_hours': 24,      # Time to live for cached results
        'cache_audio_analysis': True,
        'cache_llm_responses': True,
    }
    
    # Logging Configuration
    LOGGING = {
        'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_logging': True,
        'console_logging': True,
        'max_log_size_mb': 10,
        'backup_count': 5,
    }
    
    # Development/Debug Settings
    DEBUG_SETTINGS = {
        'save_intermediate_data': False,
        'verbose_analysis': False,
        'profile_performance': False,
        'save_spectrograms': False,
        'export_raw_features': False,
    }
    
    @classmethod
    def get_model_config(cls, model_name: str) -> dict:
        """Get optimal configuration for specific model"""
        model_configs = {
            'gemma3:27b': {
                'temperature': 0.3,
                'top_p': 0.9,
                'max_tokens': 4096,
                'context_window': 8192
            },
            'gemma2:27b': {
                'temperature': 0.3,
                'top_p': 0.9,
                'max_tokens': 4096,
                'context_window': 8192
            },
            'gemma2:9b': {
                'temperature': 0.4,
                'top_p': 0.85,
                'max_tokens': 3072,
                'context_window': 4096
            },
            'llama2:13b': {
                'temperature': 0.35,
                'top_p': 0.9,
                'max_tokens': 3584,
                'context_window': 4096
            },
            'mistral:7b': {
                'temperature': 0.4,
                'top_p': 0.8,
                'max_tokens': 2048,
                'context_window': 8192
            }
        }
        return model_configs.get(model_name, model_configs['gemma3:27b'])
    
    @classmethod
    def get_genre_config(cls, genre: str) -> dict:
        """Get analysis configuration for specific genre"""
        return cls.GENRE_PROFILES.get(genre.lower(), cls.GENRE_PROFILES['rock'])
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check paths
        try:
            cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)
            cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create directories: {e}")
        
        # Check sample rate
        if cls.AUDIO_SAMPLE_RATE < 8000 or cls.AUDIO_SAMPLE_RATE > 192000:
            errors.append(f"Invalid sample rate: {cls.AUDIO_SAMPLE_RATE}")
        
        # Check model availability
        try:
            import ollama #type: ignore
            client = ollama.Client()
            available_models = [m['name'] for m in client.list()['models']]
            if cls.DEFAULT_MODEL not in available_models:
                errors.append(f"Default model {cls.DEFAULT_MODEL} not available")
        except:
            errors.append("Cannot connect to Ollama")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True


# Environment-specific overrides
if os.getenv('AUDIO_FEEDBACK_ENV') == 'development':
    Config.DEBUG_SETTINGS['save_intermediate_data'] = True
    Config.DEBUG_SETTINGS['verbose_analysis'] = True
    Config.LOGGING['level'] = 'DEBUG'

elif os.getenv('AUDIO_FEEDBACK_ENV') == 'production':
    Config.CACHE_SETTINGS['enabled'] = True
    Config.PERFORMANCE['parallel_workers'] = 4
    Config.LOGGING['level'] = 'WARNING'

# GPU acceleration settings (if available)
if os.getenv('CUDA_VISIBLE_DEVICES'):
    Config.PERFORMANCE['gpu_acceleration'] = True
    Config.PERFORMANCE['parallel_workers'] = 1  # Less CPU parallelism when using GPU

# Example usage and validation
if __name__ == "__main__":
    print("Audio Feedback Analyzer Configuration")
    print("=" * 50)
    print(f"Sample Rate: {Config.AUDIO_SAMPLE_RATE} Hz")
    print(f"Default Model: {Config.DEFAULT_MODEL}")
    print(f"Supported Formats: {Config.SUPPORTED_FORMATS}")
    print(f"Output Directory: {Config.BASE_DIR}")
    
    print("\nValidating configuration...")
    if Config.validate_config():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has errors")
        
    print(f"\nFrequency Bands:")
    for band, (low, high) in Config.SPECTRAL_BANDS.items():
        print(f"  {band}: {low}-{high} Hz")
        
    print(f"\nGenre Profiles Available: {list(Config.GENRE_PROFILES.keys())}")
    print(f"Alternative Models: {Config.ALTERNATIVE_MODELS}")

    