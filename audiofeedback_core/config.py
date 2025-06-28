#!/usr/bin/env python3
"""
Configuration Management Module

This module contains all configuration settings for the Audio Feedback Analyzer.
It provides a centralized way to manage analysis parameters, genre profiles,
LLM settings, and other system configurations.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple


class Config:
    """
    Central configuration class for audio analysis system.
    
    This class contains all configuration parameters organized into logical groups:
    - Audio analysis settings (sample rates, formats, etc.)
    - Genre profiles for different music styles
    - LLM settings and model configurations
    - Performance and processing options
    - Quality thresholds and professional standards
    """
    
    # ============================================================================
    # AUDIO ANALYSIS SETTINGS
    # ============================================================================
    
    # Sample rate for audio processing (44.1kHz = CD quality)
    AUDIO_SAMPLE_RATE: int = 44100
    
    # Supported audio file formats
    SUPPORTED_FORMATS: set = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    
    # Analysis window size for temporal analysis (in seconds)
    ANALYSIS_WINDOW_SIZE: float = 0.1
    
    # Frequency bands for spectral analysis (Hz ranges)
    SPECTRAL_BANDS: Dict[str, Tuple[int, int]] = {
        'sub_bass': (20, 60),        # Sub-bass frequencies
        'bass': (60, 250),           # Bass frequencies  
        'low_mids': (250, 500),      # Low midrange
        'mids': (500, 2000),         # Midrange
        'high_mids': (2000, 4000),   # High midrange
        'presence': (4000, 6000),    # Presence range
        'brilliance': (6000, 20000)  # High frequencies/air
    }
    
    # ============================================================================
    # LLM SETTINGS  
    # ============================================================================
    
    # Default AI model for feedback generation
    DEFAULT_MODEL: str = "gemma3:27b"
    
    # Alternative models (ordered by quality/speed trade-off)
    ALTERNATIVE_MODELS: List[str] = [
        "gemma2:27b",    # Previous version, excellent quality
        "gemma2:9b",     # Faster processing, good quality
        "llama2:13b",    # Alternative architecture
        "mistral:7b",    # Fastest option, basic quality
    ]
    
    # Ollama connection settings
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 300  # seconds
    
    # LLM generation parameters
    LLM_TEMPERATURE: float = 0.3  # Lower = more consistent
    LLM_TOP_P: float = 0.9
    LLM_MAX_TOKENS: int = 4096
    
    # ============================================================================
    # GENRE PROFILES
    # ============================================================================
    
    # Genre-specific analysis parameters and expectations
    GENRE_PROFILES: Dict[str, Dict[str, Any]] = {
        'rock': {
            'expected_dynamic_range': (8, 15),
            'expected_rms': (-12, -8),
            'frequency_emphasis': ['low_mids', 'presence'],
            'stereo_width_target': (0.6, 1.2),
            'description': 'Balanced dynamics with emphasis on guitar frequencies'
        },
        'metal': {
            'expected_dynamic_range': (6, 12),
            'expected_rms': (-10, -6),
            'frequency_emphasis': ['bass', 'low_mids', 'presence', 'brilliance'],
            'stereo_width_target': (0.8, 1.4),
            'description': 'Aggressive sound with tight compression and wide stereo field'
        },
        'electronic': {
            'expected_dynamic_range': (6, 12),
            'expected_rms': (-10, -6),
            'frequency_emphasis': ['sub_bass', 'brilliance'],
            'stereo_width_target': (0.8, 1.5),
            'description': 'Synthetic sounds with emphasis on extremes of frequency spectrum'
        },
        'drum_and_bass': {
            'expected_dynamic_range': (8, 14),
            'expected_rms': (-8, -4),
            'frequency_emphasis': ['sub_bass', 'bass', 'brilliance'],
            'stereo_width_target': (1.0, 1.6),
            'description': 'High-energy with powerful low end and crisp highs'
        },
        'classical': {
            'expected_dynamic_range': (15, 25),
            'expected_rms': (-20, -14),
            'frequency_emphasis': ['mids', 'high_mids'],
            'stereo_width_target': (0.4, 0.8),
            'description': 'Natural dynamics with acoustic instrument frequency balance'
        },
        'jazz': {
            'expected_dynamic_range': (12, 20),
            'expected_rms': (-16, -10),
            'frequency_emphasis': ['bass', 'mids', 'presence'],
            'stereo_width_target': (0.5, 1.0),
            'description': 'Organic sound with room for instrumental expression'
        },
        'podcast': {
            'expected_dynamic_range': (6, 12),
            'expected_rms': (-18, -14),
            'frequency_emphasis': ['mids', 'high_mids'],
            'stereo_width_target': (0.0, 0.3),
            'description': 'Speech-optimized with focus on clarity and intelligibility'
        }
    }
    
    # ============================================================================
    # PROFESSIONAL STANDARDS
    # ============================================================================
    
    # Industry standard loudness targets for different platforms
    PROFESSIONAL_STANDARDS: Dict[str, Dict[str, float]] = {
        'streaming': {
            'lufs_target': -14.0,        # Spotify, Apple Music, etc.
            'peak_max': -1.0,
            'dynamic_range_min': 6.0
        },
        'cd_mastering': {
            'lufs_target': -9.0,         # Physical CD release
            'peak_max': -0.1,
            'dynamic_range_min': 8.0
        },
        'broadcast': {
            'lufs_target': -23.0,        # TV/Radio broadcast
            'peak_max': -3.0,
            'dynamic_range_min': 12.0
        }
    }
    
    # Quality assessment thresholds
    QUALITY_THRESHOLDS: Dict[str, Dict[str, float]] = {
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
    
    # Audio processing warning thresholds
    WARNING_THRESHOLDS: Dict[str, float] = {
        'clipping_detected': 0.95,           # Peak amplitude threshold
        'excessive_loudness': -6.0,          # RMS dB threshold
        'insufficient_headroom': -0.5,       # Peak dB threshold  
        'poor_dynamics': 3.0,                # Dynamic range dB threshold
        'phase_correlation_issue': 0.3,      # Stereo correlation threshold
    }
    
    # ============================================================================
    # SYSTEM SETTINGS
    # ============================================================================
    
    # Output format settings
    OUTPUT_FORMATS: Dict[str, bool] = {
        'json': True,          # Technical analysis data
        'markdown': True,      # Human-readable feedback
        'visualization': True, # Charts and graphs
        'csv': False,          # Tabular data export
    }
    
    # Visualization configuration
    VISUALIZATION_CONFIG: Dict[str, Any] = {
        'figure_size': (15, 10),
        'dpi': 300,
        'style': 'seaborn-v0_8',
        'color_palette': 'viridis',
        'font_size': 12
    }
    
    # Performance and processing settings
    PERFORMANCE: Dict[str, Any] = {
        'max_file_size_mb': 500,     # Maximum file size to process
        'chunk_size_seconds': 30,    # For very long files
        'parallel_workers': 2,       # For batch processing
        'memory_limit_gb': 8,        # Memory limit for processing
    }
    
    # File processing options
    FILE_PROCESSING: Dict[str, bool] = {
        'auto_normalize': False,     # Normalize input before analysis
        'remove_dc_offset': True,    # Remove DC bias
        'apply_fade_in': False,      # Apply fade to avoid clicks
        'apply_fade_out': False,     # Apply fade to avoid clicks
        'mono_conversion': False,    # Force mono conversion
    }
    
    # Advanced analysis features
    ADVANCED_FEATURES: Dict[str, bool] = {
        'pitch_detection': True,
        'tempo_analysis': True,
        'key_detection': False,      # Requires additional libraries
        'mood_analysis': False,      # Experimental feature
        'genre_classification': False, # Requires training data
    }
    
    # FFT settings for spectral analysis
    FFT_SETTINGS: Dict[str, Any] = {
        'window_type': 'hann',
        'overlap': 0.5,
        'nperseg': 2048,
    }
    
    # Caching settings
    CACHE_SETTINGS: Dict[str, Any] = {
        'enabled': True,
        'max_size_mb': 1000,         # Maximum cache size
        'ttl_hours': 24,             # Time to live for cached results
        'cache_audio_analysis': True,
        'cache_llm_responses': True,
    }
    
    # Logging configuration
    LOGGING: Dict[str, Any] = {
        'level': 'INFO',             # DEBUG, INFO, WARNING, ERROR
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_logging': True,
        'console_logging': True,
        'max_log_size_mb': 10,
        'backup_count': 5,
    }
    
    # Development/debug settings
    DEBUG_SETTINGS: Dict[str, bool] = {
        'save_intermediate_data': False,
        'verbose_analysis': False,
        'profile_performance': False,
        'save_spectrograms': False,
        'export_raw_features': False,
    }
    
    # Directory paths
    BASE_DIR: Path = Path(__file__).parent.parent
    TEMP_DIR: Path = BASE_DIR / "temp"
    CACHE_DIR: Path = BASE_DIR / "cache"
    
    # ============================================================================
    # PROMPT TEMPLATES
    # ============================================================================
    
    # Custom prompts for different analysis contexts
    PROMPT_TEMPLATES: Dict[str, str] = {
        'professional': """You are a professional audio engineer with 20+ years of experience in mixing and mastering. 
Analyze this audio data and provide detailed technical feedback suitable for professional audio production.""",
        
        'educational': """You are an audio engineering instructor. 
Analyze this audio data and provide educational feedback that explains concepts clearly for students learning audio production.""",
        
        'creative': """You are a creative audio producer focused on artistic expression. 
Analyze this audio data and provide feedback that balances technical quality with creative considerations.""",
        
        'broadcast': """You are a broadcast engineer ensuring content meets broadcasting standards. 
Analyze this audio data for compliance with broadcast specifications and loudness standards."""
    }
    
    # ============================================================================
    # CLASS METHODS
    # ============================================================================
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """
        Get optimal configuration parameters for a specific LLM model.
        
        Args:
            model_name: Name of the model (e.g., 'gemma3:27b')
            
        Returns:
            Dictionary containing model-specific configuration parameters
        """
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
    def get_genre_config(cls, genre: str) -> Dict[str, Any]:
        """
        Get analysis configuration for a specific music genre.
        
        Args:
            genre: Genre name (e.g., 'rock', 'metal', 'classical')
            
        Returns:
            Dictionary containing genre-specific analysis parameters
        """
        return cls.GENRE_PROFILES.get(genre.lower(), cls.GENRE_PROFILES['rock'])
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate configuration settings and system requirements.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        errors = []
        
        # Check if directories can be created
        try:
            cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)
            cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create directories: {e}")
        
        # Validate sample rate
        if cls.AUDIO_SAMPLE_RATE < 8000 or cls.AUDIO_SAMPLE_RATE > 192000:
            errors.append(f"Invalid sample rate: {cls.AUDIO_SAMPLE_RATE}")
        
        # Check Ollama connection and model availability
        try:
            import ollama
            client = ollama.Client()
            response = client.list()
            
            # Handle both possible response formats
            if hasattr(response, 'models'):
                available_models = [m.model for m in response.models]
            else:
                available_models = [m['name'] for m in response['models']]
                
            if cls.DEFAULT_MODEL not in available_models:
                errors.append(f"Default model {cls.DEFAULT_MODEL} not available")
        except Exception as e:
            errors.append(f"Cannot connect to Ollama: {e}")
        
        # Report errors if any
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    @classmethod
    def get_supported_genres(cls) -> List[str]:
        """
        Get list of supported genre names.
        
        Returns:
            List of supported genre strings
        """
        return list(cls.GENRE_PROFILES.keys())
    
    @classmethod
    def get_quality_assessment(cls, dynamic_range: float, crest_factor: float, 
                               peak_db: float) -> str:
        """
        Assess audio quality based on technical parameters.
        
        Args:
            dynamic_range: Dynamic range in dB
            crest_factor: Crest factor in dB  
            peak_db: Peak level in dB
            
        Returns:
            Quality assessment string ('excellent', 'good', 'acceptable', 'poor')
        """
        # Check against each quality threshold
        for quality, thresholds in cls.QUALITY_THRESHOLDS.items():
            if (dynamic_range >= thresholds['dynamic_range_min'] and
                crest_factor >= thresholds['crest_factor_min'] and
                peak_db <= thresholds['peak_max']):
                return quality
        
        return 'poor'


# ============================================================================
# ENVIRONMENT-SPECIFIC OVERRIDES
# ============================================================================

# Apply development environment settings
if os.getenv('AUDIO_FEEDBACK_ENV') == 'development':
    Config.DEBUG_SETTINGS['save_intermediate_data'] = True
    Config.DEBUG_SETTINGS['verbose_analysis'] = True
    Config.LOGGING['level'] = 'DEBUG'

# Apply production environment settings
elif os.getenv('AUDIO_FEEDBACK_ENV') == 'production':
    Config.CACHE_SETTINGS['enabled'] = True
    Config.PERFORMANCE['parallel_workers'] = 4
    Config.LOGGING['level'] = 'WARNING'

# Apply GPU acceleration settings if available
if os.getenv('CUDA_VISIBLE_DEVICES'):
    Config.PERFORMANCE['gpu_acceleration'] = True
    Config.PERFORMANCE['parallel_workers'] = 1  # Less CPU parallelism when using GPU


# ============================================================================
# MODULE TEST
# ============================================================================

if __name__ == "__main__":
    """Test configuration module functionality."""
    print("Audio Feedback Analyzer Configuration")
    print("=" * 50)
    print(f"Sample Rate: {Config.AUDIO_SAMPLE_RATE} Hz")
    print(f"Default Model: {Config.DEFAULT_MODEL}")
    print(f"Supported Formats: {Config.SUPPORTED_FORMATS}")
    print(f"Base Directory: {Config.BASE_DIR}")
    
    print("\nValidating configuration...")
    if Config.validate_config():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has errors")
        
    print(f"\nFrequency Bands:")
    for band, (low, high) in Config.SPECTRAL_BANDS.items():
        print(f"  {band}: {low}-{high} Hz")
        
    print(f"\nSupported Genres: {Config.get_supported_genres()}")
    print(f"Alternative Models: {Config.ALTERNATIVE_MODELS}")
