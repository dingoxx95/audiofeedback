#!/usr/bin/env python3
"""
Audio Feedback Analyzer Core Package

This package contains the core modules for the Audio Feedback Analyzer:
- audio_analyzer: Audio signal processing and analysis
- llm_generator: AI-powered feedback generation  
- visualization: Audio analysis visualization
- config: Configuration management
- genre_detector: Automatic genre detection
- utils: Utility functions and helpers

The package is designed to be modular, maintainable, and easily extensible.
"""

from .audio_analyzer import AudioAnalyzer
from .llm_generator import LLMFeedbackGenerator
from .config import Config
from .genre_detector import GenreDetector
from .visualization import AudioVisualizer
from .app import AudioFeedbackApp

__version__ = "1.0.0"
__author__ = "Audio Feedback Team"

# Export main classes for easy import
__all__ = [
    "AudioAnalyzer",
    "LLMFeedbackGenerator", 
    "Config",
    "GenreDetector",
    "AudioVisualizer",
    "AudioFeedbackApp"
]
