#!/usr/bin/env python3
"""
LLM Feedback Generator Module

This module handles the generation of professional audio feedback using Large Language Models (LLM).
It integrates with Ollama to provide detailed, contextual analysis and recommendations based on
audio analysis data and genre-specific requirements.

Author: Audio Feedback Analyzer
License: MIT
"""

import json
import time
from typing import Dict, Any, Optional, List
import warnings

try:
    import ollama  # type: ignore
except ImportError as e:
    print(f"Missing LLM libraries: {e}")
    print("Install with: pip install ollama")
    ollama = None

from .config import Config


class LLMFeedbackGenerator:
    """
    Generate professional audio feedback using Ollama/Gemma.
    
    This class handles the communication with LLM models to generate detailed,
    professional feedback based on audio analysis data. It supports multiple
    models and can adapt the feedback style based on the target genre and
    analysis type.
    
    Attributes:
        model_name (str): Name of the LLM model to use
        client: Ollama client instance for model communication
        
    Example:
        >>> generator = LLMFeedbackGenerator("gemma3:27b")
        >>> if generator.test_connection():
        ...     feedback = generator.generate_feedback(analysis_data, genre="rock")
        ...     print(feedback)
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the LLM feedback generator.
        
        Args:
            model_name (str, optional): Name of the LLM model to use.
                                      Defaults to Config.DEFAULT_MODEL.
        
        Raises:
            RuntimeError: If Ollama is not available or properly installed.
        """
        if ollama is None:
            raise RuntimeError("Ollama library not available. Please install with: pip install ollama")
            
        self.model_name = model_name or Config.DEFAULT_MODEL
        self.client = ollama.Client(host=Config.OLLAMA_HOST)
        
    def test_connection(self) -> bool:
        """
        Test if Ollama is active and the model is available.
        
        Returns:
            bool: True if connection is successful and model is available, False otherwise.
            
        Example:
            >>> generator = LLMFeedbackGenerator()
            >>> if generator.test_connection():
            ...     print("LLM is ready for use")
            ... else:
            ...     print("LLM is not available")
        """
        try:
            # Test Ollama connection
            available_models = self.client.list()
            
            # Safe extraction of model names
            models_list = available_models.get('models', [])
            model_names = []
            for model in models_list:
                if isinstance(model, dict) and 'name' in model:
                    model_names.append(model['name'])
                elif hasattr(model, 'name'):
                    model_names.append(model.name)
                elif hasattr(model, 'model'):
                    model_names.append(model.model)
                else:
                    # Try to extract model name from string representation
                    model_str = str(model)
                    if "model='" in model_str:
                        # Extract model name from format: model='gemma3:27b' ...
                        start = model_str.find("model='") + 7
                        end = model_str.find("'", start)
                        if end > start:
                            model_names.append(model_str[start:end])
                        else:
                            model_names.append(model_str)
                    else:
                        model_names.append(model_str)
            
            if self.model_name not in model_names:
                print(f"Warning: Model '{self.model_name}' not found in available models: {model_names}")
                
                # Try to find an alternative model
                for alt_model in Config.ALTERNATIVE_MODELS:
                    if alt_model in model_names:
                        print(f"Using alternative model: {alt_model}")
                        self.model_name = alt_model
                        return True
                        
                print("No suitable model found. Please pull a model first:")
                print(f"ollama pull {self.model_name}")
                return False
                
            return True
            
        except Exception as e:
            error_msg = str(e) if str(e) else f"{type(e).__name__}: {getattr(e, 'message', 'Unknown error')}"
            print(f"Cannot connect to Ollama: {error_msg}")
            print("Make sure Ollama is running: ollama serve")
            print("Or install Ollama from: https://ollama.ai/")
            return False
    
    def create_analysis_prompt(self, analysis_data: Dict[str, Any], genre: str = None) -> str:
        """
        Create detailed prompt for LLM analysis based on audio data and genre.
        
        This recreates the original comprehensive prompt structure that generates
        detailed, professional feedback in the original format.
        """
        
        # Extract genre information
        detected_genre = analysis_data.get('genre_detection', {}).get('detected_genre', 'unknown')
        genre_profile = analysis_data.get('genre_profile', {})
        
        # Use provided genre or detected genre
        target_genre = genre or detected_genre
        
        # Build genre-specific context
        genre_context = ""
        if target_genre != 'unknown' and Config:
            profile = Config.get_genre_config(target_genre)
            genre_context = f"""
GENRE CONTEXT:
Detected/Target Genre: {target_genre.replace('_', ' ').title()}
Expected Dynamic Range: {profile['expected_dynamic_range'][0]}-{profile['expected_dynamic_range'][1]} dB
Expected RMS Level: {profile['expected_rms'][0]} to {profile['expected_rms'][1]} dB
Key Frequency Bands: {', '.join(profile['frequency_emphasis'])}
Target Stereo Width: {profile['stereo_width_target'][0]}-{profile['stereo_width_target'][1]}

When analyzing, consider these genre-specific expectations and compare the audio against typical {target_genre.replace('_', ' ')} productions.
"""

        # Extract specific analysis components
        dynamics = analysis_data.get('dynamics', {})
        spectrum = analysis_data.get('frequency_spectrum', {})
        temporal = analysis_data.get('temporal_features', {})
        stereo = analysis_data.get('stereo_analysis', {})
        harmonic = analysis_data.get('harmonic_content', {})
        file_info = analysis_data.get('file_info', {})

        prompt = f"""You are a professional audio engineer and mastering specialist with 20+ years of experience.

Analyze the following technical audio data and provide comprehensive feedback as if you were reviewing a mix/master for a client.

AUDIO FILE INFORMATION:
- Filename: {file_info.get('filename', 'Unknown')}
- Duration: {file_info.get('duration_seconds', 0):.2f} seconds
- Sample Rate: {file_info.get('sample_rate', 0)} Hz
- Channels: {file_info.get('channels', 1)}
- Bit Depth: {file_info.get('bit_depth', 16)} bits

DYNAMICS ANALYSIS:
- RMS Level: {dynamics.get('rms_db', 0):.2f} dB
- Peak Level: {dynamics.get('peak_db', 0):.2f} dB
- Crest Factor: {dynamics.get('crest_factor_db', 0):.2f} dB
- Dynamic Range: {dynamics.get('dynamic_range_db', 0):.2f} dB
- LUFS (estimated): {dynamics.get('lufs_estimate', 0):.1f} LUFS

FREQUENCY SPECTRUM ANALYSIS:
- Spectral Centroid: {spectrum.get('spectral_centroid_mean', 0):.0f} Hz
- Spectral Bandwidth: {spectrum.get('spectral_bandwidth_mean', 0):.0f} Hz
- Spectral Rolloff: {spectrum.get('spectral_rolloff_mean', 0):.0f} Hz
- Zero Crossing Rate: {spectrum.get('zero_crossing_rate_mean', 0):.4f}
- Spectral Contrast: {spectrum.get('spectral_contrast_mean', 0):.2f}

FREQUENCY BAND ENERGY DISTRIBUTION:
{json.dumps(spectrum.get('frequency_bands', {}), indent=2)}

TEMPORAL CHARACTERISTICS:
- Tempo: {temporal.get('tempo', 0):.1f} BPM
- Onset Rate: {temporal.get('onset_rate', 0):.2f} onsets/second
- Onset Strength: {temporal.get('onset_strength_mean', 0):.4f}

STEREO IMAGING:
{json.dumps(stereo, indent=2) if stereo else f"Audio is {file_info.get('channels', 1)}-channel {'stereo' if file_info.get('is_stereo', False) else 'mono'}"}

HARMONIC CONTENT:
- Harmonic Ratio: {harmonic.get('harmonic_ratio', 0):.3f}
- Percussive Ratio: {harmonic.get('percussive_ratio', 0):.3f}
- Fundamental Frequency: {harmonic.get('fundamental_freq', 0):.1f} Hz (if detected)

{genre_context}

PROFESSIONAL STANDARDS FOR REFERENCE:
Streaming (Spotify/Apple): -14 LUFS, -1 dB peak, 6+ dB dynamic range
CD Mastering: -9 LUFS, -0.1 dB peak, 8+ dB dynamic range
Broadcast: -23 LUFS, -3 dB peak, 12+ dB dynamic range

QUALITY ASSESSMENT CRITERIA:
Excellent: 15+ dB dynamic range, 10+ dB crest factor, <-3 dB peak
Good: 10+ dB dynamic range, 6+ dB crest factor, <-1 dB peak
Acceptable: 6+ dB dynamic range, 3+ dB crest factor, <-0.1 dB peak

Please provide a comprehensive analysis that includes:

1. OVERALL ASSESSMENT
   - Rate the audio quality (Excellent/Good/Acceptable/Poor) with justification
   - Identify the most significant strengths and weaknesses

2. TECHNICAL ANALYSIS
   - Loudness and dynamics evaluation against professional standards
   - Frequency balance assessment (too bright, too dark, well-balanced)
   - Stereo imaging quality (if applicable)
   - Headroom and peak management

3. GENRE-SPECIFIC FEEDBACK
   - How well does this audio fit the expected characteristics for {target_genre}?
   - Genre-appropriate loudness and dynamics
   - Frequency emphasis alignment with genre expectations

4. SPECIFIC RECOMMENDATIONS
   - Concrete mixing/mastering suggestions with dB values where applicable
   - EQ recommendations for specific frequency ranges
   - Compression/limiting suggestions
   - Stereo enhancement or correction advice

5. PRIORITY ACTIONS
   - List the top 3 most important improvements in order of priority
   - Provide specific technical parameters for each recommendation

Format your response exactly like this professional audio analysis report:
- Use section headers with ★★ prefixes
- Provide detailed technical analysis with specific values
- Include frequency ranges, dB values, and technical parameters
- Give concrete, actionable recommendations
- Write as if you're an experienced mastering engineer reviewing a client's work

Keep the feedback professional, specific, and actionable. Use technical terminology appropriate for audio engineers while explaining concepts clearly. Provide concrete dB values and frequency ranges in your recommendations."""

        return prompt
    
    def generate_feedback(self, analysis_data: Dict[str, Any], genre: str = None, 
                         analysis_type: str = 'professional') -> str:
        """
        Generate comprehensive feedback using the LLM model.
        
        This method sends the analysis data to the LLM and generates detailed,
        professional feedback tailored to the specified genre and analysis type.
        
        Args:
            analysis_data (Dict[str, Any]): Complete audio analysis results
            genre (str, optional): Target genre for context-specific feedback
            analysis_type (str): Type of analysis ('professional', 'educational', 
                                'creative', 'broadcast'). Defaults to 'professional'.
                                
        Returns:
            str: Generated feedback text with detailed recommendations
            
        Raises:
            RuntimeError: If LLM generation fails or model is not available
            
        Example:
            >>> feedback = generator.generate_feedback(analysis, genre="jazz", 
            ...                                       analysis_type="educational")
            >>> print(feedback[:100])  # Preview first 100 characters
        """
        
        if not self.test_connection():
            raise RuntimeError(f"LLM model '{self.model_name}' is not available")
        
        # Create the analysis prompt with all the data
        prompt = self.create_analysis_prompt(analysis_data, genre)
        
        try:
            print(f"Generating feedback using {self.model_name}...")
            start_time = time.time()
            
            # Generate response using Ollama
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,  # Low temperature for consistent, professional output
                    'top_p': 0.9,
                    'num_predict': 2048  # Allow for detailed responses
                }
            )
            
            generation_time = time.time() - start_time
            print(f"Feedback generated in {generation_time:.1f} seconds")
            
            feedback = response['response']
            
            # Add generation metadata
            metadata = f"\n\n---\n*Feedback generated by {self.model_name} in {generation_time:.1f}s*"
            
            return feedback + metadata
            
        except Exception as e:
            error_msg = f"Error generating feedback with {self.model_name}: {e}"
            print(error_msg)
            
            # Return fallback feedback based on analysis data
            return self._generate_fallback_feedback(analysis_data, genre)
    
    def _generate_fallback_feedback(self, analysis_data: Dict[str, Any], genre: str = None) -> str:
        """
        Generate basic feedback without LLM when model is unavailable.
        
        This method provides a technical summary and basic recommendations
        based on the analysis data when the LLM model is not available.
        
        Args:
            analysis_data (Dict[str, Any]): Complete audio analysis results
            genre (str, optional): Target genre for basic recommendations
            
        Returns:
            str: Basic technical feedback and recommendations
        """
        
        dynamics = analysis_data.get('dynamics', {})
        spectrum = analysis_data.get('frequency_spectrum', {})
        file_info = analysis_data.get('file_info', {})
        
        # Basic quality assessment
        rms_db = dynamics.get('rms_db', -20)
        peak_db = dynamics.get('peak_db', -6)
        dynamic_range = dynamics.get('dynamic_range_db', 10)
        
        # Determine quality level
        if dynamic_range >= 15 and peak_db <= -3:
            quality = "Excellent"
        elif dynamic_range >= 10 and peak_db <= -1:
            quality = "Good"
        elif dynamic_range >= 6 and peak_db <= -0.1:
            quality = "Acceptable"
        else:
            quality = "Poor"
        
        feedback = f"""AUDIO ANALYSIS REPORT (Technical Summary)
{'='*50}

LLM feedback generation is currently unavailable. Here's a technical summary:

OVERALL QUALITY: {quality}

TECHNICAL SPECIFICATIONS:
- Duration: {file_info.get('duration_seconds', 0):.2f} seconds
- RMS Level: {rms_db:.2f} dB
- Peak Level: {peak_db:.2f} dB  
- Dynamic Range: {dynamic_range:.2f} dB
- Sample Rate: {file_info.get('sample_rate', 0)} Hz

BASIC RECOMMENDATIONS:

Loudness:
"""
        
        # Basic loudness recommendations
        if rms_db > -6:
            feedback += "- CRITICAL: Excessive loudness detected. Reduce overall level by at least 6 dB.\n"
        elif rms_db > -9:
            feedback += "- Consider reducing loudness for better dynamic range.\n"
        elif rms_db < -20:
            feedback += "- Audio may be too quiet. Consider increasing overall level.\n"
        else:
            feedback += "- Loudness level is appropriate.\n"
        
        # Peak level recommendations
        if peak_db > -0.1:
            feedback += "- CRITICAL: Clipping detected. Apply limiting to reduce peaks below -0.1 dB.\n"
        elif peak_db > -1:
            feedback += "- Insufficient headroom. Consider reducing peaks to -1 dB or lower.\n"
        else:
            feedback += "- Peak levels are well controlled.\n"
        
        # Dynamic range recommendations
        if dynamic_range < 6:
            feedback += "- Poor dynamic range. Reduce compression and limiting.\n"
        elif dynamic_range < 10:
            feedback += "- Limited dynamic range. Consider lighter compression.\n"
        else:
            feedback += "- Good dynamic range preservation.\n"
        
        # Genre-specific notes
        if genre and genre != 'unknown':
            genre_config = Config.get_genre_config(genre)
            expected_rms = genre_config['expected_rms']
            expected_dr = genre_config['expected_dynamic_range']
            
            feedback += f"\nGENRE CONSIDERATIONS ({genre.upper()}):\n"
            feedback += f"- Expected RMS: {expected_rms[0]} to {expected_rms[1]} dB\n"
            feedback += f"- Expected Dynamic Range: {expected_dr[0]}-{expected_dr[1]} dB\n"
            
            if rms_db < expected_rms[0]:
                feedback += "- Audio is quieter than typical for this genre.\n"
            elif rms_db > expected_rms[1]:
                feedback += "- Audio is louder than typical for this genre.\n"
            
            if dynamic_range < expected_dr[0]:
                feedback += "- Dynamic range is compressed for this genre.\n"
            elif dynamic_range > expected_dr[1]:
                feedback += "- Dynamic range is wider than typical for this genre.\n"
        
        feedback += "\n---\n*Basic technical analysis - LLM feedback unavailable*"
        
        return feedback
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from Ollama.
        
        Returns:
            List[str]: List of available model names
        """
        try:
            models = self.client.list()
            return [model['name'] for model in models['models']]
        except Exception:
            return []
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different LLM model.
        
        Args:
            model_name (str): Name of the model to switch to
            
        Returns:
            bool: True if switch was successful, False otherwise
        """
        old_model = self.model_name
        self.model_name = model_name
        
        if self.test_connection():
            print(f"Switched from {old_model} to {model_name}")
            return True
        else:
            self.model_name = old_model
            print(f"Failed to switch to {model_name}, keeping {old_model}")
            return False
