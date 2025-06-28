#!/usr/bin/env python3
"""
Audio Analysis Module

This module contains the AudioAnalyzer class responsible for comprehensive
audio signal analysis including dynamics, frequency spectrum, temporal features,
stereo properties, and harmonic content analysis.
"""

import warnings
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Suppress librosa warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import librosa
    import soundfile as sf
    from pydub import AudioSegment
except ImportError as e:
    print(f"Missing required audio libraries: {e}")
    print("Install with: pip install librosa pydub soundfile")
    raise

from .config import Config


class AudioAnalyzer:
    """
    Comprehensive audio analysis class for extracting technical parameters.
    
    This class provides methods to analyze various aspects of audio signals:
    - Dynamic range and loudness characteristics
    - Frequency spectrum and spectral features
    - Temporal characteristics (tempo, rhythm, onsets)
    - Stereo imaging and spatial properties  
    - Harmonic vs percussive content analysis
    
    The analyzer loads audio files, processes them at a consistent sample rate,
    and extracts detailed technical parameters suitable for professional
    audio feedback generation.
    """
    
    def __init__(self, sample_rate: int = None):
        """
        Initialize the audio analyzer.
        
        Args:
            sample_rate: Target sample rate for analysis. If None, uses Config default.
        """
        self.sample_rate = sample_rate or Config.AUDIO_SAMPLE_RATE
        self.audio_data = None
        self.original_sr = None
        self.duration = 0
        self.is_stereo = False
        self.left_channel = None
        self.right_channel = None
        self.mono_mix = None
        
    def load_audio(self, file_path: str) -> bool:
        """
        Load audio file and prepare it for analysis.
        
        This method loads audio using librosa for analysis compatibility,
        handles both mono and stereo files, and creates appropriate channel
        representations for different types of analysis.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            True if loading successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"Error: Audio file not found: {file_path}")
                return False
            
            print(f"🎵 Loading audio: {file_path.name}")
            
            # Load with librosa for analysis (handles most formats)
            self.audio_data, self.original_sr = librosa.load(
                str(file_path), 
                sr=self.sample_rate, 
                mono=False  # Preserve stereo information
            )
            
            # Handle stereo vs mono
            if len(self.audio_data.shape) > 1:
                # Stereo file
                self.is_stereo = True
                self.left_channel = self.audio_data[0]
                self.right_channel = self.audio_data[1]
                self.mono_mix = (self.left_channel + self.right_channel) / 2
                print(f"📊 Loaded stereo audio: {self.audio_data.shape}")
            else:
                # Mono file
                self.is_stereo = False
                self.mono_mix = self.audio_data
                self.left_channel = self.audio_data
                self.right_channel = self.audio_data
                print(f"📊 Loaded mono audio: {self.audio_data.shape}")
            
            # Calculate duration
            self.duration = librosa.get_duration(y=self.mono_mix, sr=self.sample_rate)
            
            # Load with pydub for additional format support if needed
            try:
                self.pydub_audio = AudioSegment.from_file(str(file_path))
            except Exception:
                print("⚠️  Warning: Could not load with pydub (some metadata may be unavailable)")
                self.pydub_audio = None
            
            print(f"✅ Audio loaded successfully:")
            print(f"   Duration: {self.duration:.2f} seconds")
            print(f"   Sample Rate: {self.sample_rate} Hz (original: {self.original_sr} Hz)")
            print(f"   Channels: {'Stereo' if self.is_stereo else 'Mono'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading audio file: {e}")
            return False
    
    def analyze_dynamics(self) -> Dict[str, float]:
        """
        Analyze dynamic range and loudness characteristics.
        
        This method calculates various dynamic properties of the audio:
        - RMS (Root Mean Square) levels for average loudness
        - Peak levels and crest factor
        - Dynamic range using percentile analysis
        - Loudness range using windowed analysis
        
        Returns:
            Dictionary containing dynamic analysis results
        """
        if self.mono_mix is None:
            raise ValueError("No audio data loaded. Call load_audio() first.")
            
        dynamics = {}
        
        # RMS (Root Mean Square) - average loudness
        rms = librosa.feature.rms(y=self.mono_mix)[0]
        dynamics['rms_mean'] = float(np.mean(rms))
        dynamics['rms_std'] = float(np.std(rms))
        dynamics['rms_db'] = float(20 * np.log10(dynamics['rms_mean'] + 1e-8))
        
        # Peak analysis
        peak_amplitude = float(np.max(np.abs(self.mono_mix)))
        dynamics['peak_amplitude'] = peak_amplitude
        dynamics['peak_db'] = float(20 * np.log10(peak_amplitude + 1e-8))
        
        # Crest factor (peak to RMS ratio) - indicates compression level
        dynamics['crest_factor'] = float(peak_amplitude / (dynamics['rms_mean'] + 1e-8))
        dynamics['crest_factor_db'] = float(20 * np.log10(dynamics['crest_factor']))
        
        # Dynamic range using percentile analysis (more robust than peak-to-peak)
        percentile_1 = np.percentile(np.abs(self.mono_mix), 1)
        percentile_99 = np.percentile(np.abs(self.mono_mix), 99)
        dynamics['dynamic_range_db'] = float(20 * np.log10((percentile_99 + 1e-8) / (percentile_1 + 1e-8)))
        
        # Loudness range using sliding window analysis
        window_size = int(Config.ANALYSIS_WINDOW_SIZE * self.sample_rate)  # 100ms windows
        windowed_rms = []
        
        for i in range(0, len(self.mono_mix) - window_size, window_size // 2):
            window = self.mono_mix[i:i + window_size]
            windowed_rms.append(np.sqrt(np.mean(window**2)))
        
        windowed_rms = np.array(windowed_rms)
        if len(windowed_rms) > 0:
            dynamics['loudness_range_db'] = float(20 * np.log10(
                (np.percentile(windowed_rms, 95) + 1e-8) / (np.percentile(windowed_rms, 10) + 1e-8)
            ))
        else:
            dynamics['loudness_range_db'] = 0.0
        
        return dynamics
    
    def analyze_frequency_spectrum(self) -> Dict[str, Any]:
        """
        Analyze frequency content and spectral characteristics.
        
        This method extracts spectral features including:
        - Spectral centroid, bandwidth, and rolloff
        - Zero crossing rate for harmonicity assessment
        - Spectral contrast for tonal vs noisy content
        - Energy distribution across defined frequency bands
        
        Returns:
            Dictionary containing frequency spectrum analysis results
        """
        if self.mono_mix is None:
            raise ValueError("No audio data loaded. Call load_audio() first.")
            
        spectrum = {}
        
        # Spectral centroid - indicates brightness/timbre
        spectral_centroids = librosa.feature.spectral_centroid(y=self.mono_mix, sr=self.sample_rate)[0]
        spectrum['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        spectrum['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        # Spectral bandwidth - indicates spectral spread
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=self.mono_mix, sr=self.sample_rate)[0]
        spectrum['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        
        # Spectral rolloff - frequency below which 85% of energy is contained
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.mono_mix, sr=self.sample_rate)[0]
        spectrum['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        
        # Zero crossing rate - indicates noisiness vs harmonicity
        zcr = librosa.feature.zero_crossing_rate(self.mono_mix)[0]
        spectrum['zero_crossing_rate_mean'] = float(np.mean(zcr))
        
        # Spectral contrast - tonal vs noisy components
        contrast = librosa.feature.spectral_contrast(y=self.mono_mix, sr=self.sample_rate)
        spectrum['spectral_contrast_mean'] = float(np.mean(contrast))
        
        # Frequency band energy analysis
        spectrum['frequency_bands'] = self._analyze_frequency_bands()
        
        return spectrum
    
    def _analyze_frequency_bands(self) -> Dict[str, float]:
        """
        Analyze energy distribution across defined frequency bands.
        
        Returns:
            Dictionary mapping band names to normalized energy values
        """
        # Compute STFT for frequency analysis using librosa
        stft = librosa.stft(self.mono_mix, 
                           n_fft=Config.FFT_SETTINGS['nperseg'], 
                           hop_length=Config.FFT_SETTINGS['nperseg']//4)
        magnitude = np.abs(stft)
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=Config.FFT_SETTINGS['nperseg'])
        
        # Calculate energy in each defined frequency band
        band_energies = {}
        total_energy = np.sum(magnitude)
        
        for band_name, (freq_min, freq_max) in Config.SPECTRAL_BANDS.items():
            # Find frequency bin indices for this band
            band_indices = np.where((freqs >= freq_min) & (freqs < freq_max))[0]
            
            if len(band_indices) > 0:
                # Calculate normalized energy in this band
                band_energy = np.sum(magnitude[band_indices, :])
                band_energies[f'{band_name}_energy'] = float(band_energy / (total_energy + 1e-8))
            else:
                band_energies[f'{band_name}_energy'] = 0.0
        
        return band_energies
    
    def analyze_temporal_features(self) -> Dict[str, Any]:
        """
        Analyze temporal characteristics like tempo, rhythm, and onset detection.
        
        This method extracts time-domain features including:
        - Tempo detection and beat tracking
        - Onset detection for transient analysis
        - Rhythmic stability and timing characteristics
        
        Returns:
            Dictionary containing temporal analysis results
        """
        if self.mono_mix is None:
            raise ValueError("No audio data loaded. Call load_audio() first.")
            
        temporal = {}
        
        try:
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=self.mono_mix, sr=self.sample_rate)
            temporal['tempo'] = float(tempo)
            temporal['beats_count'] = len(beats)
            temporal['beats_per_second'] = float(len(beats) / self.duration)
            
            # Beat consistency (tempo stability)
            if len(beats) > 1:
                beat_intervals = np.diff(librosa.frames_to_time(beats, sr=self.sample_rate))
                temporal['tempo_stability'] = float(1.0 - (np.std(beat_intervals) / np.mean(beat_intervals)))
            else:
                temporal['tempo_stability'] = 0.0
                
        except Exception:
            # Fallback if tempo detection fails
            temporal['tempo'] = 0.0
            temporal['beats_count'] = 0
            temporal['beats_per_second'] = 0.0
            temporal['tempo_stability'] = 0.0
        
        # Onset detection (transients/attacks)
        onset_frames = librosa.onset.onset_detect(y=self.mono_mix, sr=self.sample_rate)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)
        temporal['onset_count'] = len(onset_times)
        temporal['onset_rate'] = float(len(onset_times) / self.duration)
        
        # Onset strength (transient intensity)
        onset_strength = librosa.onset.onset_strength(y=self.mono_mix, sr=self.sample_rate)
        temporal['onset_strength_mean'] = float(np.mean(onset_strength))
        temporal['onset_strength_std'] = float(np.std(onset_strength))
        
        return temporal
    
    def analyze_stereo_properties(self) -> Dict[str, Any]:
        """
        Analyze stereo imaging and spatial properties.
        
        For stereo files, this method calculates:
        - Stereo correlation (phase relationship)
        - Stereo width (mid-side energy ratio)
        - Left-right balance
        
        For mono files, returns default values.
        
        Returns:
            Dictionary containing stereo analysis results
        """
        if not self.is_stereo:
            return {
                'stereo_width': 0.0,
                'correlation': 1.0,
                'balance': 0.0,
                'is_mono': True
            }
        
        stereo = {'is_mono': False}
        
        # Stereo correlation (phase relationship between channels)
        correlation = np.corrcoef(self.left_channel, self.right_channel)[0, 1]
        stereo['correlation'] = float(correlation)
        
        # Mid-side processing for stereo width analysis
        mid_signal = (self.left_channel + self.right_channel) / 2
        side_signal = (self.left_channel - self.right_channel) / 2
        
        # Stereo width (ratio of side to mid energy)
        mid_energy = np.mean(mid_signal**2)
        side_energy = np.mean(side_signal**2)
        stereo['stereo_width'] = float(side_energy / (mid_energy + 1e-8))
        
        # Left-right balance
        left_energy = np.mean(self.left_channel**2)
        right_energy = np.mean(self.right_channel**2)
        stereo['balance'] = float((right_energy - left_energy) / (right_energy + left_energy + 1e-8))
        
        # Stereo spread analysis
        stereo['mid_energy'] = float(mid_energy)
        stereo['side_energy'] = float(side_energy)
        
        return stereo
    
    def analyze_harmonic_content(self) -> Dict[str, Any]:
        """
        Analyze harmonic vs percussive content.
        
        This method uses harmonic-percussive separation to analyze:
        - Ratio of harmonic to percussive content
        - Pitch stability and fundamental frequency
        - Tonal vs rhythmic characteristics
        
        Returns:
            Dictionary containing harmonic content analysis results
        """
        if self.mono_mix is None:
            raise ValueError("No audio data loaded. Call load_audio() first.")
            
        harmonic = {}
        
        # Separate harmonic and percussive components
        harmonic_component, percussive_component = librosa.effects.hpss(self.mono_mix)
        
        # Calculate energy ratios
        harmonic_energy = np.mean(harmonic_component**2)
        percussive_energy = np.mean(percussive_component**2)
        total_energy = harmonic_energy + percussive_energy
        
        harmonic['harmonic_ratio'] = float(harmonic_energy / (total_energy + 1e-8))
        harmonic['percussive_ratio'] = float(percussive_energy / (total_energy + 1e-8))
        
        # Pitch analysis
        try:
            # Extract pitch information using piptrack
            pitches, magnitudes = librosa.piptrack(y=self.mono_mix, sr=self.sample_rate)
            pitch_values = []
            
            # Extract dominant pitch for each frame
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:  # Valid pitch detected
                    pitch_values.append(pitch)
            
            if pitch_values:
                harmonic['fundamental_frequency_mean'] = float(np.mean(pitch_values))
                harmonic['fundamental_frequency_std'] = float(np.std(pitch_values))
                harmonic['pitch_stability'] = float(1.0 - (np.std(pitch_values) / np.mean(pitch_values)))
            else:
                harmonic['fundamental_frequency_mean'] = 0.0
                harmonic['fundamental_frequency_std'] = 0.0
                harmonic['pitch_stability'] = 0.0
                
        except Exception:
            # Fallback if pitch detection fails
            harmonic['fundamental_frequency_mean'] = 0.0
            harmonic['fundamental_frequency_std'] = 0.0
            harmonic['pitch_stability'] = 0.0
        
        return harmonic
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze an audio file and return complete analysis results.
        
        This is the main method that combines file loading and complete analysis
        into a single convenient call.
        
        Args:
            file_path: Path to the audio file to analyze
            
        Returns:
            Dictionary containing complete analysis results, or None if loading failed
        """
        print(f"🎯 Starting analysis of: {Path(file_path).name}")
        
        # Load the audio file
        if not self.load_audio(file_path):
            return None
        
        # Perform complete analysis
        try:
            analysis = self.get_complete_analysis()
            
            # Add file path to results
            analysis['file_path'] = str(file_path)
            analysis['file_name'] = Path(file_path).name
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            return None
    
    def get_complete_analysis(self) -> Dict[str, Any]:
        """
        Perform complete audio analysis and return all results.
        
        This method orchestrates all analysis modules and returns
        a comprehensive dictionary containing all extracted features
        and technical parameters.
        
        Returns:
            Dictionary containing complete analysis results
        """
        if self.audio_data is None:
            raise ValueError("No audio data loaded. Call load_audio() first.")
        
        print("🔬 Performing comprehensive audio analysis...")
        
        # Basic file information
        analysis = {
            'file_info': {
                'duration_seconds': self.duration,
                'sample_rate': self.sample_rate,
                'original_sample_rate': self.original_sr,
                'is_stereo': self.is_stereo,
                'channels': 2 if self.is_stereo else 1
            }
        }
        
        # Run all analysis modules
        print("   📊 Analyzing dynamics...")
        analysis['dynamics'] = self.analyze_dynamics()
        
        print("   🎵 Analyzing frequency spectrum...")
        analysis['frequency_spectrum'] = self.analyze_frequency_spectrum()
        
        print("   ⏱️ Analyzing temporal features...")
        analysis['temporal_features'] = self.analyze_temporal_features()
        
        print("   🔊 Analyzing stereo properties...")
        analysis['stereo_properties'] = self.analyze_stereo_properties()
        
        print("   🎼 Analyzing harmonic content...")
        analysis['harmonic_content'] = self.analyze_harmonic_content()
        
        print("✅ Audio analysis complete!")
        
        return analysis


# ============================================================================
# MODULE TEST
# ============================================================================

if __name__ == "__main__":
    """Test audio analyzer functionality."""
    print("Audio Analyzer Module Test")
    print("=" * 30)
    
    # This would require a test audio file
    # analyzer = AudioAnalyzer()
    # success = analyzer.load_audio("test_file.wav")
    # if success:
    #     analysis = analyzer.get_complete_analysis()
    #     print("Analysis completed successfully!")
    
    print("Import successful - AudioAnalyzer class available")
