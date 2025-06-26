#!/usr/bin/env python3
"""
Audio Feedback Analyzer
A comprehensive audio analysis tool that provides professional feedback
using advanced audio analysis and LLM evaluation.
"""

import os
import sys
import json
import argparse
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Audio processing libraries
try:
    import librosa
    import librosa.display
    import numpy as np
    import scipy.signal
    from pydub import AudioSegment
    import matplotlib.pyplot as plt
    import seaborn as sns #type: ignore
except ImportError as e:
    print(f"Missing required audio libraries: {e}")
    print("Install with: pip install librosa pydub matplotlib seaborn scipy numpy")
    sys.exit(1)

# LLM integration
try:
    import requests
    import ollama #type: ignore
except ImportError as e:
    print(f"Missing LLM libraries: {e}")
    print("Install with: pip install requests ollama")
    sys.exit(1)

# Import configuration
try:
    from config_example import Config
except ImportError:
    print("Warning: Configuration file not found. Using defaults.")
    Config = None


class AudioAnalyzer:
    """Comprehensive audio analysis class"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.audio_data = None
        self.original_sr = None
        self.duration = 0
        
    def load_audio(self, file_path: str) -> bool:
        """Load audio file using librosa and pydub for compatibility"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Load with librosa for analysis
            self.audio_data, self.original_sr = librosa.load(
                str(file_path), 
                sr=self.sample_rate, 
                mono=False
            )
            
            # Handle stereo/mono
            if len(self.audio_data.shape) > 1:
                self.is_stereo = True
                self.left_channel = self.audio_data[0]
                self.right_channel = self.audio_data[1]
                self.mono_mix = librosa.to_mono(self.audio_data)
            else:
                self.is_stereo = False
                self.mono_mix = self.audio_data
                self.left_channel = self.audio_data
                self.right_channel = None
            
            self.duration = librosa.get_duration(y=self.mono_mix, sr=self.sample_rate)
            
            # Load with pydub for additional analysis
            self.pydub_audio = AudioSegment.from_file(str(file_path))
            
            return True
            
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False
    
    def analyze_dynamics(self) -> Dict[str, float]:
        """Analyze dynamic range and loudness characteristics"""
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
        
        # Crest factor (peak to RMS ratio)
        dynamics['crest_factor'] = float(peak_amplitude / (dynamics['rms_mean'] + 1e-8))
        dynamics['crest_factor_db'] = float(20 * np.log10(dynamics['crest_factor']))
        
        # Dynamic range
        percentile_1 = np.percentile(np.abs(self.mono_mix), 1)
        percentile_99 = np.percentile(np.abs(self.mono_mix), 99)
        dynamics['dynamic_range_db'] = float(20 * np.log10((percentile_99 + 1e-8) / (percentile_1 + 1e-8)))
        
        # Loudness range (using sliding window)
        window_size = int(0.1 * self.sample_rate)  # 100ms windows
        windowed_rms = []
        for i in range(0, len(self.mono_mix) - window_size, window_size // 2):
            window = self.mono_mix[i:i + window_size]
            windowed_rms.append(np.sqrt(np.mean(window**2)))
        
        windowed_rms = np.array(windowed_rms)
        dynamics['loudness_range_db'] = float(20 * np.log10(
            (np.percentile(windowed_rms, 95) + 1e-8) / (np.percentile(windowed_rms, 10) + 1e-8)
        ))
        
        return dynamics
    
    def analyze_frequency_spectrum(self) -> Dict[str, Any]:
        """Analyze frequency content and spectral characteristics"""
        spectrum = {}
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=self.mono_mix, sr=self.sample_rate)[0]
        spectrum['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        spectrum['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=self.mono_mix, sr=self.sample_rate)[0]
        spectrum['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.mono_mix, sr=self.sample_rate)[0]
        spectrum['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        
        # Zero crossing rate (indicates noisiness/harmonicity)
        zcr = librosa.feature.zero_crossing_rate(self.mono_mix)[0]
        spectrum['zero_crossing_rate_mean'] = float(np.mean(zcr))
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=self.mono_mix, sr=self.sample_rate)
        spectrum['spectral_contrast_mean'] = float(np.mean(contrast))
        
        # Frequency band analysis
        stft = librosa.stft(self.mono_mix)
        magnitude = np.abs(stft)
        
        # Define frequency bands
        freqs = librosa.fft_frequencies(sr=self.sample_rate)
        
        sub_bass = np.where((freqs >= 20) & (freqs < 60))[0]
        bass = np.where((freqs >= 60) & (freqs < 250))[0]
        low_mids = np.where((freqs >= 250) & (freqs < 500))[0]
        mids = np.where((freqs >= 500) & (freqs < 2000))[0]
        high_mids = np.where((freqs >= 2000) & (freqs < 4000))[0]
        presence = np.where((freqs >= 4000) & (freqs < 6000))[0]
        brilliance = np.where((freqs >= 6000) & (freqs < 20000))[0]
        
        spectrum['frequency_bands'] = {
            'sub_bass_energy': float(np.mean(magnitude[sub_bass, :])) if len(sub_bass) > 0 else 0.0,
            'bass_energy': float(np.mean(magnitude[bass, :])) if len(bass) > 0 else 0.0,
            'low_mids_energy': float(np.mean(magnitude[low_mids, :])) if len(low_mids) > 0 else 0.0,
            'mids_energy': float(np.mean(magnitude[mids, :])) if len(mids) > 0 else 0.0,
            'high_mids_energy': float(np.mean(magnitude[high_mids, :])) if len(high_mids) > 0 else 0.0,
            'presence_energy': float(np.mean(magnitude[presence, :])) if len(presence) > 0 else 0.0,
            'brilliance_energy': float(np.mean(magnitude[brilliance, :])) if len(brilliance) > 0 else 0.0,
        }
        
        return spectrum
    
    def analyze_temporal_features(self) -> Dict[str, Any]:
        """Analyze temporal characteristics like tempo, rhythm, onset detection"""
        temporal = {}
        
        try:
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=self.mono_mix, sr=self.sample_rate)
            temporal['tempo'] = float(tempo)
            temporal['beats_count'] = len(beats)
            temporal['beats_per_second'] = float(len(beats) / self.duration)
            
            # Beat consistency (tempo stability)
            if len(beats) > 1:
                beat_intervals = np.diff(beats) / self.sample_rate
                temporal['tempo_stability'] = float(1.0 / (np.std(beat_intervals) + 1e-8))
            else:
                temporal['tempo_stability'] = 0.0
                
        except Exception:
            temporal['tempo'] = 0.0
            temporal['beats_count'] = 0
            temporal['beats_per_second'] = 0.0
            temporal['tempo_stability'] = 0.0
        
        # Onset detection (transients)
        onset_frames = librosa.onset.onset_detect(y=self.mono_mix, sr=self.sample_rate)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)
        temporal['onset_count'] = len(onset_times)
        temporal['onset_rate'] = float(len(onset_times) / self.duration)
        
        # Onset strength (transient intensity)
        onset_strength = librosa.onset.onset_strength(y=self.mono_mix, sr=self.sample_rate)
        temporal['onset_strength_mean'] = float(np.mean(onset_strength))
        
        return temporal
    
    def analyze_stereo_properties(self) -> Dict[str, Any]:
        """Analyze stereo imaging and spatial properties"""
        if not self.is_stereo:
            return {'stereo_width': 0.0, 'correlation': 1.0, 'balance': 0.0}
        
        stereo = {}
        
        # Stereo correlation
        correlation = np.corrcoef(self.left_channel, self.right_channel)[0, 1]
        stereo['correlation'] = float(correlation)
        
        # Stereo width (difference between channels)
        mid_signal = (self.left_channel + self.right_channel) / 2
        side_signal = (self.left_channel - self.right_channel) / 2
        
        mid_energy = np.mean(mid_signal**2)
        side_energy = np.mean(side_signal**2)
        
        stereo['stereo_width'] = float(side_energy / (mid_energy + 1e-8))
        
        # Left-Right balance
        left_energy = np.mean(self.left_channel**2)
        right_energy = np.mean(self.right_channel**2)
        stereo['balance'] = float((right_energy - left_energy) / (right_energy + left_energy + 1e-8))
        
        return stereo
    
    def analyze_harmonic_content(self) -> Dict[str, Any]:
        """Analyze harmonic vs percussive content"""
        harmonic = {}
        
        # Separate harmonic and percussive components
        harmonic_component, percussive_component = librosa.effects.hpss(self.mono_mix)
        
        harmonic_energy = np.mean(harmonic_component**2)
        percussive_energy = np.mean(percussive_component**2)
        total_energy = harmonic_energy + percussive_energy
        
        harmonic['harmonic_ratio'] = float(harmonic_energy / (total_energy + 1e-8))
        harmonic['percussive_ratio'] = float(percussive_energy / (total_energy + 1e-8))
        
        # Pitch analysis
        try:
            pitches, magnitudes = librosa.piptrack(y=self.mono_mix, sr=self.sample_rate)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                harmonic['fundamental_frequency_mean'] = float(np.mean(pitch_values))
                harmonic['fundamental_frequency_std'] = float(np.std(pitch_values))
                harmonic['pitch_stability'] = float(1.0 / (np.std(pitch_values) + 1e-8))
            else:
                harmonic['fundamental_frequency_mean'] = 0.0
                harmonic['fundamental_frequency_std'] = 0.0
                harmonic['pitch_stability'] = 0.0
                
        except Exception:
            harmonic['fundamental_frequency_mean'] = 0.0
            harmonic['fundamental_frequency_std'] = 0.0
            harmonic['pitch_stability'] = 0.0
        
        return harmonic
    
    def detect_genre(self, analysis: Dict[str, Any]) -> str:
        """Detect likely genre based on audio characteristics"""
        if not Config:
            return "unknown"
        
        dynamics = analysis.get('dynamics', {})
        spectrum = analysis.get('frequency_spectrum', {})
        temporal = analysis.get('temporal_features', {})
        stereo = analysis.get('stereo_properties', {})
        
        # Extract key parameters
        dynamic_range = dynamics.get('dynamic_range_db', 0)
        rms_db = dynamics.get('rms_db', -20)
        tempo = temporal.get('tempo', 0)
        stereo_width = stereo.get('stereo_width', 0.5)
        
        # Frequency band energies
        freq_bands = spectrum.get('frequency_bands', {})
        sub_bass = freq_bands.get('sub_bass_energy', 0)
        bass = freq_bands.get('bass_energy', 0)
        brilliance = freq_bands.get('brilliance_energy', 0)
        
        # Genre detection logic
        scores = {}
        
        for genre, profile in Config.GENRE_PROFILES.items():
            score = 0
            
            # Dynamic range scoring
            dr_min, dr_max = profile['expected_dynamic_range']
            if dr_min <= dynamic_range <= dr_max:
                score += 3
            elif abs(dynamic_range - (dr_min + dr_max)/2) < 5:
                score += 1
            
            # RMS level scoring
            rms_min, rms_max = profile['expected_rms']
            if rms_min <= rms_db <= rms_max:
                score += 3
            elif abs(rms_db - (rms_min + rms_max)/2) < 3:
                score += 1
            
            # Stereo width scoring
            sw_min, sw_max = profile['stereo_width_target']
            if sw_min <= stereo_width <= sw_max:
                score += 2
            elif abs(stereo_width - (sw_min + sw_max)/2) < 0.3:
                score += 1
            
            # Frequency emphasis scoring
            freq_emphasis = profile.get('frequency_emphasis', [])
            if 'sub_bass' in freq_emphasis and sub_bass > 0.3:
                score += 1
            if 'bass' in freq_emphasis and bass > 0.4:
                score += 1
            if 'brilliance' in freq_emphasis and brilliance > 0.2:
                score += 1
            
            # Tempo-based scoring for specific genres
            if genre == 'drum_and_bass' and 160 <= tempo <= 180:
                score += 2
            elif genre == 'metal' and 100 <= tempo <= 160:
                score += 1
            elif genre == 'classical' and 60 <= tempo <= 120:
                score += 1
            
            scores[genre] = score
        
        # Return the genre with highest score
        if scores:
            detected_genre = max(scores.items(), key=lambda x: x[1])
            if detected_genre[1] >= 3:  # Minimum confidence threshold
                return detected_genre[0]
        
        return 'unknown'

    def get_complete_analysis(self) -> Dict[str, Any]:
        """Perform complete audio analysis"""
        if self.audio_data is None:
            raise ValueError("No audio data loaded. Call load_audio() first.")
        
        analysis = {
            'file_info': {
                'duration_seconds': self.duration,
                'sample_rate': self.sample_rate,
                'original_sample_rate': self.original_sr,
                'is_stereo': self.is_stereo,
                'channels': 2 if self.is_stereo else 1
            },
            'dynamics': self.analyze_dynamics(),
            'frequency_spectrum': self.analyze_frequency_spectrum(),
            'temporal_features': self.analyze_temporal_features(),
            'stereo_properties': self.analyze_stereo_properties(),
            'harmonic_content': self.analyze_harmonic_content()
        }
        
        # Add genre detection
        detected_genre = self.detect_genre(analysis)
        analysis['genre_detection'] = {
            'detected_genre': detected_genre,
            'confidence': 'high' if detected_genre != 'unknown' else 'low'
        }
        
        # Add genre-specific recommendations if genre is detected
        if Config and detected_genre != 'unknown':
            genre_profile = Config.get_genre_config(detected_genre)
            analysis['genre_profile'] = genre_profile
        
        return analysis


class LLMFeedbackGenerator:
    """Generate professional audio feedback using Ollama/Gemma"""
    
    def __init__(self, model_name: str = "gemma3:27b"):
        self.model_name = model_name
        self.client = ollama.Client()
        
    def test_connection(self) -> bool:
        """Testa se Ollama è attivo e il modello è disponibile"""
        try:
            models = self.client.list()
            # FIX: usa .model invece di ['name']
            model_names = [model.model for model in models.models]
            print(f"🤖 Modelli disponibili: {model_names}")
            
            if self.model_name not in model_names:
                print(f"❌ Modello {self.model_name} non trovato.")
                print(f"💡 Modelli disponibili: {model_names}")
                # Usa il primo modello disponibile se quello richiesto non c'è
                if model_names:
                    self.model_name = model_names[0]
                    print(f"🔄 Usando invece: {self.model_name}")
                    return True
                return False
            
            print(f"✅ Modello {self.model_name} trovato e pronto!")
            return True
            
        except Exception as e:
            print(f"❌ Impossibile connettersi a Ollama: {e}")
            print("💡 Assicurati che Ollama sia in esecuzione con: ollama serve")
            return False
    
    def create_analysis_prompt(self, analysis_data: Dict[str, Any], genre: str = None) -> str:
        """Create detailed prompt for LLM analysis"""
        
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
        
        prompt = f"""You are a professional audio engineer and mastering specialist with 20+ years of experience. 
Analyze the following technical audio data and provide comprehensive feedback as if you were reviewing a mix/master for a client.

{genre_context}

AUDIO ANALYSIS DATA:
{json.dumps(analysis_data, indent=2)}

Please provide feedback in the following structure:

## OVERALL ASSESSMENT
Rate the overall technical quality (1-10) and provide a brief summary.

## GENRE ANALYSIS
{f"Based on the detected genre ({target_genre.replace('_', ' ').title()}), assess how well this track fits the typical characteristics of this style." if target_genre != 'unknown' else "Analyze what genre this track most closely resembles and whether it meets those genre conventions."}

## DYNAMICS ANALYSIS
- Comment on the dynamic range, crest factor, and loudness levels
- Assess if the track is over-compressed or has good dynamics
- Compare RMS and peak levels to professional standards{f" and {target_genre.replace('_', ' ')} genre expectations" if target_genre != 'unknown' else ""}

## FREQUENCY BALANCE
- Analyze the frequency distribution across all bands
- Identify any problematic frequency buildups or deficiencies
- Comment on spectral balance and tonal characteristics
{f"- Compare against typical {target_genre.replace('_', ' ')} frequency emphasis: {', '.join(genre_profile.get('frequency_emphasis', []))}" if genre_profile else ""}

## STEREO IMAGING & SPATIAL CHARACTERISTICS
- Evaluate stereo width, correlation, and left-right balance
- Comment on the spatial presentation and imaging
{f"- Assess against {target_genre.replace('_', ' ')} stereo width expectations ({genre_profile.get('stereo_width_target', [0, 1])[0]}-{genre_profile.get('stereo_width_target', [0, 1])[1]})" if genre_profile else ""}

## TEMPORAL CHARACTERISTICS
- Assess tempo stability, transient handling, and rhythmic elements
- Comment on onset detection and timing characteristics

## TECHNICAL RECOMMENDATIONS
Provide specific, actionable recommendations for improvement:
- EQ suggestions (specific frequency ranges and adjustments)
- Dynamic processing recommendations
- Stereo imaging improvements
- Any other technical corrections needed
{f"- Genre-specific recommendations for {target_genre.replace('_', ' ')} productions" if target_genre != 'unknown' else ""}

## GENRE & REFERENCE CONTEXT
{f"How does this track compare to professional {target_genre.replace('_', ' ')} references? What could be improved to better fit the genre?" if target_genre != 'unknown' else "Based on the analysis, what genre does this appear to be, and how does it compare to professional references in that style?"}

## PRIORITY FIXES
List the top 3 most important issues to address first{f", considering {target_genre.replace('_', ' ')} production standards" if target_genre != 'unknown' else ""}.

Be specific with frequencies (Hz), dB values, and technical terminology. Assume the client has intermediate technical knowledge."""

        return prompt
    
    def generate_feedback(self, analysis_data: Dict[str, Any], genre: str = None) -> str:
        """Generate comprehensive feedback using the LLM"""
        if not self.test_connection():
            return "Error: Cannot connect to Ollama or model not available."
        
        prompt = self.create_analysis_prompt(analysis_data, genre)
        
        print(f"🎯 Generating feedback for detected genre: {analysis_data.get('genre_detection', {}).get('detected_genre', 'unknown')}")
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a world-class audio engineer and mastering specialist. Provide detailed, technical, and actionable feedback.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.3,  # Lower temperature for more consistent technical analysis
                    'top_p': 0.9,
                }
            )
            
            return response['message']['content']
            
        except Exception as e:
            return f"Error generating feedback: {e}"


class AudioFeedbackApp:
    """Main application class"""
    
    def __init__(self):
        self.analyzer = AudioAnalyzer()
        self.llm_generator = LLMFeedbackGenerator()
        
    def process_audio_file(self, file_path: str, output_dir: Optional[str] = None, genre: str = None) -> Dict[str, Any]:
        """Process audio file and generate complete analysis and feedback"""
        
        print(f"🎵 Loading audio file: {file_path}")
        if not self.analyzer.load_audio(file_path):
            return {"error": "Failed to load audio file"}
        
        print("🔬 Performing audio analysis...")
        try:
            analysis = self.analyzer.get_complete_analysis()
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}
        
        detected_genre = analysis.get('genre_detection', {}).get('detected_genre', 'unknown')
        print(f"🎯 Detected genre: {detected_genre.replace('_', ' ').title() if detected_genre != 'unknown' else 'Unknown'}")
        
        print("🤖 Generating LLM feedback...")
        feedback = self.llm_generator.generate_feedback(analysis, genre)
        
        # Combine results
        result = {
            "file_path": str(file_path),
            "analysis": analysis,
            "feedback": feedback,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
        # Save results if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save analysis JSON
            analysis_file = output_path / f"{Path(file_path).stem}_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(result["analysis"], f, indent=2)
            
            # Save feedback text
            feedback_file = output_path / f"{Path(file_path).stem}_feedback.md"
            with open(feedback_file, 'w') as f:
                f.write(f"# Audio Analysis Feedback - {Path(file_path).name}\n\n")
                f.write(f"**Detected Genre:** {detected_genre.replace('_', ' ').title()}\n\n")
                f.write(feedback)
            
            print(f"📁 Results saved to: {output_path}")
        
        return result
    
    def create_visualization(self, analysis: Dict[str, Any], output_path: Optional[str] = None):
        """Create visualization of audio analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Audio Analysis Visualization', fontsize=16)
            
            # Frequency bands visualization
            bands = analysis['frequency_spectrum']['frequency_bands']
            band_names = list(bands.keys())
            band_values = [bands[name] for name in band_names]
            
            axes[0, 0].bar(range(len(band_names)), band_values)
            axes[0, 0].set_title('Frequency Band Energy')
            axes[0, 0].set_xticks(range(len(band_names)))
            axes[0, 0].set_xticklabels([name.replace('_', '\n') for name in band_names], rotation=45)
            axes[0, 0].set_ylabel('Energy')
            
            # Dynamics visualization
            dynamics = analysis['dynamics']
            dyn_labels = ['RMS (dB)', 'Peak (dB)', 'Crest Factor (dB)', 'Dynamic Range (dB)']
            dyn_values = [
                dynamics['rms_db'],
                dynamics['peak_db'],
                dynamics['crest_factor_db'],
                dynamics['dynamic_range_db']
            ]
            
            axes[0, 1].bar(dyn_labels, dyn_values)
            axes[0, 1].set_title('Dynamic Characteristics')
            axes[0, 1].set_ylabel('dB')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Stereo properties (if stereo)
            if analysis['file_info']['is_stereo']:
                stereo = analysis['stereo_properties']
                stereo_labels = ['Correlation', 'Stereo Width', 'Balance']
                stereo_values = [stereo['correlation'], stereo['stereo_width'], stereo['balance']]
                
                axes[1, 0].bar(stereo_labels, stereo_values)
                axes[1, 0].set_title('Stereo Properties')
                axes[1, 0].set_ylabel('Value')
            else:
                axes[1, 0].text(0.5, 0.5, 'Mono File\n(No Stereo Analysis)', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Stereo Properties')
            
            # Harmonic content
            harmonic = analysis['harmonic_content']
            harm_labels = ['Harmonic Ratio', 'Percussive Ratio']
            harm_values = [harmonic['harmonic_ratio'], harmonic['percussive_ratio']]
            
            axes[1, 1].pie(harm_values, labels=harm_labels, autopct='%1.1f%%')
            axes[1, 1].set_title('Harmonic vs Percussive Content')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to: {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error creating visualization: {e}")


def main():
    parser = argparse.ArgumentParser(description='Audio Feedback Analyzer')
    parser.add_argument('input_file', help='Input audio file (WAV or MP3)')
    parser.add_argument('-o', '--output', help='Output directory for results')
    parser.add_argument('-v', '--visualize', action='store_true', help='Create visualization')
    parser.add_argument('-m', '--model', default='gemma3:27b', help='LLM model name (default: gemma3:27b)')
    parser.add_argument('-g', '--genre', help='Force specific genre (rock, metal, electronic, drum_and_bass, classical, jazz, podcast)', 
                        choices=['rock', 'metal', 'electronic', 'drum_and_bass', 'classical', 'jazz', 'podcast'])
    
    args = parser.parse_args()
    
    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return 1
    
    if input_file.suffix.lower() not in ['.wav', '.mp3', '.flac', '.m4a']:
        print(f"Warning: File extension {input_file.suffix} may not be supported")
    
    # Initialize app
    app = AudioFeedbackApp()
    app.llm_generator.model_name = args.model
    
    # Process file
    result = app.process_audio_file(str(input_file), args.output, args.genre)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1
    
    # Print feedback to console
    print("\n" + "="*80)
    print("AUDIO FEEDBACK REPORT")
    print("="*80)
    print(result['feedback'])
    
    # Create visualization if requested
    if args.visualize:
        viz_path = None
        if args.output:
            viz_path = Path(args.output) / f"{input_file.stem}_visualization.png"
        app.create_visualization(result['analysis'], viz_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
