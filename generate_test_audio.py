#!/usr/bin/env python3
"""
Test Audio Generator
Generate various types of test audio files for testing the audio feedback analyzer.
"""

import numpy as np
import soundfile as sf
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class TestAudioGenerator:
    """Generate test audio files with various characteristics"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def generate_sine_wave(self, frequency: float, duration: float, amplitude: float = 0.5) -> np.ndarray:
        """Generate a sine wave"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        return amplitude * np.sin(2 * np.pi * frequency * t)
    
    def generate_white_noise(self, duration: float, amplitude: float = 0.1) -> np.ndarray:
        """Generate white noise"""
        samples = int(self.sample_rate * duration)
        return amplitude * np.random.normal(0, 1, samples)
    
    def generate_pink_noise(self, duration: float, amplitude: float = 0.1) -> np.ndarray:
        """Generate pink noise (1/f noise)"""
        samples = int(self.sample_rate * duration)
        
        # Generate white noise
        white = np.random.normal(0, 1, samples)
        
        # Apply pink noise filter (approximate)
        # Using a simple filter approach
        fft = np.fft.fft(white)
        freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)
        
        # Pink noise has 1/f characteristic
        # Avoid division by zero
        pink_filter = np.where(freqs != 0, 1/np.sqrt(np.abs(freqs)), 1)
        pink_filter[0] = 1  # DC component
        
        pink_fft = fft * pink_filter
        pink = np.real(np.fft.ifft(pink_fft))
        
        # Normalize
        pink = pink / np.max(np.abs(pink)) * amplitude
        return pink
    
    def generate_sweep(self, start_freq: float, end_freq: float, duration: float, 
                      amplitude: float = 0.5, sweep_type: str = 'linear') -> np.ndarray:
        """Generate frequency sweep"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        if sweep_type == 'linear':
            # Linear frequency sweep
            freq_t = start_freq + (end_freq - start_freq) * t / duration
            phase = 2 * np.pi * start_freq * t + np.pi * (end_freq - start_freq) * t**2 / duration
        elif sweep_type == 'logarithmic':
            # Logarithmic frequency sweep
            freq_t = start_freq * (end_freq / start_freq) ** (t / duration)
            phase = 2 * np.pi * start_freq * duration / np.log(end_freq / start_freq) * \
                   (np.exp(np.log(end_freq / start_freq) * t / duration) - 1)
        else:
            raise ValueError("sweep_type must be 'linear' or 'logarithmic'")
        
        return amplitude * np.sin(phase)
    
    def generate_impulse_response(self, duration: float, decay_time: float, 
                                amplitude: float = 0.8) -> np.ndarray:
        """Generate impulse response (like a drum hit)"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Sharp attack
        attack = np.where(t < 0.001, 1, 0)
        
        # Exponential decay
        decay = np.exp(-t / decay_time)
        
        # Add some frequency content
        freq_content = (np.sin(2 * np.pi * 200 * t) + 
                       0.5 * np.sin(2 * np.pi * 400 * t) +
                       0.25 * np.sin(2 * np.pi * 800 * t))
        
        return amplitude * (attack + 0.1 * decay * freq_content) * decay
    
    def generate_complex_tone(self, fundamental: float, duration: float, 
                            harmonics: int = 5, amplitude: float = 0.5) -> np.ndarray:
        """Generate complex tone with harmonics"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        signal = np.zeros_like(t)
        
        for harmonic in range(1, harmonics + 1):
            harmonic_amplitude = amplitude / harmonic  # Decreasing amplitude
            signal += harmonic_amplitude * np.sin(2 * np.pi * fundamental * harmonic * t)
        
        return signal
    
    def apply_dynamics(self, signal: np.ndarray, compression_ratio: float = 1.0, 
                      attack_time: float = 0.003, release_time: float = 0.1) -> np.ndarray:
        """Apply simple dynamic compression"""
        if compression_ratio <= 1.0:
            return signal
        
        # Simple peak detection and gain reduction
        compressed = signal.copy()
        gain_reduction = 1.0
        
        for i in range(len(signal)):
            current_level = abs(signal[i])
            
            # Threshold detection (simple)
            threshold = 0.7
            if current_level > threshold:
                target_gain = threshold / current_level
                target_gain = 1.0 - (1.0 - target_gain) * (compression_ratio - 1.0) / compression_ratio
            else:
                target_gain = 1.0
            
            # Smooth gain changes
            if target_gain < gain_reduction:
                # Attack
                gain_reduction += (target_gain - gain_reduction) * (1.0 / (attack_time * self.sample_rate))
            else:
                # Release
                gain_reduction += (target_gain - gain_reduction) * (1.0 / (release_time * self.sample_rate))
            
            compressed[i] *= gain_reduction
        
        return compressed
    
    def create_stereo(self, left: np.ndarray, right: Optional[np.ndarray] = None, 
                     stereo_width: float = 1.0) -> np.ndarray:
        """Create stereo signal"""
        if right is None:
            right = left.copy()
        
        # Apply stereo width
        mid = (left + right) / 2
        side = (left - right) / 2
        
        side *= stereo_width
        
        new_left = mid + side
        new_right = mid - side
        
        return np.column_stack((new_left, new_right))
    
    def generate_test_suite(self, output_dir: str = "./test_audio/"):
        """Generate a complete suite of test audio files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating test audio files in: {output_path}")
        
        # 1. Simple sine wave - 440Hz (A4)
        print("Generating: 01_sine_440hz.wav")
        sine = self.generate_sine_wave(440, 10.0, 0.5)
        sf.write(output_path / "01_sine_440hz.wav", sine, self.sample_rate)
        
        # 2. Complex harmonic tone
        print("Generating: 02_complex_tone.wav")
        complex_tone = self.generate_complex_tone(220, 10.0, 8, 0.4)
        sf.write(output_path / "02_complex_tone.wav", complex_tone, self.sample_rate)
        
        # 3. White noise
        print("Generating: 03_white_noise.wav")
        white = self.generate_white_noise(10.0, 0.3)
        sf.write(output_path / "03_white_noise.wav", white, self.sample_rate)
        
        # 4. Pink noise
        print("Generating: 04_pink_noise.wav")
        pink = self.generate_pink_noise(10.0, 0.3)
        sf.write(output_path / "04_pink_noise.wav", pink, self.sample_rate)
        
        # 5. Frequency sweep
        print("Generating: 05_sweep_20hz_20khz.wav")
        sweep = self.generate_sweep(20, 20000, 10.0, 0.4, 'logarithmic')
        sf.write(output_path / "05_sweep_20hz_20khz.wav", sweep, self.sample_rate)
        
        # 6. Impulse responses (drum-like)
        print("Generating: 06_drum_hits.wav")
        drums = np.zeros(int(self.sample_rate * 10))
        
        # Add multiple drum hits at different times
        for i, decay_time in enumerate([0.1, 0.2, 0.05, 0.15]):
            start_sample = int(i * 2.5 * self.sample_rate)
            impulse = self.generate_impulse_response(1.0, decay_time, 0.8)
            end_sample = min(start_sample + len(impulse), len(drums))
            impulse_len = end_sample - start_sample
            drums[start_sample:end_sample] += impulse[:impulse_len]
        
        sf.write(output_path / "06_drum_hits.wav", drums, self.sample_rate)
        
        # 7. Dynamic content (compressed vs uncompressed)
        print("Generating: 07_dynamic_natural.wav")
        print("Generating: 07_dynamic_compressed.wav")
        
        # Create dynamic content
        dynamic_signal = np.zeros(int(self.sample_rate * 10))
        for i in range(10):
            start = int(i * self.sample_rate)
            end = int((i + 0.8) * self.sample_rate)
            amplitude = 0.1 + 0.7 * np.random.random()  # Random dynamics
            freq = 220 * (2 ** (np.random.randint(-12, 12) / 12))  # Random pitch
            tone = self.generate_sine_wave(freq, 0.8, amplitude)
            if end - start <= len(tone):
                dynamic_signal[start:end] = tone[:end-start]
        
        # Save natural version
        sf.write(output_path / "07_dynamic_natural.wav", dynamic_signal, self.sample_rate)
        
        # Save compressed version
        compressed = self.apply_dynamics(dynamic_signal, compression_ratio=4.0)
        sf.write(output_path / "07_dynamic_compressed.wav", compressed, self.sample_rate)
        
        # 8. Stereo content
        print("Generating: 08_stereo_wide.wav")
        print("Generating: 08_stereo_narrow.wav")
        
        # Create stereo content
        left_tone = self.generate_sine_wave(330, 10.0, 0.4)  # E4
        right_tone = self.generate_sine_wave(392, 10.0, 0.4)  # G4
        
        # Wide stereo
        stereo_wide = self.create_stereo(left_tone, right_tone, stereo_width=2.0)
        sf.write(output_path / "08_stereo_wide.wav", stereo_wide, self.sample_rate)
        
        # Narrow stereo
        stereo_narrow = self.create_stereo(left_tone, right_tone, stereo_width=0.3)
        sf.write(output_path / "08_stereo_narrow.wav", stereo_narrow, self.sample_rate)
        
        # 9. Mixed content (music-like)
        print("Generating: 09_musical_mix.wav")
        
        # Create a simple musical mix
        duration = 15.0
        mix = np.zeros(int(self.sample_rate * duration))
        
        # Bass line (sine wave)
        bass_notes = [55, 73.42, 82.41, 61.74]  # A1, D2, E2, B1
        for i, note in enumerate(bass_notes):
            start = int(i * 3.75 * self.sample_rate)
            end = min(int((i + 3.75) * self.sample_rate), len(mix))
            bass_tone = self.generate_sine_wave(note, 3.75, 0.3)
            mix[start:end] += bass_tone[:end-start]
        
        # Add some harmonics (chord)
        chord_notes = [220, 277.18, 329.63]  # A3, C#4, E4
        for note in chord_notes:
            chord_tone = self.generate_complex_tone(note, duration, 4, 0.15)
            mix += chord_tone
        
        # Add percussion (impulses)
        for i in range(int(duration * 2)):  # Every 0.5 seconds
            start_sample = int(i * 0.5 * self.sample_rate)
            if start_sample < len(mix):
                impulse = self.generate_impulse_response(0.3, 0.05, 0.4)
                end_sample = min(start_sample + len(impulse), len(mix))
                mix[start_sample:end_sample] += impulse[:end_sample-start_sample]
        
        # Add some high-frequency content
        hihat = self.generate_white_noise(duration, 0.05)
        # Filter to high frequencies (simple approach)
        hihat = np.diff(np.concatenate(([0], hihat)))  # High-pass filter approximation
        mix[:len(hihat)] += hihat
        
        # Normalize
        mix = mix / np.max(np.abs(mix)) * 0.8
        
        # Create stereo version
        stereo_mix = self.create_stereo(mix, mix, stereo_width=1.2)
        sf.write(output_path / "09_musical_mix.wav", stereo_mix, self.sample_rate)
        
        # 10. Problematic audio (clipping, very loud, very quiet)
        print("Generating: 10_problematic_clipped.wav")
        print("Generating: 10_problematic_quiet.wav")
        
        # Clipped audio
        loud_tone = self.generate_sine_wave(1000, 5.0, 1.5)  # Intentionally too loud
        clipped = np.clip(loud_tone, -1.0, 1.0)
        sf.write(output_path / "10_problematic_clipped.wav", clipped, self.sample_rate)
        
        # Very quiet audio
        quiet_tone = self.generate_sine_wave(1000, 5.0, 0.01)  # Very quiet
        sf.write(output_path / "10_problematic_quiet.wav", quiet_tone, self.sample_rate)
        
        print(f"\nTest audio generation complete!")
        print(f"Generated 12 test files in: {output_path}")
        print("\nTest files:")
        for file in sorted(output_path.glob("*.wav")):
            print(f"  {file.name}")
    
    def create_visualization(self, signal: np.ndarray, title: str, output_file: Optional[str] = None):
        """Create visualization of generated audio"""
        plt.figure(figsize=(12, 8))
        
        # Time domain
        plt.subplot(3, 1, 1)
        time = np.linspace(0, len(signal) / self.sample_rate, len(signal))
        plt.plot(time[:min(len(time), self.sample_rate)], signal[:min(len(signal), self.sample_rate)])
        plt.title(f"{title} - Time Domain (first 1 second)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        
        # Frequency domain
        plt.subplot(3, 1, 2)
        fft = np.fft.fft(signal[:self.sample_rate])  # First second
        freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)
        magnitude = 20 * np.log10(np.abs(fft) + 1e-8)
        
        plt.plot(freqs[:len(freqs)//2], magnitude[:len(magnitude)//2])
        plt.title(f"{title} - Frequency Domain")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True, alpha=0.3)
        plt.xlim(0, self.sample_rate//2)
        
        # Spectrogram
        plt.subplot(3, 1, 3)
        plt.specgram(signal, Fs=self.sample_rate, cmap='viridis')
        plt.title(f"{title} - Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label='Power (dB)')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate test audio files')
    parser.add_argument('-o', '--output', default='./test_audio/', 
                       help='Output directory (default: ./test_audio/)')
    parser.add_argument('-s', '--sample-rate', type=int, default=44100,
                       help='Sample rate (default: 44100)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations of generated audio')
    
    args = parser.parse_args()
    
    # Create generator
    generator = TestAudioGenerator(sample_rate=args.sample_rate)
    
    # Generate test suite
    generator.generate_test_suite(args.output)
    
    # Create visualizations if requested
    if args.visualize:
        print("\nCreating visualizations...")
        output_path = Path(args.output)
        
        # Create a few example visualizations
        examples = [
            ("01_sine_440hz.wav", "440 Hz Sine Wave"),
            ("04_pink_noise.wav", "Pink Noise"),
            ("09_musical_mix.wav", "Musical Mix")
        ]
        
        for filename, title in examples:
            file_path = output_path / filename
            if file_path.exists():
                # Load and visualize
                audio, sr = sf.read(str(file_path))
                if len(audio.shape) > 1:
                    audio = audio[:, 0]  # Use left channel for stereo
                
                viz_file = output_path / f"{filename.replace('.wav', '_visualization.png')}"
                generator.create_visualization(audio, title, str(viz_file))
                print(f"  Created: {viz_file.name}")


if __name__ == "__main__":
    main()
