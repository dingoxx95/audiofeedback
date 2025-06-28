#!/usr/bin/env python3
"""
Audio Visualization Module

This module provides comprehensive audio visualization capabilities including
spectrum analysis, waveform display, dynamics visualization, and genre-specific
visual feedback. It generates publication-quality plots and charts for
professional audio analysis reporting.

Author: Audio Feedback Analyzer
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any, List, Tuple, Optional
import warnings
import os
from pathlib import Path

try:
    import librosa
    import librosa.display
    import seaborn as sns
except ImportError as e:
    print(f"Missing visualization libraries: {e}")
    print("Install with: pip install librosa seaborn")
    librosa = None
    sns = None

from .config import Config


class AudioVisualizer:
    """
    Comprehensive audio visualization and plotting for professional analysis.
    
    This class generates various types of visual representations of audio data
    including spectrograms, waveforms, frequency analysis, dynamics plots,
    and stereo imaging visualizations. All plots are designed for professional
    reporting and analysis purposes.
    
    Features:
    - Spectrum analysis and spectrograms
    - Waveform and dynamics visualization
    - Frequency balance and genre comparison
    - Stereo imaging and correlation plots
    - Professional styling and annotations
    
    Attributes:
        style (str): Matplotlib style for professional appearance
        figure_size (Tuple[int, int]): Default figure size for plots
        dpi (int): Resolution for saved figures
        
    Example:
        >>> visualizer = AudioVisualizer()
        >>> visualizer.create_comprehensive_report(analysis_data, "output_dir")
        >>> print("Visualizations saved to output_dir/")
    """
    
    def __init__(self, style: str = 'default', figure_size: Tuple[int, int] = (12, 8), 
                 dpi: int = 300):
        """
        Initialize the audio visualizer.
        
        Args:
            style (str): Matplotlib style for plots ('default', 'seaborn', 'bmh')
            figure_size (Tuple[int, int]): Default figure size (width, height) in inches
            dpi (int): Resolution for saved figures
        """
        self.style = style
        self.figure_size = figure_size
        self.dpi = dpi
        
        # Set up professional plotting style
        self._setup_plot_style()
        
        # Color schemes for different visualizations
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'warning': '#FFD23F',
            'neutral': '#6C757D',
            'background': '#F8F9FA'
        }
        
        # Genre color mapping
        self.genre_colors = {
            'rock': '#E74C3C',
            'jazz': '#3498DB',
            'classical': '#9B59B6',
            'electronic': '#00D4AA',
            'hip_hop': '#F39C12',
            'folk': '#27AE60',
            'pop': '#E91E63',
            'unknown': '#95A5A6'
        }
    
    def _setup_plot_style(self):
        """Configure matplotlib for professional-looking plots."""
        plt.style.use(self.style)
        
        # Set professional defaults
        plt.rcParams.update({
            'figure.figsize': self.figure_size,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 9,
            'grid.alpha': 0.3,
            'axes.grid': True,
            'axes.spines.top': False,
            'axes.spines.right': False
        })
        
        if sns is not None:
            sns.set_palette("husl")
    
    def create_comprehensive_report(self, analysis_data: Dict[str, Any], 
                                  output_dir: str, file_name: str = None) -> Dict[str, str]:
        """
        Create a comprehensive visual analysis report.
        
        Generates multiple visualization plots and saves them to the specified directory.
        This includes waveform analysis, frequency spectrum, dynamics, stereo imaging,
        and genre-specific comparisons.
        
        Args:
            analysis_data (Dict[str, Any]): Complete audio analysis results
            output_dir (str): Directory to save visualization files
            file_name (str, optional): Base name for output files
            
        Returns:
            Dict[str, str]: Dictionary of plot types and their file paths
            
        Example:
            >>> files = visualizer.create_comprehensive_report(analysis, "reports")
            >>> for plot_type, file_path in files.items():
            ...     print(f"{plot_type}: {file_path}")
        """
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate base filename
        if file_name is None:
            file_info = analysis_data.get('file_info', {})
            file_name = file_info.get('filename', 'audio_analysis')
            file_name = Path(file_name).stem  # Remove extension
        
        generated_files = {}
        
        print("Generating comprehensive visual analysis...")
        
        # 1. Waveform and dynamics overview
        try:
            waveform_file = output_path / f"{file_name}_waveform_dynamics.png"
            self.plot_waveform_dynamics(analysis_data, str(waveform_file))
            generated_files['waveform_dynamics'] = str(waveform_file)
            print("✓ Waveform and dynamics plot created")
        except Exception as e:
            print(f"✗ Error creating waveform plot: {e}")
        
        # 2. Frequency spectrum analysis
        try:
            spectrum_file = output_path / f"{file_name}_frequency_spectrum.png"
            self.plot_frequency_analysis(analysis_data, str(spectrum_file))
            generated_files['frequency_spectrum'] = str(spectrum_file)
            print("✓ Frequency spectrum analysis created")
        except Exception as e:
            print(f"✗ Error creating spectrum plot: {e}")
        
        # 3. Stereo imaging (if stereo)
        if analysis_data.get('file_info', {}).get('is_stereo', False):
            try:
                stereo_file = output_path / f"{file_name}_stereo_imaging.png"
                self.plot_stereo_analysis(analysis_data, str(stereo_file))
                generated_files['stereo_imaging'] = str(stereo_file)
                print("✓ Stereo imaging analysis created")
            except Exception as e:
                print(f"✗ Error creating stereo plot: {e}")
        
        # 4. Genre comparison
        try:
            genre_file = output_path / f"{file_name}_genre_analysis.png"
            self.plot_genre_comparison(analysis_data, str(genre_file))
            generated_files['genre_analysis'] = str(genre_file)
            print("✓ Genre comparison analysis created")
        except Exception as e:
            print(f"✗ Error creating genre plot: {e}")
        
        # 5. Professional standards comparison
        try:
            standards_file = output_path / f"{file_name}_professional_standards.png"
            self.plot_professional_standards(analysis_data, str(standards_file))
            generated_files['professional_standards'] = str(standards_file)
            print("✓ Professional standards comparison created")
        except Exception as e:
            print(f"✗ Error creating standards plot: {e}")
        
        # 6. Summary dashboard
        try:
            dashboard_file = output_path / f"{file_name}_dashboard.png"
            self.create_analysis_dashboard(analysis_data, str(dashboard_file))
            generated_files['dashboard'] = str(dashboard_file)
            print("✓ Analysis dashboard created")
        except Exception as e:
            print(f"✗ Error creating dashboard: {e}")
        
        print(f"\nGenerated {len(generated_files)} visualization files in {output_dir}")
        return generated_files
    
    def plot_waveform_dynamics(self, analysis_data: Dict[str, Any], 
                              output_file: str = None) -> None:
        """
        Create waveform and dynamics visualization.
        
        Displays the audio waveform along with RMS levels, peak levels,
        and dynamic range analysis over time.
        
        Args:
            analysis_data (Dict[str, Any]): Audio analysis data
            output_file (str, optional): Path to save the plot
        """
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle('Waveform and Dynamics Analysis', fontsize=16, fontweight='bold')
        
        # Get audio data and file info
        audio_data = analysis_data.get('audio_data')
        file_info = analysis_data.get('file_info', {})
        dynamics = analysis_data.get('dynamics', {})
        
        duration = file_info.get('duration_seconds', 10)
        sample_rate = file_info.get('sample_rate', 44100)
        
        if audio_data is not None:
            # Time axis
            time = np.linspace(0, duration, len(audio_data))
            
            # Plot 1: Waveform
            axes[0].plot(time, audio_data, color=self.colors['primary'], linewidth=0.5, alpha=0.7)
            axes[0].set_title('Waveform', fontweight='bold')
            axes[0].set_ylabel('Amplitude')
            axes[0].set_xlim(0, duration)
            axes[0].grid(True, alpha=0.3)
            
            # Add peak indicators
            peak_level = dynamics.get('peak_db', -6)
            peak_threshold = 10**(peak_level/20)
            axes[0].axhline(y=peak_threshold, color=self.colors['accent'], 
                           linestyle='--', alpha=0.7, label=f'Peak: {peak_level:.1f} dB')
            axes[0].axhline(y=-peak_threshold, color=self.colors['accent'], 
                           linestyle='--', alpha=0.7)
            axes[0].legend()
            
            # Plot 2: RMS levels over time
            if len(audio_data) > sample_rate:  # If longer than 1 second
                # Calculate RMS in 0.1 second windows
                window_size = int(sample_rate * 0.1)
                rms_values = []
                rms_times = []
                
                for i in range(0, len(audio_data) - window_size, window_size):
                    window = audio_data[i:i + window_size]
                    rms = np.sqrt(np.mean(window**2))
                    rms_db = 20 * np.log10(max(rms, 1e-10))
                    rms_values.append(rms_db)
                    rms_times.append(time[i + window_size//2])
                
                axes[1].plot(rms_times, rms_values, color=self.colors['secondary'], 
                           linewidth=2, label='RMS Level')
                
                # Add reference lines
                avg_rms = dynamics.get('rms_db', -20)
                axes[1].axhline(y=avg_rms, color=self.colors['accent'], 
                               linestyle='-', alpha=0.8, label=f'Average RMS: {avg_rms:.1f} dB')
                axes[1].axhline(y=-14, color='green', linestyle='--', alpha=0.5, 
                               label='Streaming Target (-14 LUFS)')
                axes[1].axhline(y=-9, color='orange', linestyle='--', alpha=0.5,
                               label='CD Target (-9 LUFS)')
                
                axes[1].set_title('RMS Levels Over Time', fontweight='bold')
                axes[1].set_ylabel('Level (dB)')
                axes[1].set_xlim(0, duration)
                axes[1].set_ylim(-40, 0)
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Dynamic range visualization
        dr_db = dynamics.get('dynamic_range_db', 10)
        crest_factor = dynamics.get('crest_factor_db', 10)
        rms_db = dynamics.get('rms_db', -20)
        peak_db = dynamics.get('peak_db', -6)
        
        # Create bar chart for dynamics
        categories = ['Dynamic\nRange', 'Crest\nFactor', 'RMS\nLevel', 'Peak\nLevel']
        values = [dr_db, crest_factor, rms_db, peak_db]
        colors = [self.colors['primary'], self.colors['secondary'], 
                 self.colors['accent'], self.colors['success']]
        
        bars = axes[2].bar(categories, values, color=colors, alpha=0.7)
        axes[2].set_title('Dynamic Properties', fontweight='bold')
        axes[2].set_ylabel('Level (dB)')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{value:.1f} dB', ha='center', va='bottom', fontweight='bold')
        
        # Add reference lines for professional standards
        axes[2].axhline(y=-14, color='green', linestyle='--', alpha=0.5, 
                       label='Streaming Standard')
        axes[2].axhline(y=-1, color='orange', linestyle='--', alpha=0.5,
                       label='Peak Limit')
        axes[2].legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_frequency_analysis(self, analysis_data: Dict[str, Any], 
                               output_file: str = None) -> None:
        """
        Create comprehensive frequency spectrum analysis visualization.
        
        Shows frequency spectrum, spectral features, and frequency band energy
        distribution with professional audio frequency references.
        
        Args:
            analysis_data (Dict[str, Any]): Audio analysis data
            output_file (str, optional): Path to save the plot
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Frequency Spectrum Analysis', fontsize=16, fontweight='bold')
        
        spectrum = analysis_data.get('frequency_spectrum', {})
        audio_data = analysis_data.get('audio_data')
        file_info = analysis_data.get('file_info', {})
        
        # Plot 1: Frequency spectrum
        ax1 = axes[0, 0]
        if audio_data is not None and librosa is not None:
            sample_rate = file_info.get('sample_rate', 44100)
            
            # Compute power spectrum
            fft = np.fft.fft(audio_data)
            power_spectrum = np.abs(fft)**2
            freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
            
            # Use only positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_spectrum = power_spectrum[:len(power_spectrum)//2]
            
            # Convert to dB
            spectrum_db = 10 * np.log10(positive_spectrum + 1e-10)
            
            ax1.semilogx(positive_freqs[1:], spectrum_db[1:], 
                        color=self.colors['primary'], linewidth=1)
            ax1.set_xlim(20, sample_rate//2)
            ax1.set_ylim(np.min(spectrum_db[1:]), np.max(spectrum_db[1:]))
        
        ax1.set_title('Power Spectrum', fontweight='bold')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power (dB)')
        ax1.grid(True, alpha=0.3)
        
        # Add frequency reference lines
        freq_refs = [(60, 'Low'), (250, 'Low-Mid'), (1000, 'Mid'), 
                    (4000, 'High-Mid'), (10000, 'High')]
        for freq, label in freq_refs:
            ax1.axvline(x=freq, color='gray', linestyle='--', alpha=0.5)
            ax1.text(freq, ax1.get_ylim()[1], label, rotation=90, 
                    ha='right', va='top', fontsize=8)
        
        # Plot 2: Frequency band energy distribution
        ax2 = axes[0, 1]
        freq_bands = spectrum.get('frequency_bands', {})
        
        if freq_bands:
            bands = list(freq_bands.keys())
            energies = list(freq_bands.values())
            colors_bands = [self.colors['primary'], self.colors['secondary'], 
                           self.colors['accent']]
            
            wedges, texts, autotexts = ax2.pie(energies, labels=bands, autopct='%1.1f%%',
                                              colors=colors_bands, startangle=90)
            ax2.set_title('Frequency Band Energy Distribution', fontweight='bold')
            
            # Enhance text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # Plot 3: Spectral features
        ax3 = axes[1, 0]
        spectral_features = {
            'Centroid': spectrum.get('spectral_centroid_mean', 2000),
            'Bandwidth': spectrum.get('spectral_bandwidth_mean', 1000),
            'Rolloff': spectrum.get('spectral_rolloff_mean', 8000),
            'Contrast': spectrum.get('spectral_contrast_mean', 0.5)
        }
        
        # Normalize features for visualization
        features = list(spectral_features.keys())
        values = list(spectral_features.values())
        
        # Create separate scales for frequency and ratio features
        freq_features = ['Centroid', 'Bandwidth', 'Rolloff']
        ratio_features = ['Contrast']
        
        # Plot frequency features
        freq_values = [spectral_features[f] for f in freq_features if f in spectral_features]
        freq_bars = ax3.bar(freq_features, freq_values, 
                           color=[self.colors['primary'], self.colors['secondary'], self.colors['accent']])
        
        ax3.set_title('Spectral Features', fontweight='bold')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(freq_bars, freq_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.0f} Hz', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Spectral contrast on separate axis
        ax4 = axes[1, 1]
        contrast_value = spectral_features.get('Contrast', 0.5)
        zcr = spectrum.get('zero_crossing_rate_mean', 0.1)
        
        contrast_features = ['Spectral Contrast', 'Zero Crossing Rate']
        contrast_values = [contrast_value, zcr]
        
        bars = ax4.bar(contrast_features, contrast_values, 
                      color=[self.colors['success'], self.colors['warning']])
        
        ax4.set_title('Spectral Characteristics', fontweight='bold')
        ax4.set_ylabel('Ratio')
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, contrast_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_stereo_analysis(self, analysis_data: Dict[str, Any], 
                            output_file: str = None) -> None:
        """
        Create stereo imaging and spatial analysis visualization.
        
        Shows stereo width, correlation, phase relationships, and
        left/right channel balance for stereo audio files.
        
        Args:
            analysis_data (Dict[str, Any]): Audio analysis data
            output_file (str, optional): Path to save the plot
        """
        
        stereo_props = analysis_data.get('stereo_properties', {})
        
        if not stereo_props:
            print("No stereo data available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Stereo Imaging Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Stereo correlation
        ax1 = axes[0, 0]
        correlation = stereo_props.get('correlation_coefficient', 0.5)
        
        # Create correlation gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        ax1.plot(x, y, 'k-', linewidth=2)
        
        # Mark correlation value
        corr_angle = np.pi * (1 - correlation) / 2
        corr_x = np.cos(corr_angle)
        corr_y = np.sin(corr_angle)
        ax1.arrow(0, 0, corr_x, corr_y, head_width=0.1, head_length=0.1, 
                 fc=self.colors['primary'], ec=self.colors['primary'])
        
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-0.2, 1.2)
        ax1.set_aspect('equal')
        ax1.set_title(f'Stereo Correlation: {correlation:.3f}', fontweight='bold')
        ax1.text(0, -0.1, 'Mono\n(1.0)', ha='center', va='top')
        ax1.text(-1, 0.5, 'Wide\n(-1.0)', ha='center', va='center')
        ax1.text(1, 0.5, 'Narrow\n(0.0)', ha='center', va='center')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Channel balance
        ax2 = axes[0, 1]
        left_rms = stereo_props.get('left_rms_db', -20)
        right_rms = stereo_props.get('right_rms_db', -20)
        balance = stereo_props.get('balance_factor', 0)
        
        channels = ['Left', 'Right']
        levels = [left_rms, right_rms]
        colors = [self.colors['primary'], self.colors['secondary']]
        
        bars = ax2.bar(channels, levels, color=colors, alpha=0.7)
        ax2.set_title(f'Channel Balance (Factor: {balance:.3f})', fontweight='bold')
        ax2.set_ylabel('RMS Level (dB)')
        ax2.set_ylim(-40, 0)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, levels):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f} dB', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Stereo width visualization
        ax3 = axes[1, 0]
        width = stereo_props.get('stereo_width', 0.5)
        
        # Create stereo field visualization
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        ax3.add_patch(circle)
        
        # Show stereo width as an arc
        width_angle = width * np.pi
        start_angle = (np.pi - width_angle) / 2
        end_angle = (np.pi + width_angle) / 2
        
        arc_theta = np.linspace(start_angle, end_angle, 50)
        arc_x = 0.8 * np.cos(arc_theta)
        arc_y = 0.8 * np.sin(arc_theta)
        
        ax3.fill_between(arc_x, arc_y, 0, alpha=0.3, color=self.colors['primary'])
        ax3.plot(arc_x, arc_y, color=self.colors['primary'], linewidth=3)
        
        ax3.set_xlim(-1.2, 1.2)
        ax3.set_ylim(-0.2, 1.2)
        ax3.set_aspect('equal')
        ax3.set_title(f'Stereo Width: {width:.3f}', fontweight='bold')
        ax3.text(-1, -0.1, 'L', ha='center', va='top', fontweight='bold')
        ax3.text(1, -0.1, 'R', ha='center', va='top', fontweight='bold')
        ax3.text(0, -0.1, 'Center', ha='center', va='top')
        
        # Plot 4: Phase relationship
        ax4 = axes[1, 1]
        phase_coherence = stereo_props.get('phase_coherence', 0.8)
        width_factor = stereo_props.get('width_factor', 0.5)
        
        # Radar chart for stereo metrics
        metrics = ['Correlation', 'Width', 'Balance', 'Phase\nCoherence']
        values = [
            (correlation + 1) / 2,  # Normalize correlation to 0-1
            width,
            (balance + 1) / 2,  # Normalize balance to 0-1  
            phase_coherence
        ]
        
        # Close the polygon
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        ax4.plot(angles, values, 'o-', linewidth=2, color=self.colors['primary'])
        ax4.fill(angles, values, alpha=0.25, color=self.colors['primary'])
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics)
        ax4.set_ylim(0, 1)
        ax4.set_title('Stereo Metrics Overview', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_genre_comparison(self, analysis_data: Dict[str, Any], 
                             output_file: str = None) -> None:
        """
        Create genre detection and comparison visualization.
        
        Shows detected genre confidence, comparison with different genre
        profiles, and genre-specific recommendations.
        
        Args:
            analysis_data (Dict[str, Any]): Audio analysis data
            output_file (str, optional): Path to save the plot
        """
        
        genre_data = analysis_data.get('genre_detection', {})
        
        if not genre_data:
            print("No genre detection data available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Genre Analysis and Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Genre confidence scores
        ax1 = axes[0, 0]
        genre_scores = genre_data.get('genre_scores', {})
        detected_genre = genre_data.get('detected_genre', 'unknown')
        
        if genre_scores:
            genres = list(genre_scores.keys())
            scores = list(genre_scores.values())
            colors = [self.genre_colors.get(genre, self.colors['neutral']) for genre in genres]
            
            # Highlight detected genre
            for i, genre in enumerate(genres):
                if genre == detected_genre:
                    colors[i] = self.colors['accent']
            
            bars = ax1.barh(genres, scores, color=colors, alpha=0.7)
            ax1.set_title(f'Genre Detection Scores\nDetected: {detected_genre.title()}', 
                         fontweight='bold')
            ax1.set_xlabel('Confidence Score')
            ax1.set_xlim(0, 1)
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Add score labels
            for bar, score in zip(bars, scores):
                width = bar.get_width()
                ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        # Plot 2: Feature matching analysis
        ax2 = axes[0, 1]
        feature_analysis = genre_data.get('feature_analysis', {})
        
        if feature_analysis and 'category_scores' in feature_analysis:
            categories = list(feature_analysis['category_scores'].keys())
            cat_scores = list(feature_analysis['category_scores'].values())
            
            bars = ax2.bar(categories, cat_scores, 
                          color=[self.colors['primary'], self.colors['secondary'], 
                                self.colors['accent'], self.colors['success']])
            ax2.set_title('Feature Category Matching', fontweight='bold')
            ax2.set_ylabel('Match Score')
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add score labels
            for bar, score in zip(bars, cat_scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Genre characteristics comparison
        ax3 = axes[1, 0]
        if detected_genre != 'unknown':
            # Get current audio characteristics
            dynamics = analysis_data.get('dynamics', {})
            spectrum = analysis_data.get('frequency_spectrum', {})
            temporal = analysis_data.get('temporal_features', {})
            
            current_values = [
                dynamics.get('dynamic_range_db', 10),
                dynamics.get('rms_db', -20) + 40,  # Normalize for visualization
                spectrum.get('spectral_centroid_mean', 2000) / 100,  # Scale down
                temporal.get('tempo', 120)
            ]
            
            # Get genre target values  
            genre_config = Config.get_genre_config(detected_genre)
            target_values = [
                np.mean(genre_config['expected_dynamic_range']),
                np.mean(genre_config['expected_rms']) + 40,  # Normalize
                2000 / 100,  # Typical centroid scaled
                100  # Typical tempo
            ]
            
            characteristics = ['Dynamic\nRange', 'Loudness\n(Norm)', 'Brightness\n(Scaled)', 'Tempo']
            
            x = np.arange(len(characteristics))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, current_values, width, 
                           label='Current', color=self.colors['primary'], alpha=0.7)
            bars2 = ax3.bar(x + width/2, target_values, width,
                           label=f'{detected_genre.title()} Target', 
                           color=self.genre_colors[detected_genre], alpha=0.7)
            
            ax3.set_title('Current vs Genre Characteristics', fontweight='bold')
            ax3.set_ylabel('Normalized Values')
            ax3.set_xticks(x)
            ax3.set_xticklabels(characteristics)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Recommendations summary
        ax4 = axes[1, 1]
        recommendations = genre_data.get('recommendations', {})
        
        if recommendations and 'specific_advice' in recommendations:
            advice_list = recommendations['specific_advice']
            
            # Create text visualization of recommendations
            ax4.text(0.05, 0.95, 'Genre-Specific Recommendations:', 
                    transform=ax4.transAxes, fontsize=12, fontweight='bold',
                    verticalalignment='top')
            
            y_pos = 0.85
            for i, advice in enumerate(advice_list[:6]):  # Show up to 6 recommendations
                ax4.text(0.05, y_pos, f"• {advice}", 
                        transform=ax4.transAxes, fontsize=9,
                        verticalalignment='top', wrap=True)
                y_pos -= 0.12
            
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            
            # Add colored background
            bg_rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, 
                                      edgecolor=self.genre_colors.get(detected_genre, self.colors['neutral']), 
                                      facecolor=self.genre_colors.get(detected_genre, self.colors['neutral']), 
                                      alpha=0.1, transform=ax4.transAxes)
            ax4.add_patch(bg_rect)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_professional_standards(self, analysis_data: Dict[str, Any], 
                                   output_file: str = None) -> None:
        """
        Create professional audio standards comparison visualization.
        
        Compares current audio metrics against industry standards for
        streaming, broadcast, and CD mastering.
        
        Args:
            analysis_data (Dict[str, Any]): Audio analysis data  
            output_file (str, optional): Path to save the plot
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Professional Standards Comparison', fontsize=16, fontweight='bold')
        
        dynamics = analysis_data.get('dynamics', {})
        
        # Current values
        current_rms = dynamics.get('rms_db', -20)
        current_peak = dynamics.get('peak_db', -6) 
        current_dr = dynamics.get('dynamic_range_db', 10)
        current_lufs = dynamics.get('lufs', current_rms)  # Approximate if not available
        
        # Professional standards
        standards = {
            'Streaming (Spotify)': {'lufs': -14, 'peak': -1, 'dr': 6},
            'Streaming (Apple)': {'lufs': -16, 'peak': -1, 'dr': 8},
            'CD Mastering': {'lufs': -9, 'peak': -0.1, 'dr': 8},
            'Broadcast': {'lufs': -23, 'peak': -3, 'dr': 12}
        }
        
        # Plot 1: LUFS comparison
        ax1 = axes[0, 0]
        standard_names = list(standards.keys())
        lufs_targets = [standards[std]['lufs'] for std in standard_names]
        
        bars = ax1.barh(standard_names, lufs_targets, alpha=0.7, color=self.colors['primary'])
        ax1.axvline(x=current_lufs, color=self.colors['accent'], linewidth=3, 
                   label=f'Current: {current_lufs:.1f} LUFS')
        
        ax1.set_title('LUFS Target Comparison', fontweight='bold')
        ax1.set_xlabel('LUFS')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, value in zip(bars, lufs_targets):
            width = bar.get_width()
            label_x = width - 1 if width < 0 else width + 0.5
            ax1.text(label_x, bar.get_y() + bar.get_height()/2,
                    f'{value} LUFS', ha='left' if width < 0 else 'left', 
                    va='center', fontweight='bold')
        
        # Plot 2: Peak level comparison
        ax2 = axes[0, 1]
        peak_targets = [standards[std]['peak'] for std in standard_names]
        
        bars = ax2.barh(standard_names, peak_targets, alpha=0.7, color=self.colors['secondary'])
        ax2.axvline(x=current_peak, color=self.colors['accent'], linewidth=3,
                   label=f'Current: {current_peak:.1f} dB')
        
        ax2.set_title('Peak Level Comparison', fontweight='bold')  
        ax2.set_xlabel('Peak Level (dB)')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, value in zip(bars, peak_targets):
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{value} dB', ha='left', va='center', fontweight='bold')
        
        # Plot 3: Dynamic range comparison
        ax3 = axes[1, 0]
        dr_targets = [standards[std]['dr'] for std in standard_names]
        
        bars = ax3.barh(standard_names, dr_targets, alpha=0.7, color=self.colors['success'])
        ax3.axvline(x=current_dr, color=self.colors['accent'], linewidth=3,
                   label=f'Current: {current_dr:.1f} dB')
        
        ax3.set_title('Dynamic Range Comparison', fontweight='bold')
        ax3.set_xlabel('Dynamic Range (dB)')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, value in zip(bars, dr_targets):
            width = bar.get_width()
            ax3.text(width + 0.2, bar.get_y() + bar.get_height()/2,
                    f'{value} dB', ha='left', va='center', fontweight='bold')
        
        # Plot 4: Overall compliance radar
        ax4 = axes[1, 1]
        
        # Calculate compliance scores for each standard
        compliance_data = {}
        for std_name, std_values in standards.items():
            lufs_score = 1 - abs(current_lufs - std_values['lufs']) / 20  # Normalize
            peak_score = 1 - abs(current_peak - std_values['peak']) / 10  # Normalize
            dr_score = min(current_dr / std_values['dr'], 2) / 2  # Normalize, cap at 2x target
            
            # Ensure scores are between 0 and 1
            lufs_score = max(0, min(1, lufs_score))
            peak_score = max(0, min(1, peak_score))
            dr_score = max(0, min(1, dr_score))
            
            compliance_data[std_name] = [lufs_score, peak_score, dr_score]
        
        # Create radar chart
        categories = ['LUFS', 'Peak', 'Dynamic Range']
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = [self.colors['primary'], self.colors['secondary'], 
                 self.colors['accent'], self.colors['success']]
        
        for i, (std_name, scores) in enumerate(compliance_data.items()):
            scores += scores[:1]  # Complete the circle
            ax4.plot(angles, scores, 'o-', linewidth=2, 
                    label=std_name, color=colors[i % len(colors)])
            ax4.fill(angles, scores, alpha=0.1, color=colors[i % len(colors)])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 1)
        ax4.set_title('Standards Compliance Score', fontweight='bold')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_analysis_dashboard(self, analysis_data: Dict[str, Any], 
                                 output_file: str = None) -> None:
        """
        Create a comprehensive dashboard with key metrics and visualizations.
        
        Combines the most important analysis results into a single
        overview dashboard for quick assessment.
        
        Args:
            analysis_data (Dict[str, Any]): Audio analysis data
            output_file (str, optional): Path to save the plot
        """
        
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('Audio Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Get data
        file_info = analysis_data.get('file_info', {})
        dynamics = analysis_data.get('dynamics', {})
        spectrum = analysis_data.get('frequency_spectrum', {})
        genre_data = analysis_data.get('genre_detection', {})
        stereo_props = analysis_data.get('stereo_properties', {})
        
        # 1. File info and basic metrics (top section)
        ax_info = fig.add_subplot(gs[0, :2])
        ax_info.axis('off')
        
        info_text = f"""FILE INFORMATION
Filename: {file_info.get('filename', 'Unknown')}
Duration: {file_info.get('duration_seconds', 0):.1f} seconds
Sample Rate: {file_info.get('sample_rate', 0)} Hz
Channels: {file_info.get('channels', 0)} ({'Stereo' if file_info.get('is_stereo', False) else 'Mono'})
File Size: {file_info.get('file_size_mb', 0):.1f} MB

DYNAMICS SUMMARY
RMS Level: {dynamics.get('rms_db', 0):.1f} dB
Peak Level: {dynamics.get('peak_db', 0):.1f} dB  
Dynamic Range: {dynamics.get('dynamic_range_db', 0):.1f} dB
Crest Factor: {dynamics.get('crest_factor_db', 0):.1f} dB"""
        
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['background']))
        
        # 2. Quality gauge
        ax_gauge = fig.add_subplot(gs[0, 2])
        
        # Calculate overall quality score
        dr_score = min(dynamics.get('dynamic_range_db', 10) / 15, 1)
        peak_score = 1 if dynamics.get('peak_db', -6) <= -1 else 0.5
        rms_score = 1 if -20 <= dynamics.get('rms_db', -20) <= -8 else 0.5
        
        quality_score = (dr_score + peak_score + rms_score) / 3
        
        # Create quality gauge
        theta = np.linspace(0, np.pi, 100)
        ax_gauge.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=3)
        
        # Color zones
        for i, (angle_start, angle_end, color, label) in enumerate([
            (0, np.pi/3, 'red', 'Poor'),
            (np.pi/3, 2*np.pi/3, 'orange', 'Good'), 
            (2*np.pi/3, np.pi, 'green', 'Excellent')
        ]):
            theta_zone = np.linspace(angle_start, angle_end, 50)
            ax_gauge.fill_between(np.cos(theta_zone), np.sin(theta_zone), 0, 
                                alpha=0.3, color=color)
        
        # Quality needle
        needle_angle = np.pi * (1 - quality_score)
        ax_gauge.arrow(0, 0, np.cos(needle_angle), np.sin(needle_angle),
                      head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        ax_gauge.set_xlim(-1.2, 1.2)
        ax_gauge.set_ylim(-0.2, 1.2)
        ax_gauge.set_aspect('equal')
        ax_gauge.set_title(f'Quality Score: {quality_score:.1%}', fontweight='bold')
        ax_gauge.axis('off')
        
        # 3. Genre detection
        ax_genre = fig.add_subplot(gs[0, 3])
        ax_genre.axis('off')
        
        detected_genre = genre_data.get('detected_genre', 'unknown')
        confidence = genre_data.get('confidence', 0)
        
        genre_text = f"""GENRE DETECTION
Detected: {detected_genre.title()}
Confidence: {confidence:.1%}

STATUS: {'✓ Confident' if confidence > 0.5 else '? Uncertain' if confidence > 0.3 else '✗ Unknown'}"""
        
        ax_genre.text(0.05, 0.95, genre_text, transform=ax_genre.transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", 
                              facecolor=self.genre_colors.get(detected_genre, self.colors['neutral']),
                              alpha=0.3))
        
        # 4. Frequency spectrum (middle left)
        ax_spectrum = fig.add_subplot(gs[1:3, :2])
        
        freq_bands = spectrum.get('frequency_bands', {})
        if freq_bands:
            bands = ['Low\n(20-250 Hz)', 'Mid\n(250-4k Hz)', 'High\n(4k-20k Hz)']
            energies = [freq_bands.get('low', 0), freq_bands.get('mid', 0), freq_bands.get('high', 0)]
            colors_freq = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
            
            bars = ax_spectrum.bar(bands, energies, color=colors_freq, alpha=0.7)
            ax_spectrum.set_title('Frequency Energy Distribution', fontweight='bold')
            ax_spectrum.set_ylabel('Energy')
            ax_spectrum.grid(True, alpha=0.3, axis='y')
            
            for bar, energy in zip(bars, energies):
                height = bar.get_height()
                ax_spectrum.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{energy:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Stereo imaging (middle right)
        if stereo_props:
            ax_stereo = fig.add_subplot(gs[1:3, 2:])
            
            # Stereo metrics
            correlation = stereo_props.get('correlation_coefficient', 0.5)
            width = stereo_props.get('stereo_width', 0.5)
            balance = stereo_props.get('balance_factor', 0)
            
            # Radar chart
            metrics = ['Correlation', 'Width', 'Balance']
            values = [(correlation + 1) / 2, width, (abs(balance) + 1) / 2]
            values += values[:1]
            
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            ax_stereo.plot(angles, values, 'o-', linewidth=3, color=self.colors['primary'])
            ax_stereo.fill(angles, values, alpha=0.25, color=self.colors['primary'])
            ax_stereo.set_xticks(angles[:-1])
            ax_stereo.set_xticklabels(metrics)
            ax_stereo.set_ylim(0, 1)
            ax_stereo.set_title('Stereo Properties', fontweight='bold')
            ax_stereo.grid(True, alpha=0.3)
            
        # 6. Professional standards compliance (bottom)
        ax_standards = fig.add_subplot(gs[3, :])
        
        current_rms = dynamics.get('rms_db', -20)
        current_peak = dynamics.get('peak_db', -6)
        current_dr = dynamics.get('dynamic_range_db', 10)
        
        # Check compliance with different standards
        standards_check = {
            'Streaming': all([
                -16 <= current_rms <= -12,
                current_peak <= -1,
                current_dr >= 6
            ]),
            'CD': all([
                -12 <= current_rms <= -6,
                current_peak <= -0.1,
                current_dr >= 8
            ]),
            'Broadcast': all([
                -25 <= current_rms <= -20,
                current_peak <= -3,
                current_dr >= 12
            ])
        }
        
        standards_names = list(standards_check.keys())
        compliance = [1 if standards_check[std] else 0 for std in standards_names]
        colors_std = ['green' if c else 'red' for c in compliance]
        
        bars = ax_standards.bar(standards_names, compliance, color=colors_std, alpha=0.7)
        ax_standards.set_title('Professional Standards Compliance', fontweight='bold')
        ax_standards.set_ylabel('Compliant')
        ax_standards.set_ylim(0, 1.2)
        ax_standards.grid(True, alpha=0.3, axis='y')
        
        # Add compliance labels
        for bar, compliant, std_name in zip(bars, compliance, standards_names):
            label = '✓ PASS' if compliant else '✗ FAIL'
            ax_standards.text(bar.get_x() + bar.get_width()/2., 0.5,
                            label, ha='center', va='center', fontweight='bold',
                            color='white', fontsize=12)
        
        if output_file:
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_analysis_summary(self, analysis_data: Dict[str, Any], 
                             output_file: str) -> None:
        """
        Save a text summary of the analysis alongside visualizations.
        
        Args:
            analysis_data (Dict[str, Any]): Audio analysis data
            output_file (str): Path to save the summary text file
        """
        
        file_info = analysis_data.get('file_info', {})
        dynamics = analysis_data.get('dynamics', {})
        spectrum = analysis_data.get('frequency_spectrum', {})
        genre_data = analysis_data.get('genre_detection', {})
        stereo_props = analysis_data.get('stereo_properties', {})
        
        summary = f"""AUDIO ANALYSIS SUMMARY
{'='*50}

FILE INFORMATION:
- Filename: {file_info.get('filename', 'Unknown')}
- Duration: {file_info.get('duration_seconds', 0):.2f} seconds
- Sample Rate: {file_info.get('sample_rate', 0)} Hz
- Channels: {file_info.get('channels', 0)} ({'Stereo' if file_info.get('is_stereo', False) else 'Mono'})
- File Size: {file_info.get('file_size_mb', 0):.1f} MB

DYNAMIC ANALYSIS:
- RMS Level: {dynamics.get('rms_db', 0):.2f} dB
- Peak Level: {dynamics.get('peak_db', 0):.2f} dB
- Dynamic Range: {dynamics.get('dynamic_range_db', 0):.2f} dB
- Crest Factor: {dynamics.get('crest_factor_db', 0):.2f} dB
- Loudness Range: {dynamics.get('loudness_range_db', 0):.2f} dB

FREQUENCY ANALYSIS:
- Spectral Centroid: {spectrum.get('spectral_centroid_mean', 0):.0f} Hz
- Spectral Bandwidth: {spectrum.get('spectral_bandwidth_mean', 0):.0f} Hz
- Spectral Rolloff: {spectrum.get('spectral_rolloff_mean', 0):.0f} Hz
- Spectral Contrast: {spectrum.get('spectral_contrast_mean', 0):.3f}
- Zero Crossing Rate: {spectrum.get('zero_crossing_rate_mean', 0):.4f}

FREQUENCY BAND ENERGY:
"""
        
        freq_bands = spectrum.get('frequency_bands', {})
        for band, energy in freq_bands.items():
            summary += f"- {band.title()}: {energy:.3f}\n"
        
        # Genre detection
        if genre_data:
            detected_genre = genre_data.get('detected_genre', 'unknown')
            confidence = genre_data.get('confidence', 0)
            
            summary += f"""
GENRE DETECTION:
- Detected Genre: {detected_genre.title()}
- Confidence: {confidence:.1%}
"""
            
            genre_scores = genre_data.get('genre_scores', {})
            if genre_scores:
                summary += "- All Genre Scores:\n"
                for genre, score in sorted(genre_scores.items(), key=lambda x: x[1], reverse=True):
                    summary += f"  • {genre.title()}: {score:.3f}\n"
        
        # Stereo properties
        if stereo_props:
            summary += f"""
STEREO IMAGING:
- Correlation Coefficient: {stereo_props.get('correlation_coefficient', 0):.3f}
- Stereo Width: {stereo_props.get('stereo_width', 0):.3f}
- Balance Factor: {stereo_props.get('balance_factor', 0):.3f}
- Left RMS: {stereo_props.get('left_rms_db', 0):.2f} dB
- Right RMS: {stereo_props.get('right_rms_db', 0):.2f} dB
"""
        
        # Professional standards assessment
        current_rms = dynamics.get('rms_db', -20)
        current_peak = dynamics.get('peak_db', -6)
        current_dr = dynamics.get('dynamic_range_db', 10)
        
        summary += f"""
PROFESSIONAL STANDARDS ASSESSMENT:
- Streaming Compliance (-14 LUFS, -1 dB peak, 6+ dB DR): {'✓' if -16 <= current_rms <= -12 and current_peak <= -1 and current_dr >= 6 else '✗'}
- CD Compliance (-9 LUFS, -0.1 dB peak, 8+ dB DR): {'✓' if -12 <= current_rms <= -6 and current_peak <= -0.1 and current_dr >= 8 else '✗'}
- Broadcast Compliance (-23 LUFS, -3 dB peak, 12+ dB DR): {'✓' if -25 <= current_rms <= -20 and current_peak <= -3 and current_dr >= 12 else '✗'}

ANALYSIS TIMESTAMP: {analysis_data.get('analysis_timestamp', 'Unknown')}
"""
        
        # Save summary
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"Analysis summary saved to: {output_file}")
