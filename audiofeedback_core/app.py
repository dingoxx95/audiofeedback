#!/usr/bin/env python3
"""
Audio Feedback Application Module

This module contains the main application logic for the Audio Feedback Analyzer.
It orchestrates all the analysis components including audio analysis, genre detection,
LLM feedback generation, and visualization creation. This module serves as the
central coordinator for the entire analysis workflow.

Author: Audio Feedback Analyzer
License: MIT
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import librosa
    import numpy as np
except ImportError as e:
    print(f"Missing audio libraries: {e}")
    print("Install with: pip install librosa numpy")
    sys.exit(1)

from .config import Config
from .audio_analyzer import AudioAnalyzer
from .genre_detector import GenreDetector
from .llm_generator import LLMFeedbackGenerator
from .visualization import AudioVisualizer


class AudioFeedbackApp:
    """
    Main application class for the Audio Feedback Analyzer.
    
    This class coordinates all components of the audio analysis system,
    providing a unified interface for analyzing audio files and generating
    comprehensive feedback reports. It handles the complete workflow from
    audio loading to report generation.
    
    Features:
    - Complete audio analysis pipeline
    - Genre detection and classification
    - LLM-powered professional feedback
    - Comprehensive visualization reports
    - Batch processing capabilities
    - Flexible output formats
    
    Attributes:
        config (Config): Application configuration
        analyzer (AudioAnalyzer): Audio analysis engine
        genre_detector (GenreDetector): Genre detection system
        llm_generator (LLMFeedbackGenerator): LLM feedback generator
        visualizer (AudioVisualizer): Visualization engine
        
    Example:
        >>> app = AudioFeedbackApp()
        >>> result = app.analyze_file("audio.wav", generate_visuals=True)
        >>> print(result['summary'])
    """
    
    def __init__(self, config: Config = None, enable_llm: bool = True, 
                 enable_visualizations: bool = True):
        """
        Initialize the Audio Feedback Application.
        
        Args:
            config (Config, optional): Configuration object. If None, uses default config.
            enable_llm (bool): Whether to enable LLM feedback generation
            enable_visualizations (bool): Whether to enable visualization generation
        """
        self.config = config or Config()
        self.enable_llm = enable_llm
        self.enable_visualizations = enable_visualizations
        
        # Initialize components
        print("Initializing Audio Feedback Analyzer...")
        
        # Core analyzer
        self.analyzer = AudioAnalyzer()
        print("✓ Audio analyzer initialized")
        
        # Genre detector
        self.genre_detector = GenreDetector()
        print("✓ Genre detector initialized")
        
        # LLM generator (optional)
        self.llm_generator = None
        if enable_llm:
            try:
                self.llm_generator = LLMFeedbackGenerator()
                if self.llm_generator.test_connection():
                    print("✓ LLM generator initialized and ready")
                else:
                    print("⚠ LLM generator initialized but not available")
                    self.enable_llm = False
            except Exception as e:
                print(f"⚠ LLM generator failed to initialize: {e}")
                self.enable_llm = False
        
        # Visualizer (optional)
        self.visualizer = None
        if enable_visualizations:
            try:
                self.visualizer = AudioVisualizer()
                print("✓ Visualizer initialized")
            except Exception as e:
                print(f"⚠ Visualizer failed to initialize: {e}")
                self.enable_visualizations = False
        
        print("Audio Feedback Analyzer ready!\n")
    
    def analyze_file(self, file_path: str, genre: str = None, 
                    generate_llm_feedback: bool = None, 
                    generate_visuals: bool = None,
                    output_dir: str = None) -> Dict[str, Any]:
        """
        Perform complete analysis of an audio file.
        
        This method coordinates the entire analysis pipeline including:
        - Audio loading and preprocessing
        - Technical audio analysis
        - Genre detection and classification
        - LLM feedback generation (if enabled)
        - Visualization creation (if enabled)
        
        Args:
            file_path (str): Path to the audio file to analyze
            genre (str, optional): Force a specific genre for analysis
            generate_llm_feedback (bool, optional): Override LLM generation setting
            generate_visuals (bool, optional): Override visualization setting
            output_dir (str, optional): Directory for output files
            
        Returns:
            Dict[str, Any]: Complete analysis results containing:
                - analysis_data: Raw analysis data
                - genre_detection: Genre detection results
                - llm_feedback: Generated feedback (if enabled)
                - visualizations: Paths to generated visualizations (if enabled)
                - summary: Executive summary of analysis
                - recommendations: Key recommendations
                
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            ValueError: If the file format is not supported
            RuntimeError: If analysis fails
            
        Example:
            >>> result = app.analyze_file("song.wav", genre="rock")
            >>> print(result['summary']['quality_rating'])
            >>> print(result['recommendations'][:3])
        """
        
        print(f"Analyzing audio file: {file_path}")
        start_time = time.time()
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Override instance settings if specified
        use_llm = generate_llm_feedback if generate_llm_feedback is not None else self.enable_llm
        use_visuals = generate_visuals if generate_visuals is not None else self.enable_visualizations
        
        # Set up output directory
        if output_dir is None:
            output_dir = self._get_default_output_dir(file_path)
        
        results = {
            'file_path': file_path,
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_data': {},
            'genre_detection': {},
            'llm_feedback': None,
            'visualizations': {},
            'summary': {},
            'recommendations': [],
            'processing_time': 0
        }
        
        try:
            # Step 1: Core audio analysis
            print("Step 1/4: Performing audio analysis...")
            analysis_data = self.analyzer.analyze_file(file_path)
            results['analysis_data'] = analysis_data
            print("✓ Audio analysis completed")
            
            # Step 2: Genre detection
            print("Step 2/4: Detecting genre...")
            if genre:
                # Use specified genre
                genre_result = {
                    'detected_genre': genre,
                    'confidence': 1.0,
                    'genre_scores': {genre: 1.0},
                    'feature_analysis': {},
                    'recommendations': self.genre_detector.get_genre_characteristics(genre)
                }
            else:
                # Automatic detection
                genre_result = self.genre_detector.detect_genre(analysis_data)
            
            results['genre_detection'] = genre_result
            detected_genre = genre_result['detected_genre']
            confidence = genre_result['confidence']
            print(f"✓ Genre detection completed: {detected_genre} ({confidence:.1%} confidence)")
            
            # Step 3: LLM feedback generation
            if use_llm and self.llm_generator:
                print("Step 3/4: Generating AI feedback...")
                try:
                    llm_feedback = self.llm_generator.generate_feedback(
                        analysis_data, 
                        genre=detected_genre
                    )
                    results['llm_feedback'] = llm_feedback
                    print("✓ AI feedback generated")
                    
                    # Print the LLM feedback immediately
                    print("\n" + "="*60)
                    print("AUDIO FEEDBACK REPORT")
                    print("="*60)
                    print(llm_feedback)
                    
                except Exception as e:
                    print(f"⚠ LLM feedback generation failed: {e}")
                    results['llm_feedback'] = None
            else:
                print("Step 3/4: Skipping AI feedback (disabled)")
            
            # Step 4: Visualization generation
            if use_visuals and self.visualizer:
                print("Step 4/4: Creating visualizations...")
                try:
                    vis_files = self.visualizer.create_comprehensive_report(
                        analysis_data, 
                        output_dir,
                        Path(file_path).stem
                    )
                    results['visualizations'] = vis_files
                    
                    # Also save text summary
                    summary_file = os.path.join(output_dir, f"{Path(file_path).stem}_summary.txt")
                    self.visualizer.save_analysis_summary(analysis_data, summary_file)
                    results['visualizations']['summary_text'] = summary_file
                    
                    print(f"✓ Visualizations created in {output_dir}")
                except Exception as e:
                    print(f"⚠ Visualization generation failed: {e}")
                    results['visualizations'] = {}
            else:
                print("Step 4/4: Skipping visualizations (disabled)")
            
            # Generate executive summary
            results['summary'] = self._generate_executive_summary(analysis_data, genre_result)
            results['recommendations'] = self._generate_key_recommendations(analysis_data, genre_result)
            
            # Calculate processing time
            results['processing_time'] = time.time() - start_time
            
            print(f"\n✓ Analysis completed in {results['processing_time']:.1f} seconds")
            
            # Only print summary if no LLM feedback was generated
            if not results.get('llm_feedback'):
                self._print_analysis_summary(results)
            
            return results
            
        except Exception as e:
            error_msg = f"Analysis failed: {e}"
            print(f"✗ {error_msg}")
            results['error'] = error_msg
            results['processing_time'] = time.time() - start_time
            raise RuntimeError(error_msg) from e
    
    def analyze_batch(self, file_paths: List[str], output_base_dir: str = "batch_analysis",
                     genre: str = None, parallel: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple audio files in batch.
        
        Args:
            file_paths (List[str]): List of audio file paths to analyze
            output_base_dir (str): Base directory for output files
            genre (str, optional): Force genre for all files
            parallel (bool): Whether to process files in parallel (future feature)
            
        Returns:
            Dict[str, Dict[str, Any]]: Results for each file, keyed by filename
        """
        
        print(f"Starting batch analysis of {len(file_paths)} files...")
        batch_start_time = time.time()
        
        # Create base output directory
        os.makedirs(output_base_dir, exist_ok=True)
        
        results = {}
        successful = 0
        failed = 0
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"\n--- Processing file {i}/{len(file_paths)}: {Path(file_path).name} ---")
            
            try:
                # Create individual output directory
                file_output_dir = os.path.join(output_base_dir, Path(file_path).stem)
                
                # Analyze file
                result = self.analyze_file(
                    file_path, 
                    genre=genre,
                    output_dir=file_output_dir
                )
                
                results[file_path] = result
                successful += 1
                
            except Exception as e:
                print(f"✗ Failed to analyze {file_path}: {e}")
                results[file_path] = {
                    'error': str(e),
                    'file_path': file_path
                }
                failed += 1
        
        # Generate batch summary
        batch_time = time.time() - batch_start_time
        
        batch_summary = {
            'total_files': len(file_paths),
            'successful': successful,
            'failed': failed,
            'processing_time': batch_time,
            'average_time_per_file': batch_time / len(file_paths),
            'output_directory': output_base_dir
        }
        
        # Save batch results
        batch_results_file = os.path.join(output_base_dir, "batch_results.json")
        with open(batch_results_file, 'w') as f:
            json.dump({
                'batch_summary': batch_summary,
                'individual_results': {k: self._serialize_result(v) for k, v in results.items()}
            }, f, indent=2, default=str)
        
        print(f"\n--- Batch Analysis Complete ---")
        print(f"Successfully processed: {successful}/{len(file_paths)} files")
        print(f"Total time: {batch_time:.1f} seconds")
        print(f"Average per file: {batch_time/len(file_paths):.1f} seconds")
        print(f"Results saved to: {batch_results_file}")
        
        return results
    
    def _get_default_output_dir(self, file_path: str) -> str:
        """Generate default output directory for a file."""
        base_name = Path(file_path).stem
        output_dir = f"analysis_{base_name}_{int(time.time())}"
        return output_dir
    
    def _generate_executive_summary(self, analysis_data: Dict[str, Any], 
                                   genre_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an executive summary of the analysis results."""
        
        dynamics = analysis_data.get('dynamics', {})
        file_info = analysis_data.get('file_info', {})
        spectrum = analysis_data.get('frequency_spectrum', {})
        
        # Calculate quality rating
        quality_factors = []
        
        # Dynamic range quality
        dr = dynamics.get('dynamic_range_db', 10)
        if dr >= 15:
            dr_quality = 'Excellent'
            dr_score = 5
        elif dr >= 10:
            dr_quality = 'Good'
            dr_score = 4
        elif dr >= 6:
            dr_quality = 'Acceptable'
            dr_score = 3
        elif dr >= 3:
            dr_quality = 'Poor'
            dr_score = 2
        else:
            dr_quality = 'Very Poor'
            dr_score = 1
        
        quality_factors.append(dr_score)
        
        # Peak level quality
        peak = dynamics.get('peak_db', -6)
        if peak <= -3:
            peak_quality = 'Excellent'
            peak_score = 5
        elif peak <= -1:
            peak_quality = 'Good'
            peak_score = 4
        elif peak <= -0.1:
            peak_quality = 'Acceptable'
            peak_score = 3
        elif peak <= 0:
            peak_quality = 'Poor'
            peak_score = 2
        else:
            peak_quality = 'Clipping'
            peak_score = 1
        
        quality_factors.append(peak_score)
        
        # RMS level quality (context dependent)
        rms = dynamics.get('rms_db', -20)
        if -16 <= rms <= -8:
            rms_quality = 'Good'
            rms_score = 4
        elif -20 <= rms <= -6:
            rms_quality = 'Acceptable'
            rms_score = 3
        else:
            rms_quality = 'Suboptimal'
            rms_score = 2
        
        quality_factors.append(rms_score)
        
        # Overall quality
        avg_score = sum(quality_factors) / len(quality_factors)
        if avg_score >= 4.5:
            overall_quality = 'Excellent'
        elif avg_score >= 3.5:
            overall_quality = 'Good'
        elif avg_score >= 2.5:
            overall_quality = 'Acceptable'
        elif avg_score >= 1.5:
            overall_quality = 'Poor'
        else:
            overall_quality = 'Very Poor'
        
        # Determine primary issues
        issues = []
        if dr < 6:
            issues.append("Compressed dynamics")
        if peak > -1:
            issues.append("Insufficient headroom")
        if peak > 0:
            issues.append("Clipping detected")
        if rms < -25:
            issues.append("Too quiet")
        if rms > -6:
            issues.append("Too loud")
        
        # Determine strengths
        strengths = []
        if dr >= 12:
            strengths.append("Excellent dynamic range")
        if peak <= -3:
            strengths.append("Good headroom")
        if -16 <= rms <= -10:
            strengths.append("Appropriate loudness")
        
        return {
            'overall_quality': overall_quality,
            'quality_score': avg_score,
            'dynamic_range_assessment': {
                'value': dr,
                'quality': dr_quality,
                'score': dr_score
            },
            'peak_level_assessment': {
                'value': peak,
                'quality': peak_quality,
                'score': peak_score
            },
            'loudness_assessment': {
                'value': rms,
                'quality': rms_quality,
                'score': rms_score
            },
            'detected_genre': genre_data.get('detected_genre', 'unknown'),
            'genre_confidence': genre_data.get('confidence', 0),
            'primary_issues': issues,
            'strengths': strengths,
            'file_duration': file_info.get('duration_seconds', 0),
            'file_format': {
                'sample_rate': file_info.get('sample_rate', 0),
                'channels': file_info.get('channels', 0),
                'is_stereo': file_info.get('is_stereo', False)
            }
        }
    
    def _generate_key_recommendations(self, analysis_data: Dict[str, Any], 
                                    genre_data: Dict[str, Any]) -> List[str]:
        """Generate prioritized recommendations based on analysis."""
        
        recommendations = []
        dynamics = analysis_data.get('dynamics', {})
        
        rms = dynamics.get('rms_db', -20)
        peak = dynamics.get('peak_db', -6)
        dr = dynamics.get('dynamic_range_db', 10)
        
        # Priority 1: Critical issues
        if peak > 0:
            recommendations.append(f"CRITICAL: Reduce clipping - current peak: {peak:.1f} dB")
        elif peak > -0.1:
            recommendations.append(f"Reduce peak level to -0.1 dB or lower (current: {peak:.1f} dB)")
        
        # Priority 2: Loudness standards
        if rms < -20:
            recommendations.append(f"Increase overall loudness (current: {rms:.1f} dB)")
        elif rms > -6:
            recommendations.append(f"Reduce overall loudness for better dynamics (current: {rms:.1f} dB)")
        
        # Priority 3: Dynamic range
        if dr < 6:
            recommendations.append(f"Increase dynamic range - reduce compression (current: {dr:.1f} dB)")
        elif dr < 10:
            recommendations.append(f"Consider lighter compression for better dynamics (current: {dr:.1f} dB)")
        
        # Genre-specific recommendations
        detected_genre = genre_data.get('detected_genre', 'unknown')
        if detected_genre != 'unknown':
            genre_recs = genre_data.get('recommendations', {}).get('specific_advice', [])
            if genre_recs:
                recommendations.extend(genre_recs[:3])  # Add top 3 genre recommendations
        
        # If no critical issues, add enhancement suggestions
        if len(recommendations) < 3:
            if dr >= 10 and peak <= -1:
                recommendations.append("Good technical quality - consider creative enhancements")
            
            if detected_genre != 'unknown':
                recommendations.append(f"Optimize for {detected_genre.replace('_', ' ')} genre characteristics")
            else:
                recommendations.append("Consider manual genre specification for better advice")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _print_analysis_summary(self, results: Dict[str, Any]) -> None:
        """Print a concise summary of analysis results."""
        
        summary = results.get('summary', {})
        recommendations = results.get('recommendations', [])
        
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Overall Quality: {summary.get('overall_quality', 'Unknown')}")
        print(f"Detected Genre: {summary.get('detected_genre', 'unknown').title()} "
              f"({summary.get('genre_confidence', 0):.1%} confidence)")
        
        if summary.get('primary_issues'):
            print(f"Main Issues: {', '.join(summary['primary_issues'])}")
        
        if summary.get('strengths'):
            print(f"Strengths: {', '.join(summary['strengths'])}")
        
        print("\nTop Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"{i}. {rec}")
        
        if results.get('llm_feedback'):
            print("\n✓ AI feedback generated")
        
        if results.get('visualizations'):
            vis_count = len([v for v in results['visualizations'].values() if v])
            print(f"✓ {vis_count} visualizations created")
        
        print("="*60)
    
    def _serialize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize result for JSON export, handling non-serializable objects."""
        
        serialized = {}
        
        for key, value in result.items():
            if key == 'analysis_data' and isinstance(value, dict):
                # Handle audio data which might contain numpy arrays
                serialized_analysis = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serialized_analysis[k] = v.tolist()
                    elif k == 'audio_data':
                        # Skip raw audio data for JSON serialization
                        continue
                    else:
                        serialized_analysis[k] = v
                serialized[key] = serialized_analysis
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            else:
                serialized[key] = value
        
        return serialized


def main():
    """
    Command-line interface for the Audio Feedback Analyzer.
    
    This function provides a complete CLI for analyzing audio files
    with various options and output formats.
    """
    
    parser = argparse.ArgumentParser(
        description="Professional Audio Feedback Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.wav                          # Analyze with all features
  %(prog)s audio.wav --genre rock             # Force rock genre
  %(prog)s audio.wav --no-llm                 # Skip AI feedback
  %(prog)s audio.wav --no-visuals             # Skip visualizations
  %(prog)s audio.wav --output reports/        # Custom output directory
  %(prog)s *.wav --batch                      # Batch process multiple files
        """
    )
    
    parser.add_argument('files', nargs='+', help='Audio file(s) to analyze')
    parser.add_argument('--genre', choices=['rock', 'metal', 'jazz', 'classical', 'electronic', 
                                           'drum_and_bass', 'hip_hop', 'folk', 'pop', 'podcast'], 
                       help='Force specific genre for analysis')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--no-llm', action='store_true', 
                       help='Disable AI feedback generation')
    parser.add_argument('--no-visuals', action='store_true',
                       help='Disable visualization generation')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple files in batch mode')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Configure verbosity
    if not args.verbose:
        import logging
        logging.getLogger().setLevel(logging.WARNING)
    
    # Initialize application
    app = AudioFeedbackApp(
        enable_llm=not args.no_llm,
        enable_visualizations=not args.no_visuals
    )
    
    try:
        if args.batch or len(args.files) > 1:
            # Batch processing
            output_dir = args.output or "batch_analysis"
            results = app.analyze_batch(
                args.files,
                output_base_dir=output_dir,
                genre=args.genre
            )
            
            print(f"\nBatch analysis complete. Results saved to: {output_dir}")
            
        else:
            # Single file processing
            file_path = args.files[0]
            output_dir = args.output
            
            result = app.analyze_file(
                file_path,
                genre=args.genre,
                output_dir=output_dir
            )
            
            # Save results as JSON
            output_path = output_dir or app._get_default_output_dir(file_path)
            os.makedirs(output_path, exist_ok=True)
            
            results_file = os.path.join(output_path, "analysis_results.json")
            with open(results_file, 'w') as f:
                json.dump(app._serialize_result(result), f, indent=2, default=str)
            
            print(f"\nAnalysis complete. Results saved to: {output_path}")
    
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
