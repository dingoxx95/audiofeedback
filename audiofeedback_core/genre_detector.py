#!/usr/bin/env python3
"""
Genre Detection Module

This module handles automatic music genre detection and provides genre-specific
audio analysis recommendations. It uses a combination of spectral features,
temporal characteristics, and timbral properties to classify audio content
and suggest appropriate mixing/mastering parameters.

Author: Audio Feedback Analyzer
License: MIT
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import warnings

try:
    import librosa
except ImportError as e:
    print(f"Missing audio libraries: {e}")
    print("Install with: pip install librosa")
    librosa = None

from .config import Config


class GenreDetector:
    """
    Automatic music genre detection and genre-specific analysis.
    
    This class analyzes audio features to detect the most likely genre
    and provides genre-specific recommendations for mixing and mastering.
    It uses a rule-based approach combined with spectral and temporal
    feature analysis.
    
    The detection considers:
    - Spectral characteristics (brightness, energy distribution)
    - Temporal features (tempo, rhythm patterns)
    - Dynamic properties (loudness, compression)
    - Harmonic content and timbral qualities
    
    Attributes:
        confidence_threshold (float): Minimum confidence for genre detection
        feature_weights (Dict): Weights for different feature categories
        
    Example:
        >>> detector = GenreDetector()
        >>> genre_info = detector.detect_genre(analysis_data)
        >>> print(f"Detected: {genre_info['detected_genre']} ({genre_info['confidence']:.1%})")
    """
    
    def __init__(self, confidence_threshold: float = 0.3):
        """
        Initialize the genre detector.
        
        Args:
            confidence_threshold (float): Minimum confidence level for genre detection.
                                        Values below this threshold result in 'unknown' genre.
                                        Range: 0.0 to 1.0, default: 0.3
        """
        self.confidence_threshold = confidence_threshold
        
        # Feature importance weights for genre detection
        self.feature_weights = {
            'spectral': 0.35,    # Frequency characteristics
            'temporal': 0.25,    # Tempo and rhythm
            'timbral': 0.20,     # Harmonic content and texture
            'dynamic': 0.20      # Loudness and dynamics
        }
        
        # Genre feature profiles (reference characteristics)
        self.genre_profiles = self._initialize_genre_profiles()
    
    def _initialize_genre_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize reference profiles for different music genres.
        
        These profiles contain typical ranges and characteristics for each genre,
        used as reference points for the detection algorithm.
        
        Returns:
            Dict[str, Dict[str, Any]]: Genre profiles with feature ranges
        """
        return {
            'rock': {
                'tempo_range': (80, 140),
                'spectral_centroid_range': (2000, 4000),
                'energy_distribution': {'low': 0.3, 'mid': 0.4, 'high': 0.3},
                'dynamic_range': (8, 15),
                'typical_rms': (-12, -8),
                'harmonic_ratio': (0.3, 0.7),
                'spectral_rolloff_factor': 0.85,
                'onset_density': (1.5, 4.0)
            },
            'jazz': {
                'tempo_range': (60, 180),
                'spectral_centroid_range': (1500, 3500),
                'energy_distribution': {'low': 0.35, 'mid': 0.45, 'high': 0.2},
                'dynamic_range': (12, 20),
                'typical_rms': (-18, -12),
                'harmonic_ratio': (0.6, 0.9),
                'spectral_rolloff_factor': 0.75,
                'onset_density': (0.8, 3.0)
            },
            'classical': {
                'tempo_range': (50, 160),
                'spectral_centroid_range': (1200, 3000),
                'energy_distribution': {'low': 0.4, 'mid': 0.35, 'high': 0.25},
                'dynamic_range': (15, 30),
                'typical_rms': (-25, -15),
                'harmonic_ratio': (0.7, 0.95),
                'spectral_rolloff_factor': 0.70,
                'onset_density': (0.5, 2.5)
            },
            'electronic': {
                'tempo_range': (110, 140),
                'spectral_centroid_range': (2500, 5000),
                'energy_distribution': {'low': 0.4, 'mid': 0.3, 'high': 0.3},
                'dynamic_range': (6, 12),
                'typical_rms': (-10, -6),
                'harmonic_ratio': (0.2, 0.6),
                'spectral_rolloff_factor': 0.90,
                'onset_density': (2.0, 6.0)
            },
            'hip_hop': {
                'tempo_range': (70, 100),
                'spectral_centroid_range': (1800, 3500),
                'energy_distribution': {'low': 0.5, 'mid': 0.3, 'high': 0.2},
                'dynamic_range': (6, 12),
                'typical_rms': (-12, -8),
                'harmonic_ratio': (0.3, 0.7),
                'spectral_rolloff_factor': 0.80,
                'onset_density': (1.0, 3.0)
            },
            'folk': {
                'tempo_range': (60, 120),
                'spectral_centroid_range': (1500, 3000),
                'energy_distribution': {'low': 0.3, 'mid': 0.5, 'high': 0.2},
                'dynamic_range': (10, 18),
                'typical_rms': (-20, -14),
                'harmonic_ratio': (0.6, 0.85),
                'spectral_rolloff_factor': 0.75,
                'onset_density': (0.8, 2.5)
            },
            'pop': {
                'tempo_range': (90, 130),
                'spectral_centroid_range': (2000, 4000),
                'energy_distribution': {'low': 0.3, 'mid': 0.4, 'high': 0.3},
                'dynamic_range': (6, 12),
                'typical_rms': (-12, -8),
                'harmonic_ratio': (0.4, 0.7),
                'spectral_rolloff_factor': 0.85,
                'onset_density': (1.5, 4.0)
            }
        }
    
    def detect_genre(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect the most likely genre based on audio analysis data.
        
        This method analyzes spectral, temporal, timbral, and dynamic features
        to determine the most likely genre. It returns the detected genre along
        with confidence scores and detailed feature analysis.
        
        Args:
            analysis_data (Dict[str, Any]): Complete audio analysis results containing
                                          frequency spectrum, temporal features, dynamics, etc.
                                          
        Returns:
            Dict[str, Any]: Genre detection results containing:
                - detected_genre (str): Most likely genre or 'unknown'
                - confidence (float): Confidence score (0.0 to 1.0)
                - genre_scores (Dict[str, float]): Scores for all tested genres
                - feature_analysis (Dict): Detailed breakdown of feature matching
                - recommendations (Dict): Genre-specific mixing/mastering advice
                
        Example:
            >>> result = detector.detect_genre(analysis_data)
            >>> print(f"Genre: {result['detected_genre']}")
            >>> print(f"Confidence: {result['confidence']:.1%}")
            >>> for genre, score in result['genre_scores'].items():
            ...     print(f"  {genre}: {score:.2f}")
        """
        
        # Extract features from analysis data
        features = self._extract_genre_features(analysis_data)
        
        if not features:
            return self._create_unknown_result("Insufficient features for genre detection")
        
        # Calculate genre scores
        genre_scores = {}
        feature_analysis = {}
        
        for genre, profile in self.genre_profiles.items():
            score, analysis = self._calculate_genre_score(features, profile, genre)
            genre_scores[genre] = score
            feature_analysis[genre] = analysis
        
        # Find best match
        best_genre = max(genre_scores, key=genre_scores.get)
        best_score = genre_scores[best_genre]
        
        # Check confidence threshold
        if best_score < self.confidence_threshold:
            return self._create_unknown_result("Low confidence in genre detection", 
                                             genre_scores, feature_analysis)
        
        # Generate genre-specific recommendations
        recommendations = self._generate_genre_recommendations(best_genre, features, analysis_data)
        
        return {
            'detected_genre': best_genre,
            'confidence': best_score,
            'genre_scores': genre_scores,
            'feature_analysis': feature_analysis[best_genre],
            'all_feature_analysis': feature_analysis,
            'recommendations': recommendations,
            'features_used': features
        }
    
    def _extract_genre_features(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant features for genre detection from analysis data.
        
        Args:
            analysis_data (Dict[str, Any]): Complete audio analysis results
            
        Returns:
            Dict[str, Any]: Extracted features for genre classification
        """
        features = {}
        
        try:
            # Temporal features
            temporal = analysis_data.get('temporal_features', {})
            features['tempo'] = temporal.get('tempo', 120)
            features['onset_rate'] = temporal.get('onset_rate', 1.0)
            features['onset_strength'] = temporal.get('onset_strength_mean', 0.1)
            
            # Spectral features  
            spectrum = analysis_data.get('frequency_spectrum', {})
            features['spectral_centroid'] = spectrum.get('spectral_centroid_mean', 2000)
            features['spectral_bandwidth'] = spectrum.get('spectral_bandwidth_mean', 1000)
            features['spectral_rolloff'] = spectrum.get('spectral_rolloff_mean', 8000)
            features['spectral_contrast'] = spectrum.get('spectral_contrast_mean', 0.5)
            features['zero_crossing_rate'] = spectrum.get('zero_crossing_rate_mean', 0.1)
            
            # Frequency band distribution
            freq_bands = spectrum.get('frequency_bands', {})
            total_energy = sum(freq_bands.values()) if freq_bands else 1
            if total_energy > 0:
                features['energy_low'] = freq_bands.get('low', 0) / total_energy
                features['energy_mid'] = freq_bands.get('mid', 0) / total_energy  
                features['energy_high'] = freq_bands.get('high', 0) / total_energy
            else:
                features['energy_low'] = features['energy_mid'] = features['energy_high'] = 0.33
            
            # Dynamic features
            dynamics = analysis_data.get('dynamics', {})
            features['rms_db'] = dynamics.get('rms_db', -20)
            features['peak_db'] = dynamics.get('peak_db', -6)
            features['dynamic_range'] = dynamics.get('dynamic_range_db', 10)
            features['crest_factor'] = dynamics.get('crest_factor_db', 10)
            
            # Harmonic content
            harmonic = analysis_data.get('harmonic_content', {})
            features['harmonic_ratio'] = harmonic.get('harmonic_ratio', 0.5)
            features['percussive_ratio'] = harmonic.get('percussive_ratio', 0.5)
            
            return features
            
        except Exception as e:
            print(f"Error extracting genre features: {e}")
            return {}
    
    def _calculate_genre_score(self, features: Dict[str, Any], profile: Dict[str, Any], 
                              genre: str) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate how well the features match a specific genre profile.
        
        Args:
            features (Dict[str, Any]): Extracted audio features
            profile (Dict[str, Any]): Genre reference profile
            genre (str): Genre name for scoring
            
        Returns:
            Tuple[float, Dict[str, Any]]: Overall score and detailed analysis
        """
        scores = {}
        analysis = {}
        
        # Spectral scoring
        spectral_score = 0
        spectral_factors = 0
        
        # Spectral centroid
        if 'spectral_centroid' in features:
            centroid = features['spectral_centroid']
            centroid_range = profile['spectral_centroid_range']
            centroid_score = self._score_range_match(centroid, centroid_range)
            spectral_score += centroid_score
            spectral_factors += 1
            analysis['spectral_centroid_match'] = centroid_score
        
        # Energy distribution
        if all(f'energy_{band}' in features for band in ['low', 'mid', 'high']):
            energy_dist = {
                'low': features['energy_low'],
                'mid': features['energy_mid'], 
                'high': features['energy_high']
            }
            energy_score = self._score_energy_distribution(energy_dist, profile['energy_distribution'])
            spectral_score += energy_score
            spectral_factors += 1
            analysis['energy_distribution_match'] = energy_score
        
        scores['spectral'] = spectral_score / max(spectral_factors, 1)
        
        # Temporal scoring
        temporal_score = 0
        temporal_factors = 0
        
        # Tempo
        if 'tempo' in features:
            tempo = features['tempo']
            tempo_range = profile['tempo_range']
            tempo_score = self._score_range_match(tempo, tempo_range)
            temporal_score += tempo_score
            temporal_factors += 1
            analysis['tempo_match'] = tempo_score
        
        # Onset density
        if 'onset_rate' in features:
            onset_rate = features['onset_rate']
            onset_range = profile.get('onset_density', (0.5, 5.0))
            onset_score = self._score_range_match(onset_rate, onset_range)
            temporal_score += onset_score
            temporal_factors += 1
            analysis['onset_density_match'] = onset_score
        
        scores['temporal'] = temporal_score / max(temporal_factors, 1)
        
        # Timbral scoring
        timbral_score = 0
        timbral_factors = 0
        
        # Harmonic ratio
        if 'harmonic_ratio' in features:
            harmonic_ratio = features['harmonic_ratio']
            harmonic_range = profile.get('harmonic_ratio', (0.3, 0.8))
            harmonic_score = self._score_range_match(harmonic_ratio, harmonic_range)
            timbral_score += harmonic_score
            timbral_factors += 1
            analysis['harmonic_ratio_match'] = harmonic_score
        
        scores['timbral'] = timbral_score / max(timbral_factors, 1)
        
        # Dynamic scoring
        dynamic_score = 0
        dynamic_factors = 0
        
        # Dynamic range
        if 'dynamic_range' in features:
            dr = features['dynamic_range']
            dr_range = profile['dynamic_range']
            dr_score = self._score_range_match(dr, dr_range)
            dynamic_score += dr_score
            dynamic_factors += 1
            analysis['dynamic_range_match'] = dr_score
        
        # RMS level
        if 'rms_db' in features:
            rms = features['rms_db']
            rms_range = profile['typical_rms']
            rms_score = self._score_range_match(rms, rms_range)
            dynamic_score += rms_score
            dynamic_factors += 1
            analysis['rms_level_match'] = rms_score
        
        scores['dynamic'] = dynamic_score / max(dynamic_factors, 1)
        
        # Calculate weighted overall score
        overall_score = sum(scores[category] * self.feature_weights[category] 
                           for category in scores if category in self.feature_weights)
        
        analysis['category_scores'] = scores
        analysis['overall_score'] = overall_score
        
        return overall_score, analysis
    
    def _score_range_match(self, value: float, target_range: Tuple[float, float], 
                          tolerance: float = 0.2) -> float:
        """
        Score how well a value fits within a target range.
        
        Args:
            value (float): Value to score
            target_range (Tuple[float, float]): Target range (min, max)
            tolerance (float): Tolerance for partial scoring outside range
            
        Returns:
            float: Score from 0.0 (poor match) to 1.0 (perfect match)
        """
        min_val, max_val = target_range
        range_size = max_val - min_val
        
        if min_val <= value <= max_val:
            # Perfect match within range
            return 1.0
        
        # Calculate distance from nearest boundary
        if value < min_val:
            distance = min_val - value
        else:
            distance = value - max_val
        
        # Apply tolerance
        tolerance_distance = range_size * tolerance
        
        if distance <= tolerance_distance:
            # Partial score within tolerance
            return max(0.0, 1.0 - (distance / tolerance_distance))
        else:
            # Poor match outside tolerance
            return 0.0
    
    def _score_energy_distribution(self, actual: Dict[str, float], 
                                  target: Dict[str, float]) -> float:
        """
        Score how well the energy distribution matches the target.
        
        Args:
            actual (Dict[str, float]): Actual energy distribution
            target (Dict[str, float]): Target energy distribution
            
        Returns:
            float: Score from 0.0 to 1.0
        """
        total_difference = 0
        
        for band in ['low', 'mid', 'high']:
            actual_energy = actual.get(band, 0)
            target_energy = target.get(band, 0)
            total_difference += abs(actual_energy - target_energy)
        
        # Convert difference to similarity score
        return max(0.0, 1.0 - (total_difference / 2.0))
    
    def _generate_genre_recommendations(self, genre: str, features: Dict[str, Any], 
                                      analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate genre-specific mixing and mastering recommendations.
        
        Args:
            genre (str): Detected genre
            features (Dict[str, Any]): Extracted features
            analysis_data (Dict[str, Any]): Complete analysis data
            
        Returns:
            Dict[str, Any]: Genre-specific recommendations
        """
        genre_config = Config.get_genre_config(genre)
        profile = self.genre_profiles.get(genre, {})
        
        recommendations = {
            'target_loudness': genre_config['expected_rms'],
            'target_dynamic_range': genre_config['expected_dynamic_range'],
            'frequency_emphasis': genre_config['frequency_emphasis'],
            'stereo_width_target': genre_config['stereo_width_target'],
            'specific_advice': []
        }
        
        # Generate specific advice based on current vs target characteristics
        current_rms = features.get('rms_db', -20)
        target_rms_range = genre_config['expected_rms']
        
        if current_rms < target_rms_range[0]:
            recommendations['specific_advice'].append(
                f"Increase overall loudness by {target_rms_range[0] - current_rms:.1f} dB to match {genre} standards"
            )
        elif current_rms > target_rms_range[1]:
            recommendations['specific_advice'].append(
                f"Reduce overall loudness by {current_rms - target_rms_range[1]:.1f} dB for better {genre} dynamics"
            )
        
        current_dr = features.get('dynamic_range', 10)
        target_dr_range = genre_config['expected_dynamic_range']
        
        if current_dr < target_dr_range[0]:
            recommendations['specific_advice'].append(
                f"Increase dynamic range by reducing compression - target {target_dr_range[0]}+ dB for {genre}"
            )
        elif current_dr > target_dr_range[1]:
            recommendations['specific_advice'].append(
                f"Consider gentle compression to tighten dynamics for {genre} style"
            )
        
        # Frequency-specific recommendations
        energy_dist = {
            'low': features.get('energy_low', 0.33),
            'mid': features.get('energy_mid', 0.33),
            'high': features.get('energy_high', 0.33)
        }
        
        target_dist = profile.get('energy_distribution', {})
        
        for band, target_energy in target_dist.items():
            current_energy = energy_dist.get(band, 0.33)
            difference = abs(current_energy - target_energy)
            
            if difference > 0.1:  # Significant difference
                if current_energy < target_energy:
                    recommendations['specific_advice'].append(
                        f"Boost {band}-frequency content for better {genre} character"
                    )
                else:
                    recommendations['specific_advice'].append(
                        f"Reduce {band}-frequency content for cleaner {genre} mix"
                    )
        
        return recommendations
    
    def _create_unknown_result(self, reason: str, genre_scores: Dict[str, float] = None,
                             feature_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a result for unknown/undetected genre.
        
        Args:
            reason (str): Reason for unknown classification
            genre_scores (Dict[str, float], optional): Calculated genre scores
            feature_analysis (Dict[str, Any], optional): Feature analysis results
            
        Returns:
            Dict[str, Any]: Unknown genre result
        """
        return {
            'detected_genre': 'unknown',
            'confidence': 0.0,
            'genre_scores': genre_scores or {},
            'feature_analysis': {},
            'all_feature_analysis': feature_analysis or {},
            'recommendations': {
                'target_loudness': (-14, -10),  # General streaming standard
                'target_dynamic_range': (8, 15),
                'frequency_emphasis': ['balanced'],
                'stereo_width_target': (0.3, 0.7),
                'specific_advice': [
                    "Genre could not be detected - using general recommendations",
                    "Consider manual genre specification for better advice"
                ]
            },
            'detection_reason': reason
        }
    
    def get_genre_characteristics(self, genre: str) -> Dict[str, Any]:
        """
        Get detailed characteristics and expectations for a specific genre.
        
        Args:
            genre (str): Genre name
            
        Returns:
            Dict[str, Any]: Genre characteristics and recommendations
        """
        if genre not in self.genre_profiles:
            return self._create_unknown_result(f"Unknown genre: {genre}")['recommendations']
        
        profile = self.genre_profiles[genre]
        config = Config.get_genre_config(genre)
        
        return {
            'profile': profile,
            'config': config,
            'description': self._get_genre_description(genre),
            'mixing_tips': self._get_mixing_tips(genre),
            'mastering_tips': self._get_mastering_tips(genre)
        }
    
    def _get_genre_description(self, genre: str) -> str:
        """Get a brief description of the genre characteristics."""
        descriptions = {
            'rock': "Energetic with strong mid-range presence, moderate dynamics, guitar-driven",
            'jazz': "Complex harmonies, wide dynamics, acoustic instruments, sophisticated arrangements",
            'classical': "Wide dynamic range, natural acoustics, orchestral balance, minimal compression",
            'electronic': "Synthetic sounds, controlled dynamics, wide frequency spectrum, precise imaging",
            'hip_hop': "Strong low-end, rhythmic emphasis, controlled dynamics, vocal prominence",
            'folk': "Acoustic instruments, natural dynamics, warm mid-range, intimate feel",
            'pop': "Polished production, moderate dynamics, vocal-centric, radio-ready loudness"
        }
        return descriptions.get(genre, "General music characteristics")
    
    def _get_mixing_tips(self, genre: str) -> List[str]:
        """Get genre-specific mixing tips."""
        tips = {
            'rock': [
                "Emphasize guitar and drums in the mix",
                "Use parallel compression on drums",
                "High-pass vocals around 80-100 Hz",
                "Create space with panning and reverb"
            ],
            'jazz': [
                "Preserve natural instrument balance", 
                "Use minimal compression to maintain dynamics",
                "Focus on room ambience and spatial imaging",
                "Keep low-end clean and defined"
            ],
            'classical': [
                "Maintain natural orchestral balance",
                "Avoid heavy processing or compression",
                "Preserve concert hall acoustics",
                "Focus on natural stereo imaging"
            ],
            'electronic': [
                "Use precise stereo imaging and effects",
                "Layer sounds carefully in frequency spectrum",
                "Apply creative sound design and processing",
                "Control dynamics with sidechaining"
            ],
            'hip_hop': [
                "Make bass and kick the foundation",
                "Use heavy compression on drums",
                "Create space for vocal clarity",
                "Layer samples and loops effectively"
            ],
            'folk': [
                "Maintain acoustic instrument authenticity",
                "Use gentle compression and EQ",
                "Focus on vocal clarity and warmth",
                "Create intimate soundscape"
            ],
            'pop': [
                "Make vocals prominent and clear",
                "Use consistent dynamics and loudness",
                "Apply modern production techniques",
                "Focus on radio-ready sound"
            ]
        }
        return tips.get(genre, ["Apply general mixing principles"])
    
    def _get_mastering_tips(self, genre: str) -> List[str]:
        """Get genre-specific mastering tips."""
        tips = {
            'rock': [
                "Target -10 to -8 LUFS for streaming",
                "Preserve drum impact and guitar presence",
                "Use multiband compression carefully",
                "Maintain energy and excitement"
            ],
            'jazz': [
                "Target -16 to -12 LUFS for dynamics",
                "Minimal limiting to preserve dynamics",
                "Subtle EQ for tonal balance",
                "Preserve natural performance feel"
            ],
            'classical': [
                "Target -18 to -14 LUFS for wide dynamics",
                "Avoid aggressive limiting",
                "Maintain natural frequency balance",
                "Preserve concert hall acoustics"
            ],
            'electronic': [
                "Target -8 to -6 LUFS for club systems",
                "Use precise frequency shaping",
                "Apply creative stereo enhancement",
                "Ensure translation across systems"
            ],
            'hip_hop': [
                "Target -10 to -8 LUFS with strong low-end",
                "Use heavy limiting for consistency",
                "Enhance sub-bass content",
                "Maintain vocal clarity"
            ],
            'folk': [
                "Target -16 to -12 LUFS for intimacy",
                "Gentle mastering to preserve character",
                "Warm tonal balance",
                "Maintain acoustic authenticity"
            ],
            'pop': [
                "Target -11 to -8 LUFS for radio",
                "Use competitive limiting",
                "Bright, polished sound",
                "Ensure broad format compatibility"
            ]
        }
        return tips.get(genre, ["Apply general mastering principles"])
