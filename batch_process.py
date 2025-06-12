#!/usr/bin/env python3
"""
Batch Audio Processing Script
Process multiple audio files in a directory and generate comprehensive reports.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import our main analyzer (assuming it's in the same directory)
try:
    from audiofeedback import AudioFeedbackApp
except ImportError:
    # If running as standalone script
    import importlib.util
    spec = importlib.util.spec_from_file_location("audiofeedback", "audiofeedback.py")
    audiofeedback_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audiofeedback_module)
    AudioFeedbackApp = audiofeedback_module.AudioFeedbackApp


class BatchAudioProcessor:
    """Batch process multiple audio files"""
    
    def __init__(self, model_name: str = "gemma2:27b", max_workers: int = 2):
        """
        Initialize batch processor
        
        Args:
            model_name: LLM model to use
            max_workers: Maximum parallel processes (be careful with memory usage)
        """
        self.model_name = model_name
        self.max_workers = max_workers
        self.supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
        
    def find_audio_files(self, directory: str, recursive: bool = True) -> List[Path]:
        """Find all audio files in directory"""
        directory = Path(directory)
        audio_files = []
        
        if recursive:
            for ext in self.supported_formats:
                audio_files.extend(directory.rglob(f"*{ext}"))
                audio_files.extend(directory.rglob(f"*{ext.upper()}"))
        else:
            for ext in self.supported_formats:
                audio_files.extend(directory.glob(f"*{ext}"))
                audio_files.extend(directory.glob(f"*{ext.upper()}"))
        
        return sorted(audio_files)
    
    def process_single_file(self, file_path: Path, output_dir: Path, create_viz: bool = False) -> Dict[str, Any]:
        """Process a single audio file"""
        try:
            print(f"Processing: {file_path.name}")
            
            # Create subdirectory for this file
            file_output_dir = output_dir / file_path.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize analyzer for this thread
            app = AudioFeedbackApp()
            app.llm_generator.model_name = self.model_name
            
            # Process file
            result = app.process_audio_file(str(file_path), str(file_output_dir))
            
            # Create visualization if requested
            if create_viz and "analysis" in result:
                viz_path = file_output_dir / f"{file_path.stem}_visualization.png"
                app.create_visualization(result["analysis"], str(viz_path))
            
            # Add processing metadata
            result["processing_info"] = {
                "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                "model_used": self.model_name,
                "processing_time": time.time()  # Will be updated by caller
            }
            
            return {
                "file_path": str(file_path),
                "status": "success",
                "result": result
            }
            
        except Exception as e:
            return {
                "file_path": str(file_path),
                "status": "error",
                "error": str(e)
            }
    
    def process_batch(self, 
                     input_dir: str, 
                     output_dir: str, 
                     recursive: bool = True,
                     create_visualizations: bool = False,
                     parallel: bool = False) -> Dict[str, Any]:
        """
        Process all audio files in a directory
        
        Args:
            input_dir: Directory containing audio files
            output_dir: Directory to save results
            recursive: Search subdirectories
            create_visualizations: Create visualization charts
            parallel: Process files in parallel (use with caution - memory intensive)
        """
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")
        
        # Find all audio files
        audio_files = self.find_audio_files(input_path, recursive)
        
        if not audio_files:
            print(f"No audio files found in {input_path}")
            return {"processed": 0, "errors": 0, "results": []}
        
        print(f"Found {len(audio_files)} audio files to process")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process files
        results = []
        errors = []
        start_time = time.time()
        
        if parallel and len(audio_files) > 1:
            # Parallel processing
            print(f"Processing {len(audio_files)} files in parallel (max {self.max_workers} workers)")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self.process_single_file, file_path, output_path, create_visualizations): file_path
                    for file_path in audio_files
                }
                
                # Collect results
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result["status"] == "success":
                            results.append(result)
                        else:
                            errors.append(result)
                    except Exception as e:
                        errors.append({
                            "file_path": str(file_path),
                            "status": "error",
                            "error": f"Parallel processing error: {e}"
                        })
        else:
            # Sequential processing
            print(f"Processing {len(audio_files)} files sequentially")
            
            for i, file_path in enumerate(audio_files, 1):
                print(f"[{i}/{len(audio_files)}] Processing: {file_path.name}")
                
                file_start_time = time.time()
                result = self.process_single_file(file_path, output_path, create_visualizations)
                file_end_time = time.time()
                
                # Update processing time
                if result["status"] == "success":
                    result["result"]["processing_info"]["processing_time"] = file_end_time - file_start_time
                    results.append(result)
                else:
                    errors.append(result)
                
                # Print progress
                elapsed = file_end_time - file_start_time
                print(f"  Completed in {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        
        # Create summary report
        summary = {
            "batch_info": {
                "input_directory": str(input_path),
                "output_directory": str(output_path),
                "total_files_found": len(audio_files),
                "successfully_processed": len(results),
                "errors": len(errors),
                "total_processing_time": total_time,
                "model_used": self.model_name,
                "parallel_processing": parallel,
                "created_visualizations": create_visualizations
            },
            "successful_results": results,
            "errors": errors
        }
        
        # Save summary report
        summary_file = output_path / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create markdown summary
        self.create_markdown_summary(summary, output_path / "batch_summary.md")
        
        print(f"\nBatch processing complete!")
        print(f"Processed: {len(results)}/{len(audio_files)} files")
        print(f"Errors: {len(errors)}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average time per file: {total_time/len(audio_files):.1f}s")
        print(f"Results saved to: {output_path}")
        
        return summary
    
    def create_markdown_summary(self, summary: Dict[str, Any], output_file: Path):
        """Create a markdown summary report"""
        
        with open(output_file, 'w') as f:
            f.write("# Batch Audio Analysis Summary\n\n")
            
            # Overview
            batch_info = summary["batch_info"]
            f.write("## Overview\n\n")
            f.write(f"- **Input Directory**: {batch_info['input_directory']}\n")
            f.write(f"- **Output Directory**: {batch_info['output_directory']}\n")
            f.write(f"- **Total Files**: {batch_info['total_files_found']}\n")
            f.write(f"- **Successfully Processed**: {batch_info['successfully_processed']}\n")
            f.write(f"- **Errors**: {batch_info['errors']}\n")
            f.write(f"- **Total Processing Time**: {batch_info['total_processing_time']:.1f} seconds\n")
            f.write(f"- **Model Used**: {batch_info['model_used']}\n")
            f.write(f"- **Parallel Processing**: {'Yes' if batch_info['parallel_processing'] else 'No'}\n\n")
            
            # Successful results
            if summary["successful_results"]:
                f.write("## Successfully Processed Files\n\n")
                for result in summary["successful_results"]:
                    file_path = Path(result["file_path"])
                    f.write(f"### {file_path.name}\n\n")
                    
                    if "result" in result and "analysis" in result["result"]:
                        analysis = result["result"]["analysis"]
                        f.write(f"- **Duration**: {analysis['file_info']['duration_seconds']:.1f}s\n")
                        f.write(f"- **Sample Rate**: {analysis['file_info']['sample_rate']} Hz\n")
                        f.write(f"- **Channels**: {analysis['file_info']['channels']}\n")
                        f.write(f"- **RMS Level**: {analysis['dynamics']['rms_db']:.1f} dB\n")
                        f.write(f"- **Peak Level**: {analysis['dynamics']['peak_db']:.1f} dB\n")
                        f.write(f"- **Dynamic Range**: {analysis['dynamics']['dynamic_range_db']:.1f} dB\n")
                        
                        if "tempo" in analysis.get("temporal_features", {}):
                            f.write(f"- **Tempo**: {analysis['temporal_features']['tempo']:.1f} BPM\n")
                        
                        f.write(f"- **Analysis File**: `{file_path.stem}_analysis.json`\n")
                        f.write(f"- **Feedback File**: `{file_path.stem}_feedback.md`\n\n")
            
            # Errors
            if summary["errors"]:
                f.write("## Errors\n\n")
                for error in summary["errors"]:
                    file_path = Path(error["file_path"])
                    f.write(f"- **{file_path.name}**: {error['error']}\n")
                f.write("\n")
            
            # Statistics
            if summary["successful_results"]:
                f.write("## Statistics\n\n")
                
                # Calculate aggregate statistics
                durations = []
                rms_levels = []
                peak_levels = []
                dynamic_ranges = []
                tempos = []
                
                for result in summary["successful_results"]:
                    if "result" in result and "analysis" in result["result"]:
                        analysis = result["result"]["analysis"]
                        durations.append(analysis["file_info"]["duration_seconds"])
                        rms_levels.append(analysis["dynamics"]["rms_db"])
                        peak_levels.append(analysis["dynamics"]["peak_db"])
                        dynamic_ranges.append(analysis["dynamics"]["dynamic_range_db"])
                        
                        if "tempo" in analysis.get("temporal_features", {}):
                            tempo = analysis["temporal_features"]["tempo"]
                            if tempo > 0:  # Valid tempo detected
                                tempos.append(tempo)
                
                import statistics
                
                f.write(f"- **Average Duration**: {statistics.mean(durations):.1f}s\n")
                f.write(f"- **Average RMS Level**: {statistics.mean(rms_levels):.1f} dB\n")
                f.write(f"- **Average Peak Level**: {statistics.mean(peak_levels):.1f} dB\n")
                f.write(f"- **Average Dynamic Range**: {statistics.mean(dynamic_ranges):.1f} dB\n")
                
                if tempos:
                    f.write(f"- **Average Tempo**: {statistics.mean(tempos):.1f} BPM\n")
                
                # Loudness distribution
                f.write("\n### Loudness Distribution\n\n")
                loud_count = sum(1 for rms in rms_levels if rms > -14)
                medium_count = sum(1 for rms in rms_levels if -20 <= rms <= -14)
                quiet_count = sum(1 for rms in rms_levels if rms < -20)
                
                f.write(f"- **Loud (> -14 dB RMS)**: {loud_count} files\n")
                f.write(f"- **Medium (-20 to -14 dB RMS)**: {medium_count} files\n")
                f.write(f"- **Quiet (< -20 dB RMS)**: {quiet_count} files\n")


def main():
    parser = argparse.ArgumentParser(description='Batch Audio Analysis Tool')
    parser.add_argument('input_dir', help='Input directory containing audio files')
    parser.add_argument('-o', '--output', required=True, help='Output directory for results')
    parser.add_argument('-r', '--recursive', action='store_true', help='Search subdirectories recursively')
    parser.add_argument('-v', '--visualize', action='store_true', help='Create visualization charts')
    parser.add_argument('-p', '--parallel', action='store_true', help='Process files in parallel (memory intensive)')
    parser.add_argument('-w', '--workers', type=int, default=2, help='Number of parallel workers (default: 2)')
    parser.add_argument('-m', '--model', default='gemma2:27b', help='LLM model name (default: gemma2:27b)')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not Path(args.input_dir).exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1
    
    # Warning for parallel processing
    if args.parallel:
        print("WARNING: Parallel processing is memory intensive!")
        print(f"Using {args.workers} workers with model {args.model}")
        if args.model == "gemma2:27b":
            print("Consider using a smaller model (e.g., gemma2:9b) for parallel processing")
        
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0
    
    # Initialize processor
    processor = BatchAudioProcessor(
        model_name=args.model,
        max_workers=args.workers
    )
    
    try:
        # Process batch
        summary = processor.process_batch(
            input_dir=args.input_dir,
            output_dir=args.output,
            recursive=args.recursive,
            create_visualizations=args.visualize,
            parallel=args.parallel
        )
        
        # Print final summary
        print(f"\n{'='*50}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"Total files processed: {summary['batch_info']['successfully_processed']}")
        print(f"Errors: {summary['batch_info']['errors']}")
        print(f"Total time: {summary['batch_info']['total_processing_time']:.1f}s")
        print(f"Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error during batch processing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())