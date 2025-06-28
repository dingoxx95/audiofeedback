#!/usr/bin/env python3
"""
Batch Audio Processing Script
Process multiple audio files in a directory and generate comprehensive reports.

This script has been updated to use the new modular audiofeedback_core package.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import the modular core package
try:
    from audiofeedback_core import AudioFeedbackApp, Config
except ImportError as e:
    print(f"Error importing audiofeedback_core: {e}")
    print("Make sure the audiofeedback_core package is properly installed.")
    sys.exit(1)


class BatchAudioProcessor:
    """Batch process multiple audio files using the new modular architecture"""
    
    def __init__(self, model_name: str = "gemma3:27b", max_workers: int = 2, 
                 enable_llm: bool = True, enable_visualizations: bool = True):
        """
        Initialize batch processor
        
        Args:
            model_name: LLM model to use
            max_workers: Maximum parallel processes (be careful with memory usage)
            enable_llm: Whether to enable LLM feedback generation
            enable_visualizations: Whether to enable visualization generation
        """
        self.model_name = model_name
        self.max_workers = max_workers
        self.enable_llm = enable_llm
        self.enable_visualizations = enable_visualizations
        self.supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
        
        # Initialize configuration
        try:
            self.config = Config()
            if model_name:
                self.config.DEFAULT_MODEL = model_name
        except Exception as e:
            print(f"Warning: Could not load configuration: {e}")
            self.config = None
        
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
    
    def process_single_file(self, file_path: Path, output_dir: Path, genre: str = None) -> Dict[str, Any]:
        """Process a single audio file using the new modular architecture"""
        try:
            print(f"Processing: {file_path.name}")
            start_time = time.time()
            
            # Create subdirectory for this file
            file_output_dir = output_dir / file_path.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize analyzer for this thread with current settings
            app = AudioFeedbackApp(
                config=self.config,
                enable_llm=self.enable_llm,
                enable_visualizations=self.enable_visualizations
            )
            
            # Process file
            result = app.analyze_file(
                str(file_path), 
                genre=genre,
                output_dir=str(file_output_dir)
            )
            
            # Add processing metadata
            result["processing_info"] = {
                "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                "model_used": self.model_name if self.enable_llm else "N/A",
                "processing_time": time.time() - start_time,
                "features_enabled": {
                    "llm_feedback": self.enable_llm,
                    "visualizations": self.enable_visualizations
                }
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
                "error": str(e),
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def process_batch(self, 
                     input_dir: str, 
                     output_dir: str, 
                     recursive: bool = True,
                     parallel: bool = False,
                     genre: str = None) -> Dict[str, Any]:
        """
        Process all audio files in a directory
        
        Args:
            input_dir: Directory containing audio files
            output_dir: Directory to save results
            recursive: Search subdirectories
            parallel: Process files in parallel (use with caution - memory intensive)
            genre: Force specific genre for all files
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
        print(f"LLM enabled: {self.enable_llm}")
        print(f"Visualizations enabled: {self.enable_visualizations}")
        print(f"Parallel processing: {parallel}")
        if genre:
            print(f"Genre forced to: {genre}")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process files
        results = []
        successful = 0
        errors = 0
        start_time = time.time()
        
        if parallel and self.max_workers > 1:
            # Parallel processing
            print(f"Processing with {self.max_workers} workers...")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all jobs
                future_to_file = {
                    executor.submit(self.process_single_file, file_path, output_path, genre): file_path
                    for file_path in audio_files
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result["status"] == "success":
                            successful += 1
                            print(f"✓ Completed: {file_path.name}")
                        else:
                            errors += 1
                            print(f"✗ Failed: {file_path.name} - {result['error']}")
                            
                    except Exception as exc:
                        errors += 1
                        print(f"✗ Exception processing {file_path.name}: {exc}")
                        results.append({
                            "file_path": str(file_path),
                            "status": "error",
                            "error": str(exc)
                        })
        else:
            # Sequential processing
            print("Processing files sequentially...")
            
            for i, file_path in enumerate(audio_files, 1):
                print(f"\nProgress: {i}/{len(audio_files)}")
                
                result = self.process_single_file(file_path, output_path, genre)
                results.append(result)
                
                if result["status"] == "success":
                    successful += 1
                    print(f"✓ Completed: {file_path.name}")
                else:
                    errors += 1
                    print(f"✗ Failed: {file_path.name} - {result['error']}")
        
        # Calculate summary statistics
        total_time = time.time() - start_time
        avg_time = total_time / len(audio_files) if audio_files else 0
        
        # Create batch summary
        batch_summary = {
            "input_directory": str(input_path),
            "output_directory": str(output_path),
            "total_files": len(audio_files),
            "processed": successful,
            "errors": errors,
            "processing_time": total_time,
            "average_time_per_file": avg_time,
            "parallel_processing": parallel,
            "max_workers": self.max_workers if parallel else 1,
            "features_enabled": {
                "llm_feedback": self.enable_llm,
                "visualizations": self.enable_visualizations
            },
            "model_used": self.model_name if self.enable_llm else "N/A",
            "forced_genre": genre
        }
        
        # Generate detailed report
        report = {
            "batch_summary": batch_summary,
            "individual_results": results
        }
        
        # Save batch report
        report_file = output_path / "batch_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create summary text file
        summary_file = output_path / "batch_summary.txt"
        self.create_text_summary(batch_summary, results, summary_file)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total files: {len(audio_files)}")
        print(f"Successfully processed: {successful}")
        print(f"Errors: {errors}")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Average per file: {avg_time:.1f} seconds")
        print(f"Report saved to: {report_file}")
        print(f"Summary saved to: {summary_file}")
        
        return report
    
    def create_text_summary(self, batch_summary: Dict[str, Any], results: List[Dict[str, Any]], 
                           output_file: Path) -> None:
        """Create a human-readable text summary of the batch processing"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("BATCH AUDIO ANALYSIS SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            # Batch info
            f.write("BATCH INFORMATION:\n")
            f.write(f"Input Directory: {batch_summary['input_directory']}\n")
            f.write(f"Output Directory: {batch_summary['output_directory']}\n")
            f.write(f"Total Files: {batch_summary['total_files']}\n")
            f.write(f"Successfully Processed: {batch_summary['processed']}\n")
            f.write(f"Errors: {batch_summary['errors']}\n")
            f.write(f"Processing Time: {batch_summary['processing_time']:.1f} seconds\n")
            f.write(f"Average Time per File: {batch_summary['average_time_per_file']:.1f} seconds\n")
            f.write(f"LLM Model Used: {batch_summary['model_used']}\n")
            f.write(f"Parallel Processing: {'Yes' if batch_summary['parallel_processing'] else 'No'}\n")
            
            if batch_summary.get('forced_genre'):
                f.write(f"Forced Genre: {batch_summary['forced_genre']}\n")
            
            f.write("\nFEATURES ENABLED:\n")
            features = batch_summary['features_enabled']
            f.write(f"LLM Feedback: {'Yes' if features['llm_feedback'] else 'No'}\n")
            f.write(f"Visualizations: {'Yes' if features['visualizations'] else 'No'}\n")
            
            # Successful files
            successful_results = [r for r in results if r['status'] == 'success']
            if successful_results:
                f.write(f"\nSUCCESSFULLY PROCESSED FILES ({len(successful_results)}):\n")
                f.write("-" * 40 + "\n")
                
                for result in successful_results:
                    file_name = Path(result['file_path']).name
                    file_result = result['result']
                    summary = file_result.get('summary', {})
                    
                    f.write(f"\nFile: {file_name}\n")
                    f.write(f"  Quality: {summary.get('overall_quality', 'Unknown')}\n")
                    f.write(f"  Genre: {summary.get('detected_genre', 'unknown').title()}")
                    f.write(f" ({summary.get('genre_confidence', 0):.1%} confidence)\n")
                    
                    if summary.get('primary_issues'):
                        f.write(f"  Issues: {', '.join(summary['primary_issues'])}\n")
                    
                    processing_info = result['result'].get('processing_info', {})
                    if processing_info:
                        f.write(f"  Processing Time: {processing_info.get('processing_time', 0):.1f}s\n")
                        f.write(f"  File Size: {processing_info.get('file_size_mb', 0):.1f} MB\n")
            
            # Failed files
            failed_results = [r for r in results if r['status'] == 'error']
            if failed_results:
                f.write(f"\nFAILED FILES ({len(failed_results)}):\n")
                f.write("-" * 40 + "\n")
                
                for result in failed_results:
                    file_name = Path(result['file_path']).name
                    f.write(f"File: {file_name}\n")
                    f.write(f"  Error: {result['error']}\n")
            
            f.write(f"\nReport generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def main():
    """Command-line interface for batch processing"""
    
    parser = argparse.ArgumentParser(
        description="Batch Audio Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input_folder output_folder                # Process all files
  %(prog)s input_folder output_folder --parallel     # Parallel processing
  %(prog)s input_folder output_folder --no-llm       # Skip AI feedback
  %(prog)s input_folder output_folder --genre rock   # Force rock genre
  %(prog)s input_folder output_folder --workers 4    # Use 4 parallel workers
        """
    )
    
    parser.add_argument('input_dir', help='Directory containing audio files')
    parser.add_argument('output_dir', help='Directory to save analysis results')
    parser.add_argument('--recursive', '-r', action='store_true', default=True,
                       help='Search subdirectories (default: True)')
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Process files in parallel')
    parser.add_argument('--workers', '-w', type=int, default=2,
                       help='Number of parallel workers (default: 2)')
    parser.add_argument('--model', '-m', default="gemma3:27b",
                       help='LLM model to use (default: gemma3:27b)')
    parser.add_argument('--genre', choices=['rock', 'jazz', 'classical', 'electronic', 
                                           'hip_hop', 'folk', 'pop'],
                       help='Force specific genre for all files')
    parser.add_argument('--no-llm', action='store_true',
                       help='Disable AI feedback generation')
    parser.add_argument('--no-visuals', action='store_true',
                       help='Disable visualization generation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    print("Batch Audio Analysis Tool")
    print("="*50)
    
    try:
        # Initialize processor
        processor = BatchAudioProcessor(
            model_name=args.model,
            max_workers=args.workers,
            enable_llm=not args.no_llm,
            enable_visualizations=not args.no_visuals
        )
        
        # Process batch
        result = processor.process_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            recursive=args.recursive,
            parallel=args.parallel,
            genre=args.genre
        )
        
        # Summary
        summary = result['batch_summary']
        success_rate = summary['processed'] / summary['total_files'] * 100 if summary['total_files'] > 0 else 0
        
        print(f"\nBatch processing completed with {success_rate:.1f}% success rate")
        
    except KeyboardInterrupt:
        print("\nBatch processing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during batch processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
