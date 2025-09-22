#!/usr/bin/env python3
"""
RunPod Serverless Handler for VoiceAI Stem Splitter
AI-powered audio stem separation using Demucs
"""

import os
import sys
import json
import logging
import tempfile
import shutil
import uuid
import base64
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import runpod

# System monitoring imports
import torch
import torchaudio
import soundfile as sf
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
MODELS = {
    "htdemucs": {"memory_required": 7000, "segment_default": 7, "max_segment": 7.8},
    "htdemucs_ft": {"memory_required": 8000, "segment_default": 7, "max_segment": 7.8},
    "htdemucs_6s": {"memory_required": 9000, "segment_default": 7, "max_segment": 7.8},
    "mdx": {"memory_required": 4000, "segment_default": 15},
    "mdx_extra": {"memory_required": 4000, "segment_default": 15},
    "mdx_q": {"memory_required": 3000, "segment_default": 15},
    "mdx_extra_q": {"memory_required": 3000, "segment_default": 15},
}

def check_gpu_availability():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU available: {gpu_count} device(s), {gpu_memory:.1f}GB memory")
        return True
    else:
        logger.warning("No GPU available, using CPU")
        return False

def get_optimal_segment_size(model: str, gpu_available: bool) -> int:
    """Get optimal segment size based on model and hardware"""
    if not gpu_available:
        return 30  # Larger segments for CPU
    
    # Special handling for Hybrid Transformer models
    if model in ["htdemucs", "htdemucs_ft", "htdemucs_6s"]:
        return 7  # Maximum 7.8 seconds for Transformer models
    
    model_config = MODELS.get(model, MODELS["htdemucs"])
    return model_config["segment_default"]

def download_audio(url: str, temp_dir: str) -> str:
    """Download audio file from URL (synchronous)"""
    import urllib.request
    
    filename = f"input_{uuid.uuid4().hex}"
    filepath = os.path.join(temp_dir, filename)
    
    try:
        urllib.request.urlretrieve(url, filepath)
        logger.info(f"Downloaded audio to {filepath}")
        return filepath
    except Exception as e:
        raise Exception(f"Failed to download audio: {str(e)}")

def run_demucs_separation(
    input_path: str,
    output_dir: str,
    model: str,
    two_stems: Optional[str],
    segment: int,
    shifts: int,
    overlap: float,
    mp3_bitrate: int,
    float32: bool
) -> Dict[str, str]:
    """Run Demucs separation using subprocess command line"""
    
    logger.info(f"Running Demucs separation with model: {model}")
    
    try:
        # Build demucs command using subprocess
        cmd = ["python3", "-m", "demucs"]
        
        # Add output format options
        if float32:
            cmd.append("--float32")
        else:
            cmd.extend(["--mp3", "--mp3-bitrate", str(mp3_bitrate)])
        
        # Add two-stems option if specified
        if two_stems:
            cmd.extend(["--two-stems", two_stems])
        
        # Add model selection
        cmd.extend(["-n", model])
        
        # Add other options
        cmd.extend([
            "--out", output_dir,
            "--segment", str(segment),
            "--shifts", str(shifts),
            "--overlap", str(overlap),
        ])
        
        # Add input file at the end
        cmd.append(input_path)
        
        logger.info(f"Running demucs command: {' '.join(cmd)}")
        
        # Run demucs using subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Demucs failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            raise Exception(f"Demucs process failed: {result.stderr}")
        
        logger.info(f"Demucs completed successfully")
        logger.info(f"STDOUT: {result.stdout}")
        
        # Find output files in the separated directory
        separated_dir = os.path.join(output_dir, "separated", model)
        if not os.path.exists(separated_dir):
            raise Exception(f"Output directory not found: {separated_dir}")
        
        # Find the actual output directory (demucs creates subdirectories)
        actual_output_dir = None
        for item in os.listdir(separated_dir):
            item_path = os.path.join(separated_dir, item)
            if os.path.isdir(item_path):
                actual_output_dir = item_path
                break
        
        if not actual_output_dir:
            raise Exception(f"No output directory found in {separated_dir}")
        
        # Collect stem files
        stems = {}
        if two_stems:
            # Two-stem mode - look for the specific stem
            for ext in [".wav", ".mp3"]:
                stem_file = os.path.join(actual_output_dir, f"{two_stems}{ext}")
                if os.path.exists(stem_file):
                    stems[two_stems] = stem_file
                    break
        else:
            # Four-stem mode - look for all standard stems
            stem_names = ["drums", "bass", "other", "vocals"]
            for stem in stem_names:
                for ext in [".wav", ".mp3"]:
                    stem_file = os.path.join(actual_output_dir, f"{stem}{ext}")
                    if os.path.exists(stem_file):
                        stems[stem] = stem_file
                        break
        
        if not stems:
            raise Exception(f"No stem files found in {actual_output_dir}")
        
        logger.info(f"Generated stems: {list(stems.keys())}")
        return stems
        
    except subprocess.TimeoutExpired:
        logger.error("Demucs process timed out after 5 minutes")
        raise Exception("Demucs process timed out")
    except Exception as e:
        logger.error(f"Demucs separation failed: {str(e)}")
        raise

def encode_audio_to_base64(file_path: str) -> str:
    """Encode audio file to base64 string"""
    try:
        with open(file_path, 'rb') as f:
            audio_data = f.read()
            base64_data = base64.b64encode(audio_data).decode('utf-8')
            return base64_data
    except Exception as e:
        logger.error(f"Failed to encode {file_path}: {str(e)}")
        raise

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for audio stem separation
    
    Expected input format:
    {
        "input": {
            "audio_url": "https://example.com/audio.mp3",
            "model": "htdemucs",
            "two_stems": "vocals",  # Optional
            "segment": 10,  # Optional
            "shifts": 0,  # Optional
            "overlap": 0.25,  # Optional
            "mp3_bitrate": 320,  # Optional
            "float32": false  # Optional
        }
    }
    
    Returns:
    {
        "success": true,
        "stems": {
            "vocals": "base64_encoded_audio_data",
            "drums": "base64_encoded_audio_data",
            "bass": "base64_encoded_audio_data",
            "other": "base64_encoded_audio_data"
        },
        "processing_time": 45.2,
        "model_used": "htdemucs"
    }
    """
    start_time = time.time()
    
    try:
        # Extract input parameters
        input_data = event.get("input", {})
        
        # Validate required parameters
        if "audio_url" not in input_data:
            return {
                "error": "Missing required parameter: audio_url"
            }
        
        # Get parameters with defaults
        audio_url = input_data["audio_url"]
        model = input_data.get("model", "htdemucs")
        two_stems = input_data.get("two_stems")
        segment = input_data.get("segment")
        shifts = input_data.get("shifts", 0)
        overlap = input_data.get("overlap", 0.25)
        mp3_bitrate = input_data.get("mp3_bitrate", 320)
        float32 = input_data.get("float32", False)
        
        # Validate model
        if model not in MODELS:
            return {
                "error": f"Invalid model '{model}'. Available models: {list(MODELS.keys())}"
            }
        
        logger.info(f"Processing stem separation request: {model}")
        
        # Check GPU availability and set optimal parameters
        gpu_available = check_gpu_availability()
        if segment is None:
            segment = get_optimal_segment_size(model, gpu_available)
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Download audio file
            input_path = download_audio(audio_url, temp_dir)
            
            # Run Demucs separation
            stems = run_demucs_separation(
                input_path=input_path,
                output_dir=output_dir,
                model=model,
                two_stems=two_stems,
                segment=segment,
                shifts=shifts,
                overlap=overlap,
                mp3_bitrate=mp3_bitrate,
                float32=float32
            )
            
            # Encode stems to base64
            encoded_stems = {}
            for stem_name, stem_path in stems.items():
                try:
                    encoded_data = encode_audio_to_base64(stem_path)
                    encoded_stems[stem_name] = encoded_data
                    logger.info(f"Encoded {stem_name}: {len(encoded_data)} characters")
                except Exception as e:
                    logger.error(f"Failed to encode {stem_name}: {str(e)}")
                    return {
                        "error": f"Failed to encode {stem_name}: {str(e)}"
                    }
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return successful result
            result = {
                "success": True,
                "stems": encoded_stems,
                "processing_time": round(processing_time, 2),
                "model_used": model,
                "gpu_used": gpu_available,
                "segment_size": segment
            }
            
            logger.info(f"Separation completed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Separation failed: {str(e)}")
            return {
                "error": f"Separation failed: {str(e)}"
            }
        finally:
            # Cleanup temporary files
            try:
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary files")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp files: {str(e)}")
    
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {
            "error": f"Handler error: {str(e)}"
        }

# Register the handler with RunPod
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
