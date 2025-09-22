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
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import runpod

# Demucs imports
import demucs.api
import torch
import torchaudio
import soundfile as sf
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
MODELS = {
    "htdemucs": {"memory_required": 7000, "segment_default": 10},
    "htdemucs_ft": {"memory_required": 8000, "segment_default": 10},
    "htdemucs_6s": {"memory_required": 9000, "segment_default": 10},
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
    """Run Demucs separation using the API"""
    
    logger.info(f"Initializing Demucs separator with model: {model}")
    
    try:
        # Initialize Demucs separator
        separator = demucs.api.Separator(
            model=model,
            segment=segment,
            shifts=shifts,
            overlap=overlap
        )
        
        logger.info(f"Separating audio file: {input_path}")
        
        # Separate the audio file
        origin, separated = separator.separate_audio_file(input_path)
        
        logger.info(f"Separation completed. Found stems: {list(separated.keys())}")
        
        # Save stems to output directory
        stems = {}
        os.makedirs(output_dir, exist_ok=True)
        
        for stem_name, stem_audio in separated.items():
            # Determine file extension
            if float32:
                file_ext = ".wav"
            else:
                file_ext = ".mp3"
            
            # Create output path
            output_path = os.path.join(output_dir, f"{stem_name}{file_ext}")
            
            # Save the audio
            demucs.api.save_audio(
                stem_audio, 
                output_path, 
                samplerate=separator.samplerate,
                as_float=float32
            )
            
            stems[stem_name] = output_path
            logger.info(f"Saved {stem_name} to {output_path}")
        
        # If two_stems mode is requested, filter to only that stem
        if two_stems and two_stems in stems:
            stems = {two_stems: stems[two_stems]}
        
        logger.info(f"Generated stems: {list(stems.keys())}")
        return stems
        
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
