# VoiceAI Stem Splitter - RunPod Serverless Dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Demucs
RUN pip install --no-cache-dir demucs

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/demucs_output /tmp/demucs_temp

# Set environment variables
ENV PYTORCH_NO_CUDA_MEMORY_CACHING=1
ENV DEMUCS_MODEL=htdemucs
ENV MAX_SEGMENT=10
ENV PYTHONPATH=/app

# RunPod serverless entry point
CMD ["python", "runpod_handler.py"]
