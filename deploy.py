#!/usr/bin/env python3
"""
Deployment script for VoiceAI Stem Splitter on RunPod
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required tools are installed"""
    required_tools = ["docker", "git"]
    missing_tools = []
    
    for tool in required_tools:
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"❌ Missing required tools: {', '.join(missing_tools)}")
        print("Please install Docker and Git before proceeding.")
        return False
    
    print("✅ All required tools are installed")
    return True

def build_docker_image():
    """Build Docker image for RunPod deployment"""
    print("🐳 Building Docker image...")
    
    try:
        # Build the image
        result = subprocess.run([
            "docker", "build", 
            "-t", "voiceai-stem-splitter:latest",
            "."
        ], check=True, capture_output=True, text=True)
        
        print("✅ Docker image built successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker build failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def test_docker_image():
    """Test the Docker image locally"""
    print("🧪 Testing Docker image...")
    
    try:
        # Run a quick test
        result = subprocess.run([
            "docker", "run", 
            "--rm", 
            "-p", "8000:8000",
            "voiceai-stem-splitter:latest",
            "python", "-c", "import demucs; print('Demucs imported successfully')"
        ], check=True, capture_output=True, text=True, timeout=60)
        
        print("✅ Docker image test passed")
        print(f"Test output: {result.stdout}")
        return True
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"❌ Docker test failed: {e}")
        return False

def create_runpod_config():
    """Create RunPod configuration file"""
    config = {
        "name": "voiceai-stem-splitter",
        "description": "AI-powered audio stem separation using Demucs",
        "docker_image": "voiceai-stem-splitter:latest",
        "container_disk_in_gb": 50,
        "volume_in_gb": 10,
        "volume_mount_path": "/workspace",
        "ports": {
            "8000": "HTTP"
        },
        "env": {
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
            "DEMUCS_MODEL": "htdemucs",
            "MAX_SEGMENT": "10"
        },
        "startup_command": "python app.py",
        "gpu_types": [
            "RTX 3080",
            "RTX 3090", 
            "RTX 4080",
            "RTX 4090",
            "A5000",
            "A6000"
        ],
        "min_memory_in_gb": 8,
        "min_vcpu": 4
    }
    
    with open("runpod_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ RunPod configuration created: runpod_config.json")
    return config

def create_deployment_instructions():
    """Create deployment instructions"""
    instructions = """
# RunPod Deployment Instructions

## 1. Upload to RunPod

1. Go to [RunPod Console](https://console.runpod.io)
2. Create a new template
3. Upload this entire folder as a zip file
4. Use the following settings:

### Template Settings:
- **Name**: voiceai-stem-splitter
- **Docker Image**: voiceai-stem-splitter:latest
- **Container Disk**: 50 GB
- **Volume**: 10 GB
- **Port**: 8000 (HTTP)
- **Startup Command**: python app.py

### Environment Variables:
- PYTORCH_NO_CUDA_MEMORY_CACHING=1
- DEMUCS_MODEL=htdemucs
- MAX_SEGMENT=10

### GPU Requirements:
- Minimum: RTX 3080 (8GB VRAM)
- Recommended: RTX 4090 (24GB VRAM)
- Memory: 8GB+ system RAM

## 2. Test Deployment

Once deployed, test with:

```bash
curl -X GET https://your-runpod-url/health
```

Expected response:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "models_loaded": ["htdemucs", "mdx_extra", "mdx_q"],
  "active_jobs": 0
}
```

## 3. Usage Example

```bash
curl -X POST https://your-runpod-url/separate \\
  -H "Content-Type: application/json" \\
  -d '{
    "audio_url": "https://example.com/song.mp3",
    "model": "htdemucs",
    "two_stems": "vocals"
  }'
```

## 4. Monitor Usage

- Check logs in RunPod console
- Monitor GPU memory usage
- Track processing times
- Set up auto-scaling if needed

## 5. Integration with VoiceAI Studio

Update your main app to use this service:

```javascript
const response = await fetch('https://your-runpod-url/separate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    audio_url: uploadedFileUrl,
    model: 'htdemucs',
    two_stems: 'vocals'  // Optional: karaoke mode
  })
});
```

## Troubleshooting

### GPU Memory Issues:
- Reduce segment size: `"segment": 5`
- Use quantized model: `"model": "mdx_q"`
- Enable memory optimization: `PYTORCH_NO_CUDA_MEMORY_CACHING=1`

### Slow Processing:
- Use smaller models: `mdx_q` instead of `htdemucs`
- Reduce shifts: `"shifts": 0`
- Increase overlap: `"overlap": 0.1`

### File Size Issues:
- Compress input files
- Use streaming for large files
- Implement file size limits
"""
    
    with open("DEPLOYMENT.md", "w") as f:
        f.write(instructions)
    
    print("✅ Deployment instructions created: DEPLOYMENT.md")

def main():
    """Main deployment function"""
    print("🚀 VoiceAI Stem Splitter - RunPod Deployment")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Build Docker image
    if not build_docker_image():
        sys.exit(1)
    
    # Test Docker image
    if not test_docker_image():
        sys.exit(1)
    
    # Create RunPod configuration
    config = create_runpod_config()
    
    # Create deployment instructions
    create_deployment_instructions()
    
    print("\n🎉 Deployment preparation complete!")
    print("\nNext steps:")
    print("1. Upload this folder to RunPod")
    print("2. Follow instructions in DEPLOYMENT.md")
    print("3. Test the deployment")
    print("4. Update your main app to use the service")
    
    print(f"\n📋 Configuration saved to: runpod_config.json")
    print(f"📖 Instructions saved to: DEPLOYMENT.md")

if __name__ == "__main__":
    main()
