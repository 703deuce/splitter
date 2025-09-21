# VoiceAI Stem Splitter - RunPod Serverless

AI-powered audio stem separation using [Demucs](https://github.com/facebookresearch/demucs) for RunPod serverless GPU deployment.

## Features

- 🎵 **4-Stem Separation**: Vocals, Drums, Bass, Other
- 🎤 **2-Stem Separation**: Vocals + Accompaniment (Karaoke mode)
- 🚀 **GPU Acceleration**: Optimized for RunPod serverless GPU
- 📁 **Multiple Formats**: WAV, MP3, FLAC, OGG support
- ⚡ **Serverless**: Pay-per-use GPU processing
- 🔄 **Auto-scaling**: Automatic scaling based on demand

## Models Available

- `htdemucs`: Default Hybrid Transformer model (best quality)
- `htdemucs_ft`: Fine-tuned version (4x slower, better quality)
- `mdx_extra`: MDX Challenge winning model
- `mdx_q`: Quantized model (faster, smaller memory)

## RunPod Serverless API

### Run Function
Start stem separation job

**Request:**
```json
{
  "input": {
    "audio_url": "https://example.com/audio.mp3",
    "model": "htdemucs",
    "two_stems": "vocals",
    "segment": 10,
    "shifts": 0,
    "overlap": 0.25,
    "mp3_bitrate": 320,
    "float32": false
  }
}
```

**Response:**
```json
{
  "success": true,
  "stems": {
    "vocals": "base64_encoded_audio_data",
    "drums": "base64_encoded_audio_data",
    "bass": "base64_encoded_audio_data",
    "other": "base64_encoded_audio_data"
  },
  "processing_time": 45.2,
  "model_used": "htdemucs",
  "gpu_used": true,
  "segment_size": 10
}
```

### Status Check
Check job status using RunPod API

```bash
GET https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}
```

## Deployment

### RunPod Serverless Setup

1. **Create RunPod Serverless Endpoint:**
   - Upload this folder as a zip file
   - Use PyTorch 2.0+ base image
   - Minimum 8GB GPU memory recommended
   - Set handler: `runpod_handler.py`

2. **Environment Variables:**
   ```bash
   PYTORCH_NO_CUDA_MEMORY_CACHING=1
   DEMUCS_MODEL=htdemucs
   MAX_SEGMENT=10
   ```

3. **Deploy:**
   ```bash
   # Upload this folder to RunPod serverless
   # Set handler: runpod_handler.py
   # Configure GPU requirements
   ```

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Test the handler locally
python runpod_handler.py

# Test with sample data
python -c "
import json
from runpod_handler import handler
result = handler({
    'input': {
        'audio_url': 'https://example.com/sample.mp3',
        'model': 'htdemucs'
    }
})
print(json.dumps(result, indent=2))
"
```

## Configuration

### models/config.yaml
```yaml
models:
  htdemucs:
    memory_required: 7000  # MB
    segment_default: 10
    description: "Default Hybrid Transformer model"
  
  htdemucs_ft:
    memory_required: 8000  # MB
    segment_default: 10
    description: "Fine-tuned model (4x slower)"
    
  mdx_extra:
    memory_required: 4000  # MB
    segment_default: 15
    description: "MDX Challenge winning model"
```

## Performance Optimization

### GPU Memory Management
- Automatic segment size adjustment based on available GPU memory
- Model quantization for lower memory usage
- Batch processing for multiple files

### Processing Speed
- Parallel processing with multiple workers
- Async processing with job queues
- Result caching for repeated requests

## Integration with VoiceAI Studio

The main VoiceAI Studio app connects to this service via:

```javascript
// Frontend integration
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

## Monitoring

- **Health Checks**: `/health` endpoint
- **Metrics**: Processing time, memory usage, queue length
- **Logging**: Structured logging for debugging
- **Error Handling**: Graceful degradation and retry logic

## Cost Optimization

- **Auto-scaling**: Scale down during low usage
- **Model Selection**: Use quantized models for faster processing
- **Batch Processing**: Process multiple files together
- **Caching**: Cache results to avoid reprocessing

## Support

For issues and questions:
- GitHub Issues: [VoiceAI Studio Repository]
- Documentation: [VoiceAI Studio Docs]
- RunPod Support: [RunPod Documentation]
