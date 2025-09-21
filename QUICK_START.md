# VoiceAI Stem Splitter - Quick Start Guide

## 🚀 Deploy to RunPod Serverless in 5 Minutes

### 1. Prepare Your Repository

```bash
# Clone or download this folder
cd voiceai-stem-splitter

# Make deployment script executable
chmod +x deploy.py

# Run deployment preparation
python deploy.py
```

### 2. Deploy to RunPod Serverless

1. **Go to [RunPod Console](https://console.runpod.io)**
2. **Create New Serverless Endpoint**
3. **Upload this folder** as a zip file
4. **Configure Endpoint:**
   - **Name**: `voiceai-stem-splitter`
   - **Handler**: `runpod_handler.py`
   - **Container Disk**: `50 GB`
   - **Environment Variables**:
     ```
     PYTORCH_NO_CUDA_MEMORY_CACHING=1
     DEMUCS_MODEL=htdemucs
     MAX_SEGMENT=10
     ```

5. **GPU Requirements:**
   - **Minimum**: RTX 3080 (8GB VRAM)
   - **Recommended**: RTX 4090 (24GB VRAM)
   - **System RAM**: 8GB+

### 3. Test Your Deployment

```bash
# Test with RunPod API
curl -X POST https://api.runpod.ai/v2/{endpoint_id}/run \
  -H "Authorization: Bearer {your_api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio_url": "https://example.com/song.mp3",
      "model": "htdemucs",
      "two_stems": "vocals"
    }
  }'

# Response:
{
  "id": "job-uuid-here",
  "status": "IN_QUEUE"
}

# Check status
curl -X GET https://api.runpod.ai/v2/{endpoint_id}/status/{job_id} \
  -H "Authorization: Bearer {your_api_key}"
```

### 4. Use the API

```bash
# Separate vocals from a song
curl -X POST https://api.runpod.ai/v2/{endpoint_id}/run \
  -H "Authorization: Bearer {your_api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio_url": "https://example.com/song.mp3",
      "model": "htdemucs",
      "two_stems": "vocals"
    }
  }'

# Response:
{
  "success": true,
  "stems": {
    "vocals": "base64_encoded_audio_data"
  },
  "processing_time": 45.2,
  "model_used": "htdemucs"
}
```

## 🎵 Available Models

| Model | Quality | Speed | GPU Memory | Use Case |
|-------|---------|-------|------------|----------|
| `htdemucs` | ⭐⭐⭐⭐⭐ | Medium | 7GB | Best quality |
| `htdemucs_ft` | ⭐⭐⭐⭐⭐ | Slow | 8GB | Maximum quality |
| `mdx_extra` | ⭐⭐⭐⭐ | Fast | 4GB | Good quality, faster |
| `mdx_q` | ⭐⭐⭐ | Fastest | 3GB | Quick processing |

## 🔧 Integration with VoiceAI Studio

Update your main app's environment variables:

```bash
# .env.local
NEXT_PUBLIC_RUNPOD_API_KEY=your-runpod-api-key
NEXT_PUBLIC_RUNPOD_ENDPOINT_ID=your-endpoint-id
```

## 💡 Pro Tips

### For Maximum Quality:
```json
{
  "model": "htdemucs_ft",
  "shifts": 5,
  "overlap": 0.25
}
```

### For Fast Processing:
```json
{
  "model": "mdx_q",
  "shifts": 0,
  "overlap": 0.1,
  "segment": 15
}
```

### For Karaoke Mode:
```json
{
  "model": "htdemucs",
  "two_stems": "vocals"
}
```

## 🐛 Troubleshooting

### GPU Memory Issues:
- Use `mdx_q` model
- Reduce `segment` to 5
- Enable memory optimization

### Slow Processing:
- Use smaller models
- Reduce `shifts` to 0
- Increase `overlap` to 0.1

### File Size Issues:
- Compress input files
- Use streaming for large files
- Implement file size limits

## 📊 Monitoring

Monitor your deployment:
- **GPU Usage**: RunPod console
- **Processing Time**: API responses
- **Error Rate**: Logs and health checks
- **Cost**: RunPod billing

## 🔗 Next Steps

1. **Deploy to RunPod** ✅
2. **Test the API** ✅
3. **Update VoiceAI Studio** to use the service
4. **Add UI components** for stem splitting
5. **Monitor usage** and optimize costs

## 📞 Support

- **GitHub Issues**: [VoiceAI Studio Repository]
- **RunPod Docs**: [RunPod Documentation]
- **Demucs Docs**: [Demucs GitHub](https://github.com/facebookresearch/demucs)
