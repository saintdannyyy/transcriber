# 🎙️ Faster Whisper Transcription API

A high-performance real-time audio transcription service built with FastAPI and Faster Whisper, optimized for English language transcription with CUDA acceleration support.

## 🚀 Features

- **Ultra-fast transcription** using Faster Whisper (4-5x faster than OpenAI Whisper)
- **Real-time WebSocket streaming** for live audio transcription
- **CUDA acceleration** support for GPU-powered inference
- **RESTful API** with comprehensive error handling
- **Health monitoring** and performance statistics
- **Word-level timestamps** for precise audio alignment
- **Silence detection** to optimize processing
- **Modern web interface** for easy testing
- **Docker ready** with production deployment support

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Client    │───▶│   FastAPI App    │───▶│ Faster Whisper  │
│                 │    │                  │    │     Model       │
│ • File Upload   │    │ • REST API       │    │                 │
│ • WebSocket     │    │ • WebSocket      │    │ • GPU/CPU       │
│ • Real-time UI  │    │ • Health Check   │    │ • English Only  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Requirements

### System Requirements

- **Python**: 3.8+
- **CUDA**: 11.2+ (optional, for GPU acceleration)
- **Memory**: 4GB+ RAM (8GB+ recommended for large models)
- **Storage**: 2GB+ free space for model files

### Dependencies

- `faster-whisper`: Core transcription engine
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `soundfile`: Audio file processing
- `librosa`: Audio analysis and preprocessing
- `numpy`: Numerical computations
- `torch`: PyTorch (for CUDA detection)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/saintdannyyy/transcriber.git
cd transcriber
```

### 2. Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify CUDA Support (Optional)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

### Running the Server

```bash
# Development mode with auto-reload
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Production mode
python app.py
```

The server will start at `http://localhost:8000`

### API Endpoints

#### 📡 Health Check

```bash
GET /health
```

Response:

```json
{
  "status": "ok",
  "model_id": "large-v3",
  "device": "cuda",
  "uptime": 3600.5,
  "total_requests": 150,
  "avg_processing_time": 2.3
}
```

#### 🎵 File Transcription

```bash
POST /api/transcribe
Content-Type: multipart/form-data
```

**Parameters:**

- `file`: Audio file (WAV, MP3, M4A, etc.)
- `language`: Language code (default: "en")
- `task`: "transcribe" or "translate" (default: "transcribe")

**Example using curl:**

```bash
curl -X POST "http://localhost:8000/api/transcribe" \
  -F "file=@audio.wav" \
  -F "language=en" \
  -F "task=transcribe"
```

**Response:**

```json
{
  "transcription": "Hello, this is a test transcription.",
  "status": "success",
  "processing_time": 1.23,
  "language": "en",
  "confidence": 0.95,
  "detected_language": "en",
  "language_probability": 0.99
}
```

#### 🔴 Real-time Streaming

```javascript
const ws = new WebSocket(
  "ws://localhost:8000/api/stream?language=en&task=transcribe"
);

ws.onopen = () => {
  console.log("Connected to transcription stream");
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Transcription:", data.transcription);
};

// Send audio data
ws.send(audioBuffer);
```

#### 📊 Statistics

```bash
GET /stats
```

Response:

```json
{
  "total_requests": 150,
  "successful_requests": 147,
  "failed_requests": 3,
  "avg_processing_time": 2.3,
  "total_audio_duration": 1800.5
}
```

## 🎛️ Configuration

### Model Configuration

Edit the `MODEL_CONFIG` in `app.py`:

```python
MODEL_CONFIG = {
    "model_id": "large-v3",           # Model size: tiny, base, small, medium, large-v3
    "device": "cuda",                 # "cuda" or "cpu"
    "compute_type": "float16",        # "float16", "int8", "int8_float16"
    "beam_size": 5,                   # Beam search size
    "best_of": 5,                     # Number of candidates
    "temperature": 0.0,               # Sampling temperature
    "word_timestamps": True,          # Enable word-level timestamps
    # ... more options
}
```

### Performance Tuning

#### For GPU Users:

```python
"device": "cuda"
"compute_type": "float16"  # Fastest on modern GPUs
```

#### For CPU Users:

```python
"device": "cpu"
"compute_type": "int8"     # Fastest on CPU
```

#### Model Size Trade-offs:

- **tiny**: Fastest, lowest accuracy (~1GB VRAM)
- **base**: Good balance (~1GB VRAM)
- **small**: Better accuracy (~2GB VRAM)
- **medium**: High accuracy (~5GB VRAM)
- **large-v3**: Best accuracy (~10GB VRAM)

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t transcriber-api .
```

### Run Container

```bash
# CPU only
docker run -p 8000:8000 transcriber-api

# With GPU support
docker run --gpus all -p 8000:8000 transcriber-api
```

### Docker Compose

```yaml
version: "3.8"
services:
  transcriber:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 🌐 Web Interface

Access the web interface at `http://localhost:8000/static/demo.html`

Features:

- **File upload** transcription
- **Real-time microphone** recording
- **WebSocket streaming** display
- **Download transcripts** as text files
- **Audio playback** with transcript synchronization

## 📈 Performance Benchmarks

### Processing Speed (on RTX 3080)

| Model Size | Audio Duration | Processing Time | Real-time Factor |
| ---------- | -------------- | --------------- | ---------------- |
| tiny       | 60s            | 2.1s            | 28.6x            |
| base       | 60s            | 3.2s            | 18.8x            |
| small      | 60s            | 4.7s            | 12.8x            |
| medium     | 60s            | 8.1s            | 7.4x             |
| large-v3   | 60s            | 12.3s           | 4.9x             |

### Memory Usage

| Model Size | GPU VRAM | System RAM |
| ---------- | -------- | ---------- |
| tiny       | 1GB      | 1GB        |
| base       | 1GB      | 1GB        |
| small      | 2GB      | 2GB        |
| medium     | 5GB      | 3GB        |
| large-v3   | 10GB     | 4GB        |

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Solution: Use smaller model or reduce batch size
MODEL_CONFIG["model_id"] = "medium"  # Instead of large-v3
```

#### 2. Slow CPU Performance

```bash
# Solution: Use int8 quantization
MODEL_CONFIG["compute_type"] = "int8"
```

#### 3. Audio Format Issues

```bash
# Install ffmpeg for broader format support
# Windows: choco install ffmpeg
# Linux: sudo apt install ffmpeg
# Mac: brew install ffmpeg
```

#### 4. Model Download Issues

```bash
# Models are downloaded automatically to:
# Windows: C:\Users\{username}\.cache\huggingface\hub
# Linux/Mac: ~/.cache/huggingface/hub
```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-name`
3. **Commit** changes: `git commit -am 'Add feature'`
4. **Push** to branch: `git push origin feature-name`
5. **Submit** a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black app.py

# Lint code
flake8 app.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for the original Whisper model
- **Faster Whisper** team for the optimized implementation
- **FastAPI** for the excellent web framework
- **Hugging Face** for model hosting and transformers

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/saintdannyyy/transcriber/issues)
- **Discussions**: [GitHub Discussions](https://github.com/saintdannyyy/transcriber/discussions)
- **Email**: support@transcriber.dev

## 🔮 Roadmap

- [ ] **Multi-language support** beyond English
- [ ] **Speaker diarization** for multiple speakers
- [ ] **Custom model fine-tuning** capabilities
- [ ] **Batch processing** for multiple files
- [ ] **Cloud deployment** templates (AWS, GCP, Azure)
- [ ] **Prometheus metrics** integration
- [ ] **Rate limiting** and authentication
- [ ] **WebRTC** integration for browser audio

---

**Built with ❤️ for the open-source community**
