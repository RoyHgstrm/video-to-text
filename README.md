# Video to Text Converter

A powerful FastAPI web application that extracts and transcribes speech from video files using advanced speech recognition techniques. Built to work across any computing environment.

## Features

- Upload video files (MP4, MOV, AVI) to extract speech as text
- Advanced audio preprocessing for better recognition:
  - Audio normalization and noise reduction
  - Silence removal and dynamic range compression
  - High-pass filtering to reduce background noise
- Multiple speech recognition engines with smart fallback:
  - Offline recognition with Vosk (for privacy and reliability)
  - Online recognition with Google Speech API (for accuracy)
  - Optional Wit.ai integration (for challenging audio)
- Chunked audio processing for better results with long videos
- Modern, responsive web interface with real-time status updates
- Automatic cleanup of temporary files and videos after processing
- Docker support for easy deployment
- Universal compatibility with adaptive resource usage

## How It Works

This application employs a multi-layered approach to speech recognition:

1. **Video Processing**: Extracts audio track from uploaded video files
2. **Audio Preprocessing**: 
   - Converts to mono 16kHz PCM format (optimal for speech recognition)
   - Normalizes volume levels and applies noise reduction
   - Removes silent sections and balances audio levels
3. **Chunked Processing**: Breaks long audio into manageable segments with overlap
4. **Multi-Engine Recognition**: 
   - Tries multiple speech recognition engines
   - Selects the best result based on quality metrics
   - Falls back to alternative engines if primary fails
5. **Results Display**: Shows transcribed text with the engine used

## Universal Compatibility

The application is designed to work on any computing environment:

- **Adaptive Resource Usage**: Detects available system resources and adjusts accordingly
- **Graceful Degradation**: Falls back to simpler methods when resources are limited
- **Environment-Aware**: Adapts to different operating systems and architectures
- **Configurable**: Uses environment variables to customize behavior
- **Health Monitoring**: Includes `/health` endpoint for monitoring

## Installation

### Option 1: Standard Installation

1. Clone the repository:
```bash
git clone https://github.com/RoyHgstrm/video-to-text.git
cd video-to-text
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create `.env` file (optional):
```bash
cp .env.example .env
# Edit .env file to customize settings
```

### Option 2: Docker Installation

1. Clone the repository:
```bash
git clone https://github.com/RoyHgstrm/video-to-text.git
cd video-to-text
```

2. Create `.env` file for Docker setup (optional):
```bash
cp .env.example .env
# Edit .env file to customize settings
```

3. Use Docker Compose to build and start the application:
```bash
docker-compose up -d
```

This will:
- Build the Docker image with all necessary dependencies
- Create persistent volumes for uploads, models, and temporary files
- Start the application on the configured port (default: 8000)
- Configure the service to restart automatically

## Usage

### Running the Application

#### Standard Method:
```bash
# With default settings
uvicorn main:app --host 0.0.0.0 --port 8000

# With environment variables
export PORT=9000
export DEBUG=true
uvicorn main:app --host 0.0.0.0 --port $PORT
```

#### Docker Method:
```bash
# Start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

### Accessing the Application

Open your web browser and navigate to:
```
http://localhost:8000  # Local access
http://your-ip-address:8000  # Access from other devices on your network
```

### Using the Application

1. Upload a video file and click "Extract Text"
2. View the transcription results in real-time
3. Copy the text or view the transcription history

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Port to run the server on | `8000` |
| `HOST` | Host address to bind to | `0.0.0.0` |
| `PRODUCTION` | Enable production mode (affects security settings) | `false` |
| `DEBUG` | Enable debug logging | `false` |
| `MAX_UPLOAD_SIZE` | Maximum upload size in bytes | `104857600` (100MB) |
| `CORS_ORIGINS` | Comma-separated list of allowed origins for CORS | `*` |
| `WIT_AI_KEY` | API key for Wit.ai speech recognition | (none) |
| `UPLOAD_DIR` | Custom directory for uploaded files | `./uploads` |
| `TEMP_DIR` | Custom directory for temporary files | `./temp` |
| `MODELS_DIR` | Custom directory for speech recognition models | `./models` |
| `CONTAINER_MEMORY` | Memory limit for Docker container | `2G` |
| `CONTAINER_MEMORY_RESERVATION` | Memory reservation for Docker container | `512M` |

## Deployment Options

### Local Development

For local development, run with:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

Use the provided Docker configuration:

```bash
docker-compose up -d
```

### Cloud Deployment

#### Heroku

```bash
heroku create
git push heroku main
heroku config:set PRODUCTION=true
```

#### Railway/Render/Fly.io

Deploy directly from GitHub repository with the included Docker configuration.

## API Endpoints

- `GET /`: Web interface
- `POST /analyze/`: Video analysis and transcription endpoint
- `GET /status`: Status check endpoint for queued/processing files
- `GET /health`: Health check endpoint for monitoring

## Technical Details

- Built with FastAPI for high-performance async processing
- Uses Vosk for offline speech recognition
- Implements audio preprocessing with pydub and FFmpeg (if available)
- Supports chunked processing for long videos
- User-specific queues for multi-user support
- Automatic file cleanup to preserve disk space
- Docker support with optimized container configuration
- Adaptive resource detection to work on low-end devices

## Troubleshooting

### Common Issues

- **Missing FFmpeg**: The application will automatically fall back to pydub for audio processing
- **Low Memory**: On devices with limited RAM, Vosk model loading will be skipped
- **Missing Templates**: Application will automatically create required directories
- **Large Files**: Upload size is limited by `MAX_UPLOAD_SIZE` environment variable

### Error Logs

Logs are written to both console and `app.log` file for debugging.

## Notes

- The Vosk model (~50MB) will be automatically downloaded on first run
- For optimal results, use videos with clear audio and minimal background noise
- Processing time depends on video length and audio quality
- For very large files, consider using a machine with more RAM
- When using Docker, the uploads, models, and temp directories are mapped to your host machine

## License

MIT License 