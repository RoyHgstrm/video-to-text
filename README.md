# Video to Text Converter

A powerful FastAPI web application that extracts and transcribes speech from video files using advanced speech recognition techniques.

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

## Installation

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

4. Optional: For Wit.ai integration, set environment variable:
```bash
# On Windows PowerShell
$env:WIT_AI_KEY="your-wit-ai-key"

# On Windows Command Prompt
set WIT_AI_KEY=your-wit-ai-key

# On Linux/Mac
export WIT_AI_KEY=your-wit-ai-key
```

## Usage

1. Start the application:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

2. Open your web browser and navigate to:
```
http://localhost:8000  # Local access
http://your-ip-address:8000  # Access from other devices on your network
```

3. Upload a video file and click "Extract Text"
4. View the transcription results in real-time
5. Copy the text or view the transcription history

## API Endpoints

- `GET /`: Web interface
- `POST /analyze/`: Video analysis and transcription endpoint
- `GET /status`: Status check endpoint for queued/processing files

## Technical Details

- Built with FastAPI for high-performance async processing
- Uses Vosk for offline speech recognition
- Implements audio preprocessing with pydub and FFmpeg
- Supports chunked processing for long videos
- User-specific queues for multi-user support
- Automatic file cleanup to preserve disk space

## Notes

- The Vosk model (~50MB) will be automatically downloaded on first run
- For optimal results, use videos with clear audio and minimal background noise
- Processing time depends on video length and audio quality
- For very large files, consider using a machine with more RAM

## License

MIT License 