# Video to Text Converter

A powerful FastAPI web application that extracts and transcribes speech from video files using advanced speech recognition techniques.

## Features

- Upload video files (MP4, MOV, AVI) to extract speech as text
- Advanced audio preprocessing for better recognition:
  - Audio normalization and noise reduction
  - Silence removal and dynamic range compression
- Multiple speech recognition engines:
  - Offline recognition with Vosk
  - Online recognition with Google Speech API
  - Optional Wit.ai integration
- Chunked audio processing for better results with long videos
- Modern, responsive web interface with real-time status updates
- Automatic cleanup of temporary files

## Speech Recognition Details

This application uses a multi-tiered approach to speech recognition:

1. **Audio Preprocessing**: Each video's audio is extracted, normalized, filtered, and optimized for speech recognition
2. **Multi-Engine Recognition**: The system tries multiple recognition engines and selects the best result
3. **Chunked Processing**: Long videos are processed in smaller chunks for better accuracy
4. **Smart Fallback**: If one engine fails, the system automatically tries others

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-to-text.git
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
uvicorn main:app --reload
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

3. Upload a video file and click "Extract Text"
4. View the transcription results

## API Endpoints

- `GET /`: Web interface
- `POST /analyze/`: Video analysis endpoint
- `GET /status`: Status check endpoint

## Notes

- The Vosk model will be automatically downloaded on first run (~50MB)
- For optimal results, use videos with clear audio and minimal background noise 