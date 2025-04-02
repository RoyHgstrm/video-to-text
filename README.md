# File Analyzer

A simple FastAPI web application that analyzes files and displays their metadata information.

## Current Features

- Upload various file types (Video, Audio, Images, Documents)
- View detailed file metadata (size, MIME type, creation/modification dates)
- Modern, responsive web interface
- Real-time processing status updates

## Speech Recognition Support

To enable speech recognition (extracting text from video/audio), the application requires:

1. **FFmpeg**: For extracting audio from video files
   - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - On Windows, add it to your PATH

2. **Whisper**: For speech-to-text transcription
   - Install with: `pip install openai-whisper`
   - May require additional dependencies like PyTorch

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd file-analyzer
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

## Usage

1. Start the application:
```bash
uvicorn main:app --reload
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

3. Upload a file and click "Analyze File"

4. View the file metadata information displayed after processing

## API Endpoints

- `GET /`: Web interface
- `POST /analyze/`: File analysis endpoint
- `GET /status`: Status check endpoint

## Notes

- The application focuses on providing file metadata
- Speech recognition requires additional components as mentioned above
- No external dependencies required for basic file analysis functionality 