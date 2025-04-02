import os
import sys
import logging
import uuid
import shutil
from datetime import datetime
import threading
import queue
from pathlib import Path
import mimetypes
import subprocess
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
import torch
import torchaudio
from vosk import Model, KaldiRecognizer
import json
import wave
import urllib.request
import zipfile
import platform

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Environment configuration with defaults
PORT = int(os.environ.get("PORT", 8000))
HOST = os.environ.get("HOST", "0.0.0.0")
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", 100 * 1024 * 1024))  # 100MB default
PRODUCTION = os.environ.get("PRODUCTION", "false").lower() == "true"

# Configure logging
logging_level = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(
    level=logging_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("video_to_text")

# Platform detection
SYSTEM_INFO = {
    "os": platform.system(),
    "version": platform.version(),
    "python": platform.python_version(),
    "architecture": platform.machine()
}
logger.info(f"Running on: {SYSTEM_INFO}")

# Create the FastAPI app
app = FastAPI(
    title="Video to Text",
    description="Video transcription service with multiple speech recognition engines",
    version="1.0.0"
)

# Add CORS middleware with configurable origins
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))
TEMP_DIR = os.environ.get("TEMP_DIR", os.path.join(BASE_DIR, "temp"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))

# Ensure directories exist
for directory in [UPLOAD_DIR, TEMP_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Directory: {directory}")

# Set up templates
templates_dir = os.path.join(BASE_DIR, "templates")
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Serve static files if available
static_dir = os.path.join(BASE_DIR, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global processing queues and status
processing_queue = queue.PriorityQueue()
processing_status = {}
user_queues = {}

# Initialize speech recognizers
recognizer = sr.Recognizer()
# Increase the energy threshold for better noise handling
recognizer.energy_threshold = 300
# Increase dynamic energy threshold for varying audio levels
recognizer.dynamic_energy_threshold = True
# Adjust pause threshold to detect longer pauses in speech
recognizer.pause_threshold = 0.8
# Increase phrase threshold for better phrase detection
recognizer.phrase_threshold = 0.3
# Increase non-speaking duration for better silence detection
recognizer.non_speaking_duration = 0.5

vosk_model = None

def has_command(cmd):
    """Check if a command exists in the system path"""
    try:
        result = subprocess.run(
            ["which" if SYSTEM_INFO["os"] != "Windows" else "where", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=(SYSTEM_INFO["os"] == "Windows")
        )
        return result.returncode == 0
    except Exception:
        return False

# Check for FFmpeg
FFMPEG_AVAILABLE = has_command("ffmpeg")
logger.info(f"FFmpeg available: {FFMPEG_AVAILABLE}")

def download_vosk_model():
    """Download the Vosk model if it doesn't exist"""
    # Choose appropriate model based on system architecture
    is_small_device = SYSTEM_INFO["architecture"] in ["armv7l", "armv6l"]
    
    # Use a smaller model for ARM devices
    model_name = "vosk-model-small-en-us-0.15" if not is_small_device else "vosk-model-en-us-0.22-lgraph"
    model_url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
    model_path = os.path.join(MODELS_DIR, model_name)
    zip_path = os.path.join(MODELS_DIR, f"{model_name}.zip")
    
    if not os.path.exists(model_path):
        logger.info(f"Downloading Vosk model: {model_name}")
        try:
            # Create a directory for the model
            os.makedirs(model_path, exist_ok=True)
            
            # Check if we have enough disk space (rough estimate: need 500MB free for download and extraction)
            try:
                if SYSTEM_INFO["os"] != "Windows":
                    import shutil
                    total, used, free = shutil.disk_usage(MODELS_DIR)
                    if free < 500 * 1024 * 1024:  # 500MB in bytes
                        logger.warning(f"Low disk space: {free // (1024 * 1024)}MB available. Model download may fail.")
            except Exception as e:
                logger.warning(f"Could not check disk space: {e}")
            
            # Download with progress logging
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if downloaded % (5 * 1024 * 1024) == 0:  # Log every 5MB
                    percent = 100 * downloaded / total_size if total_size > 0 else 0
                    logger.info(f"Download progress: {downloaded // (1024 * 1024)}MB / "
                                f"{total_size // (1024 * 1024)}MB ({percent:.1f}%)")
            
            logger.info(f"Downloading from {model_url}")
            urllib.request.urlretrieve(model_url, zip_path, reporthook=report_progress)
            
            # Extract the model
            logger.info("Extracting Vosk model...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(MODELS_DIR)
            
            # Clean up the zip file
            os.remove(zip_path)
            
            logger.info("Vosk model downloaded and extracted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to download Vosk model: {e}")
            # Clean up partial downloads
            try:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                if os.path.exists(model_path):
                    shutil.rmtree(model_path, ignore_errors=True)
            except Exception:
                pass
            return False
    return True

def load_vosk_model():
    """Load the Vosk model with fallback"""
    global vosk_model
    
    # Choose appropriate model based on system architecture
    is_small_device = SYSTEM_INFO["architecture"] in ["armv7l", "armv6l"]
    model_name = "vosk-model-small-en-us-0.15" if not is_small_device else "vosk-model-en-us-0.22-lgraph"
    model_path = os.path.join(MODELS_DIR, model_name)
    
    if not os.path.exists(model_path):
        if not download_vosk_model():
            logger.warning("Vosk model not available, will use Google Speech Recognition only")
            return
    
    try:
        # Make sure we have enough RAM to load the model
        # The small model needs about 500MB RAM
        import psutil
        available_ram = psutil.virtual_memory().available
        if available_ram < 500 * 1024 * 1024:  # 500MB in bytes
            logger.warning(f"Low available RAM ({available_ram // (1024 * 1024)}MB). "
                          "Skipping Vosk model loading to avoid memory issues.")
            return
    except ImportError:
        # psutil not available, proceed anyway
        pass
    
    try:
        vosk_model = Model(model_path)
        logger.info("Vosk model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Vosk model: {e}")
        logger.warning("Will use Google Speech Recognition only")

def get_user_id(request: Request):
    """Get or create a user ID from the request and ensure it exists in user_queues."""
    user_id = request.cookies.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
    if user_id not in user_queues:
        user_queues[user_id] = {
            'queue': queue.PriorityQueue(),
            'current_processing': None,
            'last_processed': None
        }
    return user_id

def set_user_cookie(response: Response, user_id: str):
    """Set the user ID cookie"""
    response.set_cookie(
        key="user_id",
        value=user_id,
        httponly=True,
        secure=PRODUCTION,
        samesite="lax",
        max_age=60 * 60 * 24 * 30  # 30 days
    )

def convert_to_mono_pcm_wav(audio_path):
    """Convert audio to mono PCM WAV format for Vosk"""
    try:
        output_path = os.path.join(TEMP_DIR, f"mono_{os.path.basename(audio_path)}")
        
        # Use FFmpeg for better audio conversion if available
        if FFMPEG_AVAILABLE:
            try:
                subprocess.run([
                    'ffmpeg', '-y', 
                    '-i', audio_path, 
                    '-acodec', 'pcm_s16le', 
                    '-ac', '1', 
                    '-ar', '16000', 
                    output_path
                ], check=True, capture_output=True)
                logger.info(f"Converted audio to mono PCM using FFmpeg: {output_path}")
                return output_path
            except subprocess.SubprocessError as e:
                logger.warning(f"FFmpeg error: {e}. Falling back to pydub.")
        else:
            logger.info("FFmpeg not available, using pydub instead")
        
        # Fallback to pydub if FFmpeg command-line is not available
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(16000)  # 16kHz
        audio = audio.set_sample_width(2)  # 16-bit
        
        audio.export(output_path, format="wav")
        logger.info(f"Converted audio to mono PCM using pydub: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error converting audio to mono PCM: {e}")
        return audio_path

def preprocess_audio(audio_path):
    """Preprocess audio file for better recognition"""
    try:
        # First, convert to mono PCM WAV
        mono_path = convert_to_mono_pcm_wav(audio_path)
        
        # Load audio file
        audio = AudioSegment.from_file(mono_path)
        
        # Normalize audio (adjust volume)
        try:
            audio = audio.normalize()
        except Exception as e:
            logger.warning(f"Audio normalization failed: {e}")
        
        # Apply noise reduction
        # This is a simple high-pass filter to reduce background noise
        try:
            audio = audio.high_pass_filter(80)
        except Exception as e:
            logger.warning(f"High-pass filter failed: {e}")
        
        # Apply a slight compression to even out volume
        try:
            audio = audio.compress_dynamic_range(threshold=-20, ratio=4.0, attack=5.0, release=50.0)
        except Exception as e:
            logger.warning(f"Audio compression failed: {e}")
        
        # Remove silence
        try:
            audio = audio.strip_silence(silence_len=500, silence_thresh=-35, padding=100)
        except Exception as e:
            logger.warning(f"Silence removal failed: {e}")
        
        # Save processed audio
        processed_path = os.path.join(TEMP_DIR, f"processed_{os.path.basename(audio_path)}")
        try:
            audio.export(processed_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            logger.info(f"Audio preprocessing completed: {processed_path}")
            return processed_path
        except Exception as e:
            logger.error(f"Error exporting processed audio: {e}")
            # If exporting with parameters fails, try without parameters
            try:
                audio.export(processed_path, format="wav")
                logger.info(f"Audio preprocessing completed (without params): {processed_path}")
                return processed_path
            except Exception as e2:
                logger.error(f"Error exporting processed audio without parameters: {e2}")
                return mono_path
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        # If preprocessing fails, return the mono path or original path
        return mono_path if 'mono_path' in locals() else audio_path

def transcribe_with_vosk(audio_path):
    """Transcribe audio using Vosk"""
    if not vosk_model:
        return None
        
    try:
        # Ensure audio is in correct format
        mono_pcm_path = convert_to_mono_pcm_wav(audio_path)
        
        with wave.open(mono_pcm_path, "rb") as wf:
            # Check if the file is in the correct format
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                logger.warning("Audio file must be WAV format mono PCM, attempting to convert")
                mono_pcm_path = convert_to_mono_pcm_wav(audio_path)
                wf = wave.open(mono_pcm_path, "rb")
            
            rec = KaldiRecognizer(vosk_model, wf.getframerate())
            rec.SetWords(True)
            
            results = []
            # Process in smaller chunks for better results
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    results.append(json.loads(rec.Result()))
            
            results.append(json.loads(rec.FinalResult()))
            
            # Extract text with confidence and combine
            text_parts = []
            
            for result in results:
                if "text" in result and result["text"]:
                    text_parts.append(result["text"])
            
            # Join the text parts with proper spacing
            text = " ".join(text_parts)
            
            # Clean up if needed
            if os.path.exists(mono_pcm_path) and mono_pcm_path != audio_path:
                os.unlink(mono_pcm_path)
                
            return text.strip()
    except Exception as e:
        logger.error(f"Error in Vosk transcription: {e}")
        return None

def transcribe_with_speech_recognition(audio_path):
    """Transcribe audio using SpeechRecognition with Google"""
    try:
        # Ensure audio is in correct format
        mono_pcm_path = convert_to_mono_pcm_wav(audio_path)
        
        with sr.AudioFile(mono_pcm_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            # Record audio in chunks for better processing
            audio_data = recognizer.record(source)
            
            # Use Google's speech recognition
            text = recognizer.recognize_google(audio_data, language="en-US", show_all=False)
            
            # Clean up
            if os.path.exists(mono_pcm_path) and mono_pcm_path != audio_path:
                os.unlink(mono_pcm_path)
                
            return text
    except sr.UnknownValueError:
        logger.warning("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        logger.error(f"Google Speech Recognition service error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error in Speech Recognition transcription: {e}")
        return None

def transcribe_with_wit(audio_path, api_key=None):
    """Transcribe audio using Wit.ai (if API key is provided)"""
    if not api_key:
        # Try to get from environment
        api_key = os.environ.get('WIT_AI_KEY')
        
    if not api_key:
        logger.warning("No Wit.ai API key provided")
        return None
        
    try:
        # Ensure audio is in correct format
        mono_pcm_path = convert_to_mono_pcm_wav(audio_path)
        
        with sr.AudioFile(mono_pcm_path) as source:
            audio_data = recognizer.record(source)
            
            # Use Wit.ai for recognition
            text = recognizer.recognize_wit(audio_data, key=api_key)
            
            # Clean up
            if os.path.exists(mono_pcm_path) and mono_pcm_path != audio_path:
                os.unlink(mono_pcm_path)
                
            return text
    except Exception as e:
        logger.error(f"Error in Wit.ai transcription: {e}")
        return None

def process_audio_in_chunks(audio_path, transcribe_func):
    """Process long audio files in chunks for better results"""
    try:
        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        
        # If audio is short (less than 30 seconds), just process it directly
        if len(audio) < 30000:
            return transcribe_func(audio_path)
        
        # Otherwise split into 30-second chunks with 500ms overlap
        chunk_length = 30000  # 30 seconds
        overlap = 500  # 0.5 seconds overlap
        
        chunks = []
        transcript_parts = []
        
        # Create chunks
        for i in range(0, len(audio), chunk_length - overlap):
            chunk = audio[i:i + chunk_length]
            if len(chunk) < 5000:  # Skip very short chunks (less than 5 seconds)
                continue
                
            chunk_path = os.path.join(TEMP_DIR, f"chunk_{i}_{uuid.uuid4()}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
            
        # Process each chunk
        for i, chunk_path in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_transcript = transcribe_func(chunk_path)
            if chunk_transcript:
                transcript_parts.append(chunk_transcript)
            
            # Clean up chunk file
            try:
                os.unlink(chunk_path)
            except Exception as e:
                logger.warning(f"Failed to delete chunk file: {e}")
            
        # Combine all transcripts
        return " ".join(transcript_parts)
    except Exception as e:
        logger.error(f"Error processing audio in chunks: {e}")
        return transcribe_func(audio_path)  # Fall back to processing the whole file

def safe_delete_file(file_path):
    """Safely delete a file with error handling"""
    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
            return True
        except Exception as e:
            logger.warning(f"Failed to delete file {file_path}: {e}")
            return False
    return False

def process_queue():
    while True:
        try:
            timestamp, file_id = processing_queue.get()
            
            if file_id not in processing_status:
                logger.warning(f"File {file_id} not found in processing_status")
                processing_queue.task_done()
                continue

            status_data = processing_status[file_id]
            file_path = status_data.get("file_path")
            user_id = status_data.get("user_id")
            
            logger.info(f"Processing file {file_id} for user {user_id} from path: {file_path}")
            
            if not file_path or not os.path.isabs(file_path):
                file_path = os.path.abspath(os.path.join(UPLOAD_DIR, file_id))
                processing_status[file_id]["file_path"] = file_path
                logger.info(f"Updated to absolute path: {file_path}")
            
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                processing_status[file_id]["status"] = "error"
                processing_status[file_id]["error"] = "File not found"
                processing_queue.task_done()
                continue

            try:
                processing_status[file_id]["status"] = "processing"
                processing_status[file_id]["current_step"] = "Extracting audio"
                processing_status[file_id]["started_at"] = datetime.now().isoformat()
                
                audio_file_name = f"{uuid.uuid4()}.wav"
                audio_path = os.path.join(TEMP_DIR, audio_file_name)
                processed_audio_path = None
                
                # Extract audio from video
                try:
                    video = VideoFileClip(file_path)
                    video.audio.write_audiofile(audio_path)
                    video.close()
                    logger.info(f"Audio extracted to {audio_path}")
                except Exception as e:
                    logger.error(f"Error extracting audio: {e}")
                    raise Exception(f"Failed to extract audio: {str(e)}")
                
                # Preprocess audio
                processing_status[file_id]["current_step"] = "Preprocessing audio"
                processed_audio_path = preprocess_audio(audio_path)
                
                # Try multiple recognition engines
                processing_status[file_id]["current_step"] = "Converting speech to text"
                transcripts = []
                
                # Try Vosk first if available
                if vosk_model:
                    processing_status[file_id]["current_step"] = "Transcribing with Vosk"
                    vosk_text = process_audio_in_chunks(processed_audio_path, transcribe_with_vosk)
                    if vosk_text:
                        transcripts.append(("Vosk", vosk_text))
                
                # Try Google Speech Recognition
                if not transcripts or len(transcripts[0][1]) < 10:  # If Vosk failed or returned very little text
                    processing_status[file_id]["current_step"] = "Transcribing with Google"
                    google_text = process_audio_in_chunks(processed_audio_path, transcribe_with_speech_recognition)
                    if google_text:
                        transcripts.append(("Google", google_text))
                
                # Try Wit.ai if available and other methods didn't yield good results
                wit_api_key = os.environ.get('WIT_AI_KEY')
                if wit_api_key and (not transcripts or len(transcripts[0][1]) < 20):
                    processing_status[file_id]["current_step"] = "Transcribing with Wit.ai"
                    wit_text = process_audio_in_chunks(processed_audio_path, 
                                                       lambda path: transcribe_with_wit(path, wit_api_key))
                    if wit_text:
                        transcripts.append(("Wit.ai", wit_text))
                
                # Select best transcript
                if transcripts:
                    # Sort by length - longer transcripts are usually better
                    transcripts.sort(key=lambda x: len(x[1]), reverse=True)
                    selected_engine, final_text = transcripts[0]
                    
                    # Post-process the transcript
                    final_text = final_text.strip()
                    
                    processing_status[file_id]["transcript"] = final_text
                    processing_status[file_id]["recognition_engine"] = selected_engine
                else:
                    processing_status[file_id]["transcript"] = "Could not transcribe audio"
                    processing_status[file_id]["recognition_engine"] = "None"
                
                processing_status[file_id]["status"] = "completed"
                processing_status[file_id]["current_step"] = "Transcription completed"
                processing_status[file_id]["completed_at"] = datetime.now().isoformat()
                
                if user_id in user_queues:
                    user_queues[user_id]['current_processing'] = None
                    user_queues[user_id]['last_processed'] = file_id
                
                logger.info(f"File {file_id} processed successfully")
                
            except Exception as e:
                logger.error(f"Error processing file {file_id}: {e}")
                processing_status[file_id]["status"] = "error"
                processing_status[file_id]["error"] = str(e)
                if user_id in user_queues:
                    user_queues[user_id]['current_processing'] = None
            finally:
                # Clean up files
                try:
                    safe_delete_file(audio_path)
                    if processed_audio_path and processed_audio_path != audio_path:
                        safe_delete_file(processed_audio_path)
                    safe_delete_file(file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary files: {e}")
                processing_queue.task_done()
                
        except Exception as e:
            logger.error(f"Error in processing queue: {e}")
            try:
                processing_queue.task_done()
            except:
                pass

# Add graceful shutdown handler
def cleanup_on_shutdown():
    """Clean up temporary files on application shutdown"""
    try:
        for directory in [TEMP_DIR, UPLOAD_DIR]:
            for file in os.listdir(directory):
                try:
                    path = os.path.join(directory, file)
                    if os.path.isfile(path):
                        os.unlink(path)
                except Exception as e:
                    logger.warning(f"Error deleting file during shutdown: {e}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Application startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting application...")
    
    # Try to load Vosk model conditionally
    try:
        # Only attempt to load Vosk if running on a system with enough resources
        if SYSTEM_INFO["os"] != "Windows" or DEBUG:
            # Optional: import psutil only if it's available
            try:
                import psutil
                ram_gb = psutil.virtual_memory().total / (1024**3)
                if ram_gb < 1.0:  # Skip Vosk on systems with less than 1GB RAM
                    logger.warning(f"System has only {ram_gb:.1f}GB RAM. Skipping Vosk model loading.")
                else:
                    load_vosk_model()
            except ImportError:
                # psutil not available, load model anyway
                load_vosk_model()
        else:
            load_vosk_model()
    except Exception as e:
        logger.error(f"Error during Vosk model initialization: {e}")
    
    # Start the processing thread
    threading.Thread(target=process_queue, daemon=True).start()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down application...")
    cleanup_on_shutdown()

# Routes
@app.get("/")
async def read_root(request: Request):
    try:
        response = templates.TemplateResponse("index.html", {"request": request})
        user_id = get_user_id(request)
        set_user_cookie(response, user_id)
        return response
    except Exception as e:
        logger.error(f"Error rendering index page: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": SYSTEM_INFO,
        "vosk_available": vosk_model is not None,
        "ffmpeg_available": FFMPEG_AVAILABLE
    }

@app.post("/analyze/")
async def analyze_file(file: UploadFile = File(...), request: Request = None):
    try:
        user_id = get_user_id(request)
        
        filename = file.filename
        if not filename:
            raise HTTPException(status_code=400, detail="Missing filename")
            
        file_extension = os.path.splitext(filename)[1].lower()
        
        allowed_extensions = [".mp4", ".mov", ".avi", ".webm", ".mp3", ".wav", ".m4a", ".ogg"]
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Supported formats: {', '.join(allowed_extensions)}"
            )
        
        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}{file_extension}"
        file_path = os.path.abspath(os.path.join(UPLOAD_DIR, safe_filename))
        
        timestamp = datetime.now().isoformat()
        
        # Check file size before reading
        if request.headers.get("content-length"):
            content_length = int(request.headers.get("content-length"))
            if content_length > MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)}MB"
                )
        
        # Read content in chunks to avoid memory issues
        with open(file_path, "wb") as buffer:
            # Read in 1MB chunks
            chunk_size = 1024 * 1024
            bytes_read = 0
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                bytes_read += len(chunk)
                # Check size during read
                if bytes_read > MAX_UPLOAD_SIZE:
                    buffer.close()
                    # Try to delete the partial file
                    try:
                        os.unlink(file_path)
                    except:
                        pass
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)}MB"
                    )
                buffer.write(chunk)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
        
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
        
        processing_status[file_id] = {
            "status": "queued",
            "filename": filename,
            "file_path": file_path, 
            "file_size": f"{file_size_mb} MB",
            "queued_at": timestamp,
            "current_step": "Waiting in queue",
            "user_id": user_id,
            "queue_position": len(user_queues[user_id]['queue'].queue) + 1
        }
        
        user_queues[user_id]['queue'].put((timestamp, file_id))
        processing_queue.put((timestamp, file_id))
        logger.info(f"Added {file_id} to processing queue for user {user_id}")
        
        response = JSONResponse({
            "file_id": file_id,
            "status": "queued",
            "timestamp": timestamp,
            "queue_position": processing_status[file_id]["queue_position"]
        })
        set_user_cookie(response, user_id)
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status(request: Request = None):
    user_id = get_user_id(request)
    
    user_status = {}
    for file_id, status in processing_status.items():
        if status.get("user_id") == user_id:
            user_status[file_id] = status
    
    response = JSONResponse({
        "queue_size": user_queues[user_id]['queue'].qsize() if user_id in user_queues else 0,
        "processing_status": user_status,
        "user_id": user_id
    })
    set_user_cookie(response, user_id)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT) 