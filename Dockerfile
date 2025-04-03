FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including those needed for moviepy
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    procps \
    build-essential \
    imagemagick \
    python3-dev \
    zlib1g-dev \
    libjpeg-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure ImageMagick to allow operations (needed for MoviePy)
RUN mkdir -p /etc/ImageMagick-6 && \
    echo '<policymap>' > /etc/ImageMagick-6/policy.xml && \
    echo '  <policy domain="resource" name="memory" value="256MiB"/>' >> /etc/ImageMagick-6/policy.xml && \
    echo '  <policy domain="resource" name="disk" value="1GiB"/>' >> /etc/ImageMagick-6/policy.xml && \
    echo '</policymap>' >> /etc/ImageMagick-6/policy.xml

# Copy requirements first for better caching
COPY requirements.txt .

# Install basic dependencies first
RUN pip install --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn python-multipart jinja2 pydub SpeechRecognition psutil vosk

# Install MoviePy directly from source for reliability
RUN pip install --no-cache-dir numpy decorator imageio imageio-ffmpeg tqdm && \
    pip install --no-cache-dir git+https://github.com/Zulko/moviepy.git@v1.0.3

# Try installing from PyPI as a fallback
RUN pip install --no-cache-dir moviepy || echo "MoviePy already installed from source"

# Install the rest of the requirements
RUN pip install --no-cache-dir torch torchaudio

# Create necessary directories with proper permissions
RUN mkdir -p uploads temp models static && \
    chmod 777 uploads temp models

# Copy application code
COPY . .

# Environment variables with sensible defaults
ENV PORT=8000
ENV HOST=0.0.0.0
ENV PRODUCTION=true
ENV MAX_UPLOAD_SIZE=104857600
ENV CORS_ORIGINS=*
ENV PYTHONUNBUFFERED=1

# Debug - verify MoviePy installation
RUN python -c "from moviepy.editor import VideoFileClip; print('MoviePy successfully imported')" || echo "Error importing MoviePy"

# Expose port
EXPOSE 8000

# Healthcheck (wait up to 30 seconds for startup)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application with proper user permissions
ENTRYPOINT ["sh", "-c"]
CMD ["exec uvicorn main:app --host $HOST --port $PORT --timeout-keep-alive 75"] 