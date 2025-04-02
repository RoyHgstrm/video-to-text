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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure ImageMagick to allow PDF operations (needed for MoviePy)
RUN mkdir -p /etc/ImageMagick-6 && \
    echo '<policymap>' > /etc/ImageMagick-6/policy.xml && \
    echo '  <policy domain="resource" name="memory" value="256MiB"/>' >> /etc/ImageMagick-6/policy.xml && \
    echo '  <policy domain="resource" name="disk" value="1GiB"/>' >> /etc/ImageMagick-6/policy.xml && \
    echo '  <policy domain="resource" name="map" value="512MiB"/>' >> /etc/ImageMagick-6/policy.xml && \
    echo '  <policy domain="resource" name="width" value="16KP"/>' >> /etc/ImageMagick-6/policy.xml && \
    echo '  <policy domain="resource" name="height" value="16KP"/>' >> /etc/ImageMagick-6/policy.xml && \
    echo '  <policy domain="resource" name="area" value="128MB"/>' >> /etc/ImageMagick-6/policy.xml && \
    echo '  <policy domain="resource" name="time" value="120"/>' >> /etc/ImageMagick-6/policy.xml && \
    echo '</policymap>' >> /etc/ImageMagick-6/policy.xml

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies step by step to isolate any issues
RUN pip install --upgrade pip && \
    # Core dependencies
    pip install --no-cache-dir fastapi uvicorn python-multipart jinja2 && \
    # Audio processing
    pip install --no-cache-dir pydub && \
    # Speech recognition
    pip install --no-cache-dir SpeechRecognition vosk && \
    # Install MoviePy and its dependencies explicitly
    pip install --no-cache-dir numpy decorator>=4.0.11 imageio>=2.5 imageio-ffmpeg>=0.4.0 tqdm>=4.11.2 && \
    pip install --no-cache-dir moviepy>=1.0.3 && \
    # Other dependencies from requirements file
    pip install --no-cache-dir -r requirements.txt && \
    # System monitoring
    pip install --no-cache-dir psutil

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

# Expose port
EXPOSE 8000

# Healthcheck (wait up to 30 seconds for startup)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application with proper user permissions
ENTRYPOINT ["sh", "-c"]
CMD ["exec uvicorn main:app --host $HOST --port $PORT --timeout-keep-alive 75"] 