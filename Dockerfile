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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with explicit install of moviepy
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir psutil && \
    pip install --no-cache-dir moviepy>=1.0.3 decorator>=4.0.11 imageio>=2.5 imageio-ffmpeg>=0.4.0 tqdm>=4.11.2 numpy

# Create necessary directories
RUN mkdir -p uploads temp models static

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

# Run the application
ENTRYPOINT ["sh", "-c"]
CMD ["exec uvicorn main:app --host $HOST --port $PORT"] 