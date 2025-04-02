FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    procps \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt psutil

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