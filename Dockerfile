# Hebrew Agentic RAG System - Production Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Graphics and image processing
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Network and utilities
    curl \
    wget \
    git \
    # Build tools (for some Python packages)
    gcc \
    g++ \
    python3-dev \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (commented out for bundle deployment)
# RUN curl -fsSL https://ollama.com/install.sh | sh

# Create app user (security best practice)
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY hebrew_tools/ ./hebrew_tools/
COPY config/ ./config/
COPY *.py ./

# Copy models (if included in build context)
# COPY models/ ./models/

# Create necessary directories
RUN mkdir -p documents logs chroma_db cache models && \
    chown -R appuser:appuser /app

# Set environment variables for air-gapped deployment
ENV AGNO_TELEMETRY=false \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start script
COPY --chown=appuser:appuser start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]