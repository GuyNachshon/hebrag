FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
#COPY scripts/install_ollama.sh /tmp/
#RUN chmod +x /tmp/install_ollama.sh && /tmp/install_ollama.sh

# Copy and install Python packages
COPY packages/ /tmp/packages/
COPY requirements.txt .
RUN pip install --no-cache-dir --find-links /tmp/packages -r requirements.txt

# Copy models
COPY models/ ./models/

# Copy source code
COPY src/ ./src/
COPY hebrew_tools/ ./hebrew_tools/
COPY config/ ./config/

# Set environment variables for air-gapped deployment
ENV AGNO_TELEMETRY=false
ENV OLLAMA_HOST=0.0.0.0:11434
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

EXPOSE 8000 11434

# Start script
COPY scripts/start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
