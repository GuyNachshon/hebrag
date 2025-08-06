#!/bin/bash
echo "Configuring system..."
export AGNO_TELEMETRY=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Create .env file
cat > .env << 'ENV_EOF'
AGNO_TELEMETRY=false
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
OLLAMA_HOST=0.0.0.0:11434
ENV_EOF

echo "Phase 3 completed"
