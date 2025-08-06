#!/bin/bash
set -e
echo "Deploying Hebrew Agentic RAG System (Air-Gapped)"
echo "Phase 1: Environment setup..."
./scripts/deployment/phase1_environment.sh
echo "Phase 2: Installing dependencies..."
./scripts/deployment/phase2_dependencies.sh
echo "Phase 3: Configuring system..."
./scripts/deployment/phase3_configuration.sh
echo "Phase 4: Starting services..."
./scripts/deployment/phase4_startup.sh
echo "Deployment completed successfully!"
echo "Hebrew RAG system available at http://localhost:8000"
