#!/bin/bash

# Air-gapped Hebrew RAG System - System Dependencies Installer
# This script installs all required system packages for the Hebrew RAG system

set -e  # Exit on any error

echo "🚀 Installing Hebrew RAG System Dependencies for Air-Gapped Environment"
echo "========================================================================="

# Detect package manager
if command -v apt-get &> /dev/null; then
    PACKAGE_MANAGER="apt"
elif command -v yum &> /dev/null; then
    PACKAGE_MANAGER="yum" 
elif command -v dnf &> /dev/null; then
    PACKAGE_MANAGER="dnf"
elif command -v zypper &> /dev/null; then
    PACKAGE_MANAGER="zypper"
else
    echo "❌ No supported package manager found!"
    echo "Supported: apt-get, yum, dnf, zypper"
    exit 1
fi

echo "📦 Detected package manager: $PACKAGE_MANAGER"

# Update package lists
echo "🔄 Updating package lists..."
case $PACKAGE_MANAGER in
    "apt")
        sudo apt-get update
        ;;
    "yum"|"dnf")
        sudo $PACKAGE_MANAGER update -y
        ;;
    "zypper")
        sudo zypper refresh
        ;;
esac

# Install Python and core dependencies
echo "🐍 Installing Python and core dependencies..."
case $PACKAGE_MANAGER in
    "apt")
        sudo apt-get install -y \
            python3 \
            python3-venv \
            python3-pip \
            python3-dev \
            python3-distutils \
            build-essential \
            curl \
            wget \
            unzip \
            git
        ;;
    "yum"|"dnf")
        sudo $PACKAGE_MANAGER install -y \
            python3 \
            python3-venv \
            python3-pip \
            python3-devel \
            gcc \
            gcc-c++ \
            make \
            curl \
            wget \
            unzip \
            git
        ;;
    "zypper")
        sudo zypper install -y \
            python3 \
            python3-venv \
            python3-pip \
            python3-devel \
            gcc \
            gcc-c++ \
            make \
            curl \
            wget \
            unzip \
            git
        ;;
esac

# Install development libraries (for package compilation)
echo "🔧 Installing development libraries..."
case $PACKAGE_MANAGER in
    "apt")
        sudo apt-get install -y \
            libssl-dev \
            libffi-dev \
            libbz2-dev \
            libreadline-dev \
            libsqlite3-dev \
            libncurses5-dev \
            libncursesw5-dev \
            xz-utils \
            tk-dev \
            libxml2-dev \
            libxmlsec1-dev \
            liblzma-dev \
            zlib1g-dev \
            libgdbm-dev \
            libnss3-dev \
            libssl-dev \
            libreadline-dev \
            libffi-dev \
            libsqlite3-dev \
            libbz2-dev
        ;;
    "yum"|"dnf")
        sudo $PACKAGE_MANAGER install -y \
            openssl-devel \
            libffi-devel \
            bzip2-devel \
            readline-devel \
            sqlite-devel \
            ncurses-devel \
            xz-devel \
            tk-devel \
            libxml2-devel \
            xmlsec1-devel \
            zlib-devel \
            gdbm-devel
        ;;
    "zypper")
        sudo zypper install -y \
            libopenssl-devel \
            libffi-devel \
            libbz2-devel \
            readline-devel \
            sqlite3-devel \
            ncurses-devel \
            xz-devel \
            tk-devel \
            libxml2-devel \
            libxmlsec1-devel \
            zlib-devel \
            gdbm-devel
        ;;
esac

# Verify Python installation
echo "✅ Verifying Python installation..."
python3 --version || { echo "❌ Python3 not found!"; exit 1; }
python3 -m venv --help > /dev/null || { echo "❌ python3-venv not available!"; exit 1; }
python3 -m pip --version || { echo "❌ python3-pip not available!"; exit 1; }

echo "✅ Python verification complete:"
echo "   Python: $(python3 --version)"
echo "   Pip: $(python3 -m pip --version)"

# Install uv for faster package management (if available)
echo "⚡ Installing uv package manager..."
if command -v curl &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "✅ uv installed successfully"
else
    echo "⚠️  curl not available, skipping uv installation (will use pip)"
fi

echo ""
echo "🎉 System dependencies installation complete!"
echo "========================================================================="
echo "✅ Python 3 with venv and pip installed"
echo "✅ Development libraries installed"
echo "✅ Build tools installed"
echo "⚡ uv package manager ready (if curl was available)"
echo ""
echo "Next steps:"
echo "1. Run the main deployment script: ./deploy.sh"
echo "2. Or manually install Python packages from wheels/"
echo "========================================================================="