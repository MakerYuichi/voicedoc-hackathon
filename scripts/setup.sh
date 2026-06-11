#!/bin/bash
# Setup script for VoiceDoc Intelligence

set -e

echo "🎯 VoiceDoc Intelligence Setup"
echo "=============================="

# Check Python version (requires 3.12, NOT 3.14 which lacks wheel support)
echo "🐍 Checking Python version..."
PYTHON_BIN=$(which python3.12 2>/dev/null || echo "/opt/homebrew/bin/python3.12")

if [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ Python 3.12 not found. Install it with: brew install python@3.12"
    echo "   (Python 3.14 is NOT supported — packages like numpy have no wheels yet)"
    exit 1
fi

$PYTHON_BIN --version

# Create virtual environment
echo "📦 Creating virtual environment with Python 3.12..."
$PYTHON_BIN -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Copy environment file
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your actual credentials!"
else
    echo "✅ .env file already exists"
fi

# Make scripts executable
chmod +x scripts/*.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your credentials"
echo "2. Start services: docker-compose up -d"
echo "3. Run the app: uvicorn app.main:app --reload"
