#!/bin/bash

# Development environment setup script

set -e

echo "🚀 Setting up Spike-SNN Event Vision Kit development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "📊 Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo "📚 Installing development dependencies..."
pip install -r requirements-dev.txt

# Install package in development mode
echo "🔧 Installing package in development mode..."
pip install -e .

# Install pre-commit hooks
echo "🎯 Installing pre-commit hooks..."
pre-commit install

# Run initial checks
echo "🧪 Running initial tests..."
pytest --version
black --version
flake8 --version
mypy --version

echo "✅ Development environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  make test"
echo ""
echo "To format code:"
echo "  make format"