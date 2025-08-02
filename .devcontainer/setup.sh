#!/bin/bash
set -e

echo "🚀 Setting up Spike-SNN Event Vision Kit development environment..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -e ".[dev,docs,monitoring,cuda,ros2]"

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating development directories..."
mkdir -p models datasets data checkpoints logs

# Set up Git configuration for better commit messages
echo "🔧 Configuring Git..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input

# Install additional development tools
echo "🛠️ Installing additional development tools..."
pip install jupyterlab ipywidgets

# Set up Jupyter extensions for neuromorphic visualization
echo "🧠 Setting up Jupyter extensions..."
jupyter lab build --dev-mode=False

# Verify installation
echo "✅ Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import snntorch; print(f'SNNTorch: {snntorch.__version__}')" || echo "⚠️ SNNTorch not found - will be installed with main package"
python -c "import tonic; print(f'Tonic: {tonic.__version__}')" || echo "⚠️ Tonic not found - will be installed with main package"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if [ $? -eq 0 ]; then
    python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
fi

# Run basic tests to ensure everything works
echo "🧪 Running basic verification tests..."
python -m pytest tests/test_basic.py -v || echo "⚠️ Basic tests failed - check installation"

# Set up development aliases
echo "📝 Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# Spike-SNN Development Aliases
alias snn-test='python -m pytest tests/ -v'
alias snn-lint='pre-commit run --all-files'
alias snn-format='black src/ tests/ && isort src/ tests/'
alias snn-typecheck='mypy src/'
alias snn-docs='cd docs && make html && cd ..'
alias snn-clean='find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true'
alias snn-train='python -m spike_snn_event.cli train'
alias snn-detect='python -m spike_snn_event.cli detect'
alias snn-benchmark='python -m spike_snn_event.cli benchmark'

# Jupyter shortcuts
alias jlab='jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser'
alias nb='jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser'

# Docker shortcuts
alias dc='docker-compose'
alias dcup='docker-compose up -d'
alias dcdown='docker-compose down'
alias dclogs='docker-compose logs -f'

# Git shortcuts for neuromorphic development
alias git-snn='git log --oneline --graph --decorate --all'
alias git-clean='git clean -fd && git reset --hard HEAD'
EOF

echo "🎉 Development environment setup complete!"
echo ""
echo "🔧 Available commands:"
echo "  snn-test      - Run all tests"
echo "  snn-lint      - Run linting and formatting checks"
echo "  snn-format    - Format code with black and isort"
echo "  snn-typecheck - Run mypy type checking"
echo "  snn-docs      - Build documentation"
echo "  snn-clean     - Clean Python cache files"
echo "  jlab          - Start Jupyter Lab"
echo ""
echo "📚 Next steps:"
echo "  1. Run 'snn-test' to verify everything works"
echo "  2. Check 'snn-lint' for code quality"
echo "  3. Start coding with 'jlab' for interactive development"
echo ""
echo "🧠 Happy neuromorphic coding! 🚀"