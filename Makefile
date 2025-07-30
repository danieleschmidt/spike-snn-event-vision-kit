.PHONY: help install install-dev test test-cov lint format clean docs docs-serve build upload

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev,docs,cuda,ros2,monitoring]"

test:  ## Run tests
	pytest tests/

test-cov:  ## Run tests with coverage
	pytest --cov=spike_snn_event --cov-report=html --cov-report=term tests/

lint:  ## Run linting
	flake8 src/ tests/
	mypy src/

format:  ## Format code
	black src/ tests/ examples/
	isort src/ tests/ examples/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

docs:  ## Build documentation
	cd docs && make html

docs-serve:  ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

build:  ## Build package
	python -m build

upload:  ## Upload to PyPI (requires authentication)
	python -m twine upload dist/*

# Development shortcuts
dev-setup: install-dev  ## Complete development setup
	pre-commit install

quick-test:  ## Run quick tests (no slow or hardware tests)
	pytest -m "not slow and not hardware" tests/