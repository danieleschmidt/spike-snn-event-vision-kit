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

# Docker build targets
docker-build:  ## Build all Docker images
	docker-compose build

docker-build-dev:  ## Build development Docker image
	docker build --target development -t spike-snn-event-vision:dev .

docker-build-prod:  ## Build production Docker image
	docker build --target production -t spike-snn-event-vision:latest .

docker-build-cpu:  ## Build CPU-only Docker image
	docker build --target cpu-only -t spike-snn-event-vision:cpu .

docker-build-ros2:  ## Build ROS2 Docker image
	docker build --target ros2 -t spike-snn-event-vision:ros2 .

docker-up:  ## Start development environment
	docker-compose up -d spike-snn-dev

docker-down:  ## Stop all services
	docker-compose down

docker-logs:  ## Show logs from all services
	docker-compose logs -f

docker-shell:  ## Open shell in development container
	docker-compose exec spike-snn-dev bash

docker-test:  ## Run tests in Docker container
	docker-compose run --rm spike-snn-dev pytest tests/

docker-clean:  ## Clean Docker images and volumes
	docker-compose down -v
	docker system prune -f
	docker volume prune -f

# Security and compliance
security-scan:  ## Run security scans
	bandit -r src/
	safety check
	pip-audit

docker-security-scan:  ## Scan Docker images for vulnerabilities
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasec/trivy image spike-snn-event-vision:latest

sbom-generate:  ## Generate Software Bill of Materials
	syft packages dir:. -o spdx-json=sbom.spdx.json
	syft packages dir:. -o cyclonedx-json=sbom.cyclonedx.json

# Performance and benchmarking
benchmark:  ## Run performance benchmarks
	pytest -m benchmark tests/benchmarks/ --benchmark-only

profile:  ## Run profiling
	python -m cProfile -o profile.stats -m spike_snn_event.cli benchmark

memory-profile:  ## Run memory profiling
	mprof run python -m spike_snn_event.cli train --config configs/profile.yaml
	mprof plot

# CI/CD helpers
ci-install:  ## Install dependencies for CI
	pip install -e ".[dev,cuda,monitoring]"
	pip install pytest-xdist pytest-benchmark

ci-test:  ## Run CI test suite
	pytest tests/ -x -v --tb=short --cov=spike_snn_event --cov-report=xml

ci-lint:  ## Run CI linting
	pre-commit run --all-files

ci-build:  ## Build for CI
	python -m build
	twine check dist/*

# Release management
release-check:  ## Check if ready for release
	python -m build
	twine check dist/*
	pytest tests/ --tb=short

release-dry-run:  ## Dry run release to test PyPI
	python -m twine upload --repository testpypi dist/*

release:  ## Release to PyPI
	python -m twine upload dist/*

# Development utilities
jupyter:  ## Start Jupyter Lab
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

tensorboard:  ## Start TensorBoard
	tensorboard --logdir=logs/tensorboard --port=6006

wandb-init:  ## Initialize Weights & Biases
	wandb init --project spike-snn-event-vision

# Hardware-specific targets
gpu-test:  ## Run GPU-specific tests
	pytest -m gpu tests/

ros2-test:  ## Run ROS2 integration tests
	pytest -m ros2 tests/

hardware-test:  ## Run all hardware tests
	pytest -m "gpu or loihi or akida or ros2" tests/

# Documentation targets
docs-api:  ## Generate API documentation
	sphinx-apidoc -f -o docs/api src/

docs-clean:  ## Clean documentation build
	cd docs && make clean

docs-linkcheck:  ## Check documentation links
	cd docs && make linkcheck

# Monitoring and observability
start-monitoring:  ## Start monitoring stack
	docker-compose --profile monitoring up -d

stop-monitoring:  ## Stop monitoring stack
	docker-compose --profile monitoring down

logs-analysis:  ## Analyze application logs
	python scripts/analyze_logs.py logs/

# Multi-platform builds (requires buildx)
docker-buildx-setup:  ## Setup Docker buildx for multi-platform builds
	docker buildx create --name multibuilder --use
	docker buildx inspect --bootstrap

docker-build-multiplatform:  ## Build multi-platform images
	docker buildx build --platform linux/amd64,linux/arm64 \
		--target production -t spike-snn-event-vision:latest .

# Environment management
env-check:  ## Check environment setup
	python scripts/check_environment.py

env-info:  ## Show environment information
	python -c "import spike_snn_event; spike_snn_event.utils.system_info()"

# All-in-one targets
all-tests: test benchmark security-scan  ## Run all tests and scans

full-build: clean lint test build  ## Complete build pipeline

production-deploy: docker-build-prod docker-security-scan  ## Production deployment prep