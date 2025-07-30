.PHONY: help install test lint format clean docs

help:		## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:	## Install development dependencies
	pip install -r requirements-dev.txt
	pre-commit install

test:		## Run tests with coverage
	pytest --cov=spike_snn_event --cov-report=html --cov-report=term-missing

test-fast:	## Run tests without coverage
	pytest

lint:		## Run linting checks
	flake8 src tests
	mypy src
	black --check .
	isort --check-only .

format:		## Format code
	black .
	isort .

clean:		## Clean build and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

docs:		## Build documentation
	cd docs && make html

docs-serve:	## Serve documentation locally
	cd docs && sphinx-autobuild . _build/html

build:		## Build package
	python -m build

upload-test:	## Upload to test PyPI
	twine upload --repository testpypi dist/*

upload:		## Upload to PyPI
	twine upload dist/*

benchmark:	## Run performance benchmarks
	python scripts/benchmark_models.py

profile:	## Profile memory usage
	python -m memory_profiler examples/basic_detection.py