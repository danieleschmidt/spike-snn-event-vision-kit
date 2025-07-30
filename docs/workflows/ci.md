# Continuous Integration Workflow

This document describes the recommended GitHub Actions workflow for the Spike-SNN Event Vision Kit.

## Required CI Workflow

Create `.github/workflows/ci.yml` with the following configuration:

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
        pip install -e .
    
    - name: Lint with flake8
      run: |
        flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src tests --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
    
    - name: Type check with mypy
      run: mypy src
    
    - name: Test with pytest
      run: |
        pytest --cov=spike_snn_event --cov-report=xml --cov-report=term-missing
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Bandit security scan
      uses: securecodewarrior/github-action-bandit@v1.0.1
      with:
        args: '-r src'
    
    - name: Run Safety check
      run: |
        pip install safety
        safety check --json

  docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.9"
    
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
        pip install -e .
    
    - name: Build documentation
      run: |
        cd docs
        make html
```

## Security Workflow

Create `.github/workflows/security.yml`:

```yaml
name: Security

on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6 AM
  workflow_dispatch:

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Dependency Review
      uses: actions/dependency-review-action@v3
    
    - name: Security audit
      run: |
        pip install pip-audit
        pip-audit --desc --format=json

    - name: SBOM Generation
      uses: anchore/sbom-action@v0
      with:
        path: ./
        format: spdx-json
```

## Release Workflow

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.9"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

## Pre-commit Integration

The repository includes pre-commit hooks that run the same checks locally:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Required Secrets

Configure these secrets in GitHub repository settings:

- `PYPI_API_TOKEN`: For package publishing
- `CODECOV_TOKEN`: For coverage reporting (optional)

## Branch Protection

Recommended branch protection rules for `main`:

- Require pull request reviews
- Require status checks to pass before merging
- Required status checks: `test`, `security-scan`, `docs`
- Require branches to be up to date before merging
- Restrict pushes to matching branches