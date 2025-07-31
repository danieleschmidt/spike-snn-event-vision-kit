# CI/CD Workflow Documentation

This directory contains documentation and templates for GitHub Actions workflows that need to be manually set up due to security considerations.

## ⚠️ Important Note

**GitHub Actions workflows (.yml files) are not included in this repository for security reasons.** 

The files in this directory are **documentation templates** that describe the required workflows. You must manually create these workflows in your `.github/workflows/` directory.

## Required Workflows

### 1. Basic CI Pipeline (`ci.yml`)
**Purpose**: Run tests, linting, and basic quality checks on every PR and push.

**Triggers**: 
- Pull requests to main branch
- Pushes to main branch
- Manual workflow dispatch

**Jobs**:
- **Lint & Format**: Black, flake8, isort, mypy
- **Test Matrix**: Python 3.8-3.11 on Ubuntu/Windows/macOS
- **Coverage**: Generate and upload coverage reports
- **Security**: Bandit security analysis

**Required Secrets**: None
**Estimated Runtime**: 8-12 minutes

### 2. Hardware Testing (`hardware-tests.yml`)
**Purpose**: Run hardware-specific tests on self-hosted runners.

**Triggers**:
- Pull requests with 'hardware' label
- Manual workflow dispatch with hardware selection

**Jobs**:
- **GPU Tests**: CUDA functionality testing
- **Event Camera**: Hardware integration tests
- **Neuromorphic**: Loihi/Akida tests (if available)

**Required Secrets**: 
- Hardware access credentials
- Self-hosted runner tokens

**Estimated Runtime**: 15-30 minutes

### 3. Security Scanning (`security.yml`)
**Purpose**: Comprehensive security analysis and dependency scanning.

**Triggers**:
- Daily schedule (2 AM UTC)
- Pull requests to main branch
- Manual dispatch

**Jobs**:
- **Dependency Scan**: Check for known vulnerabilities
- **Code Analysis**: Static security analysis
- **Container Scan**: Docker image security scanning
- **SBOM Generation**: Software Bill of Materials

**Required Secrets**:
- `GITHUB_TOKEN`: For GitHub security APIs
- Security scanning service tokens (optional)

**Estimated Runtime**: 5-10 minutes

### 4. Build & Package (`build.yml`)
**Purpose**: Build distributions and container images.

**Triggers**:
- Tags matching `v*.*.*`
- Pull requests to main branch
- Manual dispatch

**Jobs**:
- **Python Package**: Build sdist and wheel
- **Docker Images**: Multi-arch container builds
- **Documentation**: Build and deploy docs
- **Artifacts**: Upload build artifacts

**Required Secrets**:
- `PYPI_API_TOKEN`: For PyPI publishing
- `DOCKER_USERNAME` & `DOCKER_PASSWORD`: For container registry
- `DOCS_DEPLOY_KEY`: For documentation deployment

**Estimated Runtime**: 10-20 minutes

### 5. Release (`release.yml`)
**Purpose**: Automated release creation and publishing.

**Triggers**:
- Tags matching `v*.*.*`
- Manual workflow dispatch

**Jobs**:
- **Create Release**: Generate release notes from PRs
- **Publish PyPI**: Upload to Python Package Index
- **Deploy Docs**: Update documentation site
- **Notify**: Send notifications to team channels

**Required Secrets**:
- `PYPI_API_TOKEN`: PyPI publishing
- `GITHUB_TOKEN`: Release creation (auto-provided)
- Notification service tokens

**Estimated Runtime**: 5-15 minutes

### 6. Performance Benchmarks (`benchmarks.yml`)
**Purpose**: Run performance benchmarks and track regressions.

**Triggers**:
- Weekly schedule (Sundays at 3 AM UTC)
- Pull requests with 'performance' label
- Manual dispatch

**Jobs**:
- **Inference Benchmarks**: Latency and throughput tests
- **Memory Profiling**: Memory usage analysis
- **Comparison**: Compare against baseline performance
- **Report**: Generate performance report

**Required Secrets**: None (uses GitHub Pages for reports)
**Estimated Runtime**: 30-60 minutes

## Setup Instructions

### 1. Create Workflow Files

For each workflow above, create the corresponding `.yml` file in `.github/workflows/`:

```bash
mkdir -p .github/workflows
touch .github/workflows/ci.yml
touch .github/workflows/hardware-tests.yml
touch .github/workflows/security.yml
touch .github/workflows/build.yml
touch .github/workflows/release.yml
touch .github/workflows/benchmarks.yml
```

### 2. Configure Secrets

Add the following secrets in your GitHub repository settings:

**Required for all repositories**:
- `PYPI_API_TOKEN`: Get from PyPI account settings
- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub access token

**Optional (for enhanced features)**:
- `SLACK_WEBHOOK_URL`: For build notifications
- `CODECOV_TOKEN`: For enhanced coverage reporting
- Hardware-specific access tokens

### 3. Environment Variables

Configure these environment variables in your workflows:

```yaml
env:
  PYTHON_VERSION: "3.10"
  PYTORCH_VERSION: "2.0.0"
  CUDA_VERSION: "11.8"
```

### 4. Self-Hosted Runners (Optional)

For hardware testing, set up self-hosted runners:

1. Go to Settings → Actions → Runners
2. Click "New self-hosted runner"
3. Follow setup instructions for your hardware
4. Label runners appropriately (`gpu`, `event-camera`, etc.)

## Workflow Templates

See the individual template files in this directory:

- [`ci-template.yml`](./ci-template.yml) - Basic CI pipeline
- [`security-template.yml`](./security-template.yml) - Security scanning
- [`build-template.yml`](./build-template.yml) - Build and packaging
- [`release-template.yml`](./release-template.yml) - Release automation
- [`benchmarks-template.yml`](./benchmarks-template.yml) - Performance testing

## Monitoring & Troubleshooting

### Workflow Status

Monitor workflow health with:
- GitHub Actions dashboard
- Workflow status badges in README
- Email notifications for failures

### Common Issues

**Test Failures**:
- Check Python version compatibility
- Verify dependency installation
- Review hardware availability

**Security Scan Failures**:
- Update vulnerable dependencies
- Review security scan reports
- Add exceptions for false positives

**Build Failures**:
- Check Docker image builds
- Verify package metadata
- Review artifact upload permissions

**Performance Regressions**:
- Compare benchmark results
- Review recent changes
- Check hardware configuration

## Customization

### Adding New Hardware Support

1. Update hardware test matrix
2. Add new runner labels
3. Create hardware-specific test jobs
4. Update documentation

### Modifying Test Coverage

1. Adjust coverage thresholds in `pytest.ini`
2. Update coverage reporting in workflows
3. Add new test categories as needed

### Security Enhancements

1. Add additional security scanners
2. Configure dependency update automation
3. Implement security policy enforcement

## Best Practices

### Workflow Design
- Keep workflows focused and single-purpose
- Use matrix strategies for multi-environment testing
- Implement proper error handling and retries
- Cache dependencies to improve performance

### Security
- Never store secrets in workflow files
- Use GitHub's secret management
- Implement least-privilege access
- Regularly rotate access tokens

### Performance
- Use workflow caching strategically
- Parallelize independent jobs
- Minimize artifact sizes
- Consider workflow scheduling to avoid peak times

### Maintenance
- Regularly update action versions
- Monitor workflow performance
- Review and update security configurations
- Keep documentation current

## Support

For workflow setup assistance:
1. Check GitHub Actions documentation
2. Review existing working examples
3. Use GitHub Community forums
4. Contact repository maintainers

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python CI/CD Best Practices](https://docs.python.org/3/using/ci.html)
- [Docker Build Actions](https://docs.docker.com/ci-cd/github-actions/)
- [Security Scanning Tools](https://github.com/marketplace?type=actions&query=security)