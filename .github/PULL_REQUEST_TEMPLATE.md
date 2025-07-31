# Pull Request

## Description

Brief description of the changes in this PR.

## Type of Change

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Maintenance (dependency updates, build changes, etc.)
- [ ] ⚡ Performance improvement
- [ ] 🧪 Test improvements
- [ ] 🏗️ Refactoring

## Related Issues

Closes #(issue)
Relates to #(issue)

## Changes Made

### Core Changes
- List the main changes made

### Files Modified
- `file1.py` - Brief description of changes
- `file2.py` - Brief description of changes

## Testing

### Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Hardware tests completed (if applicable)
- [ ] All existing tests pass

### Manual Testing
Describe any manual testing performed:

```python
# Example code used for testing
from spike_snn_event import Model
model = Model()
# Test steps...
```

### Performance Impact
- [ ] No performance impact
- [ ] Performance improved
- [ ] Minor performance regression (justified)
- [ ] Significant performance change (please explain)

**Performance details:**
<!-- Benchmark results, profiling data, etc. -->

## Hardware Compatibility

- [ ] CPU only
- [ ] NVIDIA CUDA
- [ ] Intel Loihi 2
- [ ] BrainChip Akida
- [ ] Not applicable

**Hardware testing notes:**
<!-- Any hardware-specific testing performed -->

## Documentation

- [ ] Code is self-documenting with appropriate docstrings
- [ ] README updated (if needed)
- [ ] Documentation updated (if needed)
- [ ] Examples updated/added (if needed)
- [ ] Changelog entry added

## Code Quality

### Pre-commit Checks
- [ ] `black` formatting applied
- [ ] `flake8` linting passed
- [ ] `mypy` type checking passed
- [ ] `pytest` tests pass
- [ ] Pre-commit hooks pass

### Code Review
- [ ] Code follows project style guidelines
- [ ] Complex logic is documented
- [ ] Error handling is appropriate
- [ ] Security considerations addressed

## Breaking Changes

If this PR introduces breaking changes, please describe:

### What breaks?
<!-- Describe what existing functionality will break -->

### Migration Guide
<!-- Provide steps for users to migrate their code -->

```python
# Old way
old_api_call()

# New way
new_api_call()
```

## Additional Context

### Screenshots (if applicable)
<!-- Add screenshots for UI changes or visual improvements -->

### Dependencies
- [ ] No new dependencies
- [ ] New runtime dependencies added (listed below)
- [ ] New development dependencies added (listed below)

**New dependencies:**
<!-- List any new dependencies and justification -->

### Deployment Notes
<!-- Any special deployment considerations -->

---

## Checklist

- [ ] I have read the [Contributing Guidelines](../CONTRIBUTING.md)
- [ ] My code follows the project's coding standards
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published in downstream modules

## For Maintainers

<!-- This section is for maintainers - contributors can ignore -->

### Review Priority
- [ ] High - Critical fix or major feature
- [ ] Medium - Standard change
- [ ] Low - Minor improvement

### Merge Strategy
- [ ] Squash and merge
- [ ] Create a merge commit
- [ ] Rebase and merge