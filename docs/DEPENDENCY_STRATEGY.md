# Dependency Update and Conflict Resolution Strategy

## Overview

CyberPuppy uses a constraints-based dependency management system built on **pip-tools** to ensure reproducible, secure, and stable builds while maintaining flexibility for updates.

## Architecture

### Core Components

1. **Source Files** (`*.in`)
   - `requirements.in`: Production dependencies with version ranges
   - `requirements-dev.in`: Development dependencies with version ranges

2. **Compiled Files** (`*.txt`)
   - `requirements.txt`: Locked production dependencies
   - `requirements-dev.txt`: Locked development dependencies

3. **Global Constraints** (`constraints.txt`)
   - Cross-cutting version pins for security and compatibility
   - Applied to all dependency compilations

4. **Build System Integration** (`pyproject.toml`)
   - pip-tools configuration
   - Tool-specific settings

## Dependency Update Strategy

### 1. Regular Updates (Weekly)

```bash
# Check for outdated packages
make deps-outdated

# Review and update source files as needed
vim requirements.in
vim requirements-dev.in

# Compile with latest compatible versions
make deps-upgrade

# Test and validate
make test
make deps-check
```

### 2. Security Updates (Immediate)

```bash
# For critical security issues
make deps-upgrade-package PACKAGE=urllib3
make deps-check
make test

# Commit and deploy immediately
git add requirements*.txt
git commit -m "security: upgrade urllib3 to fix CVE-XXXX-XXXX"
```

### 3. Major Version Updates (Quarterly)

```bash
# Create feature branch
git checkout -b feature/deps-major-update

# Update version ranges in source files
# Example: torch>=2.1.0 -> torch>=2.2.0

# Full recompilation
make deps-compile-all

# Comprehensive testing
make test
make test-integration
make benchmark  # Performance regression check

# Manual validation of critical paths
```

## Conflict Resolution Framework

### Level 1: Version Range Conflicts

**Problem**: Two packages require incompatible versions of the same dependency.

**Resolution Process**:

1. **Identify the Conflict**
   ```bash
   make deps-compile  # Will show resolution errors
   pip-compile --verbose requirements.in  # Detailed conflict info
   ```

2. **Analyze Constraints**
   ```bash
   pipdeptree --packages package-name
   ```

3. **Resolution Strategies** (in order of preference):
   - **Widen Range**: Adjust version ranges to find common ground
   - **Pin Intermediate**: Add explicit pin to constraints.txt
   - **Alternative Package**: Switch to compatible alternative
   - **Downgrade**: Temporarily use older version

**Example Resolution**:
```bash
# Conflict: fastapi requires pydantic>=2.0, older-package requires pydantic<2.0
echo "pydantic>=2.0.0,<3.0.0" >> constraints.txt
# Update older-package or find alternative
```

### Level 2: Transitive Dependency Issues

**Problem**: Deep dependency chains create version conflicts.

**Resolution Process**:

1. **Map Dependency Tree**
   ```bash
   pipdeptree --graph-output png > dependency-graph.png
   ```

2. **Isolate Problem Chain**
   ```bash
   pipdeptree --packages root-package --json
   ```

3. **Strategic Interventions**:
   - **Override Transitive**: Pin problematic transitive dependency
   - **Exclude Extras**: Remove optional features causing conflicts
   - **Vendor Critical**: Include problematic package in project

### Level 3: Platform-Specific Conflicts

**Problem**: Dependencies behave differently across operating systems or Python versions.

**Resolution Strategies**:

1. **Conditional Dependencies**
   ```python
   # In requirements.in
   numpy>=1.24.0,<2.0.0; python_version < "3.12"
   numpy>=1.26.0; python_version >= "3.12"
   ```

2. **Platform Markers**
   ```python
   # Windows-specific
   pywin32>=227; sys_platform == "win32"

   # Unix-specific
   uvloop>=0.17.0; sys_platform != "win32"
   ```

3. **Multi-Platform Testing**
   - GitHub Actions matrix builds
   - Docker multi-arch testing

## Emergency Procedures

### Critical Security Vulnerability

1. **Immediate Response** (< 2 hours)
   ```bash
   # Update affected package
   make deps-upgrade-package PACKAGE=vulnerable-package

   # Emergency testing
   make test-unit  # Fast feedback
   make security   # Verify fix

   # Deploy with rollback plan
   ```

2. **Full Validation** (< 24 hours)
   ```bash
   make test-integration
   make performance-test
   make deps-check
   ```

### Dependency Resolution Failure

1. **Diagnostic Steps**
   ```bash
   # Enable verbose compilation
   pip-compile --verbose --dry-run requirements.in

   # Check for dependency cycles
   pipdeptree --warn silence

   # Validate constraints
   pip-compile --constraint constraints.txt --dry-run requirements.in
   ```

2. **Resolution Process**
   - Create minimal reproduction case
   - Test without constraints
   - Identify conflicting constraints
   - Update constraints or source requirements

### Rollback Procedures

1. **Version Control**
   ```bash
   git revert HEAD  # Rollback dependency changes
   make deps-sync   # Restore previous state
   ```

2. **Container Rollback**
   ```bash
   docker pull cyberpuppy:previous-tag
   kubectl rollout undo deployment/cyberpuppy
   ```

## Monitoring and Alerting

### Automated Checks

1. **GitHub Actions**
   - Weekly dependency updates check
   - Security vulnerability scanning
   - License compatibility validation

2. **Monitoring Metrics**
   - Dependency freshness (days behind latest)
   - Security vulnerability count
   - Build success rate

### Alert Thresholds

- **Security**: Any HIGH or CRITICAL vulnerability
- **Freshness**: Dependencies > 90 days old
- **Compatibility**: Build failures > 2 consecutive runs

## Best Practices

### Source File Management

1. **Version Ranges**
   ```python
   # Good: Semantic versioning with upper bounds
   torch>=2.1.0,<3.0.0

   # Bad: No upper bound
   torch>=2.1.0

   # Bad: Too restrictive
   torch==2.1.0
   ```

2. **Comment Documentation**
   ```python
   # Security: CVE-2024-XXXX fixed in 2.5.0
   urllib3>=2.5.0,<3.0.0

   # Compatibility: Required for Python 3.12+
   numpy>=1.26.0; python_version >= "3.12"
   ```

### Constraints Management

1. **Categorization**
   ```python
   # Security constraints
   urllib3>=2.0.0,<3.0.0

   # Compatibility constraints
   setuptools>=68.0.0

   # Performance constraints
   uvloop>=0.17.0; sys_platform != "win32"
   ```

2. **Regular Review**
   - Monthly constraint audit
   - Remove obsolete constraints
   - Update security baselines

### Testing Integration

1. **Dependency-Specific Tests**
   ```python
   def test_security_packages():
       """Ensure critical security packages are current."""
       import urllib3
       assert urllib3.__version__ >= "2.0.0"
   ```

2. **Performance Regression Testing**
   ```python
   @pytest.mark.performance
   def test_model_inference_speed():
       """Ensure dependency updates don't regress performance."""
       # Benchmark critical paths
   ```

## Tools and Resources

### Development Tools
- `pip-tools`: Dependency compilation and syncing
- `pipdeptree`: Dependency visualization
- `safety`: Security vulnerability scanning
- `pip-audit`: Alternative security scanner
- `pip-licenses`: License compatibility checking

### Monitoring Tools
- GitHub Dependabot (disabled in favor of custom workflow)
- Snyk (optional enterprise security scanning)
- WhiteSource/Mend (optional license scanning)

### Reference Documentation
- [pip-tools documentation](https://pip-tools.readthedocs.io/)
- [Python Packaging Guide](https://packaging.python.org/)
- [PEP 508: Dependency specification](https://peps.python.org/pep-0508/)

## Migration Guide

### From Legacy pip to pip-tools

1. **Backup Current State**
   ```bash
   cp requirements.txt requirements-legacy.txt
   cp requirements-dev.txt requirements-dev-legacy.txt
   ```

2. **Create Source Files**
   ```bash
   # Extract top-level dependencies from locked files
   # This is a manual process requiring dependency analysis
   ```

3. **Test Migration**
   ```bash
   make deps-compile-all
   make deps-sync
   make test
   ```

4. **Validate Reproducibility**
   ```bash
   # Fresh environment test
   python -m venv test-env
   source test-env/bin/activate
   pip-sync requirements.txt
   python -c "import cyberpuppy; print('OK')"
   ```

This strategy ensures CyberPuppy maintains stable, secure, and reproducible dependency management while providing clear processes for updates and conflict resolution.