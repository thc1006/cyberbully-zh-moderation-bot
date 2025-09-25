# CyberPuppy Dependency Management Workflow

## Quick Start Guide

The CyberPuppy project uses **pip-tools** for robust, reproducible dependency management. This replaces the traditional `pip freeze` approach with a more maintainable system.

### 🚀 For New Developers

```bash
# 1. Clone and enter project
git clone <repo-url>
cd cyberbully-zh-moderation-bot

# 2. Install using new pip-tools workflow
make install

# 3. Verify installation
make deps-check
make test-unit
```

### 📦 For Existing Developers

If you've been using the old workflow:

```bash
# Backup old environment
pip freeze > old-requirements.txt

# Install with new system
make clean-all  # Remove old venv
make install    # Create new pip-tools environment

# Verify everything works
make test
```

## File Structure Overview

```
cyberbully-zh-moderation-bot/
├── requirements.in          # 🎯 Source: Production dependencies (edit this)
├── requirements.txt         # 🔒 Locked: Compiled production dependencies (generated)
├── requirements-dev.in      # 🎯 Source: Development dependencies (edit this)
├── requirements-dev.txt     # 🔒 Locked: Compiled development dependencies (generated)
├── constraints.txt          # 🛡️ Global: Cross-cutting version constraints
└── pyproject.toml          # ⚙️ Config: pip-tools and build settings
```

### Key Principles

1. **Edit `.in` files** - Never manually edit `.txt` files
2. **Compile changes** - Use `make deps-compile-all` after editing `.in` files
3. **Commit both** - Always commit both `.in` and `.txt` files
4. **Use constraints** - Global constraints in `constraints.txt` for consistency

## Daily Workflow

### Adding New Dependencies

1. **Production Dependency**
   ```bash
   # Edit requirements.in
   echo "new-package>=1.0.0" >> requirements.in

   # Compile and sync
   make deps-compile
   make deps-sync

   # Test and commit
   make test
   git add requirements.in requirements.txt
   git commit -m "feat: add new-package for feature X"
   ```

2. **Development Dependency**
   ```bash
   # Edit requirements-dev.in
   echo "new-dev-tool>=2.0.0" >> requirements-dev.in

   # Compile and sync
   make deps-dev-compile
   make deps-dev-sync

   # Test and commit
   make test
   git add requirements-dev.in requirements-dev.txt
   git commit -m "dev: add new-dev-tool for improved DX"
   ```

### Updating Dependencies

1. **Update All Dependencies**
   ```bash
   # Upgrade to latest compatible versions
   make deps-upgrade

   # Review changes
   git diff requirements*.txt

   # Test thoroughly
   make test
   make test-integration

   # Commit if tests pass
   git add requirements*.txt
   git commit -m "deps: upgrade all dependencies to latest compatible versions"
   ```

2. **Update Specific Package**
   ```bash
   # Upgrade specific package
   make deps-upgrade-package PACKAGE=torch

   # Test the change
   make test

   # Commit if successful
   git add requirements.txt
   git commit -m "deps: upgrade torch to fix performance issue"
   ```

### Checking Dependencies

```bash
# Check for conflicts and security issues
make deps-check

# View dependency tree
make deps-tree

# Check for outdated packages
make deps-outdated
```

## Make Targets Reference

### Installation Commands

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies (new pip-tools way) |
| `make install-dev` | Install only development dependencies |
| `make install-legacy` | Old installation method (backwards compatibility) |

### Compilation Commands

| Command | Description |
|---------|-------------|
| `make deps-compile` | Compile requirements.in → requirements.txt |
| `make deps-dev-compile` | Compile requirements-dev.in → requirements-dev.txt |
| `make deps-compile-all` | Compile both production and development |

### Synchronization Commands

| Command | Description |
|---------|-------------|
| `make deps-sync` | Install exact versions from requirements.txt |
| `make deps-dev-sync` | Install exact versions from requirements-dev.txt |

### Update Commands

| Command | Description |
|---------|-------------|
| `make deps-upgrade` | Upgrade all deps to latest compatible versions |
| `make deps-upgrade-package PACKAGE=name` | Upgrade specific package |

### Validation Commands

| Command | Description |
|---------|-------------|
| `make deps-check` | Check for conflicts and security issues |
| `make deps-tree` | Show dependency tree |
| `make deps-outdated` | List outdated packages |
| `make deps-constraints-update` | Update constraints.txt (use with caution) |

## Advanced Usage

### Working with Constraints

The `constraints.txt` file contains global version pins that apply to all dependency compilations. This ensures consistency across different requirement files.

```bash
# Add a security constraint
echo "urllib3>=2.0.0,<3.0.0  # Security: Fix CVE-2024-XXXX" >> constraints.txt

# Recompile everything with new constraints
make deps-compile-all

# Test and commit
make test
git add constraints.txt requirements*.txt
git commit -m "security: add urllib3 constraint to fix CVE-2024-XXXX"
```

### Environment Markers

Use environment markers for conditional dependencies:

```python
# In requirements.in or requirements-dev.in

# Python version-specific
numpy>=1.24.0,<2.0.0; python_version < "3.12"
numpy>=1.26.0; python_version >= "3.12"

# Platform-specific
uvloop>=0.17.0; sys_platform != "win32"
pywin32>=227; sys_platform == "win32"

# Extra features
torch>=2.1.0
torchvision>=0.16.0; extra == "vision"
```

### Dependency Extras

Manage optional dependencies through extras:

```python
# In requirements.in
fastapi>=0.104.0
# For CKIP support, use: pip install -e ".[ckip]"

# In pyproject.toml [project.optional-dependencies]
ckip = [
    "ckiptagger>=0.2.1",
    "ckip-transformers>=0.3.4",
]
```

## Troubleshooting

### Common Issues

1. **Compilation Fails with Conflict**
   ```bash
   # Enable verbose output to see conflict details
   pip-compile --verbose requirements.in

   # Check dependency tree for the conflicting packages
   pipdeptree --packages conflicting-package
   ```

2. **Security Vulnerabilities Found**
   ```bash
   # Check which packages have vulnerabilities
   make deps-check

   # Update affected packages
   make deps-upgrade-package PACKAGE=vulnerable-package
   ```

3. **Sync Fails After Pull**
   ```bash
   # Someone updated dependencies, sync your environment
   make deps-dev-sync

   # If you have uncommitted changes, stash first
   git stash
   make deps-dev-sync
   git stash pop
   ```

### Debug Commands

```bash
# Verbose compilation to see what's happening
$(VENV)/bin/pip-compile --verbose requirements.in

# Dry run to test compilation without writing file
$(VENV)/bin/pip-compile --dry-run requirements.in

# Check what would be installed without actually installing
$(VENV)/bin/pip-sync --dry-run requirements.txt
```

## Integration with CI/CD

### GitHub Actions Integration

The project includes automated dependency validation:

- **On PR/Push**: Validates that `.txt` files are up-to-date with `.in` files
- **Weekly**: Scans for security vulnerabilities and outdated packages
- **Scheduled**: Creates issues for available dependency updates

### Pre-commit Hooks

Set up pre-commit hooks to validate dependencies:

```bash
make install-hooks

# This will check before each commit:
# - Requirements files are up-to-date
# - No security vulnerabilities in new dependencies
# - License compatibility
```

## Migration from Legacy pip

If you're migrating from the old pip workflow:

### 1. Backup Current State
```bash
pip freeze > requirements-backup.txt
cp requirements.txt requirements-legacy.txt
```

### 2. Analyze Current Dependencies
```bash
# Use pipdeptree to understand your current dependency graph
pipdeptree > current-tree.txt
```

### 3. Create Source Files
```bash
# Extract top-level dependencies (requires manual review)
# This step requires understanding which packages are direct vs. transitive
pipdeptree --json | python -c "
import json, sys
data = json.load(sys.stdin)
top_level = [pkg for pkg in data if not any(
    pkg['package']['key'] in [dep['key'] for subpkg in data for dep in subpkg.get('dependencies', [])]
)]
for pkg in top_level:
    print(f\"{pkg['package']['package_name']}>={pkg['package']['installed_version'].split('+')[0]}\")
" > requirements-extracted.in
```

### 4. Test New System
```bash
# Test compilation
pip-compile requirements-extracted.in

# Compare with current
diff requirements.txt requirements-extracted.txt

# Test installation in fresh environment
python -m venv test-env
source test-env/bin/activate  # or test-env\Scripts\activate on Windows
pip-sync requirements-extracted.txt
# Run your tests to ensure everything works
```

## Best Practices Summary

✅ **DO**:
- Edit `.in` files, never `.txt` files directly
- Use version ranges with upper bounds: `package>=1.0.0,<2.0.0`
- Commit both `.in` and `.txt` files together
- Run `make deps-check` regularly
- Use environment markers for conditional dependencies
- Document version constraints with comments

❌ **DON'T**:
- Manually edit compiled `.txt` files
- Use exact version pins without upper bounds in `.in` files
- Commit only `.in` files without compiling
- Ignore security warnings from `make deps-check`
- Add development dependencies to `requirements.in`

## Getting Help

- **Internal Documentation**: Check `docs/DEPENDENCY_STRATEGY.md` for advanced topics
- **pip-tools Documentation**: https://pip-tools.readthedocs.io/
- **Issue Tracking**: Use GitHub issues with the `dependencies` label
- **Security Issues**: Follow security reporting guidelines in `SECURITY.md`

---

*This workflow is designed for long-term maintainability and security. Take time to understand the principles rather than just copying commands.*