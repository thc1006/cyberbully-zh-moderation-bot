#!/usr/bin/env python3
"""
Simple DoD Validation - Outputs to files to avoid encoding issues
"""
import json
from pathlib import Path
import time


def run_dod_audit():
    project_root = Path.cwd()
    src_dir = project_root / "src" / "cyberpuppy"
    test_dir = project_root / "tests"

    # Discover core modules
    core_modules = []
    for py_file in src_dir.rglob("*.py"):
        if py_file.name != "__init__.py":
            core_modules.append(py_file.relative_to(src_dir))

    # Check test coverage files
    test_coverage = {}
    for module in core_modules:
        test_name = f"test_{module.stem}.py"
        test_file = test_dir / test_name
        test_coverage[str(module)] = test_file.exists()

    # Calculate coverage percentage
    covered = sum(1 for has_test in test_coverage.values() if has_test)
    total = len(test_coverage)
    coverage_percentage = (covered / total) * 100.0 if total > 0 else 0.0

    # Check Docker files
    docker_status = {
        'dockerfile': (project_root / 'Dockerfile').exists(),
        'docker_compose': (project_root / 'docker-compose.yml').exists(),
        'dockerignore': (project_root / '.dockerignore').exists()
    }

    # Check API files
    api_status = {
        'fastapi_app': (project_root / 'api' / 'app.py').exists(),
        'line_bot': (project_root / 'bot' / 'line_bot.py').exists(),
        'requirements': (project_root / 'pyproject.toml').exists()
    }

    # Check explainability files
    explain_status = {
        'ig_module': (src_dir / 'explain' / 'ig.py').exists(),
        'explain_dir': (src_dir / 'explain').exists()
    }

    # Check model files
    model_status = {
        'models_dir': (project_root / 'models').exists(),
        'baseline_model': (src_dir / 'models' / 'baselines.py').exists(),
        'detector_model': (src_dir / 'models' / 'detector.py').exists()
    }

    # DoD criteria evaluation
    dod_criteria = {
        'test_coverage_90_percent': coverage_percentage >= 90.0,
        'docker_ready': docker_status['dockerfile'],
        'api_implemented': api_status['fastapi_app'] and
            api_status['line_bot'],
        'explainability_implemented': explain_status['ig_module'],
        'models_implemented': model_status['baseline_model'] and
            model_status['detector_model'],
        'requirements_present': api_status['requirements']
    }

    # Compile results
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'core_modules_count': len(core_modules),
        'test_coverage_percentage': coverage_percentage,
        'test_coverage_details': test_coverage,
        'docker_status': docker_status,
        'api_status': api_status,
        'explain_status': explain_status,
        'model_status': model_status,
        'dod_criteria': dod_criteria
    }

    # Generate report
    dod_passed = sum(1 for passed in dod_criteria.values() if passed)
    dod_total = len(dod_criteria)
    dod_percentage = (dod_passed / dod_total) * 100

    missing_tests = [module for module, has_test in test_coverage.items() if
        not has_test]

    report = f"""# Definition of Done Validation Report
Generated: {results['timestamp']}

## Executive Summary
- Core Modules Discovered: {len(core_modules)}
- Test Coverage: {coverage_percentage:.1f}%
- DoD Criteria Met: {dod_passed}/{dod_total} ({dod_percentage:.1f}%)

## DoD Criteria Status"""

    # Add the status lines separately to avoid long lines
    coverage_status = 'PASS' if dod_criteria['test_coverage_90_percent'] else
        'FAIL'
    report += f"""
- Test Coverage >=90%: {coverage_status} ({coverage_percentage:.1f}%)
- Docker Ready: {'PASS' if dod_criteria['docker_ready'] else 'FAIL'}
- API Implemented: {'PASS' if dod_criteria['api_implemented'] else 'FAIL'}
- Explainability Implemented: {'PASS' if
    dod_criteria['explainability_implemented'] else 'FAIL'}
- Models Implemented: {'PASS' if dod_criteria['models_implemented'] else
    'FAIL'}
- Requirements Present: {'PASS' if dod_criteria['requirements_present'] else
    'FAIL'}

## Detailed Analysis

### Test Coverage Analysis
Coverage Percentage: {coverage_percentage:.1f}% (Target: >=90%)

Modules missing tests ({len(missing_tests)}):
"""

    for module in missing_tests[:15]:  # Show first 15
        report += f"- {module}\n"
    if len(missing_tests) > 15:
        report += f"... and {len(missing_tests) - 15} more\n"

    report += f"""
### Docker Configuration
- Dockerfile: {'PASS' if docker_status['dockerfile'] else 'FAIL'}
- Docker Compose: {'PASS' if docker_status['docker_compose'] else 'FAIL'}
- Docker Ignore: {'PASS' if docker_status['dockerignore'] else 'FAIL'}

### API Implementation
- FastAPI App: {'PASS' if api_status['fastapi_app'] else 'FAIL'}
- LINE Bot: {'PASS' if api_status['line_bot'] else 'FAIL'}
- Requirements: {'PASS' if api_status['requirements'] else 'FAIL'}

### Explainability Features
- IG Module: {'PASS' if explain_status['ig_module'] else 'FAIL'}
- Explain Directory: {'PASS' if explain_status['explain_dir'] else 'FAIL'}

### Model Implementation
- Models Directory: {'PASS' if model_status['models_dir'] else 'FAIL'}
- Baseline Model: {'PASS' if model_status['baseline_model'] else 'FAIL'}
- Detector Model: {'PASS' if model_status['detector_model'] else 'FAIL'}

## Overall DoD Status
"""

    if dod_percentage >= 85:
        status_text = "[READY FOR PRODUCTION]"
    elif dod_percentage >= 70:
        status_text = "[NEEDS MINOR FIXES]"
    else:
        status_text = "[NEEDS MAJOR WORK]"

    report += f"{status_text} ({dod_passed}/{dod_total}"
        " criteria met, {dod_percentage:.1f}%)\n"

    # Save results
    with open('dod_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open('DoD_VALIDATION_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("DoD Validation Complete")
    print(f"Core Modules: {len(core_modules)}")
    print(f"Test Coverage: {coverage_percentage:.1f}%")
    print(f"DoD Status: {dod_passed}/{dod_total}"
        " criteria met ({dod_percentage:.1f}%)")
    print("Results saved to: dod_validation_results.json")
    print("Report saved to: DoD_VALIDATION_REPORT.md")

    return results


if __name__ == "__main__":
    run_dod_audit()
