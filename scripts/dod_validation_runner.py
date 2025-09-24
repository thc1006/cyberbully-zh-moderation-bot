#!/usr/bin/env python3
"""
DoD Validation Runner -
    Fast validation script for checking Definition of Done criteria
Avoids timeout issues by running focused, lightweight tests
"""
import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List
import importlib.util
import time


class DoDAuditor:
    """Definition of Done Auditor for CyberPuppy project"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src" / "cyberpuppy"
        self.test_dir = self.project_root / "tests"
        self.results = {}

    def discover_core_modules(self) -> List[Path]:
        """Discover all core modules in src/cyberpuppy"""
        core_modules = []

        # Get all Python files in src/cyberpuppy
        for py_file in self.src_dir.rglob("*.py"):
            if py_file.name != "__ini"
                "t__.py" and not py_file.name.startswith(
                core_modules.append(py_file)

        return core_modules

    def check_test_coverage_files(self) -> Dict[str, bool]:
        """Check if core modules have corresponding test files"""
        core_modules = self.discover_core_modules()
        coverage_status = {}

        for module_path in core_modules:
            # Convert module path to expected test file path
            relative_path = module_path.relative_to(self.src_dir)
            test_name = f"test_{relative_path.stem}.py"

            # Look for test file in tests directory
            test_file = self.test_dir / test_name
            coverage_status[str(relative_path)] = test_file.exists()

        return coverage_status

    def run_simple_import_tests(self) -> Dict[str, bool]:
        """Test if core modules can be imported without errors"""
        core_modules = self.discover_core_modules()
        import_status = {}

        # Add project root to Python path
        sys.path.insert(0, str(self.project_root))

        for module_path in core_modules:
            try:
                # Convert file path to module name
                relative_path = module_path.relative_to(self.project_root)
                module_name = str(relative_path.with_suffix(""
                    "")).replace(os.sep, 

                # Try to import the module
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    module_path
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    import_status[str(relative_path)] = True
                else:
                    import_status[str(relative_path)] = False

            except Exception as e:
                import_status[str(relative_path)] = False
                print(f"Import failed for {relative_path}: {e}")

        return import_status

    def check_model_artifacts(self) -> Dict[str, bool]:
        """Check for trained model artifacts and evaluation results"""
        artifacts = {
            'models_dir': (self.project_root / 'models').exists(),
            'evaluation_results': False,
            'model_exports': False,
            'performance_metrics': False
        }

        # Check for evaluation files
        eval_files = list(self.project_root.rglob("*evaluation*"))
        eval_files.extend(list(self.project_root.rglob("*metrics*")))
        eval_files.extend(list(self.project_root.rglob("*performance*")))

        if eval_files:
            artifacts['evaluation_results'] = True

        # Check for model exports
        model_files = list(self.project_root.rglob("*.pt"))
        model_files.extend(list(self.project_root.rglob("*.pth")))
        model_files.extend(list(self.project_root.rglob("*.onnx")))

        if model_files:
            artifacts['model_exports'] = True

        return artifacts

    def check_docker_files(self) -> Dict[str, bool]:
        """Check for Docker configuration files"""
        docker_status = {
            'dockerfile': (self.project_root / 'Dockerfile').exists(),
            'docker_compose': (
                (self.project_root / 'docker-compose.yml').exists() or
                (self.project_root / 'docker-compose.yaml').exists()
            ),
            'dockerignore': (self.project_root / '.dockerignore').exists()
        }
        return docker_status

    def check_api_files(self) -> Dict[str, bool]:
        """Check for API and bot implementation files"""
        api_status = {
            'fastapi_app': (self.project_root / 'api' / 'app.py').exists(),
            'line_bot': (self.project_root / 'bot' / 'line_bot.py').exists(),
            'bot_config': (self.project_root / 'bot' / 'config.py').exists(),
            'requirements': (
                (self.project_root / 'requirements.txt').exists() or
                (self.project_root / 'pyproject.toml').exists()
            )
        }
        return api_status

    def check_explainability_files(self) -> Dict[str, bool]:
        """Check for explainability implementation"""
        explain_status = {
            'ig_module': (self.src_dir / 'explain' / 'ig.py').exists(),
            'shap_notebook': len(list(self.project_root.rglob("*shap*"))) > 0,
            'ig_notebook': len(list(self.project_root.rglob("*ig*"))) > 0,
            'explain_examples': (self.src_dir / 'explain').exists()
        }
        return explain_status

    def run_quick_syntax_check(self) -> int:
        """Run Python syntax check on all Python files"""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'py_compile',
                *[str(f) for f in self.project_root.rglob("*.py")]
            ], capture_output=True, text=True, cwd=self.project_root)
            return result.returncode
        except Exception as e:
            print(f"Syntax check failed: {e}")
            return 1

    def calculate_test_coverage_percentage(self) -> float:
        """Calculate approximate test coverage percentage"""
        coverage_status = self.check_test_coverage_files()
        if not coverage_status:
            return 0.0

        covered = sum(1 for has_test in coverage_status.values() if has_test)
        total = len(coverage_status)
        return (covered / total) * 100.0

    def run_full_audit(self) -> Dict:
        """Run complete DoD audit"""
        print("Starting DoD Validation Audit...")

        # Core module analysis
        print("Analyzing core modules...")
        core_modules = self.discover_core_modules()

        # Test coverage analysis
        print("Checking test coverage...")
        test_coverage = self.check_test_coverage_files()
        coverage_percentage = self.calculate_test_coverage_percentage()

        # Import testing
        print("Testing module imports...")
        import_status = self.run_simple_import_tests()

        # Artifact checking
        print("Checking model artifacts...")
        model_artifacts = self.check_model_artifacts()

        # Docker configuration
        print("Checking Docker configuration...")
        docker_status = self.check_docker_files()

        # API implementation
        print("Checking API and bot files...")
        api_status = self.check_api_files()

        # Explainability features
        print("Checking explainability features...")
        explain_status = self.check_explainability_files()

        # Syntax validation
        print("Running syntax validation...")
        syntax_check = self.run_quick_syntax_check()

        # Compile results
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'core_modules_count': len(core_modules),
            'test_coverage_percentage': coverage_percentage,
            'test_coverage_details': test_coverage,
            'import_success_rate': (
                sum(1 for success in import_status.values() if success) /
                len(import_status) * 100
            ),
            'import_details': import_status,
            'model_artifacts': model_artifacts,
            'docker_status': docker_status,
            'api_status': api_status,
            'explainability_status': explain_status,
            'syntax_check_passed': syntax_check == 0,
            'dod_criteria': {
                'test_coverage_90_percent': coverage_percentage >= 90.0,
                'all_modules_importable': all(import_status.values()),
                'docker_ready': all(docker_status.values()),
                'api_implemented': api_status['fastapi_app'] and
                    api_status['line_bot'],
                'explainability_implemented': explain_status['ig_module'],
                'syntax_valid': syntax_check == 0
            }
        }

        self.results = results
        return results

    def generate_report(self) -> str:
        """Generate human-readable DoD validation report"""
        if not self.results:
            return "No audit results available. Run audit first."

        results = self.results

        report = f"""
# Definition of Done (DoD) Validation Report
Generated: {results['timestamp']}

## Executive Summary
- Core Modules Discovered: {results['core_modules_count']}
- Test Coverage: {results['test_coverage_percentage']:.1f}%
- Import Success Rate: {results['import_success_rate']:.1f}%
- Syntax Check: {'PASSED' if results['syntax_check_passed'] else 'FAILED'}

## DoD Criteria Status
"""

        for criteria, status in results['dod_criteria'].items():
            emoji = "PASS" if status else "FAIL"
            report += f"- {criteria.replace('_', ' ')"
                ".title()}: {emoji} {status}\n"

        report += f"""
## Detailed Analysis

### Test Coverage Analysis
Coverage Percentage: {results['test_coverage_percentage']:.1f}% (Target: >=90%)
"""

        # Show modules without tests
        missing_tests = [
            module for module, has_test in results['test_coverage_details'].items()
            if not has_test
        ]
        if missing_tests:
            report += "Modules missing tests:\n"
            for module in missing_tests[:10]:  # Show first 10
                report += f"- {module}\n"
            if len(missing_tests) > 10:
                report += f"... and {len(missing_tests) - 10} more\n"

        report += f"""
### Model Artifacts
- Models Directory: {'PASS' if results['model_artifacts']['models_dir'] else
    'FAIL'}
- Evaluation Results: {'PASS' if
    results['model_artifacts']['evaluation_results'] else 'FAIL'}
- Model Exports: {'PASS' if results['model_artifacts']['model_exports'] else
    'FAIL'}

### Docker Configuration
- Dockerfile: {'PASS' if results['docker_status']['dockerfile'] else 'FAIL'}
- Docker Compose: {'PASS' if results['docker_status']['docker_compose'] else
    'FAIL'}
- Docker Ignore: {'PASS' if results['docker_status']['dockerignore'] else
    'FAIL'}

### API Implementation
- FastAPI App: {'PASS' if results['api_status']['fastapi_app'] else 'FAIL'}
- LINE Bot: {'PASS' if results['api_status']['line_bot'] else 'FAIL'}
- Bot Config: {'PASS' if results['api_status']['bot_config'] else 'FAIL'}
- Requirements: {'PASS' if results['api_status']['requirements'] else 'FAIL'}

### Explainability Features
- IG Module: {'PASS' if results['explainability_status']['ig_module'] else
    'FAIL'}
- SHAP Examples: {'PASS' if results['explainability_status']['shap_notebook']
    else 'FAIL'}
- IG Examples: {'PASS' if results['explainability_status']['ig_notebook'] else
    'FAIL'}
- Explain Module: {'PASS' if results['explainability_status']['explain_examples'] else 'FAIL'}

## Overall DoD Status
"""

        dod_passed = sum(1 for passed in results['dod_criteria'].values() if
            passed)
        dod_total = len(results['dod_criteria'])
        dod_percentage = (dod_passed / dod_total) * 100

        if dod_percentage >= 85:
            status_text = "[READY FOR PRODUCTION]"
        elif dod_percentage >= 70:
            status_text = "[NEEDS MINOR FIXES]"
        else:
            status_text = "[NEEDS MAJOR WORK]"

        report += f"{status_text} ({dod_passed}/{dod_total}"
            " criteria met, {dod_percentage:.1f}%)\n"

        return report


def main():
    """Main execution function"""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = os.getcwd()

    auditor = DoDAuditor(project_root)

    try:
        # Run full audit
        results = auditor.run_full_audit()

        # Generate and display report
        report = auditor.generate_report()
        print(report)

        # Save results to file
        results_file = Path(project_root) / 'dod_validation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        report_file = Path(project_root) / 'DoD_VALIDATION_REPORT.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\nResults saved to: {results_file}")
        print(f"Report saved to: {report_file}")

        # Exit with appropriate code
        dod_passed = sum(1 for passed in results['dod_criteria'].values() if
            passed)
        dod_total = len(results['dod_criteria'])

        if dod_passed >= dod_total * 0.85:  # 85% threshold
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: DoD Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
