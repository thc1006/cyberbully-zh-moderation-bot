#!/usr/bin/env python3
"""
Final fix for the remaining specific issues.
"""

import os
import re
import subprocess


def fix_specific_issues():
    """Fix the specific remaining issues."""

    # F841 issues - variables that need to be prefixed with underscore
    f841_fixes = [
        ('tests/integration/bot/test_webhook_processing.py', 98, 'mock_reply',
            '_mock_reply'),
        ('tests/integration/performance/test_performance_benchmarks.py', 83,
            'monitor', '_monitor'),
        ('tests/integration/performance/test_performance_benchmarks.py', 117,
            '_burst_duration', '_burst_duration'),
        ('tests/integration/performance/test_performance_benchmarks.py', 533,
            '_result', '_result'),
        ('tests/integration/run_integration_tests.py', 118, '_process',
            '_process'),
        ('tests/integration/run_integration_tests.py', 161, '_process',
            '_process'),
        ('tests/test_explain_ig.py', 267, '_attention_mask',
            '_attention_mask'),
        ('tests/test_models_advanced.py', 380, '_default_positive',
            '_default_positive'),
        ('tests/test_models_advanced.py', 381, '_high_thresh_positive',
            '_high_thresh_positive'),
        ('tests/test_property_based.py', 109, 'test_text', '_test_text'),
    ]

    for filepath, line_num, old_var, new_var in f841_fixes:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if line_num <= len(lines):
                    line = lines[line_num - 1]
                    if old_var in line and '=' in line and old_var != new_var:
                        # Replace the variable name
                        new_line = re.sub(
                            rf'\b{re.escape(old_var)}\b',
                            new_var,
                            line
                        )
                        if new_line != line:
                            lines[line_num - 1] = new_line
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.writelines(lines)
                            print(f"Fixed F841 in {filepath}:{line"
                                "_num} ({old_var} -> {new_var})")
            except Exception as e:
                print(f"Error fixing F841 in {filepath}: {e}")

    # F541 issue - f-string without placeholders
    f541_fixes = [
        ('tests/test_explain_ig.py', 400),
    ]

    for filepath, line_num in f541_fixes:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if line_num <= len(lines):
                    line = lines[line_num - 1]
                    if 'f"' in line and '{' not in line:
                        new_line = line.replace('f"', '"')
                        if new_line != line:
                            lines[line_num - 1] = new_line
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.writelines(lines)
                            print(f"Fixed F541 in {filepath}:{line_num}")
            except Exception as e:
                print(f"Error fixing F541 in {filepath}: {e}")


def main():
    """Main function."""
    print("Applying final specific fixes...")
    fix_specific_issues()

    print("\nRunning final validation...")
    try:
        result = subprocess.run([
            'python', '-m', 'flake8', '--select=W292,F841,F541,E302,E305,E402,E301,F811', '--statistics'
        ], capture_output=True, text=True, cwd='.')

        if result.stdout.strip():
            print("Remaining issues:")
            print(result.stdout)
        else:
            print("All target issues fixed!")
    except Exception as e:
        print(f"Error running validation: {e}")


if __name__ == '__main__':
    main()
