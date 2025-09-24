#!/usr/bin/env python3
"""
Fix the remaining specific formatting issues identified by flake8.
"""

import os
import re


def fix_e305_issues():
    """Fix E305: Expected 2 blank lines after class or function definition."""
    files_to_fix = [
        ('fix_bare_except.py', 89),
        ('fix_lines.py', 106),
        ('fix_newlines.py', 72),
        ('fix_unused_vars.py', 94),
        ('remove_unused_imports.py', 112),
    ]

    for filepath, line_num in files_to_fix:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Add blank line after the specified line if it's a
                    function/class end
                if line_num <= len(lines):
                    # Insert a blank line after the function/class definition
                    lines.insert(line_num, '\n')

                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    print(f"Added blank line after funct"
                        "ion in {filepath}:{line_num}")
            except Exception as e:
                print(f"Error fixing E305 in {filepath}: {e}")


def fix_f541_issues():
    """Fix F541: f-string is missing placeholders."""
    f_string_fixes = [
        ('remove_unused_imports.py', 110),
        ('tests/integration/performance/test_performance_benchmarks.py', 162),
        ('tests/integration/performance/test_performance_benchmarks.py', 220),
        ('tests/integration/performance/test_performance_benchmarks.py', 440),
        ('tests/integration/performance/test_performance_benchmarks.py', 511),
        ('tests/integration/performance/test_performance_benchmarks.py', 589),
        ('tests/test_explain_ig.py', 400),
    ]

    for filepath, line_num in f_string_fixes:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if line_num <= len(lines):
                    line = lines[line_num - 1]
                    # Look for f-strings without placeholders
                    if (line.strip().startswith('f"') or line.stri"
                        "p().startswith("f'
                        'f"' in line or "f'" in line):
                        if '{' not in line:
                            # Remove f-prefix from strings without placeholders
                            new_line = re.sub(r'f"([^"]*)"', r'"\1"', line)
                            new_line = re.sub(r"f'([^']*)'", r"'\1'", new_line)
                            if new_line != line:
                                lines[line_num - 1] = new_line
                                with open(
                                    filepath,
                                    'w',
                                    encoding='utf-8'
                                ) as f:
                                    f.writelines(lines)
                                print(f"Fixed f-string in {"
                                    "filepath}:{line_num}")
            except Exception as e:
                print(f"Error fixing f-string in {filepath}: {e}")


def fix_f841_unused_vars():
    """Fix F841: local variable assigned but never used."""
    var_fixes = [
        ('tests/integration/bot/test_webhook_processing.py', 98, 'mock_reply',
            '_mock_reply'),
        ('tests/integration/docker/test_docker_integration.py', 331,
            '_compose_file', '_compose_file'),
        ('tests/integration/performance/test_performance_benchmarks.py', 83,
            '_monitor', '_monitor'),
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
    ]

    for filepath, line_num, old_var, new_var in var_fixes:
        if os.path.exists(filepath) and old_var != new_var:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if line_num <= len(lines):
                    line = lines[line_num - 1]
                    if old_var in line and '=' in line:
                        lines[line_num - 1] = line.replace(old_var, new_var, 1)
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.writelines(lines)
                        print(f"Fixed unused variable i"
                            "n {filepath}:{line_num}")
            except Exception as e:
                print(f"Error fixing unused variable in {filepath}: {e}")


def fix_f811_redefinition():
    """Fix F811: redefinition of unused variable."""
    redefinition_fixes = [
        ('tests/test_property_based.py', 109, 'text', 'test_text'),
        ('tests/test_property_based.py', 163, 'text', 'validation_text'),
    ]

    for filepath, line_num, old_var, new_var in redefinition_fixes:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if line_num <= len(lines):
                    line = lines[line_num - 1]
                    # Replace the variable name in assignment/function
                        parameter
                    if old_var in line:
                        lines[line_num - 1] = line.replace(old_var, new_var, 1)
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.writelines(lines)
                        print(f"Fixed variable redefinitio"
                            "n in {filepath}:{line_num}")
            except Exception as e:
                print(f"Error fixing variable redefinition in {filepath}: {e}")


def add_missing_newlines():
    """Fix W292: no newline at end of file."""
    files_to_fix = [
        'scripts/fix_all_formatting.py'
    ]

    for filepath in files_to_fix:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                if not content.endswith('\n'):
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content + '\n')
                    print(f"Added newline at end of {filepath}")
            except Exception as e:
                print(f"Error adding newline to {filepath}: {e}")


def main():
    """Main function to fix all remaining issues."""
    print("Fixing remaining formatting issues...")

    print("\n1. Fixing E305 issues...")
    fix_e305_issues()

    print("\n2. Fixing F541 issues...")
    fix_f541_issues()

    print("\n3. Fixing F841 unused variables...")
    fix_f841_unused_vars()

    print("\n4. Fixing F811 variable redefinitions...")
    fix_f811_redefinition()

    print("\n5. Adding missing newlines...")
    add_missing_newlines()

    print("\nDone!")


if __name__ == '__main__':
    main()
