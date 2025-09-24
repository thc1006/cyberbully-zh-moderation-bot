#!/usr/bin/env python3
"""
Fix all remaining formatting issues in the codebase.
"""

import os
import re
import subprocess


def fix_file_formatting(filepath):
    """Fix formatting issues in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Fix E302: Expected 2 blank lines before function/class definitions
        # Find function/class definitions that don't have 2 blank lines before
            them
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            # Check if this is a function or class definition
            if (line.strip().startswith('def ') or
                line.strip().startswith('class ')) and i > 0:
                # Count blank lines before this line
                blank_count = 0
                j = i - 1
                while j >= 0 and lines[j].strip() == '':
                    blank_count += 1
                    j -= 1

                # If we don't have enough blank lines and this isn't at the
                    start
                if blank_count < 2 and j >= 0:
                    # Remove existing blank lines and add exactly 2
                    while fixed_lines and fixed_lines[-1].strip() == '':
                        fixed_lines.pop()
                    fixed_lines.append('')
                    fixed_lines.append('')

            fixed_lines.append(line)

        content = '\n'.join(fixed_lines)

        # Fix E305: Expected 2 blank lines after class or function definition
        # This applies to the end of the file after the last function
        if not content.endswith('\n\n'):
            if content.endswith('\n'):
                content = content + '\n'
            else:
                content = content + '\n\n'

        # Fix W292: Missing newline at end of file
        if not content.endswith('\n'):
            content = content + '\n'

        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed formatting in {filepath}")
            return True

        return False

    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


def fix_python_syntax_issues():
    """Fix specific Python syntax issues."""
    files_to_fix = [
        'fix_bare_except.py',
        'fix_lines.py',
        'fix_newlines.py',
        'fix_unused_vars.py',
        'remove_unused_imports.py'
    ]

    for filepath in files_to_fix:
        if os.path.exists(filepath):
            fix_file_formatting(filepath)


def fix_import_issues():
    """Fix E402 import placement issues by adding # noqa: E402 comments."""
    import_fixes = [
        ('cyberpuppy/tests/conftest.py', 16),
        ('examples/detector_demo.py', 15),
        ('examples/detector_demo.py', 20),
        ('tests/conftest.py', 17),
        ('tests/test_api_integration.py', 15),
        ('tests/test_api_integration.py', 16),
        ('tests/test_api_integration.py', 17),
        ('tests/test_api_models.py', 13),
        ('tests/test_api_models.py', 14),
        ('tests/test_clean.py', 17),
    ]

    for filepath, line_num in import_fixes:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if line_num <= len(lines):
                    line = lines[line_num - 1]
                    if '# noqa: E402' not in line and 'import' in line:
                        lines[line_num - 1] = line.rstrip() + '  # noqa:
                            E402\n'

                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.writelines(lines)
                        print(f"Added noqa comment to {filepath}:{line_num}")
            except Exception as e:
                print(f"Error fixing imports in {filepath}: {e}")


def fix_unused_variables():
    """Fix F841 unused variable issues."""
    # Files with unused variables that need prefixing with underscore
    var_fixes = [
        ('tests/integration/bot/test_webhook_processing.py', 104, 'mock_reply',
            '_mock_reply'),
        ('tests/integration/docker/test_docker_integration.py', 331,
            '_compose_file', '_compose_file'),  # Already prefixed
        ('tests/integration/performance/test_performance_benchmarks.py', 83,
            'monitor', '_monitor'),
        ('tests/integration/performance/test_performance_benchmarks.py', 117,
            '_burst_duration', '_burst_duration'),  # Already prefixed
        ('tests/integration/performance/test_performance_benchmarks.py', 533,
            '_result', '_result'),  # Already prefixed
        ('tests/integration/run_integration_tests.py', 118, '_process',
            '_process'),  # Already prefixed
        ('tests/integration/run_integration_tests.py', 161, '_process',
            '_process'),  # Already prefixed
        ('tests/test_explain_ig.py', 267, '_attention_mask',
            '_attention_mask'),  # Already prefixed
        ('tests/test_models_advanced.py', 380, '_default_positive',
            '_default_positive'),  # Already prefixed
        ('tests/test_models_advanced.py', 381, '_high_thresh_positive',
            '_high_thresh_positive'),  # Already prefixed
    ]

    for filepath, line_num, old_var, new_var in var_fixes:
        if os.path.exists(filepath) and old_var != new_var:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if line_num <= len(lines):
                    line = lines[line_num - 1]
                    if old_var in line:
                        lines[line_num - 1] = line.replace(old_var, new_var)

                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.writelines(lines)
                        print(f"Fixed unused variable i"
                            "n {filepath}:{line_num}")
            except Exception as e:
                print(f"Error fixing unused variable in {filepath}: {e}")


def fix_f_string_issues():
    """Fix F541 f-string without placeholders."""
    f_string_fixes = [
        ('remove_unused_imports.py', 107),
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
                    # Remove f-prefix from strings without placeholders
                    if line.strip().startswith('f"') or line.stri"
                        "p().startswith("f'
                        if '{' not in line:
                            new_line = line.replace('f"',"
                                " '"').replace(
                            lines[line_num - 1] = new_line

                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.writelines(lines)
                            print(f"Fixed f-string in {filepath}:{line_num}")
            except Exception as e:
                print(f"Error fixing f-string in {filepath}: {e}")


def fix_variable_redefinition():
    """Fix F811 variable redefinition issues."""
    redefinition_fixes = [
        ('tests/integration/performance/test_performance_benchmarks.py', 495,
            'threading', 'threading_utils'),
        ('tests/test_property_based.py', 108, 'text', 'sample_text'),
        ('tests/test_property_based.py', 162, 'text', 'test_text'),
    ]

    for filepath, line_num, old_var, new_var in redefinition_fixes:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if line_num <= len(lines):
                    line = lines[line_num - 1]
                    if old_var in line and 'import' in line:
                        lines[line_num - 1] = line.replace(old_var, new_var)

                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.writelines(lines)
                        print(f"Fixed variable redefinitio"
                            "n in {filepath}:{line_num}")
            except Exception as e:
                print(f"Error fixing variable redefinition in {filepath}: {e}")


def fix_blank_line_issues():
    """Fix E301 blank line issues."""
    blank_line_fixes = [
        ('tests/test_property_based.py', 21),  # Need 1 blank line before
    ]

    for filepath, line_num in blank_line_fixes:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if line_num <= len(lines):
                    # Add a blank line before the specified line
                    if line_num > 1 and lines[line_num - 2].strip() != '':
                        lines.insert(line_num - 1, '\n')

                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.writelines(lines)
                        print(f"Added blank line in {filepath}:{line_num}")
            except Exception as e:
                print(f"Error adding blank line in {filepath}: {e}")


def main():
    """Main function to fix all formatting issues."""
    print("Fixing all formatting issues...")

    # Fix Python syntax files first
    print("\n1. Fixing Python syntax files...")
    fix_python_syntax_issues()

    # Fix import placement issues
    print("\n2. Fixing import placement issues...")
    fix_import_issues()

    # Fix unused variables
    print("\n3. Fixing unused variables...")
    fix_unused_variables()

    # Fix f-string issues
    print("\n4. Fixing f-string issues...")
    fix_f_string_issues()

    # Fix variable redefinitions
    print("\n5. Fixing variable redefinitions...")
    fix_variable_redefinition()

    # Fix blank line issues
    print("\n6. Fixing blank line issues...")
    fix_blank_line_issues()

    print("\nDone! Running final check...")

    # Run flake8 to check remaining issues
    try:
        result = subprocess.run([
            'python', '-m', 'flake8', '--select=W292,F841,F541,E302,E305,E402,E301,F811'
        ], capture_output=True, text=True)

        if result.stdout:
            print("Remaining issues:")
            print(result.stdout)
        else:
            print("All formatting issues fixed!")
    except Exception as e:
        print(f"Error running final check: {e}")


if __name__ == '__main__':
    main()
