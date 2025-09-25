#!/usr/bin/env python3
"""
Fix F841 unused variable errors by prefixing with underscore.
"""

import subprocess
import re


def get_f841_errors():
    """Get F841 errors from flake8."""
    try:
        result = subprocess.run([
            'python', '-m', 'flake8', '--select=F841', '.'
        ], capture_output=True, text=True, cwd='.')

        if result.returncode != 0 and not result.stdout:
            return {}

        errors = {}
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            # Format: ./path/file.py:line:col: F841 local variable 'var' is
                assigned to but never used
            match = re.match(
                r'([^:]+):(\d+):\d+: F841 local variable \'([^\']+)\' is assigned to but never used',
                line
            )
            if match:
                filepath = match.group(1).replace('.\\', '').replace('./', '')
                line_num = int(match.group(2))
                var_name = match.group(3)

                if filepath not in errors:
                    errors[filepath] = []
                errors[filepath].append((line_num, var_name))

        return errors

    except Exception as e:
        print(f"Error getting F841 errors: {e}")
        return {}


def fix_unused_variable(filepath, errors):
    """Fix unused variables in a file by prefixing with underscore."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified = False
        # Sort by line number (descending) so we can modify from bottom to top
        errors.sort(key=lambda x: x[0], reverse=True)

        for line_num, var_name in errors:
            if line_num <= len(lines):
                line_idx = line_num - 1
                line = lines[line_idx]

                # Check if variable is assigned on this line
                if f'{var_name} =' in line and not
                    line.strip().startswith('#'):
                    # Replace variable name with underscore prefix
                    new_line = line.replace(f'{var_name} =', f'_{var_name} =')
                    if new_line != line:
                        lines[line_idx] = new_line
                        modified = True
                        print(f"Fixed {filepath}:{line_num}"
                            " - {var_name} -> _{var_name}")

        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True

        return False

    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


def main():
    errors = get_f841_errors()

    if not errors:
        print("No F841 errors found")
        return

    print(f"Found F841 errors in {len(errors)} files")

    fixed_files = 0
    for filepath, file_errors in errors.items():
        if fix_unused_variable(filepath, file_errors):
            fixed_files += 1

    print(f"Fixed {fixed_files}/{len(errors)} files")

if __name__ == '__main__':

    main()

