#!/usr/bin/env python3
"""
Fix W292 missing newline errors automatically.
"""

import subprocess
import re


def get_w292_errors():
    """Get W292 errors from flake8."""
    try:
        result = subprocess.run([
            'python', '-m', 'flake8', '--select=W292', '.'
        ], capture_output=True, text=True, cwd='.')

        if result.returncode != 0 and not result.stdout:
            return []

        files = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            # Format: ./path/file.py:line:col: W292 no newline at end of file
            match = re.match(r'([^:]+):\d+:\d+: W292', line)
            if match:
                filepath = match.group(1).replace('.\\', '').replace('./', '')
                files.append(filepath)

        return list(set(files))  # Remove duplicates

    except Exception as e:
        print(f"Error getting W292 errors: {e}")
        return []


def fix_newline(filepath):
    """Add newline to end of file if missing."""
    try:
        with open(filepath, 'rb') as f:
            content = f.read()

        if content and not content.endswith(b'\n'):
            with open(filepath, 'ab') as f:
                f.write(b'\n')
            print(f"Fixed: {filepath}")
            return True

        return False

    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


def main():
    files = get_w292_errors()

    if not files:
        print("No W292 errors found")
        return

    print(f"Found {len(files)} files with W292 errors")

    fixed = 0
    for filepath in files:
        if fix_newline(filepath):
            fixed += 1

    print(f"Fixed {fixed}/{len(files)} files")


if __name__ == '__main__':

    main()

