#!/usr/bin/env python3
"""
Remove unused imports based on flake8 F401 errors.
"""

import re
import subprocess


def get_unused_imports():
    """Get unused imports from flake8."""
    try:
        result = subprocess.run([
            'python', '-m', 'flake8', '--select=F401', '.'
        ], capture_output=True, text=True, cwd='.')

        if result.returncode != 0 and not result.stdout:
            print("No F401 errors found or flake8 failed")
            return {}

        # Parse flake8 output
        unused_imports = {}
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            # Format: ./path/file.py:line:col: F401 'module' imported but
                unused
            match = re.match(r'([^:]+):(\d+):\d+: F401 [\'"](["
                "^\'"]+)[\'
            if match:
                filepath, line_num, import_name = match.groups()
                filepath = filepath.replace('.\\', '').replace('./', '')
                if filepath not in unused_imports:
                    unused_imports[filepath] = []
                unused_imports[filepath].append((int(line_num), import_name))

        return unused_imports

    except Exception as e:
        print(f"Error getting unused imports: {e}")
        return {}


def remove_unused_imports_from_file(filepath, unused_imports):
    """Remove unused imports from a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Sort by line number (descending) so we can remove from bottom to top
        unused_imports.sort(key=lambda x: x[0], reverse=True)

        modified = False
        for line_num, import_name in unused_imports:
            if line_num <= len(lines):
                line_idx = line_num - 1  # Convert to 0-based index
                line = lines[line_idx]

                # Check if this is really an import line with the specified
                    module
                if 'import' in line and import_name in line:
                    # Handle different import patterns
                    patterns = [
                        f'import {import_name}',
                        f'from {import_name}',
                        f'{import_name},'
                    ]

                    # If it's a standalone import, remove the entire line
                    if any(pattern in line for pattern in patterns[:2]):
                        if line.strip().startswith('import ') or
                            line.strip().startswith('from '):
                            # Check if it's a single import or part of
                                multi-import
                            if ',' not in line or line.count(',') == 1:
                                lines.pop(line_idx)
                                modified = True
                                print(f"Removed line {line_"
                                    "num}: {line.strip()}")
                            else:
                                # It's a multi-import, just remove the specific
                                    import
                                new_line = re.sub(
                                    f',
                                    ?\\s*{re.escape(import_name)}\\s*,
                                    ?',
                                    '',
                                    line
                                )
                                new_line = re.sub(
                                    r',
                                    \s*,
                                    ',
                                    ',
                                    ',
                                    new_line
                                )  # Fix double commas
                                new_line = re.sub(
                                    r'(import|from [^\\s]+)\\s*,
                                    ',
                                    r'\\1 ',
                                    new_line
                                )  # Fix leading comma
                                if new_line != line:
                                    lines[line_idx] = new_line
                                    modified = True
                                    print(f"Modified line {line_num}: {line"
                                        ".strip()} -> {new_line.strip()}")

        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True

        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    unused_imports = get_unused_imports()

    if not unused_imports:
        print("No unused imports found")
        return

    print(f"Found unused imports in {len(unused_imports)} files")

    for filepath, imports in unused_imports.items():
        print(f"Processing {filepath}...")
        if remove_unused_imports_from_file(filepath, imports):
            print(f"  ✓ Fixed {len(imports)} unused imports")
        else:
            print("  - No changes made")

if __name__ == '__main__':

    main()

