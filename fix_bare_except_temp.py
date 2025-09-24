#!/usr/bin/env python3
"""
Fix E722 bare except errors by replacing with 'except Exception:'.
"""

import subprocess
import re
/^def get_e722_errors/i
/^def fix_bare_except/i
/^def main/i

def get_e722_errors():
    """Get E722 errors from flake8."""
    try:
        result = subprocess.run([
            'python', '-m', 'flake8', '--select=E722', '.'
        ], capture_output=True, text=True, cwd='.')

        if result.returncode != 0 and not result.stdout:
            return {}

        errors = {}
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            # Format: ./path/file.py:line:col: E722 do not use bare 'except'
            match = re.match(r'([^:]+):(\d+):\d+: E722 do not use bare \'except\'', line)
            if match:
                filepath = match.group(1).replace('.\\', '').replace('./', '')
                line_num = int(match.group(2))

                if filepath not in errors:
                    errors[filepath] = []
                errors[filepath].append(line_num)

        return errors

    except Exception as e:
        print(f"Error getting E722 errors: {e}")
        return {}

def fix_bare_except(filepath, line_nums):
    """Fix bare except statements in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified = False
        for line_num in line_nums:
            if line_num <= len(lines):
                line_idx = line_num - 1
                line = lines[line_idx]

                # Replace 'except:' with 'except Exception:'
                if 'except:' in line and not line.strip().startswith('#'):
                    new_line = line.replace('except:', 'except Exception:')
                    if new_line != line:
                        lines[line_idx] = new_line
                        modified = True
                        print(f"Fixed {filepath}:{line_num} - bare except -> except Exception:")

        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True

        return False

    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    errors = get_e722_errors()

    if not errors:
        print("No E722 errors found")
        return

    print(f"Found E722 errors in {len(errors)} files")

    fixed_files = 0
    for filepath, line_nums in errors.items():
        if fix_bare_except(filepath, line_nums):
            fixed_files += 1

    print(f"Fixed {fixed_files}/{len(errors)} files")

if __name__ == '__main__':
    main()
