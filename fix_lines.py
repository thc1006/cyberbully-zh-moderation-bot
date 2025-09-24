#!/usr/bin/env python3
"""
Quick fix script for E501 line length errors
"""

import sys
from pathlib import Path


def fix_line_length(content: str, max_length: int = 79) -> str:
    """Fix line length issues in Python code."""
    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        if len(line) <= max_length:
            fixed_lines.append(line)
            continue

        # Skip comments and strings that are hard to break
        stripped = line.lstrip()
        if stripped.startswith('#') or stripped.startswith('""
            ""
            fixed_lines.append(line)
            continue

        # Try to fix common patterns
        indent = len(line) - len(line.lstrip())
        indent_str = ' ' * indent

        # Function calls with long argument lists
        if '(' in line and ')' in line and 'def ' not in line:
            # Find the opening parenthesis
            paren_pos = line.find('(')
            if paren_pos > 0:
                before_paren = line[:paren_pos+1]
                after_paren = line[paren_pos+1:]

                if len(before_paren) + 4 < max_length:
                    # Split at commas if possible
                    if ',' in after_paren and after_paren.count(',') <= 3:
                        parts = after_paren.split(',')
                        if len(parts) > 1:
                            fixed_lines.append(before_paren)
                            for i, part in enumerate(parts[:-1]):
                                fixed_lines.append(
                                    indent_str + '    ' + part.strip() + ',
                                    '
                                )
                            # Last part (usually just the closing paren)
                            last_part = parts[-1].strip()
                            if last_part:
                                fixed_lines.append(indent_str + '    ' +
                                    last_part)
                            else:
                                fixed_lines.append(indent_str + ')')
                            continue

        # String concatenation
        if 'f"' in line and len(line) > max_length:
            # Try to break f-strings
            if line.count('"') >= 2:
                # Simple break at a reasonable point
                mid_point = max_length - 4
                break_point = line.rfind(' ', indent, mid_point)
                if break_point > indent:
                    part1 = line[:break_point].rstrip() + ' \\'
                    part2 = indent_str + '    ' + line[break_point:].lstrip()
                    fixed_lines.append(part1)
                    fixed_lines.append(part2)
                    continue

        # Assignment with long right-hand side
        if ' = ' in line and not line.strip().startswith('def '):
            eq_pos = line.find(' = ')
            if eq_pos > 0 and eq_pos < max_length - 10:
                var_part = line[:eq_pos + 3]
                value_part = line[eq_pos + 3:]
                if len(var_part) + 4 < max_length:
                    fixed_lines.append(var_part + '\\')
                    fixed_lines.append(indent_str + '    ' + value_part)
                    continue

        # Default: just add the line as-is (will still trigger flake8 but won't
            break code)
        fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def fix_file(file_path: Path) -> bool:
    """Fix a single Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        fixed_content = fix_line_length(content)

        # Only write if content changed
        if fixed_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"Fixed: {file_path}")
            return True
        else:
            print(f"No changes needed: {file_path}")
            return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python fix_lines.py <file_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_file() and target.suffix == '.py':
        fix_file(target)
    elif target.is_dir():
        for py_file in target.rglob('*.py'):
            fix_file(py_file)
    else:
        print(f"Invalid target: {target}")
        sys.exit(1)

