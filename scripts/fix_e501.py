#!/usr/bin/env python3
"""
Automatically fix E501 line length violations by breaking long lines.
"""

import subprocess
import re
import os
from pathlib import Path


def get_e501_errors():
    """Get E501 errors from flake8."""
    try:
        result = subprocess.run([
            'python', '-m', 'flake8', '--select=E501', '.'
        ], capture_output=True, text=True, cwd='.')

        errors = {}
        for line in result.stdout.strip().split('\n'):
            if line.strip() and 'E501' in line:
                match = re.match(r'^(.+):(\d+):\d+: E501', line)
                if match:
                    filepath, line_num = match.groups()
                    filepath = filepath.replace('.\\', '').replace('\\', '/')
                    if filepath not in errors:
                        errors[filepath] = []
                    errors[filepath].append(int(line_num))

        return errors

    except Exception as e:
        print(f"Error getting E501 errors: {e}")
        return {}


def fix_line_length(line, max_length=79):
    """Fix a single line that's too long."""
    if len(line.rstrip()) <= max_length:
        return [line]

    # Get indentation
    indent = len(line) - len(line.lstrip())
    base_indent = line[:indent]
    content = line[indent:].rstrip()

    # Handle imports
    if content.startswith('from ') and ' import ' in content:
        parts = content.split(' import ')
        if len(parts) == 2:
            from_part = parts[0]
            import_part = parts[1]
            if '(' not in import_part:
                return [
                    f"{base_indent}{from_part} import (\n",
                    f"{base_indent}    {import_part}\n",
                    f"{base_indent})\n"
                ]

    # Handle function definitions
    if content.startswith('def ') and '(' in content and ')' in content:
        func_match = re.match(r'(def\s+\w+)\((.*)\)(\s*->.*)?(.*)', content)
        if func_match:
            func_name, params, return_type, rest = func_match.groups()
            return_type = return_type or ''
            rest = rest or ':'

            if len(params) > 40:  # Break long parameter list
                param_parts = [p.strip() for p in params.split(',')]
                lines = [f"{base_indent}{func_name}(\n"]
                for i, param in enumerate(param_parts):
                    comma = ',' if i < len(param_parts) - 1 else ''
                    lines.append(f"{base_indent}    {param}{comma}\n")
                lines.append(f"{base_indent}){return_type}{rest}\n")
                return lines

    # Handle string literals
    if ('"' in content or "'" in content) and content.count('"') >= 2:
        # Try to break string at logical points
        quote_char = '"' if '"' in content else "'"
        if content.count(quote_char) >= 2:
            parts = content.split(quote_char)
            if len(parts) >= 3:
                # Simple string concatenation
                mid_point = len(parts[1]) // 2
                first_part = parts[1][:mid_point]
                second_part = parts[1][mid_point:]

                return [
                    f"{base_indent}{parts[0]}{quote_"
                        "char}{first_part}{quote_char}\n",
                    f"{base_indent}    {quote_char}{sec"
                        "ond_part}{quote_char}{parts[2]}\n"
                ]

    # Handle expressions with operators
    for op in [' and ', ' or ', ' == ', ' != ', ' + ', ' - ', ' * ', ' / ']:
        if op in content and len(content) > max_length:
            parts = content.split(op, 1)
            if len(parts) == 2 and len(parts[0]) < max_length - indent - 10:
                return [
                    f"{base_indent}{parts[0].rstrip()}{op.rstrip()}\n",
                    f"{base_indent}    {parts[1].lstrip()}\n"
                ]

    # Handle function calls with multiple arguments
    if '(' in content and ')' in content and ',' in content:
        paren_start = content.find('(')
        paren_end = content.rfind(')')

        if paren_start > 0 and paren_end > paren_start:
            func_part = content[:paren_start + 1]
            args_part = content[paren_start + 1:paren_end]
            rest_part = content[paren_end:]

            if ',' in args_part:
                args = [arg.strip() for arg in args_part.split(',')]
                if len(args) > 1:
                    lines = [f"{base_indent}{func_part}\n"]
                    for i, arg in enumerate(args):
                        comma = ',' if i < len(args) - 1 else ''
                        lines.append(f"{base_indent}    {arg}{comma}\n")
                    lines.append(f"{base_indent}{rest_part}\n")
                    return lines

    # Fallback: break at the last space before max_length
    if len(line.rstrip()) > max_length:
        break_point = max_length - indent
        while break_point > 30 and content[break_point] != ' ':
            break_point -= 1

        if break_point > 30:
            return [
                f"{base_indent}{content[:break_point].rstrip()}\n",
                f"{base_indent}    {content[break_point:].lstrip()}\n"
            ]

    return [line]


def fix_e501_in_file(filepath, line_nums):
    """Fix E501 violations in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified = False
        new_lines = []

        for i, line in enumerate(lines, 1):
            if i in line_nums:
                fixed_lines = fix_line_length(line)
                new_lines.extend(fixed_lines)
                if len(fixed_lines) > 1 or fixed_lines[0] != line:
                    modified = True
            else:
                new_lines.append(line)

        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            return True

        return False

    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


def main():
    errors = get_e501_errors()

    if not errors:
        print("No E501 errors found")
        return

    print(f"Found E501 errors in {len(errors)} files")

    fixed_files = 0
    for filepath, line_nums in list(errors.items()):  # Fix all remaining files
        print(f"Fixing {filepath} ({len(line_nums)} violations)")
        if fix_e501_in_file(filepath, line_nums):
            fixed_files += 1

    print(f"Fixed {fixed_files} files")

    # Check remaining errors
    remaining = get_e501_errors()
    print(f"Remaining E501 errors: {sum(len(v) for v in remaining.values())}")


if __name__ == '__main__':
    main()
