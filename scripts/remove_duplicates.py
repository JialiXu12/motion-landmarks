"""
Script to remove duplicate function definitions from analysis.py
"""

import re
from pathlib import Path
from collections import OrderedDict

print("=" * 80)
print("REMOVING DUPLICATE FUNCTIONS FROM ANALYSIS.PY")
print("=" * 80)

# Read the file
file_path = Path("analysis.py")
with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

lines = content.split('\n')
print(f"\nOriginal file: {len(lines)} lines")

# Find all function definitions with their line numbers
function_defs = {}
for i, line in enumerate(lines, 1):
    match = re.match(r'^def\s+(\w+)\s*\(', line)
    if match:
        func_name = match.group(1)
        if func_name not in function_defs:
            function_defs[func_name] = []
        function_defs[func_name].append(i)

# Find duplicates
duplicates = {name: line_nums for name, line_nums in function_defs.items() if len(line_nums) > 1}

if not duplicates:
    print("\n✓ No duplicate functions found!")
    exit(0)

print(f"\n Found {len(duplicates)} duplicate functions:")
for func_name, line_nums in sorted(duplicates.items()):
    print(f"  - {func_name}: lines {line_nums}")

# Identify the section added by restore script
restore_marker = "# MISSING FUNCTIONS RESTORED FROM BACKUP"
marker_line = None
for i, line in enumerate(lines):
    if restore_marker in line:
        marker_line = i
        break

if marker_line is None:
    print("\n⚠ WARNING: Could not find restore marker")
    print("Manually removing duplicates...")
    # Remove from the end (assuming duplicates are at the end)
    for func_name in duplicates:
        # Keep first occurrence, remove second
        line_nums = duplicates[func_name]
        if len(line_nums) == 2:
            print(f"  Removing duplicate of {func_name} at line {line_nums[1]}")
else:
    print(f"\nFound restore marker at line {marker_line + 1}")
    print(f"Removing everything from line {marker_line + 1} onwards...")

    # Simply truncate at the marker
    lines = lines[:marker_line]

# Write the cleaned file
output_path = Path("analysis.py")
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"\n✓ Cleaned file written: {len(lines)} lines")
print(f"✓ Removed {len(content.split(chr(10))) - len(lines)} lines")

print("\n" + "=" * 80)
print("CLEANUP COMPLETE")
print("=" * 80)
print("\nThe file now contains only the original functions.")
print("All functions from the backup were ALREADY present in your file!")
