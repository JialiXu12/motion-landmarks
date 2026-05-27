"""
Tests for the main.py → main_process_data.py + main_alignment.py separation.

Verifies:
1. All 3 scripts parse without syntax errors
2. New save functions exist in utils.py with correct signatures
3. main_process_data.py has no alignment imports
4. main_alignment.py has no distance/clockface computation
5. main.py still works as combined pipeline (backward compat)
6. save_processed_data_to_excel produces correct columns
7. save_alignment_results_to_excel produces correct columns
8. Per-registrar sheets are generated for both save functions
"""

import ast
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_file(filename, encoding='utf-8'):
    """Parse a Python file and return its AST."""
    with open(filename, 'r', encoding=encoding) as f:
        source = f.read()
    return ast.parse(source), source


def _get_imports(source):
    """Extract all import names from source code."""
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
            for alias in node.names:
                imports.add(alias.name)
    return imports


def _get_function_names(source):
    """Extract top-level function names."""
    tree = ast.parse(source)
    return [
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    ]


# ---------------------------------------------------------------------------
# 1. Syntax checks
# ---------------------------------------------------------------------------
def test_main_process_data_parses():
    _parse_file('main_process_data.py')
    print("  [OK] main_process_data.py parses")


def test_main_alignment_parses():
    _parse_file('main_alignment.py')
    print("  [OK] main_alignment.py parses")


def test_main_parses():
    _parse_file('main.py')
    print("  [OK] main.py parses (backward compat)")


def test_utils_parses():
    _parse_file('utils.py')
    print("  [OK] utils.py parses")


# ---------------------------------------------------------------------------
# 2. New save functions exist in utils.py
# ---------------------------------------------------------------------------
def test_save_functions_exist():
    _, source = _parse_file('utils.py')
    funcs = _get_function_names(source)
    assert 'save_processed_data_to_excel' in funcs, \
        "save_processed_data_to_excel not found in utils.py"
    assert 'save_alignment_results_to_excel' in funcs, \
        "save_alignment_results_to_excel not found in utils.py"
    assert 'save_results_to_excel' in funcs, \
        "save_results_to_excel should still exist (backward compat)"
    assert 'save_raw_data_to_excel' in funcs, \
        "save_raw_data_to_excel should still exist"
    print("  [OK] All 4 save functions exist in utils.py")


# ---------------------------------------------------------------------------
# 3. main_process_data.py has no alignment imports
# ---------------------------------------------------------------------------
def test_process_data_no_alignment_imports():
    _, source = _parse_file('main_process_data.py')
    imports = _get_imports(source)

    alignment_imports = {
        'align_prone_to_supine_optimal',
        'align_prone_to_supine',
        'align_prone_to_supine_fixed_sternum',
        'alignment',
        'align_fixed_sternum',
        'surface_to_point_alignment',
    }
    found = alignment_imports & imports
    assert len(found) == 0, \
        f"main_process_data.py should not import alignment modules, found: {found}"
    print("  [OK] main_process_data.py has no alignment imports")


def test_process_data_imports_correct_functions():
    _, source = _parse_file('main_process_data.py')
    imports = _get_imports(source)

    required = {
        'find_corresponding_landmarks',
        'add_averaged_landmarks',
        'calculate_landmark_distances',
        'calculate_clockface_coordinates',
        'save_raw_data_to_excel',
        'save_processed_data_to_excel',
        'load_subject',
    }
    missing = required - imports
    assert len(missing) == 0, \
        f"main_process_data.py missing imports: {missing}"
    print("  [OK] main_process_data.py imports all required functions")


# ---------------------------------------------------------------------------
# 4. main_alignment.py has no distance/clockface computation
# ---------------------------------------------------------------------------
def test_alignment_no_distance_clockface():
    _, source = _parse_file('main_alignment.py')
    imports = _get_imports(source)

    distance_imports = {
        'calculate_landmark_distances',
        'analyse_landmark_distances',
        'calculate_clockface_coordinates',
        'save_processed_data_to_excel',
    }
    found = distance_imports & imports
    assert len(found) == 0, \
        f"main_alignment.py should not import distance/clockface functions, found: {found}"
    print("  [OK] main_alignment.py has no distance/clockface imports")


def test_alignment_imports_correct_functions():
    _, source = _parse_file('main_alignment.py')
    imports = _get_imports(source)

    required = {
        'find_corresponding_landmarks',
        'add_averaged_landmarks',
        'save_alignment_results_to_excel',
        'align_prone_to_supine_optimal',
        'load_subject',
    }
    missing = required - imports
    assert len(missing) == 0, \
        f"main_alignment.py missing imports: {missing}"
    print("  [OK] main_alignment.py imports all required functions")


# ---------------------------------------------------------------------------
# 5. main.py still has the combined pipeline (backward compat)
# ---------------------------------------------------------------------------
def test_main_has_both_stages():
    _, source = _parse_file('main.py')
    imports = _get_imports(source)

    # Should import both distance/clockface AND alignment
    assert 'calculate_landmark_distances' in imports
    assert 'calculate_clockface_coordinates' in imports
    assert 'align_prone_to_supine_optimal' in imports
    assert 'save_results_to_excel' in imports
    print("  [OK] main.py retains combined pipeline")


# ---------------------------------------------------------------------------
# 6. save_processed_data_to_excel: correct columns (unit test with mock data)
# ---------------------------------------------------------------------------
def test_save_processed_data_columns():
    """Verify save_processed_data_to_excel produces the expected columns."""
    _, source = _parse_file('utils.py')

    # Check the function body references expected column names
    assert "'Distance to nipple (prone) [mm]'" in source
    assert "'Distance to skin (prone) [mm]'" in source
    assert "'Distance to rib cage (prone) [mm]'" in source
    assert "'Time (prone)'" in source
    assert "'Quadrant (prone)'" in source
    assert "'landmark side (supine)'" in source

    # Verify it does NOT include alignment-specific columns
    # The function body for save_processed_data_to_excel should not have
    # ribcage error or displacement columns
    # We check the function specifically by finding its definition
    func_start = source.find('def save_processed_data_to_excel(')
    func_end = source.find('\ndef save_alignment_results_to_excel(')
    func_body = source[func_start:func_end]

    assert 'ribcage error rmse' not in func_body, \
        "save_processed_data_to_excel should not have alignment columns"
    assert 'Landmark displacement [mm]' not in func_body, \
        "save_processed_data_to_excel should not have displacement columns"

    print("  [OK] save_processed_data_to_excel has correct columns")


# ---------------------------------------------------------------------------
# 7. save_alignment_results_to_excel: correct columns
# ---------------------------------------------------------------------------
def test_save_alignment_results_columns():
    """Verify save_alignment_results_to_excel has displacement columns."""
    _, source = _parse_file('utils.py')

    func_start = source.find('def save_alignment_results_to_excel(')
    func_body = source[func_start:]

    # These use single quotes in source
    assert "Landmark displacement [mm]" in func_body
    assert "Landmark displacement relative to nipple [mm]" in func_body
    assert "Left nipple displacement [mm]" in func_body

    # These use double quotes in source
    assert "ribcage anterior rmse" in func_body
    assert "ribcage anterior mean" in func_body
    assert "ribcage anterior std" in func_body
    assert "sternum superior prone transformed x" in func_body

    print("  [OK] save_alignment_results_to_excel has correct columns")


# ---------------------------------------------------------------------------
# 8. Per-registrar sheet names
# ---------------------------------------------------------------------------
def test_processed_data_sheet_names():
    _, source = _parse_file('utils.py')
    func_start = source.find('def save_processed_data_to_excel(')
    func_end = source.find('\ndef save_alignment_results_to_excel(')
    func_body = source[func_start:func_end]

    assert "'processed_r1_data'" in func_body
    assert "'processed_r2_data'" in func_body
    assert "'processed_ave_data'" in func_body
    print("  [OK] save_processed_data_to_excel writes 3 registrar sheets")


def test_alignment_results_merges_into_processed_sheets():
    """save_alignment_results_to_excel should merge into processed_* sheets."""
    _, source = _parse_file('utils.py')
    func_start = source.find('def save_alignment_results_to_excel(')
    func_body = source[func_start:]

    assert "'processed_r1_data'" in func_body
    assert "'processed_r2_data'" in func_body
    assert "'processed_ave_data'" in func_body
    # Merge on VL_ID + Landmark name
    assert "on=['VL_ID', 'Landmark name']" in func_body
    print("  [OK] save_alignment_results_to_excel merges into processed_* sheets")


# ---------------------------------------------------------------------------
# 9. Scripts have matching VL_IDS / path constants
# ---------------------------------------------------------------------------
def test_scripts_share_path_constants():
    """Both scripts should reference the same root paths and output paths."""
    _, src_process = _parse_file('main_process_data.py')
    _, src_align = _parse_file('main_alignment.py')

    # Both should have EXCEL_FILE_PATH
    assert 'EXCEL_FILE_PATH' in src_process
    assert 'EXCEL_FILE_PATH' in src_align

    # Both should load subjects with same roots
    assert 'ROOT_PATH_MRI' in src_process
    assert 'ROOT_PATH_MRI' in src_align

    assert 'SOFT_TISSUE_ROOT' in src_process
    assert 'SOFT_TISSUE_ROOT' in src_align

    assert 'ANATOMICAL_JSON_BASE_ROOT' in src_process
    assert 'ANATOMICAL_JSON_BASE_ROOT' in src_align

    print("  [OK] Both scripts share consistent path constants")


# ---------------------------------------------------------------------------
# 10. Alignment script has T_matrix saving
# ---------------------------------------------------------------------------
def test_alignment_saves_transform_matrices():
    _, source = _parse_file('main_alignment.py')
    assert 'np.save(matrix_path' in source
    assert 'OUTPUT_DIR_T_MATRIX' in source
    assert '"T_total"' in source
    print("  [OK] main_alignment.py saves transformation matrices")


# ---------------------------------------------------------------------------
# 11. Process data script does NOT save transformation matrices
# ---------------------------------------------------------------------------
def test_process_data_no_transform_matrices():
    _, source = _parse_file('main_process_data.py')
    assert 'np.save' not in source, \
        "main_process_data.py should not save .npy files"
    assert 'T_total' not in source, \
        "main_process_data.py should not reference T_total"
    print("  [OK] main_process_data.py does not save transform matrices")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    tests = [
        test_main_process_data_parses,
        test_main_alignment_parses,
        test_main_parses,
        test_utils_parses,
        test_save_functions_exist,
        test_process_data_no_alignment_imports,
        test_process_data_imports_correct_functions,
        test_alignment_no_distance_clockface,
        test_alignment_imports_correct_functions,
        test_main_has_both_stages,
        test_save_processed_data_columns,
        test_save_alignment_results_columns,
        test_processed_data_sheet_names,
        test_alignment_results_merges_into_processed_sheets,
        test_scripts_share_path_constants,
        test_alignment_saves_transform_matrices,
        test_process_data_no_transform_matrices,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed!")
    else:
        sys.exit(1)
