"""
Tests for the refactored alignment module structure.

Tests that:
1. All modules parse and import correctly
2. Functions live in the right modules
3. Re-exports work (backward compatibility)
4. Algorithmic functions produce correct results
5. preprocess_for_alignment() works end-to-end with synthetic data
"""

import ast
import sys
import numpy as np


def test_ast_parse():
    """All alignment modules parse without syntax errors."""
    files = {
        'alignment.py': 'utf-8-sig',
        'alignment_utils.py': 'utf-8',
        'alignment_viz.py': 'utf-8',
        'alignment_reporting.py': 'utf-8',
        'alignment_preprocessing.py': 'utf-8',
        'alignment_deprecated.py': 'utf-8',
        'surface_to_point_alignment.py': 'utf-8',
    }
    for filename, enc in files.items():
        with open(filename, 'r', encoding=enc) as f:
            source = f.read()
        try:
            ast.parse(source)
            print(f"  [OK] {filename}")
        except SyntaxError as e:
            print(f"  [FAIL] {filename}: {e}")
            return False
    return True


def test_functions_in_correct_modules():
    """Verify functions are defined in their intended canonical modules."""
    expectations = {
        'alignment_utils.py': [
            'svd_rotation_point_to_point',
            'filter_mutual_region',
            'apply_transform_to_coords',
            'inverse_transform_to_source_frame',
            'get_surface_mesh_coords',
            'get_mesh_elements_2',
            'get_mesh_with_selected_elements',
            'plot_mesh_elements',
            'plot_filter_debug',
        ],
        'alignment_preprocessing.py': [
            'compute_initial_alignment',
            'preprocess_for_alignment',
        ],
        'alignment_reporting.py': [
            'print_alignment_accuracy_report',
            'aggregate_alignment_statistics',
            'print_cohort_alignment_report',
        ],
        'alignment_deprecated.py': [
            'cleanup_spine_region',
            'filter_anterior_by_widest_point',
            'plot_anterior_filter',
            'optimal_sternum_fixed_alignment',
            'optimal_sternum_fixed_alignment_2',
            'selected_point_cloud',
        ],
    }

    all_ok = True
    for filename, expected_funcs in expectations.items():
        enc = 'utf-8-sig' if filename == 'alignment.py' else 'utf-8'
        with open(filename, 'r', encoding=enc) as f:
            source = f.read()
        tree = ast.parse(source)
        defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

        for func_name in expected_funcs:
            if func_name in defined:
                print(f"  [OK] {func_name} in {filename}")
            else:
                print(f"  [FAIL] {func_name} NOT in {filename}")
                all_ok = False
    return all_ok


def test_alignment_is_orchestrator_only():
    """alignment.py should NOT define algorithmic functions — only the orchestrator."""
    with open('alignment.py', 'r', encoding='utf-8-sig') as f:
        source = f.read()
    tree = ast.parse(source)
    defined = {n.name for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and not n.name.startswith('_')}

    # Only align_prone_to_supine_optimal and get_px_coords (internal helper) expected
    unexpected = defined - {'align_prone_to_supine_optimal', 'get_px_coords'}
    if unexpected:
        print(f"  [FAIL] alignment.py defines unexpected functions: {unexpected}")
        return False
    print(f"  [OK] alignment.py defines only: {defined}")
    return True


def test_no_dead_functions_in_alignment():
    """alignment.py should NOT contain the moved functions."""
    with open('alignment.py', 'r', encoding='utf-8-sig') as f:
        source = f.read()
    tree = ast.parse(source)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    moved = {'cleanup_spine_region', 'svd_rotation_point_to_point',
             'optimal_sternum_fixed_alignment', 'optimal_sternum_fixed_alignment_2',
             'selected_point_cloud', 'filter_anterior_by_widest_point'}

    found = defined & moved
    if found:
        print(f"  [FAIL] alignment.py still defines moved functions: {found}")
        return False
    print(f"  [OK] No dead functions in alignment.py")
    return True


def test_svd_rotation():
    """svd_rotation_point_to_point produces a proper rotation matrix."""
    from alignment_utils import svd_rotation_point_to_point

    # Create known rotation (45 deg around Z)
    theta = np.radians(45)
    R_expected = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1],
    ])

    np.random.seed(42)
    P = np.random.randn(50, 3)
    Q = (R_expected @ P.T).T

    R_computed = svd_rotation_point_to_point(P, Q)

    # Should recover the rotation
    error = np.linalg.norm(R_computed - R_expected)
    det = np.linalg.det(R_computed)

    ok = error < 1e-10 and abs(det - 1.0) < 1e-10
    print(f"  [{'OK' if ok else 'FAIL'}] svd_rotation: error={error:.2e}, det={det:.6f}")
    return ok


def test_compute_initial_alignment():
    """compute_initial_alignment aligns one vector onto another."""
    from alignment_preprocessing import compute_initial_alignment

    # SI vectors that differ by ~30 degrees
    prone_si = np.array([0.0, 0.0, -50.0])    # straight down
    supine_si = np.array([10.0, 0.0, -48.0])   # tilted

    R_init = compute_initial_alignment(prone_si, supine_si, verbose=False)

    # Verify it's a proper rotation
    det = np.linalg.det(R_init)
    ortho_err = np.linalg.norm(R_init @ R_init.T - np.eye(3))

    # Verify it aligns the vectors
    a = prone_si / np.linalg.norm(prone_si)
    b = supine_si / np.linalg.norm(supine_si)
    a_rotated = R_init @ a
    alignment_err = np.linalg.norm(a_rotated - b)

    ok = abs(det - 1.0) < 1e-10 and ortho_err < 1e-10 and alignment_err < 1e-10
    print(f"  [{'OK' if ok else 'FAIL'}] compute_initial_alignment: "
          f"det={det:.6f}, ortho_err={ortho_err:.2e}, align_err={alignment_err:.2e}")
    return ok


def test_compute_initial_alignment_parallel():
    """compute_initial_alignment handles parallel vectors."""
    from alignment_preprocessing import compute_initial_alignment

    v = np.array([0.0, 0.0, -50.0])
    R = compute_initial_alignment(v, v, verbose=False)

    ok = np.allclose(R, np.eye(3))
    print(f"  [{'OK' if ok else 'FAIL'}] compute_initial_alignment (parallel): identity={ok}")
    return ok


def test_preprocess_for_alignment_synthetic():
    """preprocess_for_alignment works with synthetic point cloud data."""
    from alignment_preprocessing import preprocess_for_alignment

    np.random.seed(42)

    # Create synthetic "mesh" as point cloud (ndarray)
    source_pts = np.random.randn(200, 3) * 50  # prone ribcage
    target_pts = np.random.randn(300, 3) * 50  # supine ribcage

    src_ss = np.array([0.0, 10.0, 100.0])
    tgt_ss = np.array([5.0, 15.0, 95.0])
    src_si = np.array([0.0, 10.0, 50.0])
    tgt_si = np.array([5.0, 15.0, 45.0])

    prep = preprocess_for_alignment(
        mesh=source_pts,
        target_pts=target_pts,
        source_sternum_sup=src_ss,
        target_sternum_sup=tgt_ss,
        source_sternum_inf=src_si,
        target_sternum_inf=tgt_si,
        selected_elements=None,
        mutual_region_padding=20.0,
        verbose=False,
    )

    # Check return dict has expected keys
    expected_keys = {'R_init', 'src_mask', 'target_pts_filtered', 'filter_info'}
    ok_keys = set(prep.keys()) == expected_keys

    # R_init should be a proper rotation
    R_init = prep['R_init']
    det = np.linalg.det(R_init)
    ok_rotation = abs(det - 1.0) < 1e-10

    # src_mask should be boolean array
    ok_mask = (prep['src_mask'] is not None and
               prep['src_mask'].dtype == bool and
               len(prep['src_mask']) == len(source_pts))

    # Filtered target should be <= original
    ok_filter = len(prep['target_pts_filtered']) <= len(target_pts)

    ok = ok_keys and ok_rotation and ok_mask and ok_filter
    print(f"  [{'OK' if ok else 'FAIL'}] preprocess_for_alignment (synthetic): "
          f"keys={ok_keys}, rotation={ok_rotation}, mask={ok_mask}, filter={ok_filter}")
    return ok


def test_preprocess_inferior_trim():
    """preprocess_for_alignment trims inferior points correctly."""
    from alignment_preprocessing import preprocess_for_alignment

    np.random.seed(42)
    # Create target PC with known Z range
    target_pts = np.column_stack([
        np.random.randn(100) * 10,
        np.random.randn(100) * 10,
        np.linspace(0, 100, 100),  # Z from 0 to 100
    ])

    src_ss = np.array([0., 0., 50.])
    tgt_ss = np.array([0., 0., 50.])
    src_si = np.array([0., 0., 30.])
    tgt_si = np.array([0., 0., 30.])

    # Trim 20mm from inferior
    prep = preprocess_for_alignment(
        mesh=np.random.randn(50, 3) * 30,
        target_pts=target_pts,
        source_sternum_sup=src_ss,
        target_sternum_sup=tgt_ss,
        source_sternum_inf=src_si,
        target_sternum_inf=tgt_si,
        pc_inferior_trim=20.0,
        verbose=False,
    )

    # After trimming 20mm from Z=0, all points should have Z > 20
    min_z = prep['target_pts_filtered'][:, 2].min()
    ok = min_z > 20.0 - 1e-6
    n_removed = len(target_pts) - len(prep['target_pts_filtered'])
    print(f"  [{'OK' if ok else 'FAIL'}] inferior trim: min_z={min_z:.1f}, "
          f"removed {n_removed} points")
    return ok


def test_re_exports_from_alignment():
    """Functions re-exported from alignment.py are accessible."""
    with open('alignment.py', 'r', encoding='utf-8-sig') as f:
        source = f.read()

    # Check that re-exports are present in import statements
    re_exported = [
        'svd_rotation_point_to_point',
        'cleanup_spine_region',
        'optimal_sternum_fixed_alignment',
        'filter_point_cloud_asymmetric',
        'apply_transform_to_coords',
        'visualize_alignment_errors',
        'print_alignment_accuracy_report',
        'compute_initial_alignment',
    ]

    all_ok = True
    for name in re_exported:
        if name in source:
            print(f"  [OK] {name} referenced in alignment.py")
        else:
            print(f"  [FAIL] {name} NOT found in alignment.py")
            all_ok = False
    return all_ok


def test_re_exports_from_preprocessing():
    """Backward compatibility: filter_anterior_by_widest_point accessible from preprocessing."""
    with open('alignment_preprocessing.py', 'r', encoding='utf-8') as f:
        source = f.read()

    if 'filter_anterior_by_widest_point' in source:
        print(f"  [OK] filter_anterior_by_widest_point re-exported from alignment_preprocessing")
        return True
    else:
        print(f"  [FAIL] filter_anterior_by_widest_point NOT in alignment_preprocessing")
        return False


def test_surface_to_point_align_accepts_new_params():
    """surface_to_point_align signature includes src_mask and R_init."""
    with open('surface_to_point_alignment.py', 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'surface_to_point_align':
            param_names = [arg.arg for arg in node.args.args]
            has_src_mask = 'src_mask' in param_names
            has_R_init = 'R_init' in param_names
            ok = has_src_mask and has_R_init
            print(f"  [{'OK' if ok else 'FAIL'}] surface_to_point_align: "
                  f"src_mask={has_src_mask}, R_init={has_R_init}")
            return ok

    print(f"  [FAIL] surface_to_point_align not found")
    return False


def test_no_circular_imports():
    """Verify no circular import patterns exist in the module graph."""
    import_map = {}
    files = {
        'alignment.py': 'utf-8-sig',
        'alignment_utils.py': 'utf-8',
        'alignment_viz.py': 'utf-8',
        'alignment_reporting.py': 'utf-8',
        'alignment_preprocessing.py': 'utf-8',
        'alignment_deprecated.py': 'utf-8',
        'surface_to_point_alignment.py': 'utf-8',
    }

    alignment_modules = {
        'alignment', 'alignment_utils', 'alignment_viz',
        'alignment_reporting', 'alignment_preprocessing',
        'alignment_deprecated', 'surface_to_point_alignment',
    }

    for filename, enc in files.items():
        module_name = filename.replace('.py', '')
        with open(filename, 'r', encoding=enc) as f:
            source = f.read()
        tree = ast.parse(source)

        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in alignment_modules:
                imports.add(node.module)
        import_map[module_name] = imports

    # Check for cycles using DFS
    def has_cycle(start, current, visited):
        if current in visited:
            return current == start
        visited.add(current)
        for dep in import_map.get(current, []):
            if has_cycle(start, dep, visited.copy()):
                return True
        return False

    all_ok = True
    for module in alignment_modules:
        if has_cycle(module, module, set()):
            print(f"  [FAIL] Circular import detected involving {module}")
            all_ok = False

    if all_ok:
        print(f"  [OK] No circular imports detected")
        for mod, deps in sorted(import_map.items()):
            if deps:
                dep_list = ', '.join(sorted(deps))
                print(f"    {mod} -> {dep_list}")
    return all_ok


def test_alignment_calls_preprocess():
    """alignment.py's orchestrator calls preprocess_for_alignment."""
    with open('alignment.py', 'r', encoding='utf-8-sig') as f:
        source = f.read()

    ok = 'preprocess_for_alignment(' in source
    print(f"  [{'OK' if ok else 'FAIL'}] alignment.py calls preprocess_for_alignment")
    return ok


def test_alignment_passes_R_init_to_icp():
    """alignment.py passes R_init from preprocessing to surface_to_point_align."""
    with open('alignment.py', 'r', encoding='utf-8-sig') as f:
        source = f.read()

    ok_rinit = "R_init=prep['R_init']" in source
    ok_mask = "src_mask=prep['src_mask']" in source
    ok_filtered = "target_pts=prep['target_pts_filtered']" in source
    ok = ok_rinit and ok_mask and ok_filtered

    print(f"  [{'OK' if ok else 'FAIL'}] alignment.py passes preprocessing results: "
          f"R_init={ok_rinit}, src_mask={ok_mask}, filtered={ok_filtered}")
    return ok


if __name__ == "__main__":
    tests = [
        ("AST parse all modules", test_ast_parse),
        ("Functions in correct modules", test_functions_in_correct_modules),
        ("alignment.py is orchestrator only", test_alignment_is_orchestrator_only),
        ("No dead functions in alignment.py", test_no_dead_functions_in_alignment),
        ("svd_rotation_point_to_point", test_svd_rotation),
        ("compute_initial_alignment", test_compute_initial_alignment),
        ("compute_initial_alignment (parallel)", test_compute_initial_alignment_parallel),
        ("preprocess_for_alignment (synthetic)", test_preprocess_for_alignment_synthetic),
        ("preprocess inferior trim", test_preprocess_inferior_trim),
        ("Re-exports from alignment.py", test_re_exports_from_alignment),
        ("Re-exports from preprocessing", test_re_exports_from_preprocessing),
        ("surface_to_point_align new params", test_surface_to_point_align_accepts_new_params),
        ("No circular imports", test_no_circular_imports),
        ("alignment calls preprocess", test_alignment_calls_preprocess),
        ("alignment passes R_init to ICP", test_alignment_passes_R_init_to_icp),
    ]

    print("=" * 60)
    print("REFACTORED MODULE TESTS")
    print("=" * 60)

    results = []
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append((name, False))

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    n_passed = sum(1 for _, ok in results if ok)
    n_total = len(results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n{n_passed}/{n_total} tests passed")

    if n_passed < n_total:
        sys.exit(1)
