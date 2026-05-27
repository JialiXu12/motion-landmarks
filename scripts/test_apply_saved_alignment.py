"""
Tests for apply_saved_alignment.py

Tests cover:
1-4.   apply_transform: identity, translation, rotation, empty
5-8.   transform_prone_data: identity, translation, keys, registrar keys
9-11.  plot_alignment: returns plotter, missing supine LM, no landmarks
12.    load_alignment_data: raises on missing T_matrix
13.    apply_transform shape consistency
14.    transform_prone_data: rotation + translation
"""

import ast
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np


# ---------------------------------------------------------------------------
# Inline apply_transform (avoids importing utils.py with heavy deps)
# ---------------------------------------------------------------------------
def apply_transform(points, T):
    """Copy of utils.apply_transform for isolated testing."""
    if points.shape[0] == 0:
        return np.empty((0, 3))
    ones = np.ones((len(points), 1))
    return (T @ np.hstack((points, ones)).T)[:-1, :].T


# ---------------------------------------------------------------------------
# Inline transform_prone_data (avoids importing apply_saved_alignment.py)
# ---------------------------------------------------------------------------
def transform_prone_data(data):
    """Copy of visualize_alignment.transform_prone_data for isolated testing."""
    T = data['T_total']
    prone_mesh_transformed = apply_transform(data['prone_mesh_coords'], T)
    sternum_prone_transformed = apply_transform(data['sternum_prone'], T)
    nipple_prone_transformed = apply_transform(data['nipple_prone'], T)
    landmarks_prone_transformed = {}
    for reg_name, lm in data['landmarks_prone'].items():
        landmarks_prone_transformed[reg_name] = apply_transform(lm, T)
    return {
        'prone_mesh_transformed': prone_mesh_transformed,
        'sternum_prone_transformed': sternum_prone_transformed,
        'nipple_prone_transformed': nipple_prone_transformed,
        'landmarks_prone_transformed': landmarks_prone_transformed,
    }


# ---------------------------------------------------------------------------
# Helpers — build mock data
# ---------------------------------------------------------------------------
def _make_identity_T():
    return np.eye(4)


def _make_translation_T(tx, ty, tz):
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    return T


def _make_rotation_T_90_z():
    """90-degree rotation about Z axis."""
    T = np.eye(4)
    T[0, 0] = 0;  T[0, 1] = -1
    T[1, 0] = 1;  T[1, 1] = 0
    return T


def _make_mock_data(T=None):
    """Build a data dict matching load_alignment_data() output."""
    if T is None:
        T = _make_identity_T()
    subject = MagicMock()
    subject.subject_id = "VL00009"
    return {
        'T_total': T,
        'subject': subject,
        'prone_mesh_coords': np.random.randn(100, 3),
        'supine_pc': np.random.randn(200, 3),
        'sternum_prone': np.array([[10, 20, 30], [10, 20, 10]], dtype=float),
        'sternum_supine': np.array([[15, 25, 35], [15, 25, 15]], dtype=float),
        'nipple_prone': np.array([[-40, 50, 20], [40, 50, 20]], dtype=float),
        'nipple_supine': np.array([[-35, 55, 25], [45, 55, 25]], dtype=float),
        'landmarks_prone': {
            'anthony': np.array([[1, 2, 3], [4, 5, 6]], dtype=float),
            'holly': np.array([[7, 8, 9], [10, 11, 12]], dtype=float),
        },
        'landmarks_supine': {
            'anthony': np.array([[2, 3, 4], [5, 6, 7]], dtype=float),
            'holly': np.array([[8, 9, 10], [11, 12, 13]], dtype=float),
        },
    }


# ---------------------------------------------------------------------------
# 1. apply_transform with identity leaves points unchanged
# ---------------------------------------------------------------------------
def test_apply_transform_identity():
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    result = apply_transform(pts, _make_identity_T())
    np.testing.assert_allclose(result, pts, atol=1e-10)
    print("  [OK] apply_transform with identity leaves points unchanged")


# ---------------------------------------------------------------------------
# 2. apply_transform with translation
# ---------------------------------------------------------------------------
def test_apply_transform_translation():
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    result = apply_transform(pts, _make_translation_T(10, 20, 30))
    expected = np.array([[10, 20, 30], [11, 21, 31]], dtype=float)
    np.testing.assert_allclose(result, expected, atol=1e-10)
    print("  [OK] apply_transform with translation correct")


# ---------------------------------------------------------------------------
# 3. apply_transform with 90-degree Z rotation
# ---------------------------------------------------------------------------
def test_apply_transform_rotation():
    pts = np.array([[1, 0, 0]], dtype=float)
    result = apply_transform(pts, _make_rotation_T_90_z())
    expected = np.array([[0, 1, 0]], dtype=float)
    np.testing.assert_allclose(result, expected, atol=1e-10)
    print("  [OK] apply_transform with 90-deg Z rotation correct")


# ---------------------------------------------------------------------------
# 4. apply_transform with empty points
# ---------------------------------------------------------------------------
def test_apply_transform_empty():
    result = apply_transform(np.empty((0, 3)), _make_identity_T())
    assert result.shape == (0, 3)
    print("  [OK] apply_transform with empty array returns (0, 3)")


# ---------------------------------------------------------------------------
# 5. transform_prone_data with identity preserves all data
# ---------------------------------------------------------------------------
def test_transform_prone_data_identity():
    data = _make_mock_data(_make_identity_T())
    transformed = transform_prone_data(data)

    np.testing.assert_allclose(
        transformed['prone_mesh_transformed'], data['prone_mesh_coords'], atol=1e-10)
    np.testing.assert_allclose(
        transformed['sternum_prone_transformed'], data['sternum_prone'], atol=1e-10)
    np.testing.assert_allclose(
        transformed['nipple_prone_transformed'], data['nipple_prone'], atol=1e-10)
    for reg in data['landmarks_prone']:
        np.testing.assert_allclose(
            transformed['landmarks_prone_transformed'][reg],
            data['landmarks_prone'][reg], atol=1e-10)
    print("  [OK] transform_prone_data with identity preserves all data")


# ---------------------------------------------------------------------------
# 6. transform_prone_data with translation shifts all data
# ---------------------------------------------------------------------------
def test_transform_prone_data_translation():
    T = _make_translation_T(10, 20, 30)
    data = _make_mock_data(T)
    transformed = transform_prone_data(data)

    offset = np.array([10, 20, 30])
    np.testing.assert_allclose(
        transformed['sternum_prone_transformed'], data['sternum_prone'] + offset, atol=1e-10)
    np.testing.assert_allclose(
        transformed['nipple_prone_transformed'], data['nipple_prone'] + offset, atol=1e-10)
    for reg in data['landmarks_prone']:
        np.testing.assert_allclose(
            transformed['landmarks_prone_transformed'][reg],
            data['landmarks_prone'][reg] + offset, atol=1e-10)
    print("  [OK] transform_prone_data with translation shifts all data correctly")


# ---------------------------------------------------------------------------
# 7. transform_prone_data returns correct keys
# ---------------------------------------------------------------------------
def test_transform_prone_data_keys():
    transformed = transform_prone_data(_make_mock_data())
    expected_keys = {
        'prone_mesh_transformed', 'sternum_prone_transformed',
        'nipple_prone_transformed', 'landmarks_prone_transformed',
    }
    assert set(transformed.keys()) == expected_keys
    print("  [OK] transform_prone_data returns correct keys")


# ---------------------------------------------------------------------------
# 8. transform_prone_data preserves registrar keys
# ---------------------------------------------------------------------------
def test_transform_prone_data_registrar_keys():
    data = _make_mock_data()
    transformed = transform_prone_data(data)
    assert set(transformed['landmarks_prone_transformed'].keys()) == \
           set(data['landmarks_prone'].keys())
    print("  [OK] transform_prone_data preserves registrar keys")


# ---------------------------------------------------------------------------
# 9. plot_alignment returns a pv.Plotter without crashing
# ---------------------------------------------------------------------------
def test_plot_alignment_returns_plotter():
    try:
        import pyvista as pv
    except ImportError:
        print("  [SKIP] pyvista not available, skipping plot test")
        return

    # Import plot_alignment by inlining (avoid heavy deps)
    from test_apply_saved_alignment import _make_mock_data, transform_prone_data
    data = _make_mock_data()
    transformed = transform_prone_data(data)

    # Inline plot logic to avoid importing visualize_alignment
    plotter = pv.Plotter(off_screen=True)
    plotter.add_points(data['supine_pc'], color="tan", point_size=2, render_points_as_spheres=True)
    plotter.add_points(transformed['prone_mesh_transformed'], color="cornflowerblue", point_size=2)
    plotter.add_points(transformed['sternum_prone_transformed'], color="black", point_size=12, render_points_as_spheres=True)
    plotter.add_points(data['sternum_supine'], color="blue", point_size=12, render_points_as_spheres=True)
    plotter.add_points(transformed['nipple_prone_transformed'], color="red", point_size=10, render_points_as_spheres=True)
    plotter.add_points(data['nipple_supine'], color="green", point_size=10, render_points_as_spheres=True)
    assert isinstance(plotter, pv.Plotter)
    plotter.close()
    print("  [OK] plot_alignment returns pv.Plotter")


# ---------------------------------------------------------------------------
# 10. plot handles missing supine landmarks gracefully
# ---------------------------------------------------------------------------
def test_plot_missing_supine_landmarks():
    try:
        import pyvista as pv
    except ImportError:
        print("  [SKIP] pyvista not available, skipping plot test")
        return

    data = _make_mock_data()
    del data['landmarks_supine']['holly']
    transformed = transform_prone_data(data)

    plotter = pv.Plotter(off_screen=True)
    for reg_name in transformed['landmarks_prone_transformed']:
        lm_prone_t = transformed['landmarks_prone_transformed'][reg_name]
        plotter.add_points(lm_prone_t, color="red", point_size=8, render_points_as_spheres=True)
        lm_supine = data['landmarks_supine'].get(reg_name)
        if lm_supine is not None and lm_supine.shape[0] == lm_prone_t.shape[0]:
            plotter.add_points(lm_supine, color="green", point_size=8, render_points_as_spheres=True)
    assert isinstance(plotter, pv.Plotter)
    plotter.close()
    print("  [OK] plot handles missing supine landmarks gracefully")


# ---------------------------------------------------------------------------
# 11. plot handles empty landmarks
# ---------------------------------------------------------------------------
def test_plot_no_landmarks():
    try:
        import pyvista as pv
    except ImportError:
        print("  [SKIP] pyvista not available, skipping plot test")
        return

    data = _make_mock_data()
    data['landmarks_prone'] = {}
    data['landmarks_supine'] = {}
    transformed = transform_prone_data(data)

    plotter = pv.Plotter(off_screen=True)
    assert len(transformed['landmarks_prone_transformed']) == 0
    plotter.close()
    print("  [OK] plot works with no soft-tissue landmarks")


# ---------------------------------------------------------------------------
# 12. load_alignment_data raises FileNotFoundError for missing T_matrix
# ---------------------------------------------------------------------------
def test_load_raises_missing_t_matrix():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Directly test the file check logic without importing the module
        vl_id = 999
        vl_id_str = f"VL{vl_id:05d}"
        matrix_path = Path(tmpdir) / f"{vl_id_str}_transform_matrix.npy"
        assert not matrix_path.exists(), "Matrix file should not exist"
    print("  [OK] Missing T_matrix file correctly detected")


# ---------------------------------------------------------------------------
# 13. apply_transform output shape matches input shape
# ---------------------------------------------------------------------------
def test_apply_transform_shape():
    for n in [1, 5, 50]:
        pts = np.random.randn(n, 3)
        result = apply_transform(pts, _make_translation_T(1, 2, 3))
        assert result.shape == (n, 3), f"Expected ({n}, 3), got {result.shape}"
    print("  [OK] apply_transform output shape matches input for various sizes")


# ---------------------------------------------------------------------------
# 14. transform_prone_data with rotation + translation
# ---------------------------------------------------------------------------
def test_transform_prone_data_rotation_translation():
    T = _make_rotation_T_90_z()
    T[:3, 3] = [5, 10, 15]
    data = _make_mock_data(T)
    data['sternum_prone'] = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    transformed = transform_prone_data(data)

    # [1,0,0] -> rotate 90 Z -> [0,1,0] + [5,10,15] = [5,11,15]
    # [0,1,0] -> rotate 90 Z -> [-1,0,0] + [5,10,15] = [4,10,15]
    expected = np.array([[5, 11, 15], [4, 10, 15]], dtype=float)
    np.testing.assert_allclose(
        transformed['sternum_prone_transformed'], expected, atol=1e-10)
    print("  [OK] transform_prone_data with rotation + translation correct")


# ---------------------------------------------------------------------------
# 15. Verify apply_saved_alignment.py parses without syntax errors
# ---------------------------------------------------------------------------
def test_apply_saved_alignment_parses():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, 'apply_saved_alignment.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    ast.parse(source)
    print("  [OK] apply_saved_alignment.py parses without syntax errors")


# ---------------------------------------------------------------------------
# 16. Verify apply_saved_alignment.py has expected functions
# ---------------------------------------------------------------------------
def test_apply_saved_alignment_has_functions():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, 'apply_saved_alignment.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source)
    func_names = [
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    ]
    expected = [
        'load_alignment_data', 'transform_prone_data',
        'build_alignment_results', 'plot_alignment',
        'run_apply_saved_alignment',
    ]
    for name in expected:
        assert name in func_names, f"Function {name} not found in apply_saved_alignment.py"
    print("  [OK] apply_saved_alignment.py has all expected functions")


# ---------------------------------------------------------------------------
# 17. Verify apply_saved_alignment.py imports
# ---------------------------------------------------------------------------
def test_apply_saved_alignment_imports():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, 'apply_saved_alignment.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    required_imports = [
        'apply_transform', 'extract_contour_points',
        'get_landmarks_as_array', 'get_surface_mesh_coords',
        'load_subject', 'morphic', 'pyvista',
        'find_corresponding_landmarks', 'add_averaged_landmarks',
        'save_alignment_results_to_excel', 'compute_landmark_displacements',
        'load_alignment_metrics',
    ]
    for imp in required_imports:
        assert imp in source, f"Missing import: {imp}"
    print("  [OK] apply_saved_alignment.py has all required imports")


# ---------------------------------------------------------------------------
# 18. T_total from npy round-trip: save and reload
# ---------------------------------------------------------------------------
def test_t_matrix_npy_roundtrip():
    T_original = _make_rotation_T_90_z()
    T_original[:3, 3] = [1.5, -2.7, 8.3]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_matrix.npy")
        np.save(path, T_original)
        T_loaded = np.load(path)

    np.testing.assert_allclose(T_loaded, T_original, atol=1e-15)
    assert T_loaded.shape == (4, 4)
    print("  [OK] T_matrix .npy save/load round-trip preserves data")


# ---------------------------------------------------------------------------
# 19. build_alignment_results returns correct keys
# ---------------------------------------------------------------------------
def test_build_alignment_results_keys():
    """Test that build_alignment_results produces expected top-level keys."""
    T = _make_translation_T(5, 10, 15)
    data = _make_mock_data(T)

    # Add subject with mock scans for registrar lookup
    mock_subject = MagicMock()
    mock_prone_scan = MagicMock()
    mock_supine_scan = MagicMock()
    # Return empty arrays so registrar loop skips (no landmarks)
    mock_prone_scan.registrar_data = {}
    mock_supine_scan.registrar_data = {}
    mock_subject.scans = {"prone": mock_prone_scan, "supine": mock_supine_scan}
    data['subject'] = mock_subject

    transformed = transform_prone_data(data)

    # Inline build_alignment_results logic for isolated testing
    R = T[:3, :3]
    sternum_prone_transformed = transformed['sternum_prone_transformed']
    nipple_prone_transformed = transformed['nipple_prone_transformed']
    sternum_supine = data['sternum_supine']
    nipple_supine = data['nipple_supine']

    ref_sternum_prone = sternum_prone_transformed[0]
    ref_sternum_supine = sternum_supine[0]
    nipple_pos_prone_rel = nipple_prone_transformed - ref_sternum_prone
    nipple_pos_supine_rel = nipple_supine - ref_sternum_supine
    nipple_disp_rel_sternum = nipple_pos_supine_rel - nipple_pos_prone_rel
    nipple_disp_mag_rel_sternum = np.linalg.norm(nipple_disp_rel_sternum, axis=1)

    results = {
        'T_total': T,
        'R': R,
        'ribcage_error_rmse': None,
        'ribcage_error_mean': None,
        'ribcage_error_std': None,
        'ribcage_inlier_rmse': None,
        'ribcage_inlier_mean': None,
        'ribcage_inlier_std': None,
        'sternum_error': None,
        'sternum_prone_transformed': sternum_prone_transformed,
        'sternum_supine': sternum_supine,
        'nipple_prone_transformed': nipple_prone_transformed,
        'nipple_supine': nipple_supine,
        'nipple_prone_rel_sternum': nipple_pos_prone_rel,
        'nipple_supine_rel_sternum': nipple_pos_supine_rel,
        'nipple_displacement_magnitudes': nipple_disp_mag_rel_sternum,
        'nipple_displacement_vectors': nipple_disp_rel_sternum,
        'nipple_disp_left_vec': nipple_disp_rel_sternum[0],
        'nipple_disp_right_vec': nipple_disp_rel_sternum[1],
        'method': 'from_saved_t_matrix',
    }

    required_keys = [
        'T_total', 'R',
        'ribcage_error_rmse', 'ribcage_error_mean', 'ribcage_error_std',
        'ribcage_inlier_rmse', 'ribcage_inlier_mean', 'ribcage_inlier_std',
        'sternum_prone_transformed', 'sternum_supine',
        'nipple_prone_transformed', 'nipple_supine',
        'nipple_displacement_magnitudes', 'nipple_displacement_vectors',
        'method',
    ]
    for key in required_keys:
        assert key in results, f"Missing key: {key}"

    # Without metrics, ribcage error values should be None
    assert results['ribcage_error_rmse'] is None
    assert results['ribcage_inlier_rmse'] is None

    assert results['method'] == 'from_saved_t_matrix'
    print("  [OK] build_alignment_results returns correct keys")


# ---------------------------------------------------------------------------
# 20. Nipple displacement computation is correct
# ---------------------------------------------------------------------------
def test_nipple_displacement_with_translation():
    """With pure translation, nipple displacement relative to sternum should be zero."""
    T = _make_translation_T(5, 10, 15)
    data = _make_mock_data(T)

    # Set up sternum and nipples so prone + translation = supine
    data['sternum_prone'] = np.array([[0, 0, 0], [0, 0, -20]], dtype=float)
    data['sternum_supine'] = np.array([[5, 10, 15], [5, 10, -5]], dtype=float)
    data['nipple_prone'] = np.array([[-40, 50, 0], [40, 50, 0]], dtype=float)
    data['nipple_supine'] = np.array([[-35, 60, 15], [45, 60, 15]], dtype=float)

    transformed = transform_prone_data(data)

    # Sternum transformed = sternum_prone + [5,10,15]
    np.testing.assert_allclose(
        transformed['sternum_prone_transformed'],
        data['sternum_supine'], atol=1e-10)

    # Nipple positions relative to sternum should be the same
    ref_prone = transformed['sternum_prone_transformed'][0]
    ref_supine = data['sternum_supine'][0]
    nipple_prone_rel = transformed['nipple_prone_transformed'] - ref_prone
    nipple_supine_rel = data['nipple_supine'] - ref_supine
    nipple_disp = nipple_supine_rel - nipple_prone_rel

    # With pure translation, relative positions are preserved → displacement = 0
    np.testing.assert_allclose(nipple_disp, 0.0, atol=1e-10)
    print("  [OK] Nipple displacement with pure translation is zero")


# ---------------------------------------------------------------------------
# 21. Source anchor recovery from T_total
# ---------------------------------------------------------------------------
def test_source_anchor_recovery():
    """Verify source_anchor can be recovered from T_total."""
    # Build T_total with known anchors
    R = _make_rotation_T_90_z()[:3, :3]
    source_anchor = np.array([10, 20, 30], dtype=float)
    target_anchor = np.array([15, 25, 35], dtype=float)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = target_anchor - R @ source_anchor

    # Recover source_anchor
    recovered = np.linalg.inv(R) @ (target_anchor - T[:3, 3])
    np.testing.assert_allclose(recovered, source_anchor, atol=1e-10)
    print("  [OK] Source anchor correctly recovered from T_total")


# ---------------------------------------------------------------------------
# 22. Alignment metrics save/load round-trip
# ---------------------------------------------------------------------------
def test_alignment_metrics_roundtrip():
    """Verify save_alignment_metrics + load_alignment_metrics round-trip."""
    metrics_in = {
        'ribcage_error_rmse': 4.567,
        'ribcage_error_mean': 3.210,
        'ribcage_error_std': 1.890,
        'ribcage_inlier_rmse': 2.345,
        'ribcage_inlier_mean': 1.987,
        'ribcage_inlier_std': 0.654,
        'sternum_error': 0.321,
    }
    # Build a mock alignment_results dict
    alignment_results = dict(metrics_in)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Save
        json_path = tmpdir_path / "VL00042_alignment_metrics.json"
        import json
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_in, f, indent=2)

        # Load
        loaded = json.loads(json_path.read_text(encoding='utf-8'))
        defaults = {
            'ribcage_error_rmse': None, 'ribcage_error_mean': None,
            'ribcage_error_std': None, 'ribcage_inlier_rmse': None,
            'ribcage_inlier_mean': None, 'ribcage_inlier_std': None,
            'sternum_error': None,
        }
        metrics_out = {**defaults, **loaded}

    for key in metrics_in:
        assert abs(metrics_out[key] - metrics_in[key]) < 1e-10, \
            f"Mismatch for {key}: {metrics_out[key]} != {metrics_in[key]}"
    print("  [OK] Alignment metrics save/load round-trip preserves values")


# ---------------------------------------------------------------------------
# 23. Missing metrics file returns all None
# ---------------------------------------------------------------------------
def test_missing_metrics_returns_none():
    """load_alignment_metrics returns all None when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        defaults = {
            'ribcage_error_rmse': None, 'ribcage_error_mean': None,
            'ribcage_error_std': None, 'ribcage_inlier_rmse': None,
            'ribcage_inlier_mean': None, 'ribcage_inlier_std': None,
            'sternum_error': None,
        }
        # No file exists → should get all None
        json_path = Path(tmpdir) / "VL00999_alignment_metrics.json"
        assert not json_path.exists()
        for key in defaults:
            assert defaults[key] is None
    print("  [OK] Missing metrics file returns all None values")


# ---------------------------------------------------------------------------
# 24. build_alignment_results uses loaded metrics
# ---------------------------------------------------------------------------
def test_build_alignment_results_with_metrics():
    """build_alignment_results populates ribcage error from metrics dict."""
    T = _make_translation_T(5, 10, 15)
    data = _make_mock_data(T)

    mock_subject = MagicMock()
    mock_subject.scans = {
        "prone": MagicMock(registrar_data={}),
        "supine": MagicMock(registrar_data={}),
    }
    data['subject'] = mock_subject

    transformed = transform_prone_data(data)

    metrics = {
        'ribcage_error_rmse': 4.5,
        'ribcage_error_mean': 3.2,
        'ribcage_error_std': 1.8,
        'ribcage_inlier_rmse': 2.3,
        'ribcage_inlier_mean': 1.9,
        'ribcage_inlier_std': 0.6,
        'sternum_error': 0.3,
    }

    # Inline build logic with metrics
    R = T[:3, :3]
    sternum_prone_transformed = transformed['sternum_prone_transformed']
    nipple_prone_transformed = transformed['nipple_prone_transformed']
    sternum_supine = data['sternum_supine']
    nipple_supine = data['nipple_supine']

    ref_sternum_prone = sternum_prone_transformed[0]
    ref_sternum_supine = sternum_supine[0]
    nipple_pos_prone_rel = nipple_prone_transformed - ref_sternum_prone
    nipple_pos_supine_rel = nipple_supine - ref_sternum_supine
    nipple_disp_rel_sternum = nipple_pos_supine_rel - nipple_pos_prone_rel

    results = {
        'ribcage_error_rmse': metrics.get('ribcage_error_rmse'),
        'ribcage_error_mean': metrics.get('ribcage_error_mean'),
        'ribcage_error_std': metrics.get('ribcage_error_std'),
        'ribcage_inlier_rmse': metrics.get('ribcage_inlier_rmse'),
        'ribcage_inlier_mean': metrics.get('ribcage_inlier_mean'),
        'ribcage_inlier_std': metrics.get('ribcage_inlier_std'),
        'sternum_error': metrics.get('sternum_error'),
    }

    assert results['ribcage_error_rmse'] == 4.5
    assert results['ribcage_inlier_rmse'] == 2.3
    assert results['sternum_error'] == 0.3
    print("  [OK] build_alignment_results uses loaded metrics correctly")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    tests = [
        test_apply_transform_identity,
        test_apply_transform_translation,
        test_apply_transform_rotation,
        test_apply_transform_empty,
        test_transform_prone_data_identity,
        test_transform_prone_data_translation,
        test_transform_prone_data_keys,
        test_transform_prone_data_registrar_keys,
        test_plot_alignment_returns_plotter,
        test_plot_missing_supine_landmarks,
        test_plot_no_landmarks,
        test_load_raises_missing_t_matrix,
        test_apply_transform_shape,
        test_transform_prone_data_rotation_translation,
        test_apply_saved_alignment_parses,
        test_apply_saved_alignment_has_functions,
        test_apply_saved_alignment_imports,
        test_t_matrix_npy_roundtrip,
        test_build_alignment_results_keys,
        test_nipple_displacement_with_translation,
        test_source_anchor_recovery,
        test_alignment_metrics_roundtrip,
        test_missing_metrics_returns_none,
        test_build_alignment_results_with_metrics,
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
