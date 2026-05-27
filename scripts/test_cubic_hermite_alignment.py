"""
Integration test for surface_to_point_alignment.py with real cubic Hermite
meshes and supine point clouds.

Requires: morphic, SimpleITK, pyvista, mesh_tools, scipy, numpy
Run from the scripts/ directory with the correct environment activated:
    python test_cubic_hermite_alignment.py
    python -m pytest test_cubic_hermite_alignment.py -v
"""

import sys
import numpy as np
import pytest
from pathlib import Path
from numpy.testing import assert_allclose

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA = PROJECT_ROOT / "test_data"
PRONE_DIR = TEST_DATA / "prone"
SUPINE_DIR = TEST_DATA / "supine"

# Measured anatomical landmarks (sternum superior positions)
ANATOMICAL_JSON_BASE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results")

# Subject ID used for testing (smallest file)
VL_ID = 9
VL_ID_STR = f"VL{VL_ID:05d}"
PRONE_MESH_PATH = PRONE_DIR / f"{VL_ID_STR}_ribcage_prone.mesh"
SUPINE_SEG_PATH = SUPINE_DIR / f"rib_cage_{VL_ID_STR}.nii.gz"

# Skip all tests if dependencies or test data are unavailable
morphic = pytest.importorskip("morphic")
sitk = pytest.importorskip("SimpleITK")
pv = pytest.importorskip("pyvista")

# Add scripts/ to path so project imports work
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import external.breast_metadata_mdv.breast_metadata as breast_metadata
from utils import extract_contour_points
from alignment import (
    get_surface_mesh_coords,
    apply_transform_to_coords,
    visualize_alignment_errors,
)
from surface_to_point_alignment import (
    compute_mesh_points_and_normals,
    surface_to_point_align,
)
from readers import read_anatomical_landmarks


# ---------------------------------------------------------------------------
# Fixtures: load real data once per session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def prone_mesh():
    """Load the real cubic Hermite prone ribcage mesh."""
    if not PRONE_MESH_PATH.exists():
        pytest.skip(f"Test data not found: {PRONE_MESH_PATH}")
    return morphic.Mesh(str(PRONE_MESH_PATH))


@pytest.fixture(scope="session")
def supine_point_cloud():
    """Load the real supine ribcage segmentation as a point cloud."""
    if not SUPINE_SEG_PATH.exists():
        pytest.skip(f"Test data not found: {SUPINE_SEG_PATH}")
    mask = breast_metadata.readNIFTIImage(
        str(SUPINE_SEG_PATH), "RAI", swap_axes=True,
    )
    return extract_contour_points(mask, 20000)


@pytest.fixture(scope="session")
def sternum_positions():
    """
    Load measured sternum superior positions from anatomical JSON files.

    Uses the manually-picked sternal-superior landmark from the JSON
    files at ANATOMICAL_JSON_BASE_ROOT/{position}/landmarks/.
    """
    prone_json = (
        ANATOMICAL_JSON_BASE_ROOT / "prone" / "landmarks"
        / f"{VL_ID_STR}_skeleton_data_prone_t2.json"
    )
    supine_json = (
        ANATOMICAL_JSON_BASE_ROOT / "supine" / "landmarks"
        / f"{VL_ID_STR}_skeleton_data_supine_t2.json"
    )

    # Try fallback paths (combined subfolder)
    if not prone_json.exists():
        prone_json = prone_json.parent / "combined" / prone_json.name
    if not supine_json.exists():
        supine_json = supine_json.parent / "combined" / supine_json.name

    if not prone_json.exists() or not supine_json.exists():
        pytest.skip(
            f"Anatomical JSON not found for {VL_ID_STR}:\n"
            f"  prone: {prone_json}\n"
            f"  supine: {supine_json}"
        )

    anat_prone = read_anatomical_landmarks(prone_json)
    anat_supine = read_anatomical_landmarks(supine_json)

    return anat_prone.sternum_superior, anat_supine.sternum_superior


# ---------------------------------------------------------------------------
# Tests: Cubic Hermite mesh sampling
# ---------------------------------------------------------------------------
class TestCubicHermiteMeshSampling:

    def test_mesh_loads_and_has_elements(self, prone_mesh):
        """Cubic Hermite mesh should load and have elements."""
        n_elem = prone_mesh.elements.size()
        assert n_elem > 0, "Mesh has no elements"
        print(f"  Mesh has {n_elem} elements")

    def test_mesh_basis_is_cubic_hermite(self, prone_mesh):
        """All elements should use H3 (cubic Hermite) basis."""
        for element in prone_mesh.elements:
            assert element.basis == ['H3', 'H3'], (
                f"Element {element.id} has basis {element.basis}, expected ['H3', 'H3']"
            )

    def test_compute_points_and_normals(self, prone_mesh):
        """compute_mesh_points_and_normals should work with cubic Hermite."""
        Xi = prone_mesh.grid(5, method='center')
        pts, nrm = compute_mesh_points_and_normals(prone_mesh, Xi)

        n_expected = prone_mesh.elements.size() * Xi.shape[0]
        assert pts.shape == (n_expected, 3)
        assert nrm.shape == (n_expected, 3)

        # Normals should be unit length
        norms = np.linalg.norm(nrm, axis=1)
        assert_allclose(norms, 1.0, atol=1e-6)

    def test_normals_vary_within_element(self, prone_mesh):
        """
        Unlike linear elements, cubic Hermite normals should vary
        across the parametric domain of a single element.
        """
        Xi = prone_mesh.grid(5, method='center')
        pts, nrm = compute_mesh_points_and_normals(
            prone_mesh, Xi, elems=[0],
        )
        # Check that normals are NOT all identical
        normal_spread = np.std(nrm, axis=0)
        assert np.any(normal_spread > 1e-4), (
            "Cubic Hermite normals should vary within an element, "
            f"but std per axis is {normal_spread}"
        )

    def test_element_selection_reduces_points(self, prone_mesh):
        """Selecting a subset of elements should reduce point count."""
        Xi = prone_mesh.grid(5, method='center')
        n_xi = Xi.shape[0]

        pts_all, _ = compute_mesh_points_and_normals(prone_mesh, Xi)
        pts_sub, _ = compute_mesh_points_and_normals(
            prone_mesh, Xi, elems=[0, 1],
        )
        assert pts_all.shape[0] == prone_mesh.elements.size() * n_xi
        assert pts_sub.shape[0] == 2 * n_xi
        assert pts_sub.shape[0] < pts_all.shape[0]


# ---------------------------------------------------------------------------
# Tests: Full alignment with real data
# ---------------------------------------------------------------------------
class TestCubicHermiteAlignment:

    def test_alignment_runs_without_error(
        self, prone_mesh, supine_point_cloud, sternum_positions,
    ):
        """Full plane-to-point alignment should run on real data."""
        prone_ss, supine_ss = sternum_positions
        R, T, info = surface_to_point_align(
            mesh=prone_mesh,
            target_pts=supine_point_cloud,
            source_sternum_sup=prone_ss,
            target_sternum_sup=supine_ss,
            max_distance=20.0,
            max_iterations=50,
            convergence_threshold=1e-6,
            trim_percentage=0.1,
            res=5,
            verbose=True,
        )
        assert R.shape == (3, 3)
        assert T.shape == (4, 4)
        assert info['iterations'] > 0

    def test_rotation_is_proper(
        self, prone_mesh, supine_point_cloud, sternum_positions,
    ):
        """Resulting R should be orthogonal with det=+1."""
        prone_ss, supine_ss = sternum_positions
        R, _, _ = surface_to_point_align(
            mesh=prone_mesh,
            target_pts=supine_point_cloud,
            source_sternum_sup=prone_ss,
            target_sternum_sup=supine_ss,
            max_distance=20.0,
            max_iterations=50,
            res=5,
        )
        assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
        assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)

    def test_alignment_reduces_rmse(
        self, prone_mesh, supine_point_cloud, sternum_positions,
    ):
        """RMSE should decrease over iterations."""
        prone_ss, supine_ss = sternum_positions
        _, _, info = surface_to_point_align(
            mesh=prone_mesh,
            target_pts=supine_point_cloud,
            source_sternum_sup=prone_ss,
            target_sternum_sup=supine_ss,
            max_distance=20.0,
            max_iterations=50,
            res=5,
        )
        history = info['iteration_history']
        assert len(history) >= 2, "Need at least 2 iterations"
        assert history[-1]['rmse'] <= history[0]['rmse']

    def test_hybrid_objective_runs(
        self, prone_mesh, supine_point_cloud, sternum_positions,
    ):
        """Hybrid plane-to-point + point-to-point should work on real data."""
        prone_ss, supine_ss = sternum_positions
        R, T, info = surface_to_point_align(
            mesh=prone_mesh,
            target_pts=supine_point_cloud,
            source_sternum_sup=prone_ss,
            target_sternum_sup=supine_ss,
            max_distance=20.0,
            max_iterations=50,
            res=5,
            point_to_point_weight=0.3,
        )
        assert R.shape == (3, 3)
        assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
        assert info['iterations'] > 0


# ---------------------------------------------------------------------------
# Visualization (run only from __main__, not from pytest)
# ---------------------------------------------------------------------------
def visualize_alignment_result():
    """
    Run the full alignment on real cubic Hermite data and visualize
    the before/after error between prone mesh and supine point cloud.
    """
    print(f"\n{'='*60}")
    print(f"CUBIC HERMITE ALIGNMENT TEST - {VL_ID_STR}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if not PRONE_MESH_PATH.exists() or not SUPINE_SEG_PATH.exists():
        print(f"ERROR: Test data not found at {TEST_DATA}")
        return

    prone_mesh = morphic.Mesh(str(PRONE_MESH_PATH))
    mask = breast_metadata.readNIFTIImage(
        str(SUPINE_SEG_PATH), "RAI", swap_axes=True,
    )
    supine_pc = extract_contour_points(mask, 20000)

    print(f"  Prone mesh: {prone_mesh.elements.size()} cubic Hermite elements")
    print(f"  Supine point cloud: {supine_pc.shape[0]} points")

    # Load measured sternum positions from anatomical JSON files
    prone_json = (
        ANATOMICAL_JSON_BASE_ROOT / "prone" / "landmarks"
        / f"{VL_ID_STR}_skeleton_data_prone_t2.json"
    )
    supine_json = (
        ANATOMICAL_JSON_BASE_ROOT / "supine" / "landmarks"
        / f"{VL_ID_STR}_skeleton_data_supine_t2.json"
    )
    if not prone_json.exists():
        prone_json = prone_json.parent / "combined" / prone_json.name
    if not supine_json.exists():
        supine_json = supine_json.parent / "combined" / supine_json.name

    if not prone_json.exists() or not supine_json.exists():
        print(f"ERROR: Anatomical JSON not found for {VL_ID_STR}")
        return

    anat_prone = read_anatomical_landmarks(prone_json)
    anat_supine = read_anatomical_landmarks(supine_json)
    prone_ss = anat_prone.sternum_superior
    supine_ss = anat_supine.sternum_superior
    print(f"  Measured prone sternum:  {prone_ss}")
    print(f"  Measured supine sternum: {supine_ss}")

    # ------------------------------------------------------------------
    # 2. Compute BEFORE error (no alignment, just sternum shift)
    # ------------------------------------------------------------------
    prone_coords = get_surface_mesh_coords(prone_mesh, res=26)
    R_identity = np.eye(3)
    prone_before = apply_transform_to_coords(
        prone_coords, R_identity, prone_ss, supine_ss,
    )
    from scipy.spatial import cKDTree
    tree_supine = cKDTree(supine_pc)
    dists_before, _ = tree_supine.query(prone_before)
    rmse_before = np.sqrt(np.mean(dists_before ** 2))
    print(f"\n  BEFORE alignment RMSE: {rmse_before:.2f} mm")

    # ------------------------------------------------------------------
    # 3. Run alignment
    # ------------------------------------------------------------------
    print(f"\n--- Running plane-to-point ICP (hybrid w=0.3) ---")
    R, T, info = surface_to_point_align(
        mesh=prone_mesh,
        target_pts=supine_pc,
        source_sternum_sup=prone_ss,
        target_sternum_sup=supine_ss,
        max_distance=20.0,
        max_iterations=200,
        convergence_threshold=1e-6,
        trim_percentage=0.1,
        res=10,
        verbose=True,
        point_to_point_weight=0.3,
    )

    # ------------------------------------------------------------------
    # 4. Compute AFTER error
    # ------------------------------------------------------------------
    prone_after = apply_transform_to_coords(
        prone_coords, R, prone_ss, supine_ss,
    )
    dists_after, idx_after = tree_supine.query(prone_after)
    rmse_after = np.sqrt(np.mean(dists_after ** 2))

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  BEFORE alignment RMSE: {rmse_before:.2f} mm")
    print(f"  AFTER  alignment RMSE: {rmse_after:.2f} mm")
    print(f"  Improvement: {rmse_before - rmse_after:.2f} mm")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Converged: {info['converged']}")

    # ------------------------------------------------------------------
    # 5. Visualize: BEFORE vs AFTER (side by side)
    # ------------------------------------------------------------------
    print("\nOpening visualization...")

    plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 700))

    # --- Panel 1: BEFORE alignment ---
    plotter.subplot(0, 0)
    plotter.set_background("white")
    plotter.add_text(
        f"BEFORE alignment\nRMSE = {rmse_before:.1f} mm",
        font_size=12, position="upper_left",
    )

    # Supine point cloud colored by error
    sup_before = pv.PolyData(supine_pc)
    # Map error from prone→supine: for each supine point, find closest
    # transformed prone point
    tree_before = cKDTree(prone_before)
    err_at_supine_before, _ = tree_before.query(supine_pc)
    sup_before["Error (mm)"] = err_at_supine_before
    plotter.add_points(
        sup_before, scalars="Error (mm)", cmap="coolwarm",
        clim=[0, 20], point_size=3, render_points_as_spheres=True,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Error (mm)", "vertical": True},
    )

    # Transformed prone mesh (semi-transparent)
    mesh_before = pv.PolyData(prone_before)
    plotter.add_points(
        mesh_before, color="tan", point_size=1, opacity=0.3,
    )

    # --- Panel 2: AFTER alignment ---
    plotter.subplot(0, 1)
    plotter.set_background("white")
    plotter.add_text(
        f"AFTER alignment\nRMSE = {rmse_after:.1f} mm",
        font_size=12, position="upper_left",
    )

    sup_after = pv.PolyData(supine_pc)
    tree_after = cKDTree(prone_after)
    err_at_supine_after, _ = tree_after.query(supine_pc)
    sup_after["Error (mm)"] = err_at_supine_after
    plotter.add_points(
        sup_after, scalars="Error (mm)", cmap="coolwarm",
        clim=[0, 20], point_size=3, render_points_as_spheres=True,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Error (mm)", "vertical": True},
    )

    mesh_after = pv.PolyData(prone_after)
    plotter.add_points(
        mesh_after, color="tan", point_size=1, opacity=0.3,
    )

    plotter.link_views()
    plotter.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    visualize_alignment_result()
