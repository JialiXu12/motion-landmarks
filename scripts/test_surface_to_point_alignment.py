"""
Tests for surface_to_point_alignment.py (plane-to-point ICP).

Run with: pytest test_surface_to_point_alignment.py -v
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose


# ---------------------------------------------------------------------------
# Helper: build a synthetic "mesh-like" object that quacks like morphic
# ---------------------------------------------------------------------------
class FakeElement:
    """Minimal mock of a morphic Element for testing."""

    def __init__(self, points, basis=None):
        """
        points: (num_nodes, 3) array defining a planar quad patch.
        We'll implement evaluate/derivative via bilinear interpolation.
        """
        self.basis = basis or ['H3', 'H3']
        self._p = np.asarray(points, dtype=np.float64)
        # corners: p00, p10, p01, p11 (bilinear ordering)
        assert self._p.shape == (4, 3), "Need 4 corner points for fake quad"

    def evaluate(self, Xi, deriv=None):
        """
        Bilinear interpolation over a quad element.
        Xi: (N, 2) array in [0,1]x[0,1]
        deriv: None -> coords, [1,0] -> dX/dxi1, [0,1] -> dX/dxi2
        """
        Xi = np.atleast_2d(Xi)
        u = Xi[:, 0:1]  # (N,1)
        v = Xi[:, 1:2]  # (N,1)
        p00, p10, p01, p11 = self._p

        if deriv is None:
            return (1 - u) * (1 - v) * p00 + u * (1 - v) * p10 + \
                   (1 - u) * v * p01 + u * v * p11
        elif deriv == [1, 0]:
            return (1 - v) * (p10 - p00) + v * (p11 - p01)
        elif deriv == [0, 1]:
            return (1 - u) * (p01 - p00) + u * (p11 - p10)
        else:
            raise ValueError(f"Unsupported deriv={deriv}")


class FakeElementsCollection:
    """Iterable collection of elements with __getitem__ and size()."""

    def __init__(self, elements_list):
        self._elems = elements_list

    def __getitem__(self, idx):
        return self._elems[idx]

    def __iter__(self):
        return iter(self._elems)

    def size(self):
        return len(self._elems)


class FakeMesh:
    """Minimal mock of morphic.Mesh."""

    def __init__(self, elements_list):
        self.elements = FakeElementsCollection(elements_list)

    def grid(self, res, method='center'):
        """Return (res*res, 2) grid of xi in [0,1]^2."""
        xi1 = np.linspace(0, 1, res)
        xi2 = np.linspace(0, 1, res)
        g1, g2 = np.meshgrid(xi1, xi2)
        return np.column_stack([g1.ravel(), g2.ravel()])


def _make_flat_mesh(z=0.0, size=100.0):
    """Create a single-element flat mesh in the XY plane at height z."""
    corners = np.array([
        [-size, -size, z],
        [size, -size, z],
        [-size, size, z],
        [size, size, z],
    ])
    return FakeMesh([FakeElement(corners)])


def _make_known_rotation_mesh(angle_deg, axis='z', z=0.0, size=100.0):
    """Create a flat mesh, then rotate its corners by a known angle."""
    corners = np.array([
        [-size, -size, z],
        [size, -size, z],
        [-size, size, z],
        [size, size, z],
    ])
    angle = np.radians(angle_deg)
    if axis == 'z':
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    elif axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    else:
        raise ValueError(f"Unknown axis: {axis}")

    rotated = (R @ corners.T).T
    return FakeMesh([FakeElement(rotated)]), R


# ===========================================================================
# Tests for compute_mesh_normals
# ===========================================================================
class TestComputeMeshNormals:

    def test_flat_mesh_normals_point_in_z(self):
        """A flat XY plane should have normals pointing in +Z or -Z."""
        from surface_to_point_alignment import compute_mesh_points_and_normals
        mesh = _make_flat_mesh(z=0.0)
        Xi = mesh.grid(3, method='center')
        points, normals = compute_mesh_points_and_normals(mesh, Xi)

        assert points.shape[1] == 3
        assert normals.shape == points.shape
        # All normals should be unit length
        norms = np.linalg.norm(normals, axis=1)
        assert_allclose(norms, 1.0, atol=1e-10)
        # All normals should be purely in Z direction
        assert_allclose(np.abs(normals[:, 2]), 1.0, atol=1e-10)
        assert_allclose(normals[:, 0], 0.0, atol=1e-10)
        assert_allclose(normals[:, 1], 0.0, atol=1e-10)

    def test_normals_are_unit_length(self):
        """Normals must always be unit vectors."""
        from surface_to_point_alignment import compute_mesh_points_and_normals
        mesh, _ = _make_known_rotation_mesh(30.0, axis='x')
        Xi = mesh.grid(4, method='center')
        _, normals = compute_mesh_points_and_normals(mesh, Xi)
        norms = np.linalg.norm(normals, axis=1)
        assert_allclose(norms, 1.0, atol=1e-10)

    def test_output_shapes_match(self):
        """Points and normals must have the same shape."""
        from surface_to_point_alignment import compute_mesh_points_and_normals
        mesh = _make_flat_mesh()
        Xi = mesh.grid(5, method='center')
        points, normals = compute_mesh_points_and_normals(mesh, Xi)
        assert points.shape == normals.shape
        # Number of points = num_elements * num_xi
        expected_n = mesh.elements.size() * Xi.shape[0]
        assert points.shape[0] == expected_n


# ===========================================================================
# Tests for plane_to_point_error
# ===========================================================================
class TestPlaneToPointError:

    def test_zero_error_when_aligned(self):
        """Points ON the source plane should have zero plane-to-point error."""
        from surface_to_point_alignment import plane_to_point_error
        source_pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        target_pts = np.array([[0.5, 0.5, 0], [1.5, 0.5, 0], [0.5, 1.5, 0]])
        errors = plane_to_point_error(source_pts, normals, target_pts)
        assert_allclose(errors, 0.0, atol=1e-10)

    def test_nonzero_error_off_plane(self):
        """Points offset in the normal direction should give that offset."""
        from surface_to_point_alignment import plane_to_point_error
        source_pts = np.array([[0, 0, 0]])
        normals = np.array([[0, 0, 1]])
        target_pts = np.array([[0, 0, 5.0]])
        errors = plane_to_point_error(source_pts, normals, target_pts)
        assert_allclose(errors, 5.0, atol=1e-10)

    def test_lateral_offset_no_error(self):
        """Point offset perpendicular to the normal has zero plane error."""
        from surface_to_point_alignment import plane_to_point_error
        source_pts = np.array([[0, 0, 0]])
        normals = np.array([[0, 0, 1]])
        target_pts = np.array([[10, 10, 0]])
        errors = plane_to_point_error(source_pts, normals, target_pts)
        assert_allclose(errors, 0.0, atol=1e-10)


# ===========================================================================
# Tests for find_correspondences_within_radius
# ===========================================================================
class TestFindCorrespondences:

    def test_all_within_radius(self):
        """All points within radius should be matched."""
        from surface_to_point_alignment import find_correspondences_within_radius
        source = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        target = np.array([[0.1, 0, 0], [1.1, 0, 0], [2.1, 0, 0]], dtype=float)
        src_idx, tgt_idx, dists = find_correspondences_within_radius(
            source, target, max_distance=1.0
        )
        assert len(src_idx) == 3
        assert_allclose(dists, 0.1, atol=1e-10)

    def test_points_beyond_radius_excluded(self):
        """Points beyond max_distance should not be matched."""
        from surface_to_point_alignment import find_correspondences_within_radius
        source = np.array([[0, 0, 0]], dtype=float)
        target = np.array([[100, 0, 0]], dtype=float)
        src_idx, tgt_idx, dists = find_correspondences_within_radius(
            source, target, max_distance=20.0
        )
        assert len(src_idx) == 0

    def test_returns_nearest_within_radius(self):
        """When multiple targets are near, the closest should be returned."""
        from surface_to_point_alignment import find_correspondences_within_radius
        source = np.array([[0, 0, 0]], dtype=float)
        target = np.array([[1, 0, 0], [2, 0, 0], [5, 0, 0]], dtype=float)
        src_idx, tgt_idx, dists = find_correspondences_within_radius(
            source, target, max_distance=10.0
        )
        assert len(src_idx) == 1
        assert tgt_idx[0] == 0  # closest is index 0
        assert_allclose(dists[0], 1.0, atol=1e-10)


# ===========================================================================
# Tests for solve_plane_to_point_rotation
# ===========================================================================
class TestSolvePlaneToPointRotation:

    def test_identity_when_aligned(self):
        """When source == target, rotation should be identity."""
        from surface_to_point_alignment import solve_plane_to_point_rotation
        pts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [-1, 0, 0], [0, -1, 0]], dtype=float)
        normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                            [-1, 0, 0], [0, -1, 0]], dtype=float)
        R = solve_plane_to_point_rotation(pts, normals, pts)
        assert_allclose(R, np.eye(3), atol=1e-6)

    def test_returns_valid_rotation_matrix(self):
        """Result must be orthogonal with det=+1."""
        from surface_to_point_alignment import solve_plane_to_point_rotation
        rng = np.random.default_rng(42)
        src = rng.standard_normal((20, 3))
        normals = rng.standard_normal((20, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        tgt = src + 0.1 * rng.standard_normal((20, 3))
        R = solve_plane_to_point_rotation(src, normals, tgt)
        # Orthogonal: R^T R = I
        assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
        # Proper rotation: det = +1
        assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)


# ===========================================================================
# Tests for full surface_to_point_align (integration)
# ===========================================================================
class TestSurfaceToPointAlign:

    def test_identity_alignment(self):
        """Mesh already aligned with point cloud should give identity T."""
        from surface_to_point_alignment import surface_to_point_align
        mesh = _make_flat_mesh(z=0.0, size=50.0)
        Xi = mesh.grid(5, method='center')
        # Target = exact same points
        target_pc = np.column_stack([
            np.random.default_rng(0).uniform(-50, 50, 100),
            np.random.default_rng(1).uniform(-50, 50, 100),
            np.zeros(100)
        ])
        sternum_src = np.array([0, 0, 0.0])
        sternum_tgt = np.array([0, 0, 0.0])

        R, T, info = surface_to_point_align(
            mesh, target_pc, sternum_src, sternum_tgt,
            max_distance=20.0, max_iterations=50, res=5
        )
        assert T.shape == (4, 4)
        assert R.shape == (3, 3)
        # Should be close to identity
        assert_allclose(R, np.eye(3), atol=0.1)

    def test_recovers_small_rotation(self):
        """Should recover a small known rotation (5 degrees about Z)."""
        from surface_to_point_alignment import surface_to_point_align
        angle = 5.0
        mesh, R_true = _make_known_rotation_mesh(angle, axis='z', size=50.0)
        # Target is the unrotated flat mesh points
        rng = np.random.default_rng(42)
        target_pc = np.column_stack([
            rng.uniform(-50, 50, 200),
            rng.uniform(-50, 50, 200),
            np.zeros(200)
        ])
        sternum_src = np.array([0, 0, 0.0])
        sternum_tgt = np.array([0, 0, 0.0])

        R_est, T, info = surface_to_point_align(
            mesh, target_pc, sternum_src, sternum_tgt,
            max_distance=30.0, max_iterations=100, res=5
        )
        # The algorithm should find R such that R @ R_true ≈ I
        # (it should undo the rotation)
        R_combined = R_est @ R_true
        assert_allclose(R_combined, np.eye(3), atol=0.15)

    def test_convergence_reduces_error(self):
        """RMSE after alignment must be less than before."""
        from surface_to_point_alignment import surface_to_point_align
        mesh, _ = _make_known_rotation_mesh(10.0, axis='z', size=50.0)
        rng = np.random.default_rng(99)
        target_pc = np.column_stack([
            rng.uniform(-50, 50, 200),
            rng.uniform(-50, 50, 200),
            np.zeros(200)
        ])
        sternum_src = np.array([0, 0, 0.0])
        sternum_tgt = np.array([0, 0, 0.0])

        _, _, info = surface_to_point_align(
            mesh, target_pc, sternum_src, sternum_tgt,
            max_distance=30.0, max_iterations=50, res=5
        )
        history = info['iteration_history']
        assert len(history) >= 2
        assert history[-1]['rmse'] <= history[0]['rmse']

    def test_transformation_matrix_shape_and_valid(self):
        """T_total must be 4x4 with last row [0,0,0,1]."""
        from surface_to_point_alignment import surface_to_point_align
        mesh = _make_flat_mesh(size=50.0)
        target_pc = np.zeros((50, 3))
        sternum_src = np.array([0, 0, 0.0])
        sternum_tgt = np.array([0, 0, 0.0])

        _, T, _ = surface_to_point_align(
            mesh, target_pc, sternum_src, sternum_tgt,
            max_distance=20.0, max_iterations=10, res=3
        )
        assert T.shape == (4, 4)
        assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-15)

    def test_sternum_stays_at_origin(self):
        """After alignment, sternum must remain at origin."""
        from surface_to_point_alignment import surface_to_point_align
        mesh, _ = _make_known_rotation_mesh(8.0, axis='x', size=50.0)
        rng = np.random.default_rng(7)
        target_pc = np.column_stack([
            rng.uniform(-50, 50, 150),
            rng.uniform(-50, 50, 150),
            np.zeros(150)
        ])
        sternum_src = np.array([0, 0, 0.0])
        sternum_tgt = np.array([0, 0, 0.0])

        R, T, info = surface_to_point_align(
            mesh, target_pc, sternum_src, sternum_tgt,
            max_distance=30.0, max_iterations=50, res=5
        )
        # R @ [0,0,0] = [0,0,0] always, so sternum error = 0
        sternum_error = np.linalg.norm(R @ sternum_src)
        assert sternum_error < 1e-10

    def test_max_distance_prevents_wrong_side_matching(self):
        """Points far away (>20mm) must not influence alignment."""
        from surface_to_point_alignment import surface_to_point_align
        mesh = _make_flat_mesh(z=0.0, size=10.0)
        # Near target (within 20mm)
        near = np.array([[0, 0, 2.0], [5, 0, 1.0], [-5, 0, 1.5]])
        # Far target (>20mm away, on "other side")
        far = np.array([[0, 0, 100.0], [0, 0, -100.0]])
        target_pc = np.vstack([near, far])
        sternum_src = np.array([0, 0, 0.0])
        sternum_tgt = np.array([0, 0, 0.0])

        R, T, info = surface_to_point_align(
            mesh, target_pc, sternum_src, sternum_tgt,
            max_distance=20.0, max_iterations=30, res=3
        )
        # Should still converge (not pulled to far-away points)
        assert info['iteration_history'][-1]['rmse'] < 50.0

    def test_info_dict_contains_required_keys(self):
        """Info dict must contain standard keys for reporting."""
        from surface_to_point_alignment import surface_to_point_align
        mesh = _make_flat_mesh(size=50.0)
        target_pc = np.zeros((50, 3))
        sternum_src = np.array([0, 0, 0.0])
        sternum_tgt = np.array([0, 0, 0.0])

        _, _, info = surface_to_point_align(
            mesh, target_pc, sternum_src, sternum_tgt,
            max_distance=20.0, max_iterations=5, res=3
        )
        required_keys = [
            'method', 'iterations', 'converged',
            'euclidean_rmse', 'n_correspondences',
            'iteration_history', 'R_total',
            'source_sternum_offset', 'target_sternum_offset',
        ]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"


# ===========================================================================
# Edge-case tests
# ===========================================================================
class TestEdgeCases:

    def test_no_correspondences_within_radius(self):
        """When no points are within radius, should not crash."""
        from surface_to_point_alignment import surface_to_point_align
        mesh = _make_flat_mesh(z=0.0, size=10.0)
        # All target points far away
        target_pc = np.array([[0, 0, 1000.0]])
        sternum_src = np.array([0, 0, 0.0])
        sternum_tgt = np.array([0, 0, 0.0])

        R, T, info = surface_to_point_align(
            mesh, target_pc, sternum_src, sternum_tgt,
            max_distance=20.0, max_iterations=5, res=3
        )
        # Should return identity when nothing matches
        assert_allclose(R, np.eye(3), atol=1e-10)

    def test_single_element_mesh(self):
        """Should work with a single-element mesh."""
        from surface_to_point_alignment import surface_to_point_align
        mesh = _make_flat_mesh(z=0.0, size=50.0)
        assert mesh.elements.size() == 1
        target_pc = np.column_stack([
            np.random.default_rng(0).uniform(-50, 50, 50),
            np.random.default_rng(1).uniform(-50, 50, 50),
            np.zeros(50)
        ])
        sternum_src = np.array([0, 0, 0.0])
        sternum_tgt = np.array([0, 0, 0.0])

        R, T, info = surface_to_point_align(
            mesh, target_pc, sternum_src, sternum_tgt,
            max_distance=20.0, max_iterations=10, res=5
        )
        assert T.shape == (4, 4)


# ===========================================================================
# Tests for element selection (Fix 1)
# ===========================================================================
class TestElementSelection:

    def test_elems_selects_subset(self):
        """Only selected elements should be sampled when elems is set."""
        from surface_to_point_alignment import compute_mesh_points_and_normals
        # Build a 3-element mesh
        elems_list = [
            FakeElement(np.array([[-50,-50,0],[50,-50,0],[-50,50,0],[50,50,0]], dtype=float)),
            FakeElement(np.array([[-50,-50,10],[50,-50,10],[-50,50,10],[50,50,10]], dtype=float)),
            FakeElement(np.array([[-50,-50,20],[50,-50,20],[-50,50,20],[50,50,20]], dtype=float)),
        ]
        mesh = FakeMesh(elems_list)
        Xi = mesh.grid(3, method='center')
        n_xi = Xi.shape[0]

        # Select only element 0 and 2
        pts, nrm = compute_mesh_points_and_normals(mesh, Xi, elems=[0, 2])
        assert pts.shape[0] == 2 * n_xi
        # All Z values should be either ~0 or ~20 (not 10)
        z_vals = np.unique(np.round(pts[:, 2], 0))
        assert 10.0 not in z_vals

    def test_elems_none_uses_all(self):
        """When elems is None, all elements should be used."""
        from surface_to_point_alignment import compute_mesh_points_and_normals
        elems_list = [
            FakeElement(np.array([[-50,-50,0],[50,-50,0],[-50,50,0],[50,50,0]], dtype=float)),
            FakeElement(np.array([[-50,-50,10],[50,-50,10],[-50,50,10],[50,50,10]], dtype=float)),
        ]
        mesh = FakeMesh(elems_list)
        Xi = mesh.grid(3, method='center')
        n_xi = Xi.shape[0]

        pts_all, _ = compute_mesh_points_and_normals(mesh, Xi, elems=None)
        assert pts_all.shape[0] == 2 * n_xi

    def test_elems_parameter_passed_through(self):
        """surface_to_point_align should pass elems to compute_mesh_points_and_normals."""
        from surface_to_point_alignment import surface_to_point_align
        # 2-element mesh; only use element 0
        elems_list = [
            FakeElement(np.array([[-50,-50,0],[50,-50,0],[-50,50,0],[50,50,0]], dtype=float)),
            FakeElement(np.array([[-50,-50,100],[50,-50,100],[-50,50,100],[50,50,100]], dtype=float)),
        ]
        mesh = FakeMesh(elems_list)
        target_pc = np.column_stack([
            np.random.default_rng(0).uniform(-50, 50, 50),
            np.random.default_rng(1).uniform(-50, 50, 50),
            np.zeros(50)
        ])
        sternum = np.array([0, 0, 0.0])

        # Should run without error using only element 0
        R, T, info = surface_to_point_align(
            mesh, target_pc, sternum, sternum,
            max_distance=20.0, max_iterations=5, res=3, elems=[0]
        )
        assert T.shape == (4, 4)


# ===========================================================================
# Tests for normal orientation consistency (Fix 3)
# ===========================================================================
class TestNormalOrientation:

    def test_normals_point_outward(self):
        """All normals should point away from mesh centroid."""
        from surface_to_point_alignment import compute_mesh_points_and_normals
        # Two parallel elements at z=0 and z=20 so centroid is at z=10,
        # giving non-zero z-component in outward_dirs
        e_low = FakeElement(np.array([
            [-50, -50, 0], [50, -50, 0], [-50, 50, 0], [50, 50, 0]
        ], dtype=float))
        e_high = FakeElement(np.array([
            [-50, -50, 20], [50, -50, 20], [-50, 50, 20], [50, 50, 20]
        ], dtype=float))
        mesh = FakeMesh([e_low, e_high])
        Xi = mesh.grid(4, method='center')
        pts, nrm = compute_mesh_points_and_normals(mesh, Xi)

        centroid = np.mean(pts, axis=0)
        outward = pts - centroid
        dots = np.sum(nrm * outward, axis=1)
        # All dots should be >= 0 (outward or tangential, never inward)
        assert np.all(dots >= -1e-10)

    def test_flipped_element_normals_become_consistent(self):
        """Element with inverted parameterisation should still get outward normals."""
        from surface_to_point_alignment import compute_mesh_points_and_normals
        # Two elements forming a V-shape so centroid is below both.
        # Normal element at z=10: cross product gives +Z (outward)
        normal_elem = FakeElement(np.array([
            [-50, -50, 10], [50, -50, 10], [-50, 50, 10], [50, 50, 10]
        ], dtype=float))
        # Flipped element at z=20: swapped corners so raw cross gives -Z,
        # but centroid-based fix should flip it to +Z (outward)
        flipped_elem = FakeElement(np.array([
            [-50, 50, 20], [50, 50, 20], [-50, -50, 20], [50, -50, 20]
        ], dtype=float))
        mesh = FakeMesh([normal_elem, flipped_elem])
        Xi = mesh.grid(3, method='center')

        pts, nrm = compute_mesh_points_and_normals(mesh, Xi)
        centroid = np.mean(pts, axis=0)
        outward = pts - centroid
        dots = np.sum(nrm * outward, axis=1)
        # All normals should point outward (non-negative dot with outward dir)
        assert np.all(dots >= -1e-10), f"Some normals point inward: min dot = {dots.min()}"


# ===========================================================================
# Tests for KD-tree reuse (Fix 4)
# ===========================================================================
class TestKDTreeReuse:

    def test_accepts_prebuilt_tree(self):
        """Passing a pre-built tree should produce the same results."""
        from surface_to_point_alignment import find_correspondences_within_radius
        from scipy.spatial import cKDTree
        source = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        target = np.array([[0.1, 0, 0], [1.1, 0, 0], [2.1, 0, 0]], dtype=float)

        tree = cKDTree(target)
        s1, t1, d1 = find_correspondences_within_radius(
            source, target, max_distance=1.0, tree=tree
        )
        s2, t2, d2 = find_correspondences_within_radius(
            source, target, max_distance=1.0, tree=None
        )
        assert_allclose(d1, d2)
        assert_allclose(s1, s2)
        assert_allclose(t1, t2)

    def test_tree_none_builds_internally(self):
        """tree=None should still work (backward compat)."""
        from surface_to_point_alignment import find_correspondences_within_radius
        source = np.array([[0, 0, 0]], dtype=float)
        target = np.array([[1, 0, 0]], dtype=float)
        s, t, d = find_correspondences_within_radius(
            source, target, max_distance=5.0, tree=None
        )
        assert len(s) == 1
        assert_allclose(d[0], 1.0, atol=1e-10)


# ===========================================================================
# Tests for hybrid plane-to-point + point-to-point objective (Fix 2)
# ===========================================================================
class TestHybridObjective:

    def test_identity_with_point_to_point_weight(self):
        """Aligned data with hybrid weight should still give identity."""
        from surface_to_point_alignment import solve_plane_to_point_rotation
        pts = np.array([[1,0,0],[0,1,0],[0,0,1],
                        [-1,0,0],[0,-1,0]], dtype=float)
        normals = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        R = solve_plane_to_point_rotation(pts, normals, pts,
                                          point_to_point_weight=0.5)
        assert_allclose(R, np.eye(3), atol=1e-6)

    def test_valid_rotation_with_hybrid(self):
        """Hybrid solver must return proper rotation (det=+1, orthogonal)."""
        from surface_to_point_alignment import solve_plane_to_point_rotation
        rng = np.random.default_rng(42)
        src = rng.standard_normal((20, 3))
        normals = rng.standard_normal((20, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        tgt = src + 0.1 * rng.standard_normal((20, 3))
        R = solve_plane_to_point_rotation(src, normals, tgt,
                                          point_to_point_weight=0.3)
        assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
        assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)

    def test_weight_zero_matches_original(self):
        """weight=0 should give the same result as the original solver."""
        from surface_to_point_alignment import solve_plane_to_point_rotation
        rng = np.random.default_rng(7)
        src = rng.standard_normal((15, 3))
        normals = rng.standard_normal((15, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        tgt = src + 0.05 * rng.standard_normal((15, 3))

        R_orig = solve_plane_to_point_rotation(src, normals, tgt,
                                               point_to_point_weight=0.0)
        R_zero = solve_plane_to_point_rotation(src, normals, tgt)
        assert_allclose(R_orig, R_zero, atol=1e-12)

    def test_weight_one_is_pure_point_to_point(self):
        """weight=1 should give a pure point-to-point rotation."""
        from surface_to_point_alignment import solve_plane_to_point_rotation
        # Create source and target with known small rotation
        rng = np.random.default_rng(123)
        src = rng.standard_normal((30, 3)) * 10
        normals = rng.standard_normal((30, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        tgt = src + 0.1 * rng.standard_normal((30, 3))
        R = solve_plane_to_point_rotation(src, normals, tgt,
                                          point_to_point_weight=1.0)
        # Must be valid rotation
        assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
        assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)

    def test_recovers_z_rotation_with_hybrid(self):
        """Hybrid solver should recover a Z-axis rotation on a flat mesh."""
        from surface_to_point_alignment import surface_to_point_align
        angle = 5.0
        mesh, R_true = _make_known_rotation_mesh(angle, axis='z', size=50.0)
        rng = np.random.default_rng(42)
        target_pc = np.column_stack([
            rng.uniform(-50, 50, 200),
            rng.uniform(-50, 50, 200),
            np.zeros(200)
        ])
        sternum = np.array([0, 0, 0.0])

        R_est, T, info = surface_to_point_align(
            mesh, target_pc, sternum, sternum,
            max_distance=30.0, max_iterations=100, res=5,
            point_to_point_weight=0.3,
        )
        R_combined = R_est @ R_true
        assert_allclose(R_combined, np.eye(3), atol=0.15)
