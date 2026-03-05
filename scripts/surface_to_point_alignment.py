"""
Plane-to-Point ICP Alignment.

Aligns a prone ribcage morphic mesh (source) to a supine ribcage point
cloud (target) by minimizing the plane-to-point distance: for each sampled
surface patch on the source mesh, the error is measured FROM the surface
plane ALONG its normal TOWARD the nearest target point.

    error_i = n_i . (q_i - p_i)

where p_i is a point on the source surface, n_i is the surface normal at
p_i (computed from the mesh), and q_i is the nearest target point.

This differs from standard point-to-plane ICP where normals come from the
target.  Here the normals come from the source mesh surface (the "plane"),
and the error measures toward the target (the "point").

Key features:
    - Normals recomputed from the source morphic mesh at every iteration
    - Sternum superior fixed at origin (rotation-only around sternum)
    - Nearest-neighbor search limited to a configurable radius (default 20mm)
      to prevent matching to the wrong side of the ribcage
    - Returns 4x4 transformation matrix and rotation matrix

Morphic mesh requirements:
    - 2D surface elements with basis ['H3', 'H3']
    - element.evaluate(Xi)            -> (N, 3) surface coordinates
    - element.evaluate(Xi, deriv=[1,0]) -> (N, 3) tangent dX/dxi1
    - element.evaluate(Xi, deriv=[0,1]) -> (N, 3) tangent dX/dxi2
"""

import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
from typing import Tuple, Dict

# Conditional import for plot_all - only needed when running alignment
try:
    from utils_plot import plot_all
except ImportError:
    plot_all = None  # Will be imported in __main__ block
import external.breast_metadata_mdv.breast_metadata as breast_metadata

from utils import plot_evaluate_alignment

# ---------------------------------------------------------------------------
# 0. Helper: Extract Euler angles from rotation matrix
# ---------------------------------------------------------------------------
def rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract Euler angles (in degrees) from a 3x3 rotation matrix.

    Uses the ZYX convention:
        R = Rz @ Ry @ Rx

    Args:
        R: (3, 3) rotation matrix

    Returns:
        (angle_x, angle_y, angle_z) in degrees - rotation around X, Y, Z axes
    """
    # Check for gimbal lock (pitch = ±90°)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        angle_x = np.arctan2(R[2, 1], R[2, 2])   # Rotation around X
        angle_y = np.arctan2(-R[2, 0], sy)       # Rotation around Y
        angle_z = np.arctan2(R[1, 0], R[0, 0])   # Rotation around Z
    else:
        # Gimbal lock case
        angle_x = np.arctan2(-R[1, 2], R[1, 1])
        angle_y = np.arctan2(-R[2, 0], sy)
        angle_z = 0

    # Convert to degrees
    return (np.degrees(angle_x), np.degrees(angle_y), np.degrees(angle_z))


def print_rotation_angles(R: np.ndarray, label: str = "Rotation"):
    """Print rotation matrix as Euler angles."""
    angle_x, angle_y, angle_z = rotation_matrix_to_euler_angles(R)
    total_angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"  {label}:")
    print(f"    Angle X:  {angle_x:+.3f}°")
    print(f"    Angle Y:  {angle_y:+.3f}°")
    print(f"    Angle Z:  {angle_z:+.3f}°")
    print(f"    Total rotation angle: {total_angle:.3f}°")


def plot_convergence_diagram(info: Dict, save_path: str = None):
    """
    Plot optimization convergence diagram showing RMSE and rotation angles.

    Args:
        info: dict returned by surface_to_point_align containing iteration_history
        save_path: optional path to save the figure (can be relative or absolute)
    """
    import matplotlib
    import matplotlib.pyplot as plt


    print(f"  Plotting convergence diagram (backend: {matplotlib.get_backend()})")

    history = info.get('iteration_history', [])
    if not history:
        print("  No iteration history to plot")
        return

    print(f"  Found {len(history)} iterations to plot")

    iterations = [h['iteration'] for h in history]
    rmse_values = [h['rmse'] for h in history]
    angle_x_values = [h.get('angle_x_deg', 0) for h in history]
    angle_y_values = [h.get('angle_y_deg', 0) for h in history]
    angle_z_values = [h.get('angle_z_deg', 0) for h in history]
    total_angle_values = [h.get('total_angle_deg', 0) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Surface-to-Point ICP Convergence', fontsize=14, fontweight='bold')

    # Plot 1: RMSE convergence
    ax1 = axes[0, 0]
    ax1.plot(iterations, rmse_values, 'b-o', linewidth=2, markersize=4, label='RMSE')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('RMSE (mm)', fontsize=11)
    ax1.set_title('RMSE Convergence', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(iterations) + 1)

    # Add convergence annotation
    if len(rmse_values) > 1:
        initial_rmse = rmse_values[0]
        final_rmse = rmse_values[-1]
        reduction = (initial_rmse - final_rmse) / initial_rmse * 100 if initial_rmse > 0 else 0
        ax1.annotate(f'Initial: {initial_rmse:.4f} mm\nFinal: {final_rmse:.4f} mm\nReduction: {reduction:.1f}%',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Total rotation angle
    ax2 = axes[0, 1]
    ax2.plot(iterations, total_angle_values, 'r-s', linewidth=2, markersize=4, label='Total Angle')
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Total Rotation (°)', fontsize=11)
    ax2.set_title('Total Rotation Angle Convergence', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(iterations) + 1)

    # Add final angle annotation
    if total_angle_values:
        ax2.annotate(f'Final: {total_angle_values[-1]:.3f}°',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    # Plot 3: Individual Euler angles (X, Y, Z)
    ax3 = axes[1, 0]
    ax3.plot(iterations, angle_x_values, 'g-^', linewidth=1.5, markersize=3, label='Angle X')
    ax3.plot(iterations, angle_y_values, 'b-v', linewidth=1.5, markersize=3, label='Angle Y')
    ax3.plot(iterations, angle_z_values, 'r-o', linewidth=1.5, markersize=3, label='Angle Z')
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Angle (°)', fontsize=11)
    ax3.set_title('Euler Angles (X, Y, Z)', fontsize=12)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max(iterations) + 1)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    # Add final angles annotation
    if angle_x_values and angle_y_values and angle_z_values:
        ax3.annotate(f'Final:\n  X: {angle_x_values[-1]:+.3f}°\n  Y: {angle_y_values[-1]:+.3f}°\n  Z: {angle_z_values[-1]:+.3f}°',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Plot 4: Rotation change per iteration (convergence rate)
    ax4 = axes[1, 1]
    rotation_changes = [h.get('rotation_change', 0) for h in history]
    ax4.semilogy(iterations, rotation_changes, 'm-d', linewidth=2, markersize=4, label='Rotation Change')
    ax4.set_xlabel('Iteration', fontsize=11)
    ax4.set_ylabel('||R_delta - I|| (log scale)', fontsize=11)
    ax4.set_title('Rotation Update Magnitude (Convergence Rate)', fontsize=12)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(0, max(iterations) + 1)

    # Add convergence threshold line
    ax4.axhline(y=1e-4, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Convergence threshold')
    ax4.legend(loc='best', fontsize=9)

    plt.tight_layout()

    # Save the figure
    if save_path:
        try:
            save_path = Path(save_path)
            if not save_path.is_absolute():
                # Assume relative to script directory
                script_dir = Path(__file__).parent
                save_path = script_dir / save_path

            # Ensure the directory exists
            save_dir = save_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Saving to: {save_path}")

            plt.savefig(str(save_path), dpi=150, bbox_inches='tight')

            # Verify the file was actually saved
            if save_path.exists():
                file_size = save_path.stat().st_size
                print(f"  Convergence diagram saved: {save_path} ({file_size:,} bytes)")
            else:
                print(f"  WARNING: File was not created at {save_path}")
        except Exception as e:
            print(f"  ERROR saving convergence diagram: {e}")
            import traceback
            traceback.print_exc()

    # Display the figure
    try:
        plt.show(block=False)  # Non-blocking show
        plt.pause(1.0)  # Display for 1 second
        print("  Convergence diagram displayed")
    except Exception as e:
        print(f"  Warning: Could not display figure interactively: {e}")
    finally:
        plt.close(fig)  # Close the figure to free memory



# ---------------------------------------------------------------------------
# 0. Mutual region filtering
# ---------------------------------------------------------------------------
def filter_mutual_region(
        src_pts: np.ndarray,
        tgt_pts: np.ndarray,
        src_ss: np.ndarray,
        tgt_ss: np.ndarray,
        padding: float = 15.0,
        padding_inferior: float = 0.0,
        verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter target point cloud and source mesh points to their mutual overlap.

    Both datasets are shifted to sternum-centered space before comparison.

    Step 1 — Filter target to source bbox (with asymmetric padding).
    Step 2 — Filter source to the (filtered) target bbox (reciprocal,
             no extra padding so mesh points beyond the PC extent are removed).

    Args:
        src_pts: (N, 3) source mesh points (world coords)
        tgt_pts: (M, 3) target point cloud (world coords)
        src_ss:  (3,) source sternum superior position
        tgt_ss:  (3,) target sternum superior position
        padding: extra margin (mm) on each side of the source bounding box
        padding_inferior: margin (mm) for inferior (Z-min) side only;
                          overrides ``padding`` for that boundary
        verbose: print filtering summary

    Returns:
        src_mask: (N,) bool — True for source points inside the mutual region
        tgt_filtered: (K, 3) filtered target points (world coords, K <= M)
        info: dict with bbox details for debugging
    """
    src_centered = src_pts - src_ss
    tgt_centered = tgt_pts - tgt_ss

    # --- Step 1: filter target to source bbox ---
    padding_min = np.array([padding, padding, padding_inferior])
    padding_max = np.array([padding, padding, padding])
    src_bbox_min = src_centered.min(axis=0) - padding_min
    src_bbox_max = src_centered.max(axis=0) + padding_max

    tgt_inside = np.all(
        (tgt_centered >= src_bbox_min) & (tgt_centered <= src_bbox_max),
        axis=1,
    )
    tgt_filtered = tgt_pts[tgt_inside]

    if verbose:
        print(f"  Target region filtering (to source bbox):")
        print(f"    Original: {len(tgt_pts)} pts  ->  "
              f"Filtered: {len(tgt_filtered)} pts "
              f"({len(tgt_filtered)/len(tgt_pts)*100:.0f}% kept)")
        print(f"    Source bbox (sternum-centered, padded): "
              f"X[{src_bbox_min[0]:.0f},{src_bbox_max[0]:.0f}] "
              f"Y[{src_bbox_min[1]:.0f},{src_bbox_max[1]:.0f}] "
              f"Z[{src_bbox_min[2]:.0f},{src_bbox_max[2]:.0f}]")

    # --- Step 2: reciprocal — filter source to (filtered) target bbox ---
    tgt_filt_centered = tgt_filtered - tgt_ss
    tgt_bbox_min = tgt_filt_centered.min(axis=0)
    tgt_bbox_max = tgt_filt_centered.max(axis=0)

    src_inside = np.all(
        (src_centered >= tgt_bbox_min) & (src_centered <= tgt_bbox_max),
        axis=1,
    )

    if verbose:
        n_removed = int(np.sum(~src_inside))
        print(f"  Reciprocal source filtering (to target extent):")
        print(f"    Source: {len(src_pts)} pts  ->  "
              f"{int(np.sum(src_inside))} pts kept, "
              f"{n_removed} removed")
        print(f"    Target bbox (sternum-centered, no padding): "
              f"X[{tgt_bbox_min[0]:.0f},{tgt_bbox_max[0]:.0f}] "
              f"Y[{tgt_bbox_min[1]:.0f},{tgt_bbox_max[1]:.0f}] "
              f"Z[{tgt_bbox_min[2]:.0f},{tgt_bbox_max[2]:.0f}]")

    info = {
        'src_bbox_min': src_bbox_min,
        'src_bbox_max': src_bbox_max,
        'tgt_bbox_min': tgt_bbox_min,
        'tgt_bbox_max': tgt_bbox_max,
    }

    return src_inside, tgt_filtered, info


# ---------------------------------------------------------------------------
# 1. Sample mesh surface points and normals
# ---------------------------------------------------------------------------
def compute_mesh_points_and_normals(
        mesh,
        Xi: np.ndarray,
        elems: list = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate surface coordinates and outward normals for every element.

    For each element, the two tangent vectors dX/dxi1 and dX/dxi2 are
    evaluated at the parametric positions Xi.  The normal is their cross
    product, normalised to unit length.

    Args:
        mesh: morphic.Mesh (or any object with .elements iterable whose
              items support .evaluate(Xi, deriv=...))
        Xi: (M, 2) parametric coordinates in [0,1]^2
        elems: optional list of element indices to use; None = all elements

    Returns:
        points:  (N, 3) surface coordinates   (N = num_selected_elements * M)
        normals: (N, 3) unit surface normals
    """
    all_pts = []
    all_nrm = []

    if elems is not None:
        elements_iter = [mesh.elements[idx] for idx in elems]
    else:
        elements_iter = mesh.elements

    for element in elements_iter:
        pts = element.evaluate(Xi)               # (M, 3)
        t1 = element.evaluate(Xi, deriv=[1, 0])  # dX/dxi1
        t2 = element.evaluate(Xi, deriv=[0, 1])  # dX/dxi2

        raw_normals = np.cross(t1, t2)           # (M, 3)
        magnitudes = np.linalg.norm(raw_normals, axis=1, keepdims=True)
        # Guard against degenerate (zero-area) patches
        magnitudes = np.where(magnitudes < 1e-15, 1.0, magnitudes)
        normals = raw_normals / magnitudes

        all_pts.append(pts)
        all_nrm.append(normals)

    all_points = np.vstack(all_pts)
    all_normals = np.vstack(all_nrm)

    # Orient normals outward: flip any that point toward the mesh centroid
    centroid = np.mean(all_points, axis=0)
    outward_dirs = all_points - centroid
    dot_products = np.sum(all_normals * outward_dirs, axis=1)
    flip_signs = np.where(dot_products < -1e-10, -1.0, 1.0)[:, np.newaxis]
    all_normals = all_normals * flip_signs
    # for debug
    #plot_all(mesh_points=all_points)
    return all_points, all_normals


# ---------------------------------------------------------------------------
# 1b. Estimate normals from a point cloud (no mesh required)
# ---------------------------------------------------------------------------
def estimate_normals_from_points(
        points: np.ndarray,
        k: int = 20,
) -> np.ndarray:
    """
    Estimate outward-pointing unit normals from a point cloud via local PCA.

    For each point, the k nearest neighbours define a local covariance
    matrix.  The eigenvector with the smallest eigenvalue is the normal.
    Normals are then oriented outward using the centroid heuristic.

    Args:
        points: (N, 3) point cloud
        k: number of neighbours for local PCA (default 20)

    Returns:
        normals: (N, 3) unit normals oriented outward
    """
    tree = cKDTree(points)
    k_actual = min(k, len(points))
    _, idx = tree.query(points, k=k_actual)

    normals = np.zeros_like(points)
    for i in range(len(points)):
        neighbours = points[idx[i]]
        centroid_local = neighbours.mean(axis=0)
        cov = (neighbours - centroid_local).T @ (neighbours - centroid_local)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]  # smallest eigenvalue = normal direction

    # Normalise
    magnitudes = np.linalg.norm(normals, axis=1, keepdims=True)
    magnitudes = np.where(magnitudes < 1e-15, 1.0, magnitudes)
    normals = normals / magnitudes

    # Orient outward using centroid heuristic
    centroid = np.mean(points, axis=0)
    outward_dirs = points - centroid
    dot_products = np.sum(normals * outward_dirs, axis=1)
    flip_signs = np.where(dot_products < -1e-10, -1.0, 1.0)[:, np.newaxis]
    normals = normals * flip_signs

    return normals


# ---------------------------------------------------------------------------
# 2. Plane-to-point error
# ---------------------------------------------------------------------------
def plane_to_point_error(
        source_pts: np.ndarray,
        normals: np.ndarray,
        target_pts: np.ndarray,
) -> np.ndarray:
    """
    Signed plane-to-point distance: n_i . (q_i - p_i).

    Measures the distance FROM the source surface plane (defined by p_i and
    its normal n_i) TOWARD the target point q_i, projected onto the normal.

    Args:
        source_pts: (N, 3) points on the source surface (the "plane")
        normals:    (N, 3) unit normals at source surface points
        target_pts: (N, 3) corresponding target points (the "point")

    Returns:
        (N,) signed distances (positive = target is on the normal side)
    """
    diff = target_pts - source_pts
    return np.sum(diff * normals, axis=1)


# ---------------------------------------------------------------------------
# 3. Correspondence search with radius limit
# ---------------------------------------------------------------------------
def find_correspondences_within_radius(
        source_pts: np.ndarray,
        target_pts: np.ndarray,
        max_distance: float = 20.0,
        tree: cKDTree = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each source point, find the nearest target point within max_distance.

    Args:
        source_pts: (N, 3) source points
        target_pts: (M, 3) target point cloud
        max_distance: maximum allowed distance (mm)
        tree: pre-built cKDTree on target_pts; if None, one is built internally

    Returns:
        src_idx:  (K,) indices into source_pts that have a match
        tgt_idx:  (K,) indices into target_pts (nearest neighbours)
        dists:    (K,) Euclidean distances
    """
    if tree is None:
        tree = cKDTree(target_pts)
    dists, idxs = tree.query(source_pts)

    valid = dists <= max_distance
    src_idx = np.where(valid)[0]
    tgt_idx = idxs[valid]
    dists_out = dists[valid]

    return src_idx, tgt_idx, dists_out


# ---------------------------------------------------------------------------
# 4. Solve for rotation using linearised plane-to-point
# ---------------------------------------------------------------------------
def solve_plane_to_point_rotation(
        source_pts: np.ndarray,
        normals: np.ndarray,
        target_pts: np.ndarray,
        point_to_point_weight: float = 0.0,
) -> np.ndarray:
    """
    Solve the linearised plane-to-point minimisation for a small rotation.

    The plane-to-point error at each correspondence is:
        e_i = n_i . (q_i - R p_i)
    where n_i is the source surface normal (rotated with the source),
    p_i is the source surface point, and q_i is the target point.

    For small angles (a, b, c) around X, Y, Z the rotation is approximated as:

        R ≈ I + [[0, -c,  b],
                  [c,  0, -a],
                  [-b, a,  0]]

    This linearises to:
        n_i . (q_i - p_i - [a,b,c] x p_i) = 0

    which is a 3x3 linear system  A @ [a,b,c] = b  solved by least squares.

    When ``point_to_point_weight`` > 0, a hybrid system is built:
        - Plane-to-point rows (N):   sqrt(1-w) * [p_i x n_i] @ x = sqrt(1-w) * n_i.(q_i-p_i)
        - Point-to-point rows (3N):  sqrt(w) * skew(p_i) @ x   = sqrt(w) * (q_i-p_i)
    This constrains the tangential sliding that pure plane-to-point misses.

    The small-angle result is then projected to the nearest proper rotation
    via SVD to keep det(R) = +1.

    Args:
        source_pts: (N, 3) source surface points (already sternum-centred)
        normals:    (N, 3) unit normals at source surface points
        target_pts: (N, 3) matched target points (already sternum-centred)
        point_to_point_weight: blend weight in [0, 1]; 0 = pure plane-to-point

    Returns:
        (3, 3) rotation matrix (proper, det = +1)
    """
    w = float(np.clip(point_to_point_weight, 0.0, 1.0))

    # diff = q - p
    d = target_pts - source_pts   # (N, 3)

    # --- Plane-to-point rows ---
    # cross product:  p x n  (per row)
    # n . ([a,b,c] x p) = [a,b,c] . (p x n)
    cn = np.cross(source_pts, normals)  # (N, 3)
    rhs_ptp = np.sum(d * normals, axis=1)  # (N,)

    if w < 1e-12:
        # Pure plane-to-point (original path)
        A = cn
        rhs = rhs_ptp
    else:
        # Build (3N, 3) skew-symmetric matrix for point-to-point rows
        # [a,b,c] x p_i gives 3 equations per correspondence
        px, py, pz = source_pts[:, 0], source_pts[:, 1], source_pts[:, 2]
        zeros = np.zeros_like(px)
        col0 = np.column_stack([zeros, -pz,  py]).ravel()
        col1 = np.column_stack([ pz, zeros, -px]).ravel()
        col2 = np.column_stack([-py,    px, zeros]).ravel()
        A_pt = np.column_stack([col0, col1, col2])
        rhs_pt = d.ravel()

        if w > 1.0 - 1e-12:
            # Pure point-to-point
            A = A_pt
            rhs = rhs_pt
        else:
            # Hybrid: weighted stack of plane-to-point and point-to-point
            w_plane = np.sqrt(1.0 - w)
            w_point = np.sqrt(w)
            A = np.vstack([w_plane * cn, w_point * A_pt])
            rhs = np.concatenate([w_plane * rhs_ptp, w_point * rhs_pt])

    # A @ x = rhs  →  solve by least squares
    result, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    a, b, c = result

    # Build skew-symmetric matrix and approximate R
    R_approx = np.array([
        [1,  -c,  b],
        [c,   1, -a],
        [-b,  a,  1],
    ])

    # Project to nearest proper rotation via SVD
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    return R


# ---------------------------------------------------------------------------
# 5. Main alignment loop
# ---------------------------------------------------------------------------
# 4b. Plot many-to-one correspondence diagnostic
# ---------------------------------------------------------------------------
def _plot_many_to_one(
        src_centered: np.ndarray,
        tgt_centered: np.ndarray,
        src_idx: np.ndarray,
        tgt_idx: np.ndarray,
        counts: np.ndarray,
        unique_targets: np.ndarray,
) -> None:
    """
    Visualise many-to-one correspondences using PyVista.

    Shows all source and target points involved in correspondences, with
    many-to-one pairs highlighted in red and connected by lines.

    Args:
        src_centered: (N, 3) full source points (sternum-centred)
        tgt_centered: (M, 3) full target points (sternum-centred)
        src_idx: (K,) indices of matched source points
        tgt_idx: (K,) indices of matched target points
        counts: per-unique-target match counts from np.unique
        unique_targets: unique target indices from np.unique
    """
    try:
        import pyvista as pv
    except ImportError:
        print("  WARNING: pyvista not available, skipping many-to-one plot")
        return

    # Identify many-to-one correspondences (target matched by >1 source)
    shared_mask = counts > 1
    shared_tgt_set = set(unique_targets[shared_mask])

    # Separate 1-to-1 and many-to-one correspondence indices
    is_many = np.array([t in shared_tgt_set for t in tgt_idx])
    m2o_src_idx = src_idx[is_many]
    m2o_tgt_idx = tgt_idx[is_many]
    one2one_src_idx = src_idx[~is_many]
    one2one_tgt_idx = tgt_idx[~is_many]

    plotter = pv.Plotter()
    plotter.set_background('white')

    # All source points (small, grey)
    plotter.add_points(
        pv.PolyData(src_centered[src_idx]),
        color='grey', point_size=3,
        render_points_as_spheres=True, label='Source (1-to-1)',
    )

    # All target points (small, blue)
    plotter.add_points(
        pv.PolyData(tgt_centered[one2one_tgt_idx]),
        color='steelblue', point_size=3,
        render_points_as_spheres=True, label='Target (1-to-1)',
    )

    # Many-to-one source points (red)
    if len(m2o_src_idx) > 0:
        plotter.add_points(
            pv.PolyData(src_centered[m2o_src_idx]),
            color='red', point_size=8,
            render_points_as_spheres=True, label='Source (many-to-1)',
        )

        # Shared target points (orange, larger)
        shared_tgt_unique = np.array(list(shared_tgt_set))
        plotter.add_points(
            pv.PolyData(tgt_centered[shared_tgt_unique]),
            color='orange', point_size=10,
            render_points_as_spheres=True, label='Target (shared)',
        )

        # Draw lines from each many-to-one source to its target
        lines_pts = []
        lines_conn = []
        for i in range(len(m2o_src_idx)):
            base = len(lines_pts)
            lines_pts.append(src_centered[m2o_src_idx[i]])
            lines_pts.append(tgt_centered[m2o_tgt_idx[i]])
            lines_conn.append([2, base, base + 1])

        if lines_pts:
            lines_mesh = pv.PolyData(
                np.array(lines_pts),
                lines=np.hstack(lines_conn),
            )
            plotter.add_mesh(
                lines_mesh, color='red', line_width=1.5,
                opacity=0.6, label='Many-to-1 links',
            )

    # Origin marker (sternum)
    plotter.add_points(
        pv.PolyData(np.zeros((1, 3))),
        color='green', point_size=15,
        render_points_as_spheres=True, label='Sternum (origin)',
    )

    plotter.add_legend(face=None)
    plotter.add_axes()
    plotter.add_title('Many-to-One Correspondence Diagnostic (Iter 1)')
    plotter.show()


# ---------------------------------------------------------------------------
# 4c. Plot aligned mesh and point cloud (only alignment regions)
# ---------------------------------------------------------------------------
def plot_alignment_result(
        aligned_mesh_pts: np.ndarray,
        target_pts: np.ndarray,
        sternum_pos: np.ndarray = None,
        title: str = "Plane-to-Point ICP Alignment Result",
        save_path: str = None,
        show_correspondences: bool = False,
        max_distance: float = 15.0,
) -> None:
    """
    Visualise the aligned mesh overlaid on the target point cloud.

    Shows only the regions used for alignment (filtered mesh elements
    and filtered target point cloud).

    Args:
        aligned_mesh_pts: (N, 3) aligned source mesh points (in target frame)
        target_pts: (M, 3) target point cloud (filtered to alignment region)
        sternum_pos: (3,) sternum position to mark (optional)
        title: plot title
        save_path: if provided, save screenshot to this path
        show_correspondences: if True, draw lines between closest pairs
        max_distance: max distance for correspondence lines (mm)
    """
    try:
        import pyvista as pv
    except ImportError:
        print("  WARNING: pyvista not available, skipping alignment plot")
        return

    plotter = pv.Plotter()
    plotter.set_background('white')

    # Target point cloud (supine) - blue
    plotter.add_points(
        pv.PolyData(target_pts),
        color='steelblue',
        point_size=4,
        render_points_as_spheres=True,
        opacity=0.6,
        label=f'Target PC ({len(target_pts)} pts)',
    )

    # Aligned source mesh (prone) - red
    plotter.add_points(
        pv.PolyData(aligned_mesh_pts),
        color='red',
        point_size=5,
        render_points_as_spheres=True,
        opacity=0.9,
        label=f'Aligned Mesh ({len(aligned_mesh_pts)} pts)',
    )

    # Draw correspondence lines if requested
    if show_correspondences:
        tree = cKDTree(target_pts)
        dists, indices = tree.query(aligned_mesh_pts, k=1)

        # Only show correspondences within max_distance
        valid_mask = dists <= max_distance
        valid_src = aligned_mesh_pts[valid_mask]
        valid_tgt = target_pts[indices[valid_mask]]

        if len(valid_src) > 0:
            # Subsample for cleaner visualization (max 500 lines)
            n_lines = min(500, len(valid_src))
            step = max(1, len(valid_src) // n_lines)

            lines_pts = []
            lines_conn = []
            for i in range(0, len(valid_src), step):
                base = len(lines_pts)
                lines_pts.append(valid_src[i])
                lines_pts.append(valid_tgt[i])
                lines_conn.append([2, base, base + 1])

            if lines_pts:
                lines_mesh = pv.PolyData(
                    np.array(lines_pts),
                    lines=np.hstack(lines_conn),
                )
                plotter.add_mesh(
                    lines_mesh, color='yellow', line_width=1.0,
                    opacity=0.4, label='Correspondences',
                )

    # Sternum marker (if provided)
    if sternum_pos is not None:
        plotter.add_points(
            pv.PolyData(sternum_pos.reshape(1, 3)),
            color='green',
            point_size=15,
            render_points_as_spheres=True,
            label='Sternum Superior',
        )

    # Calculate and display alignment statistics
    tree = cKDTree(target_pts)
    dists, _ = tree.query(aligned_mesh_pts, k=1)
    rmse = np.sqrt(np.mean(dists ** 2))
    mean_dist = np.mean(dists)

    stats_text = (
        f"RMSE: {rmse:.2f} mm\n"
        f"Mean: {mean_dist:.2f} mm\n"
        f"Mesh pts: {len(aligned_mesh_pts)}\n"
        f"Target pts: {len(target_pts)}"
    )
    plotter.add_text(stats_text, position='upper_right', font_size=10, color='black')

    plotter.add_legend(face=None, bcolor='white')
    plotter.add_axes()
    plotter.add_title(title)

    if save_path:
        plotter.show(screenshot=save_path)
        print(f"  Alignment plot saved to: {save_path}")
    else:
        plotter.show()


# ---------------------------------------------------------------------------
# 5. Main alignment loop
# ---------------------------------------------------------------------------
def surface_to_point_align(
        vl_id,
        mesh,
        target_pts: np.ndarray,
        source_sternum_sup: np.ndarray,
        target_sternum_sup: np.ndarray,
        max_distance: float = 20.0,
        max_iterations: int = 200,
        convergence_threshold: float = 1e-6,
        trim_percentage: float = 0.1,
        res: int = 10,
        verbose: bool = False,
        elems: list = None,
        point_to_point_weight: float = 0.0,
        target_region_filter: bool = True,
        target_region_padding: float = 15.0,
        target_region_padding_inferior: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Plane-to-point ICP with sternum superior fixed at origin.

    IMPORTANT: When using selected elements (via `elems` parameter), the supine
    point cloud should also be filtered to match the same anatomical region.
    Set `target_region_filter=True` (default) to automatically filter the target
    point cloud to the bounding box of the selected mesh region.

    Algorithm per iteration:
        1. Evaluate surface points and normals from the source mesh
        2. Find correspondences within max_distance
        3. Optionally trim worst correspondences
        4. Solve linearised plane-to-point for incremental rotation
        5. Accumulate rotation (applied around origin = sternum)

    Args:
        mesh: morphic.Mesh with 2D H3×H3 elements, OR (N, 3) numpy array
              of source points.  When an ndarray is passed, normals are
              estimated from the point cloud via local PCA.
        target_pts: (M, 3) supine ribcage point cloud
        source_sternum_sup: (3,) prone sternum superior position
        target_sternum_sup: (3,) supine sternum superior position
        max_distance: radius for correspondence search (mm, default 20)
        max_iterations: ICP iteration limit
        convergence_threshold: stop when RMSE change < this
        trim_percentage: reject this fraction of worst correspondences
        res: sampling resolution per element axis (res×res per element)
        verbose: print iteration progress
        elems: optional list of element indices to sample from the mesh;
               None = use all elements. IGNORED when mesh is an ndarray.
        point_to_point_weight: blend weight in [0, 1] for point-to-point
               regularisation; 0 = pure plane-to-point (default),
               1 = pure point-to-point
        target_region_filter: if True and elems is provided, filter target
               point cloud to match the mesh region bounding box
        target_region_padding: padding (mm) around mesh bounding box for
               target point cloud filtering (applied to all sides except
               inferior Z)
        target_region_padding_inferior: padding (mm) for the inferior (Z-min)
               boundary. Default 0 to exclude noisy inferior points.

    Returns:
        R_total: (3, 3) accumulated rotation matrix
        T_total: (4, 4) homogeneous transformation matrix
        info: dict with convergence metrics and history
    """
    # prone_ribcage_mesh_coords = get_surface_mesh_coords(mesh, res=26)
    #
    # # Get element centers and visualize (use this to decide which elements to select)
    # centers_array, num_elements = get_mesh_elements_2(mesh)
    # plot_mesh_elements(prone_ribcage_mesh_coords, centers_array, range(num_elements))
    #
    # selected_elements = [0, 1, 6, 7, 8, 9, 14, 15, 16, 17, 22, 23]
    # if selected_elements is not None:
    #     prone_ribcage_alignment_coords = get_mesh_with_selected_elements(
    #         prone_ribcage, selected_elements, res=26
    #     )
    #     if verbose:
    #         print(f"  Using {len(selected_elements)}/{num_elements} elements for alignment")
    #         print(f"  Selected elements: {selected_elements}")
    #         print(f"  Alignment points: {prone_ribcage_alignment_coords.shape[0]} "
    #               f"(full mesh: {prone_ribcage_mesh_coords.shape[0]})")
    #
    #     selected_centers = centers_array[selected_elements]
    #     plot_mesh_elements(prone_ribcage_alignment_coords, selected_centers,
    #                            selected_elements)
    # else:
    #     # No selection - use all elements
    #     prone_ribcage_alignment_coords = prone_ribcage_mesh_coords
    #     if verbose:
    #         print(f"  Using all {num_elements} elements for alignment")

    target_pts = np.asarray(target_pts, dtype=np.float64)
    src_ss = np.asarray(source_sternum_sup, dtype=np.float64).flatten()
    tgt_ss = np.asarray(target_sternum_sup, dtype=np.float64).flatten()

    # Detect whether source is a mesh or a point cloud
    use_point_cloud = isinstance(mesh, np.ndarray)
    if use_point_cloud:
        source_pts_fixed = np.asarray(mesh, dtype=np.float64)
        Xi = None
        if elems is not None and verbose:
            print("  WARNING: 'elems' parameter is ignored when mesh is an ndarray.")
            print("           Pass a morphic.Mesh to use element selection.")
    else:
        Xi = mesh.grid(res, method='center')

    # --- Sample initial source points for region filtering ---
    if use_point_cloud:
        src_pts_initial = source_pts_fixed
    else:
        src_pts_initial, _ = compute_mesh_points_and_normals(mesh, Xi, elems=elems)

    # --- Mutual region filtering ---
    # Step 1: filter target PC to source mesh bbox (with asymmetric padding)
    # Step 2: filter source mesh points to target PC extent (reciprocal)
    # This ensures both datasets cover the same anatomical region.
    src_mask = None
    if target_region_filter and elems is not None and not use_point_cloud:
        src_mask, target_pts, filter_info = filter_mutual_region(
            src_pts=src_pts_initial,
            tgt_pts=target_pts,
            src_ss=src_ss,
            tgt_ss=tgt_ss,
            padding=target_region_padding,
            padding_inferior=target_region_padding_inferior,
            verbose=verbose,
        )

    # Centre target on its sternum
    tgt_centered = target_pts - tgt_ss

    # Build KD-tree on centered target once (reused every iteration)
    tgt_tree = cKDTree(tgt_centered)

    # Initialise
    R_total = np.eye(3)
    prev_rmse = np.inf
    iteration_history = []

    if verbose:
        print("\n  Initial rotation (identity matrix):")
        print_rotation_angles(R_total, "Initial angles")

    for it in range(max_iterations):
        # --- 1. Sample current surface points & normals ---
        if use_point_cloud:
            src_pts_raw = source_pts_fixed
            normals_raw = estimate_normals_from_points(
                (R_total @ (source_pts_fixed - src_ss).T).T
            )
            # normals_raw are already in rotated frame; un-rotate so the
            # rotate step below produces the correct result
            normals_raw = (R_total.T @ normals_raw.T).T
        else:
            src_pts_raw, normals_raw = compute_mesh_points_and_normals(
                mesh, Xi, elems=elems
            )

        # Apply reciprocal source mask (exclude mesh pts outside target extent)
        if src_mask is not None:
            src_pts_raw = src_pts_raw[src_mask]
            normals_raw = normals_raw[src_mask]

        # Centre on source sternum, then rotate to current alignment
        src_centered = (R_total @ (src_pts_raw - src_ss).T).T
        normals_rot = (R_total @ normals_raw.T).T

        # --- 2. Find correspondences within radius ---
        src_idx, tgt_idx, dists = find_correspondences_within_radius(
            src_centered, tgt_centered, max_distance, tree=tgt_tree
        )

        n_corr = len(src_idx)
        if n_corr < 6:
            if verbose:
                print(f"  Iter {it+1}: only {n_corr} correspondences, stopping")
            break

        # --- 3. Trim worst correspondences ---
        if trim_percentage > 0 and n_corr > 20:
            threshold = np.percentile(dists, (1.0 - trim_percentage) * 100)
            keep = dists <= threshold
            src_idx = src_idx[keep]
            tgt_idx = tgt_idx[keep]
            dists = dists[keep]
            n_corr = len(src_idx)

        # --- 3b. Many-to-one diagnostic ---
        unique_targets, counts = np.unique(tgt_idx, return_counts=True)
        n_unique = len(unique_targets)
        n_duplicated = int(np.sum(counts > 1))
        max_count = int(counts.max())
        if verbose and (it < 3 or (it + 1) % 50 == 0):
            print(f"  Iter {it+1} correspondences: {n_corr} total, "
                  f"{n_unique} unique targets ({n_unique/n_corr*100:.0f}%), "
                  f"{n_duplicated} shared, worst={max_count}-to-1")

            # Detail shared targets: distance spread and boundary check
            if n_duplicated > 0:
                shared_mask = counts > 1
                shared_tgt_ids = unique_targets[shared_mask]
                shared_counts = counts[shared_mask]

                # Build target local density (neighbours within 5mm)
                n_neighbors_5mm = tgt_tree.query_ball_point(
                    tgt_centered[shared_tgt_ids], r=5.0,
                    return_length=True
                )

                print(f"    Shared target details (top 5 worst):")
                sort_idx = np.argsort(-shared_counts)
                for rank, si in enumerate(sort_idx[:5]):
                    tid = shared_tgt_ids[si]
                    cnt = shared_counts[si]
                    match_mask = tgt_idx == tid
                    match_dists = dists[match_mask]
                    closest = match_dists.min()
                    farthest = match_dists.max()
                    spread = farthest - closest
                    n_local = n_neighbors_5mm[si]
                    boundary_flag = " (BOUNDARY)" if n_local < 10 else ""
                    loc = tgt_centered[tid]
                    print(f"      tgt[{tid}] at ({loc[0]:+.1f}, {loc[1]:+.1f}, {loc[2]:+.1f}): "
                          f"{cnt}-to-1, "
                          f"dists=[{closest:.1f}, {farthest:.1f}] mm, "
                          f"spread={spread:.1f} mm, "
                          f"local_density={n_local} pts{boundary_flag}")

        # --- 3c. Plot many-to-one diagnostic (first iteration only) ---
        if verbose and it == 0 and n_duplicated > 0:
            _plot_many_to_one(
                src_centered, tgt_centered, src_idx, tgt_idx,
                counts, unique_targets,
            )

        P = src_centered[src_idx]
        N = normals_rot[src_idx]
        Q = tgt_centered[tgt_idx]

        # --- 4. Solve plane-to-point for incremental rotation ---
        R_delta = solve_plane_to_point_rotation(
            P, N, Q, point_to_point_weight=point_to_point_weight
        )

        # --- 5. Accumulate rotation ---
        R_total = R_delta @ R_total

        # --- 6. Compute RMSE and track angles ---
        new_src = (R_delta @ P.T).T
        plane_err = plane_to_point_error(new_src, (R_delta @ N.T).T, Q)
        rmse = np.sqrt(np.mean(plane_err ** 2))

        # Extract current Euler angles from accumulated rotation
        angle_x, angle_y, angle_z = rotation_matrix_to_euler_angles(R_total)
        total_angle = np.degrees(np.arccos(np.clip((np.trace(R_total) - 1) / 2, -1, 1)))

        iteration_history.append({
            'iteration': it + 1,
            'rmse': rmse,
            'n_correspondences': n_corr,
            'rotation_change': float(np.linalg.norm(R_delta - np.eye(3))),
            'angle_x_deg': angle_x,
            'angle_y_deg': angle_y,
            'angle_z_deg': angle_z,
            'total_angle_deg': total_angle,
        })

        if verbose and (it < 5 or (it + 1) % 10 == 0):
            print(f"  Iter {it+1}: RMSE={rmse:.4f} mm, corr={n_corr}, "
                  f"angles=(X:{angle_x:+.2f}°, Y:{angle_y:+.2f}°, Z:{angle_z:+.2f}°)")

        # --- 7. Convergence check ---
        if abs(prev_rmse - rmse) < convergence_threshold:
            if verbose:
                print(f"  Converged at iteration {it+1}")
            break
        prev_rmse = rmse

    # --- Build outputs ---
    # Final RMSE on all correspondences
    if use_point_cloud:
        src_final = source_pts_fixed
    else:
        src_final, _ = compute_mesh_points_and_normals(
            mesh, Xi, elems=elems
        )
    # Apply same mask used during iterations
    if verbose:
        print(f"\n  [Debug] src_final before mask: {len(src_final)} points")
        print(f"  [Debug] src_mask is None: {src_mask is None}")
    if src_mask is not None:
        if verbose:
            print(f"  [Debug] src_mask sum (True count): {np.sum(src_mask)}")
        src_final = src_final[src_mask]
    if verbose:
        print(f"  [Debug] src_final after mask: {len(src_final)} points")
        print(f"  [Debug] target_pts (filtered): {len(target_pts)} points")
    src_final_c = (R_total @ (src_final - src_ss).T).T
    s_idx, t_idx, _ = find_correspondences_within_radius(
        src_final_c, tgt_centered, max_distance, tree=tgt_tree
    )
    if len(s_idx) > 0:
        final_dists = np.linalg.norm(
            src_final_c[s_idx] - tgt_centered[t_idx], axis=1
        )
        final_rmse = float(np.sqrt(np.mean(final_dists ** 2)))
        final_mean = float(np.mean(final_dists))
        final_std = float(np.std(final_dists))
    else:
        final_rmse = np.inf
        final_mean = np.inf
        final_std = np.inf

    # 4x4 homogeneous matrix: T maps original source coords to target frame
    T_total = np.eye(4)
    T_total[:3, :3] = R_total
    T_total[:3, 3] = tgt_ss - R_total @ src_ss

    converged = (len(iteration_history) > 0 and
                 iteration_history[-1].get('rotation_change', 1.0) < 1e-4)

    info = {
        'method': 'plane_to_point_icp',
        'iterations': len(iteration_history),
        'converged': converged,
        'euclidean_rmse': final_rmse,
        'euclidean_mean': final_mean,
        'euclidean_std': final_std,
        'n_correspondences': len(s_idx),
        'R_total': R_total,
        'source_sternum_offset': src_ss,
        'target_sternum_offset': tgt_ss,
        'iteration_history': iteration_history,
        'target_pts_filtered': target_pts,  # filtered target PC used for alignment
    }

    if verbose:
        print(f"\nFinal: RMSE={final_rmse:.4f} mm, "
              f"correspondences={len(s_idx)}, "
              f"iterations={len(iteration_history)}")
        print_rotation_angles(R_total, "Final rotation angles")

    # Add Euler angles to info dict
    angle_x, angle_y, angle_z = rotation_matrix_to_euler_angles(R_total)
    total_angle = np.degrees(np.arccos(np.clip((np.trace(R_total) - 1) / 2, -1, 1)))
    info['euler_angles_deg'] = {'angle_x': angle_x, 'angle_y': angle_y, 'angle_z': angle_z}
    info['total_rotation_deg'] = total_angle

    convergence_save_path = Path(__file__).parent.parent / "output" / "figs" / f"convergence_{vl_id}.png"
    plot_convergence_diagram(info, save_path=str(convergence_save_path))

    # --- Visualize aligned mesh and target point cloud (alignment regions only) ---
    # Transform aligned mesh to target coordinate frame (add back sternum offset)
    aligned_mesh_in_target_frame = src_final_c + tgt_ss
    target_pts_in_target_frame = tgt_centered + tgt_ss

    # alignment_plot_save_path = Path(__file__).parent.parent / "output" / "figs" / f"alignment_result_{vl_id}.png"
    plot_alignment_result(
        aligned_mesh_pts=aligned_mesh_in_target_frame,
        target_pts=target_pts_in_target_frame,
        sternum_pos=tgt_ss,
        title=f"Plane-to-Point ICP Alignment - {vl_id}",
        save_path=None,
        show_correspondences=False,
    )
    plot_all(
        point_cloud=target_pts_in_target_frame,
        mesh_points=aligned_mesh_in_target_frame,
        anat_landmarks=tgt_ss,
    )

    error, mapped_idx = breast_metadata.closest_distances(
        aligned_mesh_in_target_frame, target_pts_in_target_frame
    )
    rib_error_mag = np.linalg.norm(error, axis=1)

    # Error visualization - full mesh (prone tosupine)
    plot_evaluate_alignment(
        supine_pts=target_pts_in_target_frame,
        transformed_prone_mesh=aligned_mesh_in_target_frame,
        distances=rib_error_mag,
        idxs=mapped_idx,
        worst_n=60,
        cmap="viridis",
        point_size=3,
        arrow_scale=20,
        show_scalar_bar=True,
        return_data=False
    )

    return R_total, T_total, info


# ---------------------------------------------------------------------------
# 6. Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import morphic
    import external.breast_metadata_mdv.breast_metadata as breast_metadata
    from utils import extract_contour_points, get_landmarks_as_array
    from alignment import (
        get_surface_mesh_coords,
        apply_transform_to_coords,
        filter_point_cloud_asymmetric,
        cleanup_spine_region,
        # select_elements_by_region,
        get_mesh_with_selected_elements,
        print_alignment_accuracy_report,
    )
    from utils_plot import plot_all
    from structures import Subject
    from readers import load_subject
    from clean_point_cloud import clean_ribcage_point_cloud

    # ======================================================================
    # Configuration
    # ======================================================================
    vl_ids = [9]
    ROOT_PATH_MRI = Path(r'U:\projects\volunteer_camri\old_data\mri_t2')
    SOFT_TISSUE_ROOT = Path(r'U:\projects\dashboard\picker_points')
    ANATOMICAL_JSON_BASE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results")
    PRONE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh")
    SUPINE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage")
    CLEAN_PC_CONFIG_ROOT = Path(__file__).parent.parent / "output" / "clean_up_pc_config"

    orientation_flag = 'RAI'

    # Element selection: None = all elements, or a list of indices
    ELEMENT_INDICES = None
    # Region-based selection: set to a dict of kwargs for select_elements_by_region,
    # or None to skip. Overrides ELEMENT_INDICES when set.
    ELEMENT_REGION = None  # e.g. {'y_region': 'anterior', 'y_percentile': 60}
    # Hybrid objective weight: 0 = pure plane-to-point, 1 = pure point-to-point
    POINT_TO_POINT_WEIGHT = 0.0
    # Target region padding: set to False to skip bounding-box filtering,
    # or True to filter supine PC to mesh region with padding
    USE_TARGET_REGION_PADDING = True

    for vl_id in vl_ids:
        vl_id_str = f"VL{vl_id:05d}"
        print(f"\n{'='*60}")
        print(f"PLANE-TO-POINT ICP ALIGNMENT")
        print(f"Subject: {vl_id_str}")
        print(f"{'='*60}")

        # ==============================================================
        # 1. Load subject data
        # ==============================================================
        subject = load_subject(
            vl_id=vl_id,
            positions=["prone", "supine"],
            dicom_root=ROOT_PATH_MRI,
            anatomical_json_base_root=ANATOMICAL_JSON_BASE_ROOT,
            soft_tissue_root=SOFT_TISSUE_ROOT,
        )

        anat_prone = subject.scans["prone"].anatomical_landmarks
        anat_supine = subject.scans["supine"].anatomical_landmarks

        if anat_prone.sternum_superior is None or anat_supine.sternum_superior is None:
            raise ValueError(f"Subject {vl_id_str} missing sternum superior landmarks")

        sternum_prone = np.vstack([anat_prone.sternum_superior, anat_prone.sternum_inferior])
        sternum_supine = np.vstack([anat_supine.sternum_superior, anat_supine.sternum_inferior])

        # ==============================================================
        # 2. Load prone ribcage mesh and supine point cloud
        # ==============================================================
        prone_mesh_file = PRONE_RIBCAGE_ROOT / f"{vl_id_str}_ribcage_prone.mesh"
        supine_seg_file = SUPINE_RIBCAGE_ROOT / f"rib_cage_{vl_id_str}.nii.gz"

        prone_ribcage = morphic.Mesh(str(prone_mesh_file))

        supine_ribcage_mask = breast_metadata.readNIFTIImage(
            str(supine_seg_file), orientation_flag, swap_axes=True
        )
        supine_ribcage_pc = extract_contour_points(supine_ribcage_mask, 20000)

        # # Clean up supine point cloud using per-subject config
        # clean_config_path = CLEAN_PC_CONFIG_ROOT / f"{vl_id_str}_config.json"
        # if clean_config_path.exists():
        #     image_grid = breast_metadata.SCANToPyvistaImageGrid(
        #         supine_ribcage_mask, orientation_flag
        #     )
        #     supine_ribcage_pc = clean_ribcage_point_cloud(
        #         pc_data=supine_ribcage_pc,
        #         image_grid=image_grid,
        #         config_filepath=str(clean_config_path),
        #         run_plot_all=True,
        #     )
        #     print(f"  Cleaned supine point cloud: {supine_ribcage_pc.shape[0]} points")
        # else:
        #     print(f"  WARNING: No clean config found at {clean_config_path}, "
        #           f"using raw point cloud")

        print(f"  Prone mesh elements: {prone_ribcage.elements.size()}")
        print(f"  Supine point cloud: {supine_ribcage_pc.shape[0]} points")

        # ==============================================================
        # 3. Determine element selection
        # ==============================================================
        # if ELEMENT_REGION is not None:
        #     elem_indices = select_elements_by_region(
        #         prone_ribcage, **ELEMENT_REGION
        #     )
        #     print(f"  Selected {len(elem_indices)} elements by region")
        # else:
        #     elem_indices = ELEMENT_INDICES
        #
        # if elem_indices is not None:
        #     print(f"  Using {len(elem_indices)} of "
        #           f"{prone_ribcage.elements.size()} elements")
        # else:
        #     print(f"  Using all {prone_ribcage.elements.size()} elements")

        # ==============================================================
        # 4. Run plane-to-point alignment
        # ==============================================================
        R, T_total, info = surface_to_point_align(
            vl_id=vl_id_str,
            mesh=prone_ribcage,
            target_pts=supine_ribcage_pc,
            source_sternum_sup=sternum_prone[0],
            target_sternum_sup=sternum_supine[0],
            max_distance=20.0,
            max_iterations=200,
            convergence_threshold=1e-6,
            trim_percentage=0.1,
            res=10,
            verbose=True,
            # elems=elem_indices,
            point_to_point_weight=POINT_TO_POINT_WEIGHT,
            target_region_filter=USE_TARGET_REGION_PADDING,
        )

        # ==============================================================
        # 5. Transform prone data and calculate errors
        # ==============================================================
        source_anchor = sternum_prone[0]
        target_anchor = sternum_supine[0]

        prone_mesh_coords = get_surface_mesh_coords(prone_ribcage, res=26)
        prone_transformed = apply_transform_to_coords(
            prone_mesh_coords, R, source_anchor, target_anchor
        )
        sternum_prone_transformed = apply_transform_to_coords(
            sternum_prone, R, source_anchor, target_anchor
        )

        sternum_error = np.linalg.norm(sternum_prone_transformed[0] - sternum_supine[0])

        # Query direction: prone→supine (for each transformed prone point,
        # find nearest supine point). Consistent with ICP internal direction.
        error_vecs, _ = breast_metadata.closest_distances(
            prone_transformed, supine_ribcage_pc
        )
        rib_error_mag = np.linalg.norm(error_vecs, axis=1)

        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"{'='*60}")
        print(f"  Method: {info['method']}")
        print(f"  Converged: {info['converged']}")
        print(f"  Iterations: {info['iterations']}")
        print(f"  Sternum error: {sternum_error:.6f} mm")
        print(f"  Ribcage RMSE: {np.sqrt(np.mean(rib_error_mag**2)):.4f} mm")
        print(f"  Ribcage Mean +/- SD: {np.mean(rib_error_mag):.4f} +/- {np.std(rib_error_mag):.4f} mm")

        print_alignment_accuracy_report(rib_error_mag, sternum_error, info)

        # ==============================================================
        # 6. Plot convergence diagram
        # ==============================================================
        convergence_save_path = Path(__file__).parent.parent / "output" / "figs" / f"convergence_{vl_id_str}.png"
        plot_convergence_diagram(info, save_path=str(convergence_save_path))

        # ==============================================================
        # 7. Visualize
        # ==============================================================
        plot_all(
            point_cloud=supine_ribcage_pc,
            mesh_points=prone_transformed,
            anat_landmarks=[sternum_prone_transformed, sternum_supine],
        )
