"""
Deprecated Alignment Functions

Functions moved here during refactoring.
These functions have no external callers or are superseded by newer implementations.
Kept for reference only.

Moved on: 2026-03-09
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Dict

from alignment_utils import (
    filter_point_cloud_asymmetric,
    inverse_transform_to_source_frame,
    get_surface_mesh_coords,
    svd_rotation_point_to_point,
)


# ---------------------------------------------------------------------------
# cleanup_spine_region — moved from alignment.py
# Used by: surface_to_point_alignment.py __main__ block,
#          alignment_past_commit.py (own copy)
# ---------------------------------------------------------------------------
def cleanup_spine_region(
        pc_data: np.ndarray,
        x_spine_offset: float = 20.0,
        y_spine_offset: float = 50.0,
        run_plot_all: bool = False,
        verbose: bool = False
) -> np.ndarray:
    """
    Remove points around the spine (posterior/back region).

    The spine region is highly variable and not part of the ribcage surface
    we want to align. This removes points:
    - Within +/-x_spine_offset of the median X coordinate (spine centerline)
    - Within y_spine_offset of the posterior edge

    Args:
        pc_data: (N, 3) point cloud coordinates
        x_spine_offset: lateral offset from median to define spine region (mm)
        y_spine_offset: anterior offset from posterior edge for spine (mm)
        run_plot_all: show debug plots
        verbose: print progress

    Returns:
        (M, 3) cleaned point cloud (M < N)
    """
    if verbose:
        print(f"Removing spine region - current size: {pc_data.shape[0]}")

    x_median = np.median(pc_data[:, 0])
    pc_data = pc_data[
        (pc_data[:, 0] <= (x_median - x_spine_offset)) |  # Left of spine
        (pc_data[:, 0] >= (x_median + x_spine_offset)) |  # Right of spine
        (pc_data[:, 1] < np.max(pc_data[:, 1]) - y_spine_offset)  # Anterior region
    ]

    if verbose:
        print(f"After removing spine region: {pc_data.shape[0]} points")

    if run_plot_all:
        from utils_plot import plot_all
        plot_all(point_cloud=pc_data)

    return pc_data


# ---------------------------------------------------------------------------
# filter_anterior_by_widest_point — moved from alignment_preprocessing.py
# ---------------------------------------------------------------------------
def filter_anterior_by_widest_point(
        points: np.ndarray,
        padding: float = 5.0,
        n_slices: int = 20,
        slice_thickness: float = None,
        verbose: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    DEPRECATED: Moved from alignment_preprocessing.py.

    Select the anterior region of a ribcage by finding the axial slice
    with the widest lateral (X) extent, then using the Y values of the
    two widest points in that slice as the anterior/posterior boundary.
    """
    pts = np.asarray(points, dtype=np.float64)

    z_min = pts[:, 2].min()
    z_max = pts[:, 2].max()

    if slice_thickness is not None:
        edges = np.arange(z_min, z_max + slice_thickness, slice_thickness)
    else:
        edges = np.linspace(z_min, z_max, n_slices + 1)

    best_width = -1.0
    best_slice_idx = 0
    slice_stats = []

    for i in range(len(edges) - 1):
        z_lo, z_hi = edges[i], edges[i + 1]
        mask_slice = (pts[:, 2] >= z_lo) & (pts[:, 2] < z_hi)
        n_pts = np.sum(mask_slice)

        if n_pts < 2:
            slice_stats.append({
                'z_lo': z_lo, 'z_hi': z_hi, 'n_pts': int(n_pts),
                'x_width': 0.0,
            })
            continue

        slice_pts = pts[mask_slice]
        x_width = slice_pts[:, 0].max() - slice_pts[:, 0].min()
        slice_stats.append({
            'z_lo': z_lo, 'z_hi': z_hi, 'n_pts': int(n_pts),
            'x_width': x_width,
        })

        if x_width > best_width:
            best_width = x_width
            best_slice_idx = i

    z_lo = edges[best_slice_idx]
    z_hi = edges[best_slice_idx + 1]
    widest_mask = (pts[:, 2] >= z_lo) & (pts[:, 2] < z_hi)
    widest_pts = pts[widest_mask]

    x_min_idx = np.argmin(widest_pts[:, 0])
    x_max_idx = np.argmax(widest_pts[:, 0])
    x_min_pt = widest_pts[x_min_idx]
    x_max_pt = widest_pts[x_max_idx]

    y_at_x_min = x_min_pt[1]
    y_at_x_max = x_max_pt[1]

    y_cutoff = max(y_at_x_min, y_at_x_max) + padding

    keep_mask = pts[:, 1] <= y_cutoff
    filtered = pts[keep_mask]

    info = {
        'x_min_pt': x_min_pt,
        'x_max_pt': x_max_pt,
        'y_at_x_min': y_at_x_min,
        'y_at_x_max': y_at_x_max,
        'y_cutoff': y_cutoff,
        'padding': padding,
        'n_original': len(pts),
        'n_filtered': len(filtered),
        'widest_slice_z': (z_lo, z_hi),
        'widest_slice_x_width': best_width,
        'widest_slice_pts': widest_pts,
        'slice_stats': slice_stats,
    }

    if verbose:
        print(f"  Anterior filter (slice-based widest-point method):")
        print(f"    Z range: [{z_min:.1f}, {z_max:.1f}], {len(edges)-1} slices")
        print(f"    Widest slice: Z=[{z_lo:.1f}, {z_hi:.1f}], "
              f"X width={best_width:.1f} mm, {len(widest_pts)} pts")
        print(f"    Y cutoff: {y_cutoff:.1f} (max Y of widest + {padding:.0f}mm padding)")
        print(f"    Points: {len(pts)} -> {len(filtered)} "
              f"({len(filtered)/len(pts)*100:.0f}% kept)")

    return filtered, info


def plot_anterior_filter(
        points: np.ndarray,
        filtered: np.ndarray,
        info: dict,
        title: str = "Anterior Filter (Widest Slice)",
) -> None:
    """
    DEPRECATED: Moved from alignment_preprocessing.py.

    Visualize the anterior filtering result.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("  WARNING: pyvista not available, skipping plot")
        return

    plotter = pv.Plotter()
    plotter.set_background('white')

    plotter.add_points(
        pv.PolyData(points), color='gray', point_size=2, opacity=0.2,
        label=f'Original ({info["n_original"]} pts)',
    )
    plotter.add_points(
        pv.PolyData(filtered), color='steelblue', point_size=3, opacity=0.8,
        label=f'Anterior ({info["n_filtered"]} pts)',
    )

    widest_pts = info.get('widest_slice_pts')
    if widest_pts is not None and len(widest_pts) > 0:
        z_lo, z_hi = info['widest_slice_z']
        plotter.add_points(
            pv.PolyData(widest_pts), color='yellow', point_size=5, opacity=0.9,
            label=f'Widest slice Z=[{z_lo:.0f},{z_hi:.0f}]',
        )

    plotter.add_points(
        pv.PolyData(info['x_min_pt'].reshape(1, 3)),
        color='red', point_size=15, render_points_as_spheres=True,
        label=f'X-min (Y={info["y_at_x_min"]:.1f})',
    )
    plotter.add_points(
        pv.PolyData(info['x_max_pt'].reshape(1, 3)),
        color='orange', point_size=15, render_points_as_spheres=True,
        label=f'X-max (Y={info["y_at_x_max"]:.1f})',
    )

    y_cut = info['y_cutoff']
    x_range = [points[:, 0].min() - 10, points[:, 0].max() + 10]
    z_range = [points[:, 2].min() - 10, points[:, 2].max() + 10]
    plane_pts = np.array([
        [x_range[0], y_cut, z_range[0]],
        [x_range[1], y_cut, z_range[0]],
        [x_range[1], y_cut, z_range[1]],
        [x_range[0], y_cut, z_range[1]],
    ])
    plane_faces = np.array([4, 0, 1, 2, 3])
    plane_mesh = pv.PolyData(plane_pts, faces=plane_faces)
    plotter.add_mesh(
        plane_mesh, color='green', opacity=0.3,
        label=f'Y cutoff = {y_cut:.1f}',
    )

    plotter.add_legend(face=None, bcolor='white')
    plotter.add_axes()
    plotter.add_title(title)
    plotter.show()


# ---------------------------------------------------------------------------
# optimal_sternum_fixed_alignment — moved from alignment.py
# Used by: test_alignment_cohort.py, test_alignment_parameters.py,
#          test_alignment_comparison.py, analyze_convergence.py
# ---------------------------------------------------------------------------
def optimal_sternum_fixed_alignment(
        source_pts: np.ndarray,
        target_pts: np.ndarray,
        source_sternum_sup: np.ndarray,
        target_sternum_sup: np.ndarray,
        max_correspondence_distance: float = 15.0,
        max_iterations: int = 200,
        convergence_threshold: float = 1e-6,
        trim_percentage: float = 0.1,
        verbose: bool = False,
        visualize_iterations: bool = False,
        visualize_every_n: int = 10
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    DEPRECATED: Point-to-point ICP with sternum fixed at origin.
    Superseded by surface_to_point_align (plane-to-point ICP).
    Kept for backward compatibility with test scripts.

    Original location: alignment.py
    """
    source_pts = np.asarray(source_pts, dtype=np.float64)
    target_pts = np.asarray(target_pts, dtype=np.float64)
    source_ss = np.asarray(source_sternum_sup).flatten()
    target_ss = np.asarray(target_sternum_sup).flatten()

    src_centered = source_pts - source_ss
    tgt_centered = target_pts - target_ss

    if verbose:
        print(f"Centered {len(source_pts)} source and {len(target_pts)} target points")

    tree = cKDTree(tgt_centered)

    src = src_centered.copy()
    R_total = np.eye(3)
    prev_rmse = np.inf
    iteration_history = []

    it = 0
    for it in range(max_iterations):
        dists, idxs = tree.query(src)

        valid = dists <= max_correspondence_distance

        if trim_percentage > 0 and np.sum(valid) > 100:
            valid_dists = dists[valid]
            threshold = np.percentile(valid_dists, (1.0 - trim_percentage) * 100)
            valid = valid & (dists <= threshold)

        n_valid = np.sum(valid)
        if n_valid < 10:
            if verbose:
                print(f"Iteration {it + 1}: Not enough correspondences ({n_valid})")
            break

        P = src[valid]
        Q = tgt_centered[idxs[valid]]

        R_delta = svd_rotation_point_to_point(P, Q)

        src = (R_delta @ src.T).T
        R_total = R_delta @ R_total

        dists_new, idxs_new = tree.query(src)
        valid_new = dists_new <= max_correspondence_distance
        rmse = np.sqrt(np.mean(dists_new[valid_new] ** 2)) if np.any(valid_new) else np.inf

        iteration_history.append({
            "iteration": it + 1,
            "rmse": rmse,
            "n_inliers": int(np.sum(valid_new)),
            "rotation_change": np.linalg.norm(R_delta - np.eye(3))
        })

        if verbose and (it < 5 or (it + 1) % 10 == 0):
            print(f"  Iter {it + 1}: RMSE={rmse:.4f} mm, inliers={np.sum(valid_new)}")

        if abs(prev_rmse - rmse) < convergence_threshold:
            if verbose:
                print(f"  Converged at iteration {it + 1}")
            break

        prev_rmse = rmse

    dists_final, idxs_final = tree.query(src)
    valid_final = dists_final <= max_correspondence_distance
    final_rmse = np.sqrt(np.mean(dists_final[valid_final] ** 2)) if np.any(valid_final) else np.inf

    sternum_error = np.linalg.norm(R_total @ np.zeros(3))

    info = {
        "method": "optimal_sternum_fixed_svd",
        "iterations": it + 1,
        "converged": it < max_iterations - 1,
        "sternum_error_mm": sternum_error,
        "euclidean_rmse": final_rmse,
        "n_inliers": int(np.sum(valid_final)),
        "n_total_source": len(src),
        "inlier_fraction": float(np.sum(valid_final)) / len(src),
        "trim_percentage": trim_percentage,
        "R_total": R_total,
        "source_sternum_offset": source_ss,
        "target_sternum_offset": target_ss,
        "iteration_history": iteration_history,
        "target_centered": tgt_centered
    }

    if verbose:
        print(f"\nFinal results:")
        print(f"  RMSE (inlier): {final_rmse:.4f} mm")
        print(f"  Sternum error: {sternum_error:.10f} mm (exact zero)")
        print(f"  Inliers: {np.sum(valid_final)} / {len(src)}")

    return R_total, src, info


def selected_point_cloud(pc_data, params, run_plot_all):
    """
    DEPRECATED: No external callers found. Kept for reference.

    Original location: alignment.py
    """

    z_voxels_bottom = params['z_voxels_bottom']
    z_voxels_top = params['z_voxels_top']
    x_spine_offset = params['x_spine_offset']
    y_spine_offset = params['y_spine_offset']
    x_lateral_margin = params['x_lateral_margin']
    y_posterior_margin = params['y_posterior_margin']
    z_inferior_margin = params['z_inferior_margin']
    sor_k_neighbors = params['sor_k_neighbors']
    sor_std_multiplier = params['sor_std_multiplier']

    # ----------------------------------------------------
    # II. Filtering Logic
    # ----------------------------------------------------
    print("original point cloud size:", pc_data.shape)
    if run_plot_all:
        from utils_plot import plot_all
        plot_all(point_cloud=pc_data)

    # remove points on the top and bottom axial slices
    supine_ribcage_pc = filter_point_cloud_asymmetric(
        points=pc_data,
        reference=pc_data,
        tol_min=0,
        tol_max=5,
        axis=2
    )
    print("remove top and bottom axial slices, point cloud size:", pc_data.shape)
    if run_plot_all:
        plot_all(point_cloud=pc_data)

    # remove points on the back side of the ribcage around the spine
    supine_ribcage_pc = cleanup_spine_region(
        pc_data=pc_data,
        x_spine_offset=25,
        y_spine_offset=60,
        run_plot_all=True,
        verbose=True
    )
    print("remove spine region, point cloud size:", pc_data.shape)
    if run_plot_all:
        plot_all(point_cloud=pc_data)

    x_median = np.median(pc_data[:, 0])
    pc_data = pc_data[
        (pc_data[:, 0] <= (x_median - x_lateral_margin)) |
        (pc_data[:, 0] >= (x_median + x_lateral_margin)) |
        (pc_data[:, 1] < np.max(pc_data[:, 1]) - y_posterior_margin) |
        (pc_data[:, 2] >= (np.min(pc_data[:, 2] + z_inferior_margin)))
        ]

    print("remove posterior inferior points, point cloud size:", pc_data.shape)
    if run_plot_all:
        plot_all(point_cloud=pc_data)


def optimal_sternum_fixed_alignment_2(
        source_pts: np.ndarray,
        target_pts: np.ndarray,
        source_sternum_sup: np.ndarray,
        target_sternum_sup: np.ndarray,
        max_correspondence_distance: float = 15.0,
        max_iterations: int = 150,
        convergence_threshold: float = 1e-6,
        trim_percentage: float = 0.1,
        patience: int = 10,
        rotation_threshold: float = 1e-6,
        monitor_std: bool = True,
        verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    DEPRECATED: Superseded by surface_to_point_align. No external callers.
    Kept for reference.

    Original location: alignment.py

    Optimal alignment with sternum superior strictly fixed at origin (0,0,0).
    Uses point-to-point ICP with SVD rotation.
    """

    source_pts = np.asarray(source_pts, dtype=np.float64)
    target_pts = np.asarray(target_pts, dtype=np.float64)
    source_ss = np.asarray(source_sternum_sup).flatten()
    target_ss = np.asarray(target_sternum_sup).flatten()

    # Step 1: Center both on sternum superior (sternum = origin)
    src_centered = source_pts - source_ss
    tgt_centered = target_pts - target_ss

    if verbose:
        print(f"Centered {len(source_pts)} source and {len(target_pts)} target points")
        print(
            f"Source sternum at origin: {np.linalg.norm(src_centered.mean(axis=0) - (source_pts.mean(axis=0) - source_ss)):.10f}")

    # Build KD-tree for fast correspondence
    tree = cKDTree(tgt_centered)

    # Initialize
    src = src_centered.copy()
    R_total = np.eye(3)
    prev_rmse = np.inf
    iteration_history = []
    it = 0  # Initialize iteration counter

    # Early stopping tracking
    best_rmse = np.inf
    best_R = np.eye(3)
    best_src = src.copy()
    best_iteration = 0
    no_improvement_count = 0
    std_increasing = False  # Track overfitting

    # STD tracking for overfitting detection
    rmse_window = []
    std_window = []

    for it in range(max_iterations):
        # Find correspondences
        dists, idxs = tree.query(src)

        # Filter by max distance
        valid = dists <= max_correspondence_distance

        # Trimmed ICP: reject worst correspondences for robustness
        if trim_percentage > 0 and np.sum(valid) > 100:
            valid_dists = dists[valid]
            threshold = np.percentile(valid_dists, (1.0 - trim_percentage) * 100)
            valid = valid & (dists <= threshold)

        n_valid = np.sum(valid)
        if n_valid < 10:
            if verbose:
                print(f"Iteration {it + 1}: Not enough correspondences ({n_valid})")
            break

        P = src[valid]
        Q = tgt_centered[idxs[valid]]

        # Compute optimal rotation using SVD (closed-form)
        R_delta = svd_rotation_point_to_point(P, Q)

        # Apply rotation around origin (sternum stays at 0,0,0)
        src = (R_delta @ src.T).T
        R_total = R_delta @ R_total

        # Compute RMSE and STD
        dists_new, _ = tree.query(src)
        valid_new = dists_new <= max_correspondence_distance
        rmse = np.sqrt(np.mean(dists_new[valid_new] ** 2)) if np.any(valid_new) else np.inf
        std_error = np.std(dists_new[valid_new]) if np.any(valid_new) else np.inf
        rotation_change = np.linalg.norm(R_delta - np.eye(3))

        iteration_history.append({
            "iteration": it + 1,
            "rmse": rmse,
            "std": std_error,
            "n_inliers": int(np.sum(valid_new)),
            "rotation_change": rotation_change
        })

        # Track windowed metrics for overfitting detection
        rmse_window.append(rmse)
        std_window.append(std_error)
        if len(rmse_window) > 5:
            rmse_window.pop(0)
            std_window.pop(0)

        if verbose and (it < 5 or (it + 1) % 10 == 0):
            print(f"  Iter {it + 1}: RMSE={rmse:.4f} mm, STD={std_error:.4f} mm, "
                  f"inliers={np.sum(valid_new)}, rot_change={rotation_change:.6f}")

        # Update best solution
        if rmse < best_rmse:
            best_rmse = rmse
            best_R = R_total.copy()
            best_src = src.copy()
            best_iteration = it + 1
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Convergence Check 1: RMSE convergence
        rmse_converged = abs(prev_rmse - rmse) < convergence_threshold

        # Convergence Check 2: Rotation convergence
        rotation_converged = rotation_change < rotation_threshold

        # Convergence Check 3: Early stopping with patience
        patience_exceeded = no_improvement_count >= patience

        # Convergence Check 4: STD increasing (overfitting detection)
        std_increasing = False
        if monitor_std and len(std_window) >= 5:
            # Check if STD has been increasing over last 5 iterations
            std_trend = np.polyfit(range(len(std_window)), std_window, 1)[0]
            std_increasing = std_trend > 0.01  # STD increasing by >0.01 mm per iteration

        # Stop if converged or patience exceeded
        if rmse_converged and rotation_converged:
            if verbose:
                print(f"  Converged at iteration {it + 1} (RMSE and rotation stable)")
            break
        elif patience_exceeded:
            if verbose:
                print(f"  Early stopping at iteration {it + 1} (no improvement for {patience} iterations)")
                print(f"  Returning best solution from iteration {best_iteration}")
            # Return best solution instead of current
            R_total = best_R
            src = best_src
            rmse = best_rmse
            break
        elif std_increasing and it > 50:
            if verbose:
                print(f"   STD increasing detected at iteration {it + 1} (potential overfitting)")
                print(f"  Returning best solution from iteration {best_iteration}")
            # Return best solution to avoid overfitting
            R_total = best_R
            src = best_src
            rmse = best_rmse
            break

        prev_rmse = rmse

    # Final metrics
    dists_final, idxs_final = tree.query(src)
    valid_final = dists_final <= max_correspondence_distance
    final_rmse = np.sqrt(np.mean(dists_final[valid_final] ** 2)) if np.any(valid_final) else np.inf
    final_std = np.std(dists_final[valid_final]) if np.any(valid_final) else np.inf

    # Sternum error is exactly 0 (both are at origin by construction)
    sternum_error = np.linalg.norm(R_total @ np.zeros(3))  # Always 0

    # Determine stopping reason
    stop_reason = "max_iterations"
    if it < max_iterations - 1:
        if no_improvement_count >= patience:
            stop_reason = "early_stopping_patience"
        elif std_increasing and it > 50:
            stop_reason = "std_increasing_overfitting"
        else:
            stop_reason = "rmse_rotation_converged"

    info = {
        "method": "optimal_sternum_fixed_svd",
        "iterations": it + 1,
        "best_iteration": best_iteration,
        "converged": it < max_iterations - 1,
        "stop_reason": stop_reason,
        "sternum_error_mm": sternum_error,
        "euclidean_rmse": final_rmse,
        "euclidean_std": final_std,
        "best_rmse": best_rmse,
        "n_inliers": int(np.sum(valid_final)),
        "n_total_source": len(src),
        "inlier_fraction": float(np.sum(valid_final)) / len(src),
        "R_total": R_total,
        "source_sternum_offset": source_ss,
        "target_sternum_offset": target_ss,
        "iteration_history": iteration_history,
        "target_centered": tgt_centered
    }

    if verbose:
        print(f"\nFinal results:")
        print(f"  RMSE: {final_rmse:.4f} mm (Best: {best_rmse:.4f} at iter {best_iteration})")
        print(f"  STD: {final_std:.4f} mm")
        print(f"  Sternum error: {sternum_error:.10f} mm (exact zero)")
        print(f"  Inliers: {np.sum(valid_final)} / {len(src)}")
        print(f"  Stopped: {stop_reason}")

    return R_total, src, info


# =====================================================================
# Commented-out function: plot_mesh_with_transformed_pointcloud
# Was already commented out in alignment.py. Preserved here for reference.
# =====================================================================

# def plot_mesh_with_transformed_pointcloud(morphic_mesh, target_point_cloud, R,
#                                            source_anchor, target_anchor,
#                                            mesh_color='#FFCCCC', mesh_opacity=0.5,
#                                            pc_color='blue',
#                                            title='Prone Mesh with Inverse-Transformed Supine Point Cloud'):
#     """
#     Plot the original prone mesh with element labels, alongside the supine point cloud
#     that has been inverse-transformed to the prone coordinate system.
#     """
#     transformed_pc = inverse_transform_to_source_frame(
#         target_point_cloud, R, source_anchor, target_anchor
#     )
#     num_elements = morphic_mesh.elements.size()
#     centers = []
#     for i in range(num_elements):
#         elem_coords = get_surface_mesh_coords(morphic_mesh, 3, elems=[i])
#         center_idx = elem_coords.shape[0] // 2
#         center = elem_coords[center_idx, :]
#         centers.append(center)
#     centers_array = np.array(centers)
#     mesh_coords = get_surface_mesh_coords(morphic_mesh, res=10, elems=[])
#     # ... (visualization with pyvista omitted for brevity)
#     return {
#         'element_centers': centers_array,
#         'transformed_pc': transformed_pc
#     }
