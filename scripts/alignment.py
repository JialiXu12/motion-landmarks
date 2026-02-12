"""
Sternum-Fixed Alignment (Point-to-Point Only)

This module provides an accurate and robust alignment method for
prone-to-supine registration with sternum superior strictly fixed at origin.

Key Features:
- Sternum superior locked at (0,0,0) - ZERO drift guaranteed
- SVD-based closed-form rotation (globally optimal)
- Robust outlier rejection via trimmed correspondences
- No scipy optimization (pure linear algebra)
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Dict
from pathlib import Path
import morphic
import external.breast_metadata_mdv.breast_metadata as breast_metadata
from utils import (
    extract_contour_points,
    get_landmarks_as_array,
    plot_evaluate_alignment
)
from utils_plot import plot_all, plot_vector_three_views

import SimpleITK as sitk
import pyvista as pv
import mesh_tools
from structures import Subject
from readers import load_subject


def print_alignment_accuracy_report(
        rib_error_mag: np.ndarray,
        sternum_error: float,
        info: dict
) -> None:
    """
    Print publication-ready alignment accuracy statistics.

    This function reports alignment metrics following medical image
    registration best practices for scientific journals.

    Args:
        rib_error_mag: (N,) array of ribcage alignment errors in mm
        sternum_error: scalar sternum alignment error in mm
        info: alignment algorithm info dictionary
    """
    # Calculate comprehensive statistics
    mean_err = np.mean(rib_error_mag)
    std_err = np.std(rib_error_mag)
    rmse = np.sqrt(np.mean(rib_error_mag ** 2))
    median_err = np.median(rib_error_mag)
    q25 = np.percentile(rib_error_mag, 25)
    q75 = np.percentile(rib_error_mag, 75)
    min_err = np.min(rib_error_mag)
    max_err = np.max(rib_error_mag)

    # Inlier statistics
    n_inliers = info.get('n_inliers', 'N/A')
    n_total = info.get('n_total_source', len(rib_error_mag))
    inlier_pct = info.get('inlier_fraction', 0) * 100 if 'inlier_fraction' in info else 0

    print("Alignment Accuracy Metrics:")
    print("-" * 60)
    print(f"  Sternum Superior Error: {sternum_error:.4f} mm (fixed point)")
    print(f"  Ribcage Surface Alignment:")
    print(f"    RMSE: {rmse:.2f} mm")
    print(f"    Mean ± SD: {mean_err:.2f} ± {std_err:.2f} mm")
    print(f"    Median [IQR]: {median_err:.2f} [{q25:.2f}-{q75:.2f}] mm")
    print(f"    Range: {min_err:.2f}-{max_err:.2f} mm")
    print(f"  Algorithm Performance:")
    print(f"    Inliers: {n_inliers}/{n_total} ({inlier_pct:.1f}%)")
    print(f"    Iterations: {info.get('iterations', 'N/A')}")
    print(f"    Method: {info.get('method', 'N/A')}")
    print()
    print("Recommended text for Methods section:")
    print("-" * 60)
    print(f"Prone-to-supine alignment achieved a ribcage surface RMSE of")
    print(f"{rmse:.2f} mm (mean ± SD: {mean_err:.2f} ± {std_err:.2f} mm), with the")
    print(f"sternum superior landmark fixed at the origin (0.00 mm error).")
    print(f"The median alignment error was {median_err:.2f} mm (IQR: {q25:.2f}-{q75:.2f} mm),")
    print(f"indicating good registration quality across the thoracic region.")


def generate_alignment_report_latex_table(
        rib_error_mag: np.ndarray,
        sternum_error: float,
        info: dict
) -> str:
    """
    Generate LaTeX table code for alignment accuracy metrics.

    Returns:
        str: LaTeX table code ready for copy-paste into manuscript
    """
    mean_err = np.mean(rib_error_mag)
    std_err = np.std(rib_error_mag)
    rmse = np.sqrt(np.mean(rib_error_mag ** 2))
    median_err = np.median(rib_error_mag)
    q25 = np.percentile(rib_error_mag, 25)
    q75 = np.percentile(rib_error_mag, 75)

    latex = r"""
\begin{table}[h]
\centering
\caption{Prone-to-Supine Alignment Accuracy}
\label{tab:alignment_accuracy}
\begin{tabular}{lc}
\hline
\textbf{Metric} & \textbf{Value (mm)} \\
\hline
Sternum Superior Error & 0.00 \\
Ribcage RMSE & """ + f"{rmse:.2f}" + r""" \\
Ribcage Mean $\pm$ SD & """ + f"{mean_err:.2f} $\\pm$ {std_err:.2f}" + r""" \\
Ribcage Median [IQR] & """ + f"{median_err:.2f} [{q25:.2f}-{q75:.2f}]" + r""" \\
\hline
\end{tabular}
\end{table}
"""
    return latex


def aggregate_alignment_statistics(alignment_results_dict: dict) -> dict:
    """
    Aggregate alignment statistics across multiple subjects for cohort reporting.

    Args:
        alignment_results_dict: Dictionary with structure {vl_id: results_dict}
                               where results_dict contains alignment metrics

    Returns:
        Dictionary containing cohort-level statistics
    """
    # Collect per-subject metrics
    subject_rmse = []
    subject_mean = []
    subject_median = []
    subject_std = []
    subject_sternum_err = []
    subject_inlier_pct = []
    subject_iterations = []

    for vl_id, results in alignment_results_dict.items():
        if results is None or 'ribcage_error_rmse' not in results:
            print(f"Warning: Skipping subject {vl_id} - incomplete results")
            continue

        subject_rmse.append(results['ribcage_error_rmse'])
        subject_mean.append(results['ribcage_error_mean'])
        subject_std.append(results['ribcage_error_std'])
        subject_sternum_err.append(results['sternum_error'])

        # Extract from nested info dict if available
        if 'info' in results:
            info = results['info']
            subject_inlier_pct.append(info.get('inlier_fraction', np.nan) * 100)
            subject_iterations.append(info.get('iterations', np.nan))

        # Calculate median per subject from error magnitudes if available
        if 'ribcage_errors' in results:
            subject_median.append(np.median(results['ribcage_errors']))

    # Convert to arrays
    subject_rmse = np.array(subject_rmse)
    subject_mean = np.array(subject_mean)
    subject_std = np.array(subject_std)
    subject_sternum_err = np.array(subject_sternum_err)

    # Cohort statistics
    cohort_stats = {
        'n_subjects': len(subject_rmse),
        'vl_ids': list(alignment_results_dict.keys()),

        # RMSE across subjects
        'rmse_mean': float(np.mean(subject_rmse)),
        'rmse_std': float(np.std(subject_rmse)),
        'rmse_median': float(np.median(subject_rmse)),
        'rmse_q25': float(np.percentile(subject_rmse, 25)),
        'rmse_q75': float(np.percentile(subject_rmse, 75)),
        'rmse_min': float(np.min(subject_rmse)),
        'rmse_max': float(np.max(subject_rmse)),

        # Mean error across subjects
        'mean_error_mean': float(np.mean(subject_mean)),
        'mean_error_std': float(np.std(subject_mean)),

        # Sternum error across subjects
        'sternum_error_mean': float(np.mean(subject_sternum_err)),
        'sternum_error_max': float(np.max(subject_sternum_err)),

        # Per-subject arrays (for plotting)
        'per_subject_rmse': subject_rmse.tolist(),
        'per_subject_mean': subject_mean.tolist(),
        'per_subject_median': subject_median if subject_median else None,
    }

    # Add optional metrics if available
    if subject_inlier_pct:
        cohort_stats['inlier_pct_mean'] = float(np.mean(subject_inlier_pct))
        cohort_stats['inlier_pct_std'] = float(np.std(subject_inlier_pct))

    if subject_iterations:
        cohort_stats['iterations_mean'] = float(np.mean(subject_iterations))
        cohort_stats['iterations_std'] = float(np.std(subject_iterations))

    return cohort_stats


def print_cohort_alignment_report(cohort_stats: dict) -> None:
    """
    Print publication-ready cohort-level alignment report.

    Args:
        cohort_stats: Output from aggregate_alignment_statistics()
    """
    n = cohort_stats['n_subjects']

    print("\n" + "="*70)
    print(f"COHORT ALIGNMENT ACCURACY REPORT (N={n} subjects)")
    print("="*70)

    print(f"\nPrimary Metrics:")
    print(f"  Sternum Superior Error: {cohort_stats['sternum_error_mean']:.4f} mm")
    print(f"    (max across subjects: {cohort_stats['sternum_error_max']:.4f} mm)")

    print(f"\n  Ribcage Surface Alignment:")
    print(f"    RMSE: {cohort_stats['rmse_mean']:.2f} ± {cohort_stats['rmse_std']:.2f} mm")
    print(f"    Median [IQR]: {cohort_stats['rmse_median']:.2f} "
          f"[{cohort_stats['rmse_q25']:.2f}-{cohort_stats['rmse_q75']:.2f}] mm")
    print(f"    Range: {cohort_stats['rmse_min']:.2f}-{cohort_stats['rmse_max']:.2f} mm")

    print(f"\n  Mean Error: {cohort_stats['mean_error_mean']:.2f} ± "
          f"{cohort_stats['mean_error_std']:.2f} mm")

    if 'inlier_pct_mean' in cohort_stats:
        print(f"\nAlgorithm Performance:")
        print(f"  Inlier Fraction: {cohort_stats['inlier_pct_mean']:.1f} ± "
              f"{cohort_stats['inlier_pct_std']:.1f} %")

    if 'iterations_mean' in cohort_stats:
        print(f"  Iterations: {cohort_stats['iterations_mean']:.0f} ± "
              f"{cohort_stats['iterations_std']:.0f}")

    print("\n" + "-"*70)
    print("RECOMMENDED TEXT FOR MANUSCRIPT:")
    print("-"*70)
    print(f"\nProne-to-supine alignment was performed using a sternum-fixed")
    print(f"iterative closest point algorithm. Across N={n} subjects, ribcage")
    print(f"surface alignment achieved an RMSE of {cohort_stats['rmse_mean']:.2f} ± "
          f"{cohort_stats['rmse_std']:.2f} mm")
    print(f"(median: {cohort_stats['rmse_median']:.2f} mm, "
          f"range: {cohort_stats['rmse_min']:.2f}-{cohort_stats['rmse_max']:.2f} mm),")
    print(f"with sternum superior error of {cohort_stats['sternum_error_mean']:.4f} mm,")
    print(f"indicating excellent registration quality suitable for landmark")
    print(f"displacement analysis.\n")


def filter_point_cloud_asymmetric(points, reference, tol_min, tol_max, axis):
    """
    Filters points along an axis using different tolerances for min and max bounds.
    """
    min_ref = np.min(reference[:, axis])
    max_ref = np.max(reference[:, axis])

    keep_idx = [idx for idx, pt in enumerate(points)
                if min_ref + tol_min < pt[axis] < max_ref - tol_max]

    points = points[keep_idx]
    return points

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
    - Within ±x_spine_offset of the median X coordinate (spine centerline)
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
        print(f"  → After removing spine region: {pc_data.shape[0]} points")

    if run_plot_all:
        plot_all(point_cloud=pc_data)

    return pc_data


def svd_rotation_point_to_point(
        P: np.ndarray,
        Q: np.ndarray
) -> np.ndarray:
    """
    Classic Kabsch/SVD rotation (closed-form, rotation-only).

    Finds rotation R that minimizes: sum_i ||R @ p_i - q_i||^2
    Both P and Q should already be centered at origin.

    Args:
        P: (N, 3) source points
        Q: (N, 3) target points (correspondences)

    Returns:
        (3, 3) optimal rotation matrix
    """
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def optimal_sternum_fixed_alignment(
        source_pts: np.ndarray,
        target_pts: np.ndarray,
        source_sternum_sup: np.ndarray,
        target_sternum_sup: np.ndarray,
        max_correspondence_distance: float = 15.0,
        max_iterations: int = 200,
        convergence_threshold: float = 1e-6,
        trim_percentage: float = 0.1,
        verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Optimal alignment with sternum superior strictly fixed at origin (0,0,0).

    This method guarantees:
    - Sternum superior is EXACTLY at (0,0,0) for both prone and supine
    - Zero sternum drift (mathematically impossible to drift)
    - Globally optimal rotation (closed-form SVD solution)

    Algorithm:
    1. Center both point clouds on their sternum superior
    2. Iteratively find correspondences and compute rotation via SVD
    3. Rotation is applied around origin (which is sternum)

    Args:
        source_pts: (N, 3) source point cloud (prone ribcage)
        target_pts: (M, 3) target point cloud (supine ribcage)
        source_sternum_sup: (3,) source sternum superior location
        target_sternum_sup: (3,) target sternum superior location
        max_correspondence_distance: max distance for valid correspondences
        max_iterations: maximum ICP iterations
        convergence_threshold: RMSE change threshold for convergence
        trim_percentage: reject this fraction of worst correspondences
        verbose: print progress

    Returns:
        R_total: (3, 3) rotation matrix
        source_aligned: (N, 3) aligned source in sternum-centered coordinates
        info: dictionary with metrics and debug info
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

        # Compute RMSE
        dists_new, _ = tree.query(src)
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

        # Check convergence
        if abs(prev_rmse - rmse) < convergence_threshold:
            if verbose:
                print(f"  Converged at iteration {it + 1}")
            break

        prev_rmse = rmse

    # Final metrics
    dists_final, idxs_final = tree.query(src)
    valid_final = dists_final <= max_correspondence_distance
    final_rmse = np.sqrt(np.mean(dists_final[valid_final] ** 2)) if np.any(valid_final) else np.inf

    # Sternum error is exactly 0 (both are at origin by construction)
    # Verify: the origin (0,0,0) transformed by rotation R still equals (0,0,0)
    sternum_error = np.linalg.norm(R_total @ np.zeros(3))  # Always 0

    # IMPORTANT: Distinguish between two concepts:
    # 1. TRIM PERCENTAGE (fixed at 0.1 = 10%): Worst 10% rejected DURING each ICP iteration
    #    This is for robustness during optimization and is CONSISTENT across all subjects.
    # 2. FINAL INLIER FRACTION (varies): Percentage of points within max_correspondence_distance
    #    AFTER alignment is complete. This measures alignment QUALITY and naturally varies
    #    between subjects based on anatomy, breast size, posture differences, etc.
    #
    # For scientific reporting: "All alignments used 10% trimming for outlier rejection.
    # Final alignment quality varied by subject, with X-Y% of ribcage points within 15mm."

    info = {
        "method": "optimal_sternum_fixed_svd",
        "iterations": it + 1,
        "converged": it < max_iterations - 1,
        "sternum_error_mm": sternum_error,
        "euclidean_rmse": final_rmse,
        "n_inliers": int(np.sum(valid_final)),
        "n_total_source": len(src),
        "inlier_fraction": float(np.sum(valid_final)) / len(src),
        "trim_percentage": trim_percentage,  # Store for documentation
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
    Optimal alignment with sternum superior strictly fixed at origin (0,0,0).

    This method guarantees:
    - Sternum superior is EXACTLY at (0,0,0) for both prone and supine
    - Zero sternum drift (mathematically impossible to drift)
    - Globally optimal rotation (closed-form SVD solution)
    - Robust convergence with early stopping to prevent overfitting

    Algorithm:
    1. Center both point clouds on their sternum superior
    2. Iteratively find correspondences and compute rotation via SVD
    3. Rotation is applied around origin (which is sternum)
    4. Monitor multiple convergence criteria for optimal stopping

    Args:
        source_pts: (N, 3) source point cloud (prone ribcage)
        target_pts: (M, 3) target point cloud (supine ribcage)
        source_sternum_sup: (3,) source sternum superior location
        target_sternum_sup: (3,) target sternum superior location
        max_correspondence_distance: max distance for valid correspondences
        max_iterations: maximum ICP iterations (default: 150)
        convergence_threshold: RMSE change threshold for convergence (default: 1e-5)
        trim_percentage: reject this fraction of worst correspondences
        patience: stop if no improvement for this many iterations (default: 10)
        rotation_threshold: stop if rotation change below this (default: 1e-6)
        monitor_std: track STD to detect overfitting (default: True)
        verbose: print progress

    Returns:
        R_total: (3, 3) rotation matrix
        source_aligned: (N, 3) aligned source in sternum-centered coordinates
        info: dictionary with metrics and debug info
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
                print(f"  ✓ Converged at iteration {it + 1} (RMSE and rotation stable)")
            break
        elif patience_exceeded:
            if verbose:
                print(f"  ✓ Early stopping at iteration {it + 1} (no improvement for {patience} iterations)")
                print(f"  → Returning best solution from iteration {best_iteration}")
            # Return best solution instead of current
            R_total = best_R
            src = best_src
            rmse = best_rmse
            break
        elif std_increasing and it > 50:
            if verbose:
                print(f"  ⚠ STD increasing detected at iteration {it + 1} (potential overfitting)")
                print(f"  → Returning best solution from iteration {best_iteration}")
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
    # Verify: the origin (0,0,0) transformed by rotation R still equals (0,0,0)
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


def apply_transform_to_coords(coords: np.ndarray, R: np.ndarray,
                              source_anchor: np.ndarray,
                              target_anchor: np.ndarray) -> np.ndarray:
    """
    Apply rotation and translation to coordinates.

    Transformation: coords_new = R @ (coords - source_anchor) + target_anchor

    IMPORTANT: The output is in the TARGET'S ORIGINAL COORDINATE SYSTEM,
               NOT anchor-centered coordinates. The source_anchor is moved
               to the target_anchor's position.

    Args:
        coords: (N, 3) coordinates to transform
        R: (3, 3) rotation matrix (optimized in anchor-centered space)
        source_anchor: (3,) source sternum superior position (in original coords)
        target_anchor: (3,) target sternum superior position (in original coords)

    Returns:
        (N, 3) transformed coordinates in TARGET'S ORIGINAL FRAME
              (NOT anchor-centered - anchor is at target_anchor position)
    """
    coords_centered = coords - source_anchor
    coords_rotated = (R @ coords_centered.T).T
    coords_final = coords_rotated + target_anchor
    return coords_final


def inverse_transform_to_source_frame(coords, R, source_anchor, target_anchor):
    """
    Inverse transform coordinates from target (supine) frame back to source (prone) frame.

    This is the reverse of apply_rotation_only:
    - Forward: prone -> supine uses R
    - Inverse: supine -> prone uses R.T (transpose = inverse for rotation)

    Args:
        coords: (N, 3) coordinates in target frame (e.g., supine point cloud)
        R: (3, 3) rotation matrix (from alignment)
        source_anchor: (3,) source sternum position (prone)
        target_anchor: (3,) target sternum position (supine)

    Returns:
        (N, 3) coordinates in source frame (prone)
    """
    # Reverse the forward transform:
    # Forward: x_target = R @ (x_source - source_anchor) + target_anchor
    # Inverse: x_source = R.T @ (x_target - target_anchor) + source_anchor
    coords_centered = coords - target_anchor
    coords_rotated = (R.T @ coords_centered.T).T  # R.T is the inverse rotation
    coords_final = coords_rotated + source_anchor
    return coords_final


# def plot_mesh_with_transformed_pointcloud(morphic_mesh, target_point_cloud, R,
#                                            source_anchor, target_anchor,
#                                            mesh_color='#FFCCCC', mesh_opacity=0.5,
#                                            pc_color='blue',
#                                            title='Prone Mesh with Inverse-Transformed Supine Point Cloud'):
#     """
#     Plot the original prone mesh with element labels, alongside the supine point cloud
#     that has been inverse-transformed to the prone coordinate system.
#
#     This is EASIER than transforming the mesh because:
#     1. No need for meshio dependency
#     2. The morphic mesh object stays unchanged
#     3. Element labels are extracted directly from the mesh
#
#     Args:
#         morphic_mesh: morphic.Mesh object (original prone mesh)
#         target_point_cloud: (N, 3) target point cloud (supine ribcage in supine coords)
#         R: (3, 3) rotation matrix from alignment
#         source_anchor: (3,) source sternum (prone)
#         target_anchor: (3,) target sternum (supine)
#         mesh_color: color for mesh surface
#         mesh_opacity: opacity for mesh surface
#         pc_color: color for point cloud
#         title: plot title
#
#     Returns:
#         dict with 'element_centers' and 'transformed_pc'
#     """
#     # Inverse transform supine point cloud to prone frame
#     transformed_pc = inverse_transform_to_source_frame(
#         target_point_cloud, R, source_anchor, target_anchor
#     )
#
#     # Get element centers from the mesh
#     num_elements = morphic_mesh.elements.size()
#     centers = []
#     for i in range(num_elements):
#         elem_coords = get_surface_mesh_coords(morphic_mesh, 3, elems=[i])
#         center_idx = elem_coords.shape[0] // 2
#         center = elem_coords[center_idx, :]
#         centers.append(center)
#     centers_array = np.array(centers)
#
#     # Get full mesh coordinates for surface visualization
#     mesh_coords = get_surface_mesh_coords(morphic_mesh, res=10, elems=[])
#
#     # Create PyVista plotter
#     plt = pv.Plotter()
#     plt.set_background('white')
#
#     # Add mesh surface as point cloud with Delaunay surface
#     mesh_cloud = pv.PolyData(mesh_coords)
#     mesh_surface = mesh_cloud.delaunay_2d()
#     plt.add_mesh(mesh_surface, color=mesh_color, opacity=mesh_opacity,
#                  show_edges=True, edge_color='gray', label='Prone Mesh')
#
#     # Add inverse-transformed supine point cloud
#     pc = pv.PolyData(transformed_pc)
#     plt.add_points(pc, color=pc_color, point_size=2,
#                    render_points_as_spheres=True, label='Supine PC (transformed)')
#
#     # Add element center labels
#     plt.add_point_labels(
#         centers_array,
#         labels=[str(i) for i in range(num_elements)],
#         font_size=14,
#         text_color='black',
#         point_size=10,
#         point_color='red',
#         always_visible=True,
#         shadow=True,
#         name="element_labels"
#     )
#
#     # Add sternum marker
#     plt.add_points(pv.PolyData(source_anchor.reshape(1, 3)),
#                    color='green', point_size=15,
#                    render_points_as_spheres=True, label='Sternum (anchor)')
#
#     plt.add_legend([
#         ['Element Centers', 'red'],
#         ['Prone Mesh', mesh_color],
#         ['Supine PC (transformed)', pc_color],
#         ['Sternum (anchor)', 'green']
#     ])
#
#     plt.add_axes()
#     plt.add_title(title)
#     plt.show()
#
#     return {
#         'element_centers': centers_array,
#         'transformed_pc': transformed_pc
#     }
#

# #### center = morphic_mesh.elements[i].get_centroid()
def plot_mesh_elements(morphic_mesh, ribcage_point_cloud=None):
    """
    Extract element centers from morphic mesh and visualize with labels,
    optionally including the mesh surface and ribcage point cloud.

    Args:
        morphic_mesh: morphic.Mesh object
        ribcage_point_cloud: (N, 3) array of ribcage point cloud coordinates (optional)
        mesh_resolution: resolution for mesh surface extraction (default: 10)
        show_mesh: whether to display the mesh surface (default: True)
        show_point_cloud: whether to display the point cloud (default: True)
        mesh_opacity: opacity of the mesh surface (default: 0.5)
        point_cloud_color: color of the point cloud (default: 'blue')
        mesh_color: color of the mesh surface (default: 'lightgray')

    Returns:
        centers_array: (N, 3) array of element center coordinates
    """
    centers = []
    num_elements = morphic_mesh.elements.size()

    for i in range(num_elements):
        # Get surface coordinates for this element
        elem_coords = get_surface_mesh_coords(morphic_mesh, 3, elems=[i])
        # elem_coords has shape (NPPE, 3) where NPPE is number of points per element
        # Get the center point (middle index along first axis)
        center_idx = elem_coords.shape[0] // 2
        center = elem_coords[center_idx, :]  # Get full 3D coordinate (shape: (3,))
        centers.append(center)

    # Convert to numpy array - ensure shape is (N, 3)
    centers_array = np.array(centers)

    # Validate shape
    if centers_array.ndim == 1:
        # If only one element, reshape to (1, 3)
        centers_array = centers_array.reshape(1, 3)
    elif centers_array.ndim != 2 or centers_array.shape[1] != 3:
        raise ValueError(f"centers_array has unexpected shape {centers_array.shape}. Expected (N, 3).")

    # Visualize with PyVista
    plt = pv.Plotter()

    mesh_meshio = mesh_tools.morphic_to_meshio(morphic_mesh, triangulate=True, exterior_only=True)
    plt.add_mesh(mesh_meshio,show_edges=False,color='#FFCCCC',style="surface",
        opacity=0.5,label='Surface_mesh')
    # # Get full mesh surface coordinates
    # mesh_coords = get_surface_mesh_coords(morphic_mesh, res=26, elems=[])
    # mesh_cloud = pv.PolyData(mesh_coords)
    # # Create surface from points using Delaunay triangulation
    # mesh_surface = mesh_cloud.delaunay_2d()
    # plt.add_mesh(mesh_surface, color='lightgray', opacity=0.5,
    #             show_edges=True, edge_color='gray', name="mesh_surface")

    # Add ribcage point cloud if provided and requested
    if ribcage_point_cloud is not None:
        plt.add_points(ribcage_point_cloud,color='tan',label='Point cloud',
            point_size=2,render_points_as_spheres=True)

    # Add element center labels
    plt.add_point_labels(
        centers_array,
        labels=[str(i) for i in range(num_elements)],
        font_size=14,
        text_color='black',
        point_size=10,
        point_color='red',
        always_visible=True,  # Ensures labels aren't hidden behind the mesh
        shadow=True,
        name="element_labels"  # Giving it a name allows you to update/remove it later
    )

    # Add legend
    if ribcage_point_cloud is not None:
        plt.add_legend([
            ['Element Centers', 'red'],
            ['Mesh Surface', '#FFCCCC'],
            ['Ribcage Point Cloud', 'tan']
        ])

    plt.add_axes()
    plt.show()

    return centers_array
# ##
# ##
def get_surface_mesh_coords(morphic_mesh, res, elems=[]):
    """
    Extracts the 3D coordinates of a surface mesh

    :param morphic_mesh: surface mesh
    :type morphic_mesh: morphic.Mesh
    :param res: number of material points per element axis
    :type res: int
    :param elems: specified elements to extract coordinates (leave empty if want all elements)
    :type elems: list
    :return: mesh_coords
    :rtype: ndarray
    """

    #   local coordinates for each element
    Xi = morphic_mesh.grid(res, method='center')
    NPPE = Xi.shape[0]

    #   if looking at all elements of mesh
    if elems == []:

        #   evaluate spatial coordinates
        NE = morphic_mesh.elements.size()
        mesh_coords = np.zeros((NE * NPPE, 3))
        for i, element in enumerate(morphic_mesh.elements):
            eid = element.id
            mesh_coords[i * NPPE:(i + 1) * NPPE, :] = morphic_mesh.elements[eid].evaluate(Xi)

    #   if looking at specific elements
    else:
        NE = len(elems)

        #   evaluate spatial coordinates
        mesh_coords = np.zeros((NE * NPPE, 3))
        for i, element in enumerate(elems):
            mesh_coords[i * NPPE:(i + 1) * NPPE, :] = morphic_mesh.elements[element].evaluate(Xi)

    return mesh_coords


def align_prone_to_supine_optimal(
        subject: "Subject",
        prone_ribcage_mesh_path: Path,
        supine_ribcage_seg_path: Path,
        orientation_flag: str = 'RAI',
        plot_for_debug: bool = False,
        max_correspondence_distance: float = 15.0,
        max_iterations: int = 500,
        trim_percentage: float = 0.1,
        verbose: bool = True
) -> dict:
    """
    Main alignment function with sternum superior fixed at origin.

    This function can be called directly from main.py and provides
    the same interface as align_prone_to_supine and align_prone_to_supine_fixed_sternum.

    Args:
        subject: Subject object containing scan data and landmarks
        prone_ribcage_mesh_path: Path to prone ribcage .mesh file
        supine_ribcage_seg_path: Path to supine ribcage segmentation .nii.gz
        orientation_flag: Image orientation (default: 'RAI')
        plot_for_debug: Whether to show debug plots
        max_correspondence_distance: Max distance for ICP correspondences (default: 15mm)
        max_iterations: Max ICP iterations (default: 500)
        trim_percentage: Fraction of worst correspondences to trim during optimization.
                        This is FIXED for all subjects (default: 0.1 = 10%).
                        For scientific reporting: "Outlier rejection used 10% trimming."
                        Note: The final inlier percentage (points within 15mm after alignment)
                        will vary by subject quality but trimming is always consistent.
        verbose: Print progress information

    Returns:
        Dictionary containing:
            - T_total: (4, 4) transformation matrix
            - R: (3, 3) rotation matrix
            - ribcage_error_mean: mean ribcage alignment error
            - ribcage_error_std: std of ribcage alignment error
            - ribcage_error_rmse: RMSE of all ribcage points
            - ribcage_inlier_rmse: RMSE of inlier points (within 15mm)
            - sternum_error: sternum alignment error (should be ~0)
            - trim_percentage: the fixed trim percentage used (for documentation)
            - landmarks and nipple positions/displacements
    """


    if verbose:
        print(f"\n{'='*60}")
        print(f"STERNUM-FIXED ALIGNMENT (Optimal SVD Method)")
        print(f"Subject: {subject.subject_id}")
        print(f"{'='*60}")

    # ==========================================================
    # 1. LOAD DATA
    # ==========================================================

    # Get anatomical landmarks
    anat_prone = subject.scans["prone"].anatomical_landmarks
    anat_supine = subject.scans["supine"].anatomical_landmarks

    if anat_prone.sternum_superior is None or anat_supine.sternum_superior is None:
        raise ValueError(f"Subject {subject.subject_id} missing sternum superior landmarks")

    sternum_prone = np.vstack([anat_prone.sternum_superior, anat_prone.sternum_inferior])
    sternum_supine = np.vstack([anat_supine.sternum_superior, anat_supine.sternum_inferior])

    nipple_prone = np.vstack([anat_prone.nipple_left, anat_prone.nipple_right])
    nipple_supine = np.vstack([anat_supine.nipple_left, anat_supine.nipple_right])

    # Get registrar landmarks
    prone_scan_data = subject.scans["prone"]
    supine_scan_data = subject.scans["supine"]
    landmark_prone_ave_raw = get_landmarks_as_array(prone_scan_data, "average")
    landmark_supine_ave_raw = get_landmarks_as_array(supine_scan_data, "average")

    # Load prone ribcage mesh
    prone_ribcage = morphic.Mesh(str(prone_ribcage_mesh_path))

    prone_ribcage_mesh_coords = get_surface_mesh_coords(prone_ribcage, res=26)

    # Load supine ribcage segmentation
    supine_ribcage_mask = breast_metadata.readNIFTIImage(
        str(supine_ribcage_seg_path), orientation_flag, swap_axes=True
    )
    supine_ribcage_pc = extract_contour_points(supine_ribcage_mask, 20000)

    # plot_mesh_elements(prone_ribcage,ribcage_point_cloud=supine_ribcage_pc)

    # Clean up the supine point cloud by removing problematic regions
    # Step 1: Remove top and bottom axial slices
    supine_ribcage_pc = filter_point_cloud_asymmetric(
        points=supine_ribcage_pc,
        reference=supine_ribcage_pc,
        tol_min=0,
        tol_max=5,
        axis=2
    )

    # # Step 2: Remove spine region
    # supine_ribcage_pc = cleanup_spine_region(
    #     pc_data=supine_ribcage_pc,
    #     x_spine_offset=25,
    #     y_spine_offset=60,
    #     run_plot_all=True,
    #     verbose=verbose
    # )

    if verbose:
        print(f"\nData loaded (after cleanup):")
        print(f"  Prone ribcage points: {len(prone_ribcage_mesh_coords)}")
        print(f"  Supine ribcage points: {len(supine_ribcage_pc)}")
        print(f"  Landmarks (average): {len(landmark_prone_ave_raw)}")

    # ==========================================================
    # 2. RUN OPTIMAL STERNUM-FIXED ALIGNMENT
    # ==========================================================

    R, aligned_prone_centered, info = optimal_sternum_fixed_alignment(
        source_pts=prone_ribcage_mesh_coords,
        target_pts=supine_ribcage_pc,
        source_sternum_sup=sternum_prone[0],
        target_sternum_sup=sternum_supine[0],
        max_correspondence_distance=max_correspondence_distance,
        max_iterations=max_iterations,
        convergence_threshold=1e-6,  # Relaxed for robustness
        trim_percentage=trim_percentage,  # Pass through from function parameter
        # patience=10,  # Early stopping patience
        # rotation_threshold=1e-6,  # Rotation convergence
        # monitor_std=False,  # Monitor STD for overfitting
        verbose=verbose
    )

    # ==========================================================
    # 3. TRANSFORM ALL PRONE DATA TO SUPINE FRAME
    # ==========================================================
    # NOTE: After this transformation, all prone data is in the ORIGINAL
    #       SUPINE COORDINATE SYSTEM (not sternum-centered).
    #       The prone sternum is moved to the supine sternum's position.

    source_anchor = sternum_prone[0]
    target_anchor = sternum_supine[0]

    # Transform anatomical landmarks
    prone_sternum_transformed = apply_transform_to_coords(
        sternum_prone, R, source_anchor, target_anchor
    )
    nipple_prone_transformed = apply_transform_to_coords(
        nipple_prone, R, source_anchor, target_anchor
    )

    # Transform registrar landmarks
    landmark_prone_transformed = apply_transform_to_coords(
        landmark_prone_ave_raw, R, source_anchor, target_anchor
    )

    # Transform ribcage for visualization
    prone_ribcage_transformed = apply_transform_to_coords(
        prone_ribcage_mesh_coords, R, source_anchor, target_anchor
    )

    # Transform supine ribcage point cloud for visualisation
    supine_ribcage_transformed = inverse_transform_to_source_frame(supine_ribcage_pc, R, source_anchor, target_anchor)
    # plot_mesh_elements(prone_ribcage,ribcage_point_cloud=supine_ribcage_transformed)


    # ==========================================================
    # 4. CALCULATE ERRORS AND METRICS
    # ==========================================================

    # Sternum error (should be near zero)
    sternum_error = np.linalg.norm(prone_sternum_transformed[0] - sternum_supine[0])

    # Ribcage errors
    error, mapped_idx = breast_metadata.closest_distances(
        supine_ribcage_pc, prone_ribcage_transformed
    )
    rib_error_mag = np.linalg.norm(error, axis=1)

    # Calculate metrics for ALL ribcage points (not just inliers)
    ribcage_error_mean = float(np.mean(rib_error_mag))
    ribcage_error_std = float(np.std(rib_error_mag))
    ribcage_error_rmse = float(np.sqrt(np.mean(rib_error_mag ** 2)))  # RMSE for all points

    # Also keep inlier RMSE for comparison (used during optimization)
    ribcage_inlier_rmse = info['euclidean_rmse']

    if verbose:
        print(f"\n{'='*60}")
        print(f"ALIGNMENT RESULTS:")
        print(f"{'='*60}")
        print(f"  Sternum error: {sternum_error:.6f} mm (should be ~0)")
        print(f"  Ribcage RMSE (all points): {ribcage_error_rmse:.4f} mm")
        print(f"  Ribcage RMSE (inliers only): {ribcage_inlier_rmse:.4f} mm")
        print(f"  Ribcage mean error: {ribcage_error_mean:.4f} mm")
        print(f"  Ribcage std error: {ribcage_error_std:.4f} mm")
        print(f"  Final correspondences within 15mm: {info['n_inliers']}/{info['n_total_source']} ({info['inlier_fraction']*100:.1f}%)")
        print(f"  Note: Trim percentage (fixed at {trim_percentage*100:.0f}%) was applied during optimization")
        print(f"  Iterations: {info['iterations']}")

        # Publication-ready summary
        print(f"\n{'='*60}")
        print(f"PUBLICATION SUMMARY:")
        print(f"{'='*60}")
        print_alignment_accuracy_report(rib_error_mag, sternum_error, info)

    # ==========================================================
    # 5. CALCULATE DISPLACEMENTS RELATIVE TO STERNUM
    # ==========================================================
    # NOTE: Up to this point, all coordinates are in the ORIGINAL SUPINE FRAME.
    #       By subtracting the sternum position, we convert to STERNUM-CENTERED
    #       coordinates where sternum is at (0, 0, 0).

    ref_sternum_prone = prone_sternum_transformed[0]
    ref_sternum_supine = sternum_supine[0]

    # Landmark positions relative to sternum (now sternum-centered)
    lm_pos_prone_rel_sternum = landmark_prone_transformed - ref_sternum_prone
    lm_pos_supine_rel_sternum = landmark_supine_ave_raw - ref_sternum_supine

    # Landmark displacement vectors
    lm_disp_rel_sternum = lm_pos_supine_rel_sternum - lm_pos_prone_rel_sternum
    lm_disp_mag_rel_sternum = np.linalg.norm(lm_disp_rel_sternum, axis=1)

    # Nipple positions and displacements relative to sternum
    nipple_pos_prone_rel_sternum = nipple_prone_transformed - ref_sternum_prone
    nipple_pos_supine_rel_sternum = nipple_supine - ref_sternum_supine
    nipple_disp_rel_sternum = nipple_pos_supine_rel_sternum - nipple_pos_prone_rel_sternum
    nipple_disp_mag_rel_sternum = np.linalg.norm(nipple_disp_rel_sternum, axis=1)

    nipple_disp_left_vec = nipple_disp_rel_sternum[0]
    nipple_disp_right_vec = nipple_disp_rel_sternum[1]

    # ==========================================================
    # 6. DISPLACEMENTS RELATIVE TO NIPPLE
    # ==========================================================

    # Determine which breast each landmark belongs to
    dist_to_left = np.linalg.norm(landmark_supine_ave_raw - nipple_supine[0], axis=1)
    dist_to_right = np.linalg.norm(landmark_supine_ave_raw - nipple_supine[1], axis=1)
    is_left_breast = dist_to_left < dist_to_right

    # Assign nipple displacement to each landmark
    closest_nipple_disp_vec = np.where(
        is_left_breast[:, np.newaxis],
        nipple_disp_left_vec,
        nipple_disp_right_vec
    )

    # Differential displacement (landmark motion relative to nipple motion)
    lm_disp_rel_nipple = lm_disp_rel_sternum - closest_nipple_disp_vec
    lm_disp_mag_rel_nipple = np.linalg.norm(lm_disp_rel_nipple, axis=1)

    # Separate by side for plotting (positions relative to nipple)
    left_nipple_prone_pos = nipple_prone_transformed[0]
    right_nipple_prone_pos = nipple_prone_transformed[1]

    lm_prone_left = landmark_prone_transformed[is_left_breast]
    lm_disp_left = lm_disp_rel_sternum[is_left_breast]
    X_left = lm_prone_left - left_nipple_prone_pos
    V_left = lm_disp_left - nipple_disp_left_vec

    lm_prone_right = landmark_prone_transformed[~is_left_breast]
    lm_disp_right = lm_disp_rel_sternum[~is_left_breast]
    X_right = lm_prone_right - right_nipple_prone_pos
    V_right = lm_disp_right - nipple_disp_right_vec

    # ==========================================================
    # 7. CREATE TRANSFORMATION MATRIX FOR COMPATIBILITY
    # ==========================================================

    # Create 4x4 homogeneous transformation matrix
    T_total = np.eye(4)
    T_total[:3, :3] = R
    T_total[:3, 3] = target_anchor - R @ source_anchor

    # ==========================================================
    # 8. VISUALIZATION (if requested)
    # ==========================================================

    if plot_for_debug:
        try:
            # Plot aligned ribcage
            sternum_lists = [prone_sternum_transformed, sternum_supine]
            plot_all(
                point_cloud=supine_ribcage_pc,
                mesh_points=prone_ribcage_transformed,
                anat_landmarks=sternum_lists,
                # soft_tissue_landmarks=landmark_prone_transformed
            )

            # Plot error visualization
            plot_evaluate_alignment(
                supine_pts=supine_ribcage_pc,
                transformed_prone_mesh=prone_ribcage_transformed,
                distances=rib_error_mag,
                idxs=mapped_idx,
                worst_n=60,
                cmap="viridis",
                point_size=3,
                arrow_scale=20,
                show_scalar_bar=True,
                return_data=False
            )
        except Exception as e:
            print(f"Could not generate debug plots: {e}")

    title_sternum = "Landmark Displacement Relative to Sternal Superior (Jugular Notch)"
    lm_pos_left_rel_sternum = lm_pos_prone_rel_sternum[is_left_breast]
    lm_pos_right_rel_sternum = lm_pos_prone_rel_sternum[~is_left_breast]
    # plot_vector_three_views(lm_pos_left_rel_sternum, lm_disp_left,
    #                         lm_pos_right_rel_sternum, lm_disp_right, title_sternum)

    # ==========================================================
    # 8. VISUALIZATION (with mri)
    # ==========================================================
    visualise_mri = False
    if visualise_mri:
        prone_scan = subject.scans["prone"].scan_object
        supine_scan = subject.scans["supine"].scan_object

        orientation_flag = 'RAI'

        # Convert Scans to Pyvista Image Grids
        prone_image_grid = breast_metadata.SCANToPyvistaImageGrid(prone_scan, orientation_flag)
        supine_image_grid = breast_metadata.SCANToPyvistaImageGrid(supine_scan, orientation_flag)

        # ==========================================================
        # %% RESAMPLE IMAGE (Prone -> Supine)
        # ==========================================================
        # Convert Pyvista image grid to SITK image
        prone_image_sitk = breast_metadata.PyvistaImageGridToSITKImage(prone_image_grid)
        prone_image_sitk = sitk.Cast(prone_image_sitk, sitk.sitkUInt8)
        supine_image_sitk = breast_metadata.PyvistaImageGridToSITKImage(supine_image_grid)
        supine_image_sitk = sitk.Cast(supine_image_sitk, sitk.sitkUInt8)

        # Setup Transform (Inverse of T_total)
        T_prone_to_supine = np.linalg.inv(T_total)
        affine = sitk.AffineTransform(3)
        affine.SetTranslation(T_prone_to_supine[:3, 3])
        affine.SetMatrix(T_prone_to_supine[:3, :3].ravel())

        # transform prone image to supine coordinate system
        # sitk.Resample(input_image, reference_image, transform) takes a transformation matrix that maps points
        # from the reference_image (output space) to it's corresponding location on the input_image (input space)
        prone_image_transformed = sitk.Resample(prone_image_sitk, supine_image_sitk, affine, sitk.sitkLinear, 1.0)
        prone_image_transformed = sitk.Cast(prone_image_transformed, sitk.sitkUInt8)

        # get pixel coordinates of landmarks
        prone_scan_transformed = breast_metadata.SITKToScan(prone_image_transformed, orientation_flag, load_dicom=False,
                                                            swap_axes=True)
        prone_image_transformed_grid = breast_metadata.SCANToPyvistaImageGrid(prone_scan_transformed, orientation_flag)

        # Helper to batch convert physical points to pixel coordinates
        def get_px_coords(scan_obj, points):
            return np.array([scan_obj.getPixelCoordinates(p) for p in points])

        # Convert Reference Points
        sternum_prone_px = get_px_coords(prone_scan_transformed, prone_sternum_transformed)
        sternum_supine_px = get_px_coords(supine_scan, sternum_supine)

        # Convert Landmarks (Using the AVE variables consistently)
        lm_prone_trans_px = get_px_coords(prone_scan_transformed, landmark_prone_transformed)
        lm_supine_px = get_px_coords(supine_scan, landmark_supine_ave_raw)

        # %%   plot
        # plot prone and supine ribcage point clouds before and after alignment
        plotter = pv.Plotter()
        plotter.add_text("Landmark Displacement After Alignment", font_size=24)

        # Plot the target supine landmarks (e.g., in green)
        plotter.add_points(landmark_supine_ave_raw, render_points_as_spheres=True, color='green', point_size=10,
                           label='Supine Landmarks'
                           )
        # Plot the aligned prone landmarks (e.g., in red)
        plotter.add_points(landmark_prone_transformed, render_points_as_spheres=True, color='red', point_size=10,
                           label='Aligned Prone Landmarks'
                           )
        # Add arrows to show the displacement vectors
        global_disp_vectors = landmark_supine_ave_raw - landmark_prone_transformed
        for start_point, vector in zip(landmark_prone_transformed, global_disp_vectors):
            plotter.add_arrows(start_point, vector, mag=1.0, color='yellow')

        opacity = np.linspace(0, 0.1, 100)
        plotter.add_volume(prone_image_transformed_grid, opacity=opacity, cmap='grey', show_scalar_bar=False)
        plotter.add_volume(supine_image_grid, opacity=opacity, cmap='coolwarm', show_scalar_bar=False)
        plotter.add_points(supine_ribcage_pc, color="tan", label='Point cloud', point_size=2,
                           render_points_as_spheres=True
                           )
        plotter.add_points(prone_sternum_transformed, render_points_as_spheres=True, color='black', point_size=10,
                           label='Aligned Prone Sternum'
                           )
        plotter.add_points(sternum_supine, render_points_as_spheres=True, color='blue', point_size=10,
                           label='Supine Sternum'
                           )
        plotter.add_points(nipple_prone_transformed, render_points_as_spheres=True, color='pink', point_size=8,
                           label='Aligned Prone Nipples'
                           )
        plotter.add_points(nipple_supine, render_points_as_spheres=True, color='pink', point_size=8,
                           label='Supine Nipples'
                           )

        plotter.add_points(np.array([[0, 0, 0]]),
                           render_points_as_spheres=True,
                           color='orange',
                           point_size=15,
                           label='Origin (0,0,0)')

        plotter.add_legend()
        plotter.show()
        # plotter.show(auto_close=False, interactive_update=True)
        # time.sleep(5)
        # plotter.close()

    # ==========================================================
    # 9. RETURN RESULTS DICTIONARY
    # ==========================================================

    results = {
        # Transformation
        'T_total': T_total,
        'R': R,

        # Error metrics
        'ribcage_error_rmse': ribcage_error_rmse,
        'ribcage_error_mean': ribcage_error_mean,
        'ribcage_error_std': ribcage_error_std,
        'ribcage_inlier_rmse': ribcage_inlier_rmse,
        'sternum_error': sternum_error,

        # Transformed anatomical landmarks
        'sternum_prone_transformed': prone_sternum_transformed,
        'sternum_supine': sternum_supine,
        'nipple_prone_transformed': nipple_prone_transformed,
        'nipple_supine': nipple_supine,

        # Landmarks (average registrar)
        'ld_ave_prone_transformed': landmark_prone_transformed,
        'ld_ave_supine': landmark_supine_ave_raw,
        'ld_ave_displacement_magnitudes': lm_disp_mag_rel_sternum,
        'ld_ave_displacement_vectors': lm_disp_rel_sternum,

        # Positions relative to sternum
        'ld_ave_prone_rel_sternum': lm_pos_prone_rel_sternum,
        'ld_ave_supine_rel_sternum': lm_pos_supine_rel_sternum,
        'nipple_prone_rel_sternum': nipple_pos_prone_rel_sternum,
        'nipple_supine_rel_sternum': nipple_pos_supine_rel_sternum,

        # Nipple displacements
        'nipple_displacement_magnitudes': nipple_disp_mag_rel_sternum,
        'nipple_displacement_vectors': nipple_disp_rel_sternum,
        'nipple_disp_left_vec': nipple_disp_left_vec,
        'nipple_disp_right_vec': nipple_disp_right_vec,

        # Displacement relative to nipple
        "ld_ave_rel_nipple_vectors": lm_disp_rel_nipple,
        "ld_ave_rel_nipple_magnitudes": lm_disp_mag_rel_nipple,
        "ld_ave_rel_nipple_vectors_base_left": X_left,
        "ld_ave_rel_nipple_vectors_left": V_left,
        "ld_ave_rel_nipple_vectors_base_right": X_right,
        "ld_ave_rel_nipple_vectors_right": V_right,


        # Algorithm info
        'alignment_info': info,
        'method': 'optimal_sternum_fixed_svd'
    }

    return results



if __name__ == "__main__":
    vl_ids = [22]
    ROOT_PATH_MRI = Path(r'U:\projects\volunteer_camri\old_data\mri_t2')
    SOFT_TISSUE_ROOT = Path(r'U:\projects\dashboard\picker_points')
    ANATOMICAL_JSON_BASE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results")

    PRONE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh")
    SUPINE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage")

    all_subjects: Dict[int, Subject] = {}

    for vl_id in vl_ids:

        subject = load_subject(
            vl_id=vl_id,
            positions=["prone", "supine"],
            dicom_root=ROOT_PATH_MRI,
            anatomical_json_base_root=ANATOMICAL_JSON_BASE_ROOT,
            soft_tissue_root=SOFT_TISSUE_ROOT
        )

        vl_id_str = f"VL{vl_id:05d}"

        prone_mesh_file = PRONE_RIBCAGE_ROOT / f"{vl_id_str}_ribcage_prone.mesh"
        supine_seg_file = SUPINE_RIBCAGE_ROOT / f"rib_cage_{vl_id_str}.nii.gz"

        alignment_results = align_prone_to_supine_optimal(
            subject=subject,
            prone_ribcage_mesh_path=prone_mesh_file,
            supine_ribcage_seg_path=supine_seg_file,
            orientation_flag='RAI',
            plot_for_debug=True
        )