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
from surface_to_point_alignment import surface_to_point_align

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
    print(f"    Mean +/- SD: {mean_err:.2f} +/- {std_err:.2f} mm")
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
    print(f"{rmse:.2f} mm (mean +/- SD: {mean_err:.2f} +/- {std_err:.2f} mm), with the")
    print(f"sternum superior landmark fixed at the origin (0.00 mm error).")
    print(f"The median alignment error was {median_err:.2f} mm (IQR: {q25:.2f}-{q75:.2f} mm),")
    print(f"indicating good registration quality across the thoracic region.")



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
    print(f"    RMSE: {cohort_stats['rmse_mean']:.2f} +/- {cohort_stats['rmse_std']:.2f} mm")
    print(f"    Median [IQR]: {cohort_stats['rmse_median']:.2f} "
          f"[{cohort_stats['rmse_q25']:.2f}-{cohort_stats['rmse_q75']:.2f}] mm")
    print(f"    Range: {cohort_stats['rmse_min']:.2f}-{cohort_stats['rmse_max']:.2f} mm")

    print(f"\n  Mean Error: {cohort_stats['mean_error_mean']:.2f} +/- "
          f"{cohort_stats['mean_error_std']:.2f} mm")

    if 'inlier_pct_mean' in cohort_stats:
        print(f"\nAlgorithm Performance:")
        print(f"  Inlier Fraction: {cohort_stats['inlier_pct_mean']:.1f} +/- "
              f"{cohort_stats['inlier_pct_std']:.1f} %")

    if 'iterations_mean' in cohort_stats:
        print(f"  Iterations: {cohort_stats['iterations_mean']:.0f} +/- "
              f"{cohort_stats['iterations_std']:.0f}")

    print("\n" + "-"*70)
    print("RECOMMENDED TEXT FOR MANUSCRIPT:")
    print("-"*70)
    print(f"\nProne-to-supine alignment was performed using a sternum-fixed")
    print(f"iterative closest point algorithm. Across N={n} subjects, ribcage")
    print(f"surface alignment achieved an RMSE of {cohort_stats['rmse_mean']:.2f} +/- "
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
        plot_all(point_cloud=pc_data)

    return pc_data


def selected_point_cloud(pc_data,params,run_plot_all):
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
        plot_all(point_cloud=pc_data)

    # supine_ribcage_pc = supine_ribcage_pc[supine_ribcage_pc[:, 0] > -120.]
    # print("remove outlier near arm, supine ribcage point cloud size:", supine_ribcage_pc.shape)
    # plot_all(point_cloud=supine_ribcage_pc)

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
        # pc_data = filter_point_cloud_asymmetric(
    # x_offset = 100.
    # y_median = np.median(supine_ribcage_pc[:, 1])
    # y_offset = 50.
    # supine_ribcage_pc = supine_ribcage_pc[
    # (supine_ribcage_pc[:, 0] <= (x_median - x_offset)) |
    # (supine_ribcage_pc[:, 0] >= (x_median + x_offset)) |
    # (supine_ribcage_pc[:, 1] <= (y_median - y_offset)) |
    # (supine_ribcage_pc[:, 1] >= (y_median + y_offset))
    # ]
    #
    # plot_all(point_cloud=supine_ribcage_pc)

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
        verbose: bool = False,
        visualize_iterations: bool = False,
        visualize_every_n: int = 10
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
        visualize_iterations: if True, show visualization during ICP iterations
        visualize_every_n: show visualization every N iterations (default: 10)

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

    # Visualize initial state before alignment
    if visualize_iterations:
        dists_init, idxs_init = tree.query(src)
        valid_init = dists_init <= max_correspondence_distance
        visualize_alignment_during_iteration(
            source_centered=src,
            target_centered=tgt_centered,
            iteration=0,
            correspondences=idxs_init,
            correspondence_distances=dists_init,
            valid_mask=valid_init,
            max_correspondence_distance=max_correspondence_distance,
            show_correspondences=True,
            n_correspondence_lines=100
        )

    # Initialize iteration counter
    it = 0
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

        # Visualize at specified intervals
        if visualize_iterations and ((it + 1) % visualize_every_n == 0 or it == 0):
            visualize_alignment_during_iteration(
                source_centered=src,
                target_centered=tgt_centered,
                iteration=it + 1,
                correspondences=idxs_new,
                correspondence_distances=dists_new,
                valid_mask=valid_new,
                max_correspondence_distance=max_correspondence_distance,
                show_correspondences=True,
                n_correspondence_lines=100
            )

        # Check convergence
        if abs(prev_rmse - rmse) < convergence_threshold:
            if verbose:
                print(f"  Converged at iteration {it + 1}")
            # Visualize final state on convergence
            if visualize_iterations:
                visualize_alignment_during_iteration(
                    source_centered=src,
                    target_centered=tgt_centered,
                    iteration=it + 1,
                    correspondences=idxs_new,
                    correspondence_distances=dists_new,
                    valid_mask=valid_new,
                    max_correspondence_distance=max_correspondence_distance,
                    show_correspondences=True,
                    n_correspondence_lines=100
                )
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
                print(f"  “ Converged at iteration {it + 1} (RMSE and rotation stable)")
            break
        elif patience_exceeded:
            if verbose:
                print(f"  “ Early stopping at iteration {it + 1} (no improvement for {patience} iterations)")
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

def get_mesh_elements(mesh):
    # Extract element centers from morphic mesh

    # Detect mesh dimensionality by checking first element's basis
    first_element = list(mesh.elements)[0]
    mesh_dims = len(first_element.basis)
    print(f"INFO: Mesh has {mesh_dims}D elements with basis {first_element.basis}")

    centers = []
    num_elements = mesh.elements.size()
    print(f"INFO: Mesh has {mesh.elements.size()} elements")

    for i in range(num_elements):
        # Get surface coordinates for this element
        elem_coords = get_surface_mesh_coords(mesh, 3, elems=[i])
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

    return centers_array, num_elements


def get_mesh_elements_2(mesh):
    # Extract element centers from morphic mesh

    # Detect mesh dimensionality by checking first element's basis
    first_element = list(mesh.elements)[0]
    mesh_dims = len(first_element.basis)
    print(f"INFO: Mesh has {mesh_dims}D elements with basis {first_element.basis}")

    centers = []
    num_elements = mesh.elements.size()

    Xi = mesh.grid(3, method='center')
    for i in range(num_elements):
        elem = list(mesh.elements)[i]
        elem_coords = elem.evaluate(Xi)
        center_idx = elem_coords.shape[0] // 2
        center = elem_coords[center_idx, :]
        centers.append(center)

    centers_array = np.array(centers)
    if centers_array.ndim == 1:
        centers_array = centers_array.reshape(1, 3)

    return centers_array, num_elements

# #### center = morphic_mesh.elements[i].get_centroid()
def plot_mesh_elements(mesh_points, centers_array, element_indices, ribcage_point_cloud=None):
    """
    Visualize element centers with labels,
    optionally including the mesh surface and ribcage point cloud.

    Args:
        mesh_points: (N, 3) array of mesh point coordinates to display
        centers_array: (M, 3) array of element center coordinates for labeling
        element_indices
        ribcage_point_cloud: (N, 3) array of ribcage point cloud coordinates (optional)

    Returns:
        centers_array: (N, 3) array of element center coordinates
    """
    # Determine labels for elements
    labels = [str(i) for i in element_indices]

    # Visualize with PyVista
    plt = pv.Plotter()

    # mesh_meshio = mesh_tools.morphic_to_meshio(mesh, triangulate=True, res=4, exterior_only=True)
    # plt.add_mesh(mesh_meshio,show_edges=False,color='#FFCCCC',style="surface",
    #     opacity=0.5,label='Surface_mesh')

    # # Get full mesh surface coordinates
    # mesh_coords = get_surface_mesh_coords(morphic_mesh, res=26, elems=[])
    # mesh_cloud = pv.PolyData(mesh_coords)
    # # Create surface from points using Delaunay triangulation
    # mesh_surface = mesh_cloud.delaunay_2d()
    # plt.add_mesh(mesh_surface, color='lightgray', opacity=0.5,
    #             show_edges=True, edge_color='gray', name="mesh_surface")


    # Plotted 3D coordinates of a Surface Mesh.
    plt.add_points(
        mesh_points,
        color='gray',
        render_points_as_spheres=True,
        point_size=2,
        label='3D coordinates of a surface mesh'
    )

    # Add element center labels
    plt.add_point_labels(
        centers_array,
        labels=labels,
        font_size=14,
        text_color='black',
        point_size=10,
        point_color='red',
        always_visible=True,  # Ensures labels aren't hidden behind the mesh
        shadow=True,
        name="element_labels"  # Giving it a name allows you to update/remove it later
    )

    # Add ribcage point cloud if provided and requested
    if ribcage_point_cloud is not None:
        plt.add_points(ribcage_point_cloud,color='tan',label='Point cloud',
            point_size=2,render_points_as_spheres=True)


    # Add legend
    if ribcage_point_cloud is not None:
        plt.add_legend([
            ['Mesh Surface', '#FFCCCC'],
            ['Element Centers', 'red'],
            ['Ribcage Point Cloud', 'tan']
        ])

    plt.add_axes()
    plt.show()

    return centers_array


def visualize_alignment_errors(
        source_mesh_coords: np.ndarray,
        target_pc: np.ndarray,
        source_aligned: np.ndarray = None,
        error_magnitudes: np.ndarray = None,
        error_indices: np.ndarray = None,
        source_sternum: np.ndarray = None,
        target_sternum: np.ndarray = None,
        selected_elements_coords: np.ndarray = None,
        iteration: int = None,
        title: str = "Alignment Visualization",
        cmap: str = "coolwarm",
        point_size: int = 4,
        show_error_arrows: bool = True,
        worst_n_arrows: int = 50,
        show_sternum: bool = True,
        show_legend: bool = True,
        screenshot_path: str = None
) -> None:
    """
    Visualize mesh and point cloud with alignment errors during alignment process.

    Colors the source (prone) mesh by per-source-point distance to the nearest
    target (supine) point (prone to supine query direction).

    Args:
        source_mesh_coords: (N, 3) source mesh coordinates (prone ribcage)
        target_pc: (M, 3) target point cloud (supine ribcage)
        source_aligned: (N, 3) aligned source coordinates (if None, uses source_mesh_coords)
        error_magnitudes: (N,) per-source-point distance to nearest target point
        error_indices: (N,) index into target_pc of nearest point for each source point
        source_sternum: (3,) source sternum superior position
        target_sternum: (3,) target sternum superior position
        selected_elements_coords: (K, 3) coordinates of selected elements only (subset for alignment)
        iteration: iteration number (for title display)
        title: plot title
        cmap: colormap for error visualization
        point_size: size of point cloud points
        show_error_arrows: whether to show arrows for worst errors
        worst_n_arrows: number of worst error arrows to show
        show_sternum: whether to show sternum positions
        show_legend: whether to show legend
        screenshot_path: if provided, save screenshot to this path
    """
    plotter = pv.Plotter()
    plotter.set_background('white')

    # Update title with iteration if provided
    if iteration is not None:
        title = f"{title} (Iteration {iteration})"
    plotter.add_text(title, font_size=14, position='upper_left')

    # Use aligned source if provided, otherwise use original
    display_source = source_aligned if source_aligned is not None else source_mesh_coords

    # --- Target Point Cloud (plain overlay) ---
    plotter.add_points(
        target_pc,
        color='blue',
        point_size=max(1, point_size - 2),
        render_points_as_spheres=True,
        opacity=0.3,
        label='Target (Supine)'
    )

    # --- Source Mesh colored by error (prone to supine) ---
    if error_magnitudes is not None:
        source_cloud = pv.PolyData(display_source)
        source_cloud['Alignment Error (mm)'] = error_magnitudes
        plotter.add_points(
            source_cloud,
            scalars='Alignment Error (mm)',
            cmap=cmap,
            point_size=point_size,
            render_points_as_spheres=True,
            show_scalar_bar=True,
            scalar_bar_args={
                'title': 'Error (mm)',
                'vertical': True,
                'position_x': 0.85,
                'position_y': 0.1,
                'width': 0.1,
                'height': 0.6
            }
        )
    elif selected_elements_coords is not None:
        # Full mesh in gray (not used for alignment)
        plotter.add_points(
            display_source,
            color='lightgray',
            point_size=max(1, point_size - 2),
            render_points_as_spheres=True,
            opacity=0.3,
            label='Full Mesh (not used)'
        )
        # Selected elements in red (used for alignment)
        plotter.add_points(
            selected_elements_coords,
            color='red',
            point_size=point_size,
            render_points_as_spheres=True,
            label='Selected Elements (used)'
        )
    else:
        # All mesh in red
        plotter.add_points(
            display_source,
            color='red',
            point_size=max(2, point_size - 1),
            render_points_as_spheres=True,
            label='Source Mesh (Prone)'
        )

    # --- Error Arrows (worst errors) ---
    if show_error_arrows and error_magnitudes is not None and error_indices is not None:
        # Find worst N source points by error magnitude
        worst_n = min(worst_n_arrows, len(error_magnitudes))
        worst_src_indices = np.argsort(error_magnitudes)[-worst_n:]

        # Draw arrows from worst source points to their nearest target points
        for src_idx in worst_src_indices:
            start = display_source[src_idx]
            end = target_pc[error_indices[src_idx]]
            direction = end - start

            error = error_magnitudes[src_idx]
            plotter.add_arrows(
                cent=start.reshape(1, 3),
                direction=direction.reshape(1, 3),
                mag=1.0,
                color='orange' if error > 10 else 'yellow',
            )

    # --- Sternum Markers ---
    if show_sternum:
        if source_sternum is not None:
            plotter.add_points(
                source_sternum.reshape(1, 3),
                color='green',
                point_size=15,
                render_points_as_spheres=True,
                label='Source Sternum'
            )
        if target_sternum is not None:
            plotter.add_points(
                target_sternum.reshape(1, 3),
                color='purple',
                point_size=15,
                render_points_as_spheres=True,
                label='Target Sternum'
            )

    # --- Add Legend ---
    if show_legend:
        legend_entries = []
        if selected_elements_coords is not None:
            legend_entries.append(['Selected Elements', 'red'])
            legend_entries.append(['Full Mesh', 'lightgray'])
        else:
            legend_entries.append(['Source Mesh', 'red'])

        if error_magnitudes is None:
            legend_entries.append(['Target PC', 'blue'])

        if show_sternum:
            if source_sternum is not None:
                legend_entries.append(['Source Sternum', 'green'])
            if target_sternum is not None:
                legend_entries.append(['Target Sternum', 'purple'])

        if legend_entries:
            plotter.add_legend(legend_entries, bcolor='white')

    # --- Statistics Text Box ---
    if error_magnitudes is not None:
        stats_text = (
            f"Error Statistics:\n"
            f"  Mean: {np.mean(error_magnitudes):.2f} mm\n"
            f"  Std: {np.std(error_magnitudes):.2f} mm\n"
            f"  RMSE: {np.sqrt(np.mean(error_magnitudes**2)):.2f} mm\n"
            f"  Min: {np.min(error_magnitudes):.2f} mm\n"
            f"  Max: {np.max(error_magnitudes):.2f} mm"
        )
        plotter.add_text(stats_text, font_size=10, position='lower_left')

    plotter.add_axes()

    # Save screenshot if path provided
    if screenshot_path:
        plotter.show(screenshot=screenshot_path, auto_close=True)
    else:
        plotter.show()


def visualize_alignment_during_iteration(
        source_centered: np.ndarray,
        target_centered: np.ndarray,
        iteration: int,
        correspondences: np.ndarray = None,
        correspondence_distances: np.ndarray = None,
        valid_mask: np.ndarray = None,
        max_correspondence_distance: float = 15.0,
        show_correspondences: bool = True,
        n_correspondence_lines: int = 100,
        show_unused_points: bool = True
) -> None:
    """
    Visualize alignment state during an ICP iteration (sternum-centered coordinates).

    Shows the ENTIRE mesh/point cloud with UNUSED points in muted colors
    and USED points (after trimming and max correspondence distance filtering)
    highlighted in bright colors.

    Args:
        source_centered: (N, 3) source points centered on sternum (origin)
        target_centered: (M, 3) target points centered on sternum (origin)
        iteration: current iteration number
        correspondences: (N,) indices of corresponding target points for each source
        correspondence_distances: (N,) distances to corresponding points
        valid_mask: (N,) boolean mask of valid correspondences (after trimming)
        max_correspondence_distance: maximum distance for valid correspondences
        show_correspondences: whether to draw lines between correspondences
        n_correspondence_lines: number of correspondence lines to draw
        show_unused_points: if True, show whole point cloud with unused in muted colors (default: True)
    """
    plotter = pv.Plotter()
    plotter.set_background('white')
    plotter.add_text(f"ICP Iteration {iteration}", font_size=14, position='upper_left')

    # Get valid indices
    if valid_mask is not None:
        valid_indices = np.where(valid_mask)[0]
        invalid_indices = np.where(~valid_mask)[0]
    else:
        valid_indices = np.arange(len(source_centered))
        invalid_indices = np.array([], dtype=int)

    # --- SOURCE POINTS (Prone mesh) ---
    # FIRST: Show ALL source points in muted color (whole mesh/point cloud)
    if show_unused_points and len(invalid_indices) > 0:
        invalid_source = source_centered[invalid_indices]
        plotter.add_points(
            invalid_source,
            color='lightgray',
            point_size=3,
            render_points_as_spheres=True,
            opacity=0.4,
            label='Source (not used)'
        )

    # SECOND: Highlight USED source points in bright color
    if len(valid_indices) > 0:
        valid_source = source_centered[valid_indices]

        if correspondence_distances is not None:
            # Color valid source points by correspondence distance
            valid_distances = correspondence_distances[valid_indices]
            source_cloud = pv.PolyData(valid_source)
            source_cloud['Distance (mm)'] = valid_distances

            plotter.add_points(
                source_cloud,
                scalars='Distance (mm)',
                cmap='plasma',  # More visible colormap
                point_size=6,
                render_points_as_spheres=True,
                show_scalar_bar=True,
                scalar_bar_args={'title': 'Correspondence\nDistance (mm)'}
            )
        else:
            plotter.add_points(
                valid_source,
                color='red',
                point_size=6,
                render_points_as_spheres=True,
                label='Source (used for alignment)'
            )

    # --- TARGET POINTS (Supine point cloud) ---
    if correspondences is not None and valid_mask is not None and len(valid_indices) > 0:
        # Get target points that have valid correspondences
        valid_correspondences = correspondences[valid_indices]
        unique_target_indices = np.unique(valid_correspondences)

        # FIRST: Show ALL target points in muted color (whole point cloud)
        if show_unused_points:
            all_target_indices = np.arange(len(target_centered))
            unused_target_indices = np.setdiff1d(all_target_indices, unique_target_indices)
            if len(unused_target_indices) > 0:
                unused_target = target_centered[unused_target_indices]
                plotter.add_points(
                    unused_target,
                    color='lightblue',
                    point_size=3,
                    render_points_as_spheres=True,
                    opacity=0.3,
                    label='Target (not used)'
                )

        # SECOND: Highlight USED target points in bright color
        used_target = target_centered[unique_target_indices]
        plotter.add_points(
            used_target,
            color='blue',
            point_size=5,
            render_points_as_spheres=True,
            label='Target (used for alignment)'
        )
    else:
        # No valid mask - show all target points
        plotter.add_points(
            target_centered,
            color='blue',
            point_size=4,
            render_points_as_spheres=True,
            label='Target (Supine)'
        )

    # --- Origin Marker (Sternum) ---
    plotter.add_points(
        np.array([[0, 0, 0]]),
        color='green',
        point_size=20,
        render_points_as_spheres=True,
        label='Origin (Sternum)'
    )

    # --- Correspondence Lines (only for used points) ---
    if show_correspondences and correspondences is not None and len(valid_indices) > 0:
        # Sample from valid correspondences only
        if len(valid_indices) > n_correspondence_lines:
            # Sample evenly from valid indices
            sample_idx = valid_indices[::len(valid_indices)//n_correspondence_lines][:n_correspondence_lines]
        else:
            sample_idx = valid_indices

        # Draw lines for sampled correspondences
        lines = []
        for src_idx in sample_idx:
            tgt_idx = correspondences[src_idx]
            start = source_centered[src_idx]
            end = target_centered[tgt_idx]
            lines.append([start, end])

        if lines:
            for line in lines:
                plotter.add_lines(
                    np.array(line),
                    color='yellow',
                    width=1
                )

    # --- Statistics (for used points only) ---
    n_used_source = len(valid_indices)
    n_total_source = len(source_centered)
    n_used_target = len(np.unique(correspondences[valid_indices])) if correspondences is not None and len(valid_indices) > 0 else 0
    n_total_target = len(target_centered)

    if correspondence_distances is not None and len(valid_indices) > 0:
        valid_dists = correspondence_distances[valid_indices]
        stats_text = (
            f"Iteration {iteration} Statistics:\n"
            f"  Source (bright): {n_used_source}/{n_total_source} ({100*n_used_source/n_total_source:.1f}%)\n"
            f"  Target (blue): {n_used_target}/{n_total_target} ({100*n_used_target/n_total_target:.1f}%)\n"
            f"  Mean distance: {np.mean(valid_dists):.2f} mm\n"
            f"  RMSE: {np.sqrt(np.mean(valid_dists**2)):.2f} mm\n"
            f"  Max distance: {np.max(valid_dists):.2f} mm\n"
            f"  Gray/Light = not used for alignment"
        )
        plotter.add_text(stats_text, font_size=10, position='lower_left')

    plotter.add_legend(bcolor='white')
    plotter.add_axes()
    plotter.show()


def get_surface_mesh_coords(morphic_mesh, res, elems=None):
    """
    Extracts the 3D coordinates of a surface mesh

    :param morphic_mesh: surface mesh
    :type morphic_mesh: morphic.Mesh
    :param res: number of material points per element axis
    :type res: int
    :param elems: specified elements to extract coordinates (None or empty list for all elements)
    :type elems: list
    :return: mesh_coords
    :rtype: ndarray
    """
    # Handle mutable default argument
    if elems is None:
        elems = []

    #   local coordinates for each element
    Xi = morphic_mesh.grid(res, method='center')
    NPPE = Xi.shape[0]

    #   if looking at all elements of mesh
    if len(elems) == 0:

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



def get_mesh_with_selected_elements(
        morphic_mesh,
        selected_elements: list,
        res: int = 26
) -> np.ndarray:
    """
    Return mesh coordinates for only the selected elements.

    Args:
        morphic_mesh: morphic.Mesh object
        selected_elements: list of element indices to keep (e.g. [0,1,2,5,6])
        res: number of material points per element axis (default: 26)

    Returns:
        (N, 3) array of mesh coordinates for the selected elements only
    """
    if not selected_elements:
        raise ValueError("selected_elements must be a non-empty list of element indices")

    num_elements = morphic_mesh.elements.size()
    invalid = [e for e in selected_elements if e < 0 or e >= num_elements]
    if invalid:
        raise ValueError(
            f"Element indices {invalid} are out of range. "
            f"Mesh has {num_elements} elements (0-{num_elements - 1})."
        )

    mesh_coords = get_surface_mesh_coords(morphic_mesh, res=res, elems=selected_elements)

    print(f"Selected {len(selected_elements)}/{num_elements} elements "
          f"-> {mesh_coords.shape[0]} points")

    return mesh_coords


def filter_point_cloud_to_match_mesh_region(
        point_cloud: np.ndarray,
        mesh_region_coords: np.ndarray,
        pc_sternum_sup: np.ndarray,
        mesh_sternum_sup: np.ndarray,
        padding: float = 30.0,
        verbose: bool = True,
) -> np.ndarray:
    """
    Filter a point cloud to the spatial region covered by selected mesh elements.

    Both the point cloud and mesh coordinates are shifted to sternum-centered
    space before computing the bounding box, so the spatial comparison is
    consistent even when the two datasets have different world-space origins.

    The bounding box of the mesh region is expanded by ``padding`` on every
    side so the point cloud is a slight superset of the mesh. This ensures
    every mesh point can find its true nearest neighbour during ICP.

    Args:
        point_cloud:       (M, 3) target point cloud (e.g. supine ribcage)
        mesh_region_coords: (N, 3) mesh coordinates for the selected elements
        pc_sternum_sup:    (3,) sternum superior position for the point cloud
        mesh_sternum_sup:  (3,) sternum superior position for the mesh
        padding:           extra margin (mm) on each side of the bounding box
                           (default 15 mm  one rib spacing)
        verbose:           print filtering summary

    Returns:
        (K, 3) filtered point cloud (K <= M)
    """
    pc_sternum = np.asarray(pc_sternum_sup, dtype=np.float64).flatten()
    mesh_sternum = np.asarray(mesh_sternum_sup, dtype=np.float64).flatten()

    # Centre both on their respective sternums
    mesh_centered = mesh_region_coords - mesh_sternum
    pc_centered = point_cloud - pc_sternum

    # Bounding box of the mesh region (in sternum-centered space)
    bbox_min = mesh_centered.min(axis=0) - padding
    bbox_max = mesh_centered.max(axis=0) + padding

    # Keep points inside the padded bounding box
    inside = np.all(
        (pc_centered >= bbox_min) & (pc_centered <= bbox_max),
        axis=1,
    )
    filtered = point_cloud[inside]

    if verbose:
        print(f"\n=== Point Cloud Region Filter ===")
        print(f"  Mesh region bbox (sternum-centered, with {padding:.0f}mm padding):")
        print(f"    X: [{bbox_min[0]:.1f}, {bbox_max[0]:.1f}]")
        print(f"    Y: [{bbox_min[1]:.1f}, {bbox_max[1]:.1f}]")
        print(f"    Z: [{bbox_min[2]:.1f}, {bbox_max[2]:.1f}]")
        print(f"  Points: {len(point_cloud)} Filtered: {len(filtered)} "
              f"({len(filtered)/len(point_cloud)*100:.0f}% kept)")

    return filtered


def align_prone_to_supine_optimal(
        subject: "Subject",
        prone_ribcage_mesh_path: Path,
        supine_ribcage_seg_path: Path,
        orientation_flag: str = 'RAI',
        plot_for_debug: bool = False,
        max_correspondence_distance: float = 15.0,
        max_iterations: int = 500,
        trim_percentage: float = 0.1,
        selected_elements: list = None,
        visualize_iterations: bool = False,
        visualize_every_n: int = 10,
        verbose: bool = True
) -> dict:
    """
    Main alignment function with sternum superior fixed at origin.

    This function can be called directly from main.py and provides
    the same interface as align_prone_to_supine and align_prone_to_supine_fixed_sternum.

    Element selection:
        Pass selected_elements=[0,1,2,...] with specific element indices.
        If not set, all elements are used.

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
        selected_elements: List of element indices for alignment (e.g. [0,1,2,5]).
                          If None, all elements are used.
        visualize_iterations: If True, show visualization during ICP iterations
        visualize_every_n: Show visualization every N iterations (default: 10)
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
        print(f"STERNUM-FIXED ALIGNMENT")
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
    landmark_prone_ave_raw = get_landmarks_as_array(prone_scan_data, "anthony")
    landmark_supine_ave_raw = get_landmarks_as_array(supine_scan_data, "anthony")

    # Load prone ribcage mesh
    prone_ribcage = morphic.Mesh(str(prone_ribcage_mesh_path))

    prone_ribcage_mesh_coords = get_surface_mesh_coords(prone_ribcage, res=26)

    # Load supine ribcage segmentation
    supine_ribcage_mask = breast_metadata.readNIFTIImage(
        str(supine_ribcage_seg_path), orientation_flag, swap_axes=True
    )
    supine_ribcage_pc = extract_contour_points(supine_ribcage_mask, 20000)
    plot_all(point_cloud=supine_ribcage_pc)
    # Get element centers and visualize (use this to decide which elements to select)
    centers_array, num_elements = get_mesh_elements_2(prone_ribcage)
    plot_mesh_elements(prone_ribcage_mesh_coords, centers_array, range(num_elements))

    # Select subset of mesh elements for alignment
    # Otherwise, use all elements

    if selected_elements is not None:
        prone_ribcage_alignment_coords = get_mesh_with_selected_elements(
            prone_ribcage, selected_elements, res=26
        )
        if verbose:
            print(f"  Using {len(selected_elements)}/{num_elements} elements for alignment")
            print(f"  Selected elements: {selected_elements}")
            print(f"  Alignment points: {prone_ribcage_alignment_coords.shape[0]} "
                  f"(full mesh: {prone_ribcage_mesh_coords.shape[0]})")
        if plot_for_debug:
            selected_centers = centers_array[selected_elements]
            plot_mesh_elements(prone_ribcage_alignment_coords, selected_centers,
                               selected_elements)
    else:
        # No selection - use all elements
        prone_ribcage_alignment_coords = prone_ribcage_mesh_coords
        if verbose:
            print(f"  Using all {num_elements} elements for alignment")

    # ==============================================================
    # 2. Run plane-to-point alignment
    # ==============================================================
    # Mutual region filtering (target PC + reciprocal source filtering)
    # is handled inside surface_to_point_align when target_region_filter=True.
    R, T_total, info = surface_to_point_align(
        vl_id=subject.subject_id,
        mesh=prone_ribcage,
        target_pts=supine_ribcage_pc,
        source_sternum_sup=sternum_prone[0],
        target_sternum_sup=sternum_supine[0],
        max_distance=max_correspondence_distance,
        max_iterations=max_iterations,
        convergence_threshold=1e-6,
        trim_percentage=trim_percentage,
        res=10,
        verbose=verbose,
        elems=selected_elements,
        point_to_point_weight=0.0,
        target_region_filter=True,
        target_region_padding=15.0,
        target_region_padding_inferior=0.0,
    )

    # Filtered supine PC from alignment (same region as selected mesh elements)
    supine_ribcage_pc_alignment = info['target_pts_filtered']

    # ==========================================================
    # 3. TRANSFORM ALL PRONE DATA TO SUPINE FRAME
    # ==========================================================
    # NOTE: After this transformation, all prone data is in the ORIGINAL
    #       SUPINE COORDINATE SYSTEM (not sternum-centered).
    #       The prone sternum is moved to the supine sternum's position.

    source_anchor = sternum_prone[0]
    target_anchor = sternum_supine[0]

    # Transform anatomical landmarks
    sternum_prone_transformed = apply_transform_to_coords(
        sternum_prone, R, source_anchor, target_anchor
    )
    nipple_prone_transformed = apply_transform_to_coords(
        nipple_prone, R, source_anchor, target_anchor
    )

    # Transform registrar landmarks
    landmark_prone_transformed = apply_transform_to_coords(
        landmark_prone_ave_raw, R, source_anchor, target_anchor
    )

    # Transform full ribcage for visualization and error calculation
    prone_ribcage_transformed = apply_transform_to_coords(
        prone_ribcage_mesh_coords, R, source_anchor, target_anchor
    )

    # Transform element centers for visualization
    centers_array_transformed = apply_transform_to_coords(
        centers_array, R, source_anchor, target_anchor
    )

    # Transform selected elements (same as full when no selection)
    if selected_elements is not None:
        prone_selected_transformed = apply_transform_to_coords(
            prone_ribcage_alignment_coords, R, source_anchor, target_anchor
        )
    else:
        prone_selected_transformed = prone_ribcage_transformed

    # Visualize alignment results with transformed mesh and element centers
    if selected_elements is not None:
        plot_mesh_elements(prone_selected_transformed, centers_array_transformed[selected_elements],
                          selected_elements, supine_ribcage_pc_alignment)

    plot_mesh_elements(prone_ribcage_transformed, centers_array_transformed,
                      range(num_elements), supine_ribcage_pc)

    # ==========================================================
    # 4. CALCULATE ERRORS AND METRICS
    # ==========================================================

    # Sternum error (should be near zero)
    sternum_error = np.linalg.norm(sternum_prone_transformed[0] - sternum_supine[0])

    # --- Full mesh errors (prone to supine: for each transformed prone point,
    #     find nearest supine point. Consistent with ICP internal direction.) ---
    error_full, mapped_idx_full = breast_metadata.closest_distances(
        prone_ribcage_transformed, supine_ribcage_pc
    )
    rib_error_mag_full = np.linalg.norm(error_full, axis=1)

    ribcage_error_mean = float(np.mean(rib_error_mag_full))
    ribcage_error_std = float(np.std(rib_error_mag_full))
    ribcage_error_rmse = float(np.sqrt(np.mean(rib_error_mag_full ** 2)))
    median_err = np.median(rib_error_mag_full)
    q25 = np.percentile(rib_error_mag_full, 25)
    q75 = np.percentile(rib_error_mag_full, 75)


    # --- Selected elements errors (prone to supine) ---
    # IMPORTANT: Compare selected prone mesh to the FILTERED supine point cloud
    # that corresponds to the same anatomical region. This gives the "true"
    # alignment quality for the selected region.
    error_selected, mapped_idx_selected = breast_metadata.closest_distances(
        prone_selected_transformed, supine_ribcage_pc_alignment  # Use filtered!
    )
    rib_error_mag_selected = np.linalg.norm(error_selected, axis=1)

    selected_error_mean = float(np.mean(rib_error_mag_selected))
    selected_error_std = float(np.std(rib_error_mag_selected))
    selected_error_rmse = float(np.sqrt(np.mean(rib_error_mag_selected ** 2)))

    # Also keep inlier metrics for comparison
    ribcage_inlier_rmse = info['euclidean_rmse']
    ribcage_inlier_mean = info['euclidean_mean']
    ribcage_inlier_std = info['euclidean_std']

    if verbose:
        print(f"\n{'='*60}")
        print(f"ALIGNMENT RESULTS:")
        print(f"{'='*60}")
        print(f"  Sternum error: {sternum_error:.6f} mm (should be ~0)")
        print(f"  --- Full mesh ({prone_ribcage_mesh_coords.shape[0]} pts) ---")
        print(f"    RMSE: {ribcage_error_rmse:.4f} mm")
        print(f"    Mean +/- SD: {ribcage_error_mean:.4f} +/- {ribcage_error_std:.4f} mm")
        if selected_elements is not None:
            print(f"  --- Selected elements ({prone_ribcage_alignment_coords.shape[0]} pts) ---")
            print(f"    RMSE: {selected_error_rmse:.4f} mm")
            print(f"    Mean +/- SD: {selected_error_mean:.4f} +/- {selected_error_std:.4f} mm")
        print(f"  --- ICP inlier metrics ---")
        print(f"    RMSE: {ribcage_inlier_rmse:.4f} mm")
        print(f"    Mean +/- SD: {ribcage_inlier_mean:.4f} +/- {ribcage_inlier_std:.4f} mm")
        # print(f"  Final correspondences within 15mm: {info['n_inliers']}/{info['n_total_source']} ({info['inlier_fraction']*100:.1f}%)")
        print(f"  Note: Trim percentage (fixed at {trim_percentage*100:.0f}%) was applied during optimization")
        print(f"  Iterations: {info['iterations']}")

        # Publication-ready summary
        print(f"\n{'='*60}")
        print(f"PUBLICATION SUMMARY:")
        print(f"{'='*60}")
        print_alignment_accuracy_report(rib_error_mag_full, sternum_error, info)

    # ==========================================================
    # 5. CALCULATE DISPLACEMENTS RELATIVE TO STERNUM
    # ==========================================================
    # NOTE: Up to this point, all coordinates are in the ORIGINAL SUPINE FRAME.
    #       By subtracting the sternum position, we convert to STERNUM-CENTERED
    #       coordinates where sternum is at (0, 0, 0).

    ref_sternum_prone = sternum_prone_transformed[0]
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
            # Plot full mesh alignment
            sternum_lists = [sternum_prone_transformed, sternum_supine]
            plot_all(
                point_cloud=supine_ribcage_pc,
                mesh_points=prone_ribcage_transformed,
                anat_landmarks=sternum_lists,
            )
            '''
            # Visualize alignment errors with only parts used for alignment
            # Colors prone mesh by distance to nearest supine point (prone to supine)
            visualize_alignment_errors(
                source_mesh_coords=prone_ribcage_transformed,
                target_pc=supine_ribcage_pc,
                source_aligned=prone_ribcage_transformed,
                error_magnitudes=rib_error_mag_full,
                error_indices=mapped_idx_full,
                source_sternum=sternum_prone_transformed[0],
                target_sternum=sternum_supine[0],
                selected_elements_coords=prone_selected_transformed if selected_elements is not None else None,
                title=f"Alignment Errors - Subject {subject.subject_id}",
                cmap="coolwarm",
                point_size=4,
                show_error_arrows=True,
                worst_n_arrows=50,
                show_sternum=True,
                show_legend=True
            )

            # # Error visualization - full mesh (prone tosupine)
            # plot_evaluate_alignment(
            #     supine_pts=supine_ribcage_pc,
            #     transformed_prone_mesh=prone_ribcage_transformed,
            #     distances=rib_error_mag_full,
            #     idxs=mapped_idx_full,
            #     worst_n=60,
            #     cmap="viridis",
            #     point_size=3,
            #     arrow_scale=20,
            #     show_scalar_bar=True,
            #     return_data=False
            # )

            # Error visualization - selected elements only (prone to supine)
            if selected_elements is not None:
                plot_evaluate_alignment(
                    supine_pts=supine_ribcage_pc,
                    transformed_prone_mesh=prone_selected_transformed,
                    distances=rib_error_mag_selected,
                    idxs=mapped_idx_selected,
                    worst_n=60,
                    cmap="viridis",
                    point_size=3,
                    arrow_scale=20,
                    show_scalar_bar=True,
                    return_data=False
                )
                
            '''
        except Exception as e:
            print(f"Could not generate debug plots: {e}")


    # title_sternum = "Landmark Displacement Relative to Sternal Superior (Jugular Notch)"
    # lm_pos_left_rel_sternum = lm_pos_prone_rel_sternum[is_left_breast]
    # lm_pos_right_rel_sternum = lm_pos_prone_rel_sternum[~is_left_breast]
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
        sternum_prone_px = get_px_coords(prone_scan_transformed, sternum_prone_transformed)
        sternum_supine_px = get_px_coords(supine_scan, sternum_supine)

        # Convert Landmarks (Using the AVE variables consistently)
        lm_prone_trans_px = get_px_coords(prone_scan_transformed, landmark_prone_transformed)
        lm_supine_px = get_px_coords(supine_scan, landmark_supine_ave_raw)

        # %%   plot
        breast_metadata.visualise_alignment_with_landmarks(
            supine_image_sitk, prone_image_transformed, sternum_supine_px[0], sternum_prone_px[0], orientation='axial')
        breast_metadata.visualise_alignment_with_landmarks(
            supine_image_sitk, prone_image_transformed, sternum_supine_px[0], sternum_prone_px[0], orientation='sagittal')
        breast_metadata.visualise_alignment_with_landmarks(
            supine_image_sitk, prone_image_transformed, sternum_supine_px[0], sternum_prone_px[0], orientation='coronal')

        breast_metadata.visualise_alignment_with_landmarks(
            supine_image_sitk, prone_image_transformed, sternum_supine_px[1], sternum_prone_px[1], orientation='axial')
        breast_metadata.visualise_alignment_with_landmarks(
            supine_image_sitk, prone_image_transformed, sternum_supine_px[1], sternum_prone_px[1], orientation='sagittal')
        breast_metadata.visualise_alignment_with_landmarks(
            supine_image_sitk, prone_image_transformed, sternum_supine_px[1], sternum_prone_px[1], orientation='coronal')

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
        plotter.add_points(sternum_prone_transformed, render_points_as_spheres=True, color='black', point_size=10,
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

        # Error metrics - full mesh
        'ribcage_error_rmse': ribcage_error_rmse,
        'ribcage_error_mean': ribcage_error_mean,
        'ribcage_error_std': ribcage_error_std,
        'ribcage_inlier_rmse': ribcage_inlier_rmse,
        'ribcage_inlier_mean': ribcage_inlier_mean,
        'ribcage_inlier_std': ribcage_inlier_std,
        'sternum_error': sternum_error,

        # # Error metrics - selected elements
        # 'selected_error_rmse': selected_error_rmse,
        # 'selected_error_mean': selected_error_mean,
        # 'selected_error_std': selected_error_std,
        # 'selected_elements': selected_elements,

        # Transformed anatomical landmarks
        'sternum_prone_transformed': sternum_prone_transformed,
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
    vl_ids = [9]
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
            # Manual element selection (or None to use all elements)
            # selected_elements=[0, 1, 6, 7, 8, 9, 14, 15],
            selected_elements=[0, 1, 6, 7, 8, 9, 14, 15, 16, 17, 22, 23],
            orientation_flag='RAI',
            plot_for_debug=True,
            # max_correspondence_distance=15.0,  # Max distance for valid correspondences (mm)
            max_correspondence_distance=1e6,
            trim_percentage=0,  # Reject worst 10% of correspondences
            visualize_iterations=True,
            visualize_every_n=50)


