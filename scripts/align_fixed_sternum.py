"""
Prone-to-Supine Alignment with Fixed Sternum Superior

This module implements a mathematically rigorous alignment approach where the
sternum superior landmark is locked in place during optimization.

Key Principle:
By centering all coordinates on the sternum superior (moving it to origin),
we ensure that rotations cannot displace this anatomically stable landmark.

Date: February 3, 2026
"""

from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from typing import Tuple, Dict, Optional
import external.breast_metadata_mdv.breast_metadata as breast_metadata
from structures import Subject
from utils import apply_transform, extract_contour_points, filter_point_cloud_asymmetric
from utils import get_landmarks_as_array, plot_evaluate_alignment
from utils_plot import plot_all



def apply_rotation_only(points: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Apply rotation matrix to points (no translation).

    Args:
        points: (N, 3) array of 3D coordinates
        R: (3, 3) rotation matrix

    Returns:
        Rotated points (N, 3)
    """
    return (R @ points.T).T


def rotation_matrix_from_euler(angles: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (rx, ry, rz in radians) to rotation matrix.
    Uses ZYX convention (yaw-pitch-roll).

    Args:
        angles: [rx, ry, rz] in radians

    Returns:
        3x3 rotation matrix
    """
    rx, ry, rz = angles

    # Rotation around X-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    # Rotation around Y-axis
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # Rotation around Z-axis
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    # Combined rotation: Rz * Ry * Rx
    return Rz @ Ry @ Rx


def rotation_only_objective(
    angles: np.ndarray,
    prone_ribcage_centered: np.ndarray,
    supine_ribcage_centered: np.ndarray,
    prone_sternum_inf_centered: np.ndarray,
    supine_sternum_inf_centered: np.ndarray,
    w_rib: float = 1.0,
    w_sternum: float = 100.0
) -> float:
    """
    Objective function for rotation-only alignment.
    Both prone ribcage and sternum inferior have been pre-centered on sternum superior.

    This ensures sternum superior remains at origin (fixed) during rotation.

    Args:
        angles: [rx, ry, rz] Euler angles in radians
        prone_ribcage_centered: Ribcage points centered on sternum superior
        supine_ribcage_centered: Target ribcage centered on sternum superior
        prone_sternum_inf_centered: Sternum inferior centered on sternum superior
        supine_sternum_inf_centered: Target sternum inferior centered
        w_rib: Weight for ribcage term
        w_sternum: Weight for sternum inferior term (high to lock sternum axis)

    Returns:
        Weighted sum of squared distances
    """
    R = rotation_matrix_from_euler(angles)

    # Rotate the prone data (centered on sternum superior)
    prone_rib_rotated = apply_rotation_only(prone_ribcage_centered, R)
    prone_sternum_inf_rotated = apply_rotation_only(prone_sternum_inf_centered, R)

    # Compute ribcage fit (nearest neighbor distances)
    from scipy.spatial import cKDTree
    tree = cKDTree(supine_ribcage_centered)
    distances, _ = tree.query(prone_rib_rotated, k=1)
    rib_error = np.sum(distances**2)

    # Compute sternum inferior fit (should align along chest axis)
    sternum_error = np.sum((prone_sternum_inf_rotated - supine_sternum_inf_centered)**2)

    # Combined objective
    total_error = w_rib * rib_error + w_sternum * sternum_error

    return total_error


def estimate_normals_from_neighbors(points: np.ndarray, k_neighbors: int = 50) -> np.ndarray:
    """
    Estimate surface normals using PCA on local neighborhoods.

    Args:
        points: (N, 3) point cloud
        k_neighbors: Number of neighbors for normal estimation

    Returns:
        (N, 3) array of unit normal vectors
    """
    tree = cKDTree(points)
    normals = np.zeros_like(points)

    for i in range(len(points)):
        # Find k nearest neighbors
        distances, indices = tree.query(points[i], k=min(k_neighbors, len(points)))
        neighbors = points[indices]

        # Compute covariance matrix
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered

        # Normal is eigenvector with smallest eigenvalue
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]  # Smallest eigenvalue

        # Orient normal consistently (towards negative Z if ambiguous)
        if normal[2] > 0:
            normal = -normal

        normals[i] = normal

    return normals


def huber_loss(residuals: np.ndarray, delta: float = 1.0) -> float:
    """
    Huber loss function for robust optimization.

    Args:
        residuals: Array of residual values
        delta: Threshold for switching from quadratic to linear

    Returns:
        Total Huber loss
    """
    abs_residuals = np.abs(residuals)
    quadratic = abs_residuals <= delta
    linear = abs_residuals > delta

    loss = np.sum(0.5 * residuals[quadratic]**2)
    loss += np.sum(delta * (abs_residuals[linear] - 0.5 * delta))

    return loss


def run_fixed_sternum_icp(
    source_pts_centered: np.ndarray,
    target_pts_centered: np.ndarray,
    max_correspondence_distance: float = 10.0,
    max_iterations: int = 50,
    huber_delta: float = 2.0,
    convergence_threshold: float = 1e-6,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Robust point-to-plane ICP with rotation-only constraint (no translation).
    Uses Huber loss for outlier rejection and iterative refinement.

    This is applied to data that has been pre-centered on sternum superior,
    ensuring the sternum cannot slide during refinement.

    Args:
        source_pts_centered: Source points (centered on sternum superior)
        target_pts_centered: Target points (centered on sternum superior)
        max_correspondence_distance: Max distance for correspondences
        max_iterations: Max ICP iterations
        huber_delta: Threshold for Huber loss (robustness parameter)
        convergence_threshold: Convergence criterion for angle change
        verbose: Print debug info

    Returns:
        (R, source_aligned, info): Rotation matrix, aligned points, metrics
    """
    source_pts_centered = np.asarray(source_pts_centered, dtype=np.float64)
    target_pts_centered = np.asarray(target_pts_centered, dtype=np.float64)

    if source_pts_centered.size == 0 or target_pts_centered.size == 0:
        return np.eye(3), source_pts_centered.copy(), {"fitness": 0.0, "inlier_rmse": np.nan}

    # Estimate normals
    print("  Estimating surface normals...")
    target_normals = estimate_normals_from_neighbors(target_pts_centered, k_neighbors=50)

    # Build KD-Tree for fast nearest neighbor search
    tree = cKDTree(target_pts_centered)

    # Initialize rotation
    R_cumulative = np.eye(3)

    # Iterative ICP refinement
    iteration_history = []

    for iteration in range(max_iterations):
        # Apply current rotation
        source_rotated = apply_rotation_only(source_pts_centered, R_cumulative)

        # Find correspondences
        distances, indices = tree.query(source_rotated, k=1)

        # Filter by max correspondence distance
        valid_mask = distances < max_correspondence_distance
        valid_source = source_rotated[valid_mask]
        valid_target = target_pts_centered[indices[valid_mask]]
        valid_normals = target_normals[indices[valid_mask]]

        if len(valid_source) < 10:
            if verbose:
                print(f"  Warning: Only {len(valid_source)} correspondences found")
            break

        # Compute point-to-plane residuals
        diff = valid_source - valid_target
        ptp_residuals = np.sum(diff * valid_normals, axis=1)

        # Compute fitness metrics
        fitness = len(valid_source) / len(source_rotated)
        inlier_rmse = np.sqrt(np.mean(ptp_residuals**2))

        iteration_history.append({
            "fitness": fitness,
            "inlier_rmse": inlier_rmse,
            "num_inliers": len(valid_source)
        })

        # Define objective function with Huber loss
        def rotation_objective(angles: np.ndarray) -> float:
            """Objective: Huber loss on point-to-plane distances"""
            R_delta = rotation_matrix_from_euler(angles)
            R_test = R_delta @ R_cumulative
            source_test = apply_rotation_only(source_pts_centered, R_test)

            # Use same correspondences (stable within iteration)
            source_test_valid = source_test[valid_mask]
            diff_test = source_test_valid - valid_target
            residuals = np.sum(diff_test * valid_normals, axis=1)

            # Huber loss for robustness
            return huber_loss(residuals, delta=huber_delta)

        # Optimize rotation increment
        result = minimize(
            rotation_objective,
            np.zeros(3),  # Start from no change
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-8}
        )

        # Check convergence
        angle_change = np.linalg.norm(result.x)

        if angle_change < convergence_threshold:
            if verbose:
                print(f"  Converged at iteration {iteration + 1} (angle change: {np.degrees(angle_change):.6f}°)")
            break

        # Update cumulative rotation
        R_delta = rotation_matrix_from_euler(result.x)
        R_cumulative = R_delta @ R_cumulative

        if verbose and (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1}: fitness={fitness:.4f}, RMSE={inlier_rmse:.2f} mm, "
                  f"angle change={np.degrees(angle_change):.4f}°")

    # Final alignment
    source_aligned = apply_rotation_only(source_pts_centered, R_cumulative)

    # Final metrics
    distances_final, indices_final = tree.query(source_aligned, k=1)
    valid_mask_final = distances_final < max_correspondence_distance

    fitness_final = np.sum(valid_mask_final) / len(source_aligned)
    inlier_rmse_final = np.sqrt(np.mean(distances_final[valid_mask_final]**2)) if np.any(valid_mask_final) else np.nan

    info = {
        "fitness": float(fitness_final),
        "inlier_rmse": float(inlier_rmse_final),
        "inlier_source_pts": source_aligned[valid_mask_final],
        "convergence": iteration < max_iterations - 1,
        "iterations": iteration + 1,
        "iteration_history": iteration_history
    }

    if verbose:
        print(f"  Final: fitness={fitness_final:.4f}, inlier_rmse={inlier_rmse_final:.2f} mm, "
              f"iterations={iteration + 1}")

    return R_cumulative, source_aligned, info


def align_prone_to_supine_fixed_sternum(
    subject: Subject,
    prone_ribcage_mesh_path: Path,
    supine_ribcage_seg_path: Path,
    orientation_flag: str = 'RAI',
    plot_for_debug: bool = False,
    w_rib: float = 1.0,
    w_sternum: float = 100.0
) -> dict:
    """
    Align prone to supine with sternum superior as a FIXED anchor point.

    Mathematical Approach:
    1. Translate both prone and supine so sternum superior is at origin (0,0,0)
    2. Optimize rotation ONLY (no translation allowed)
    3. Since origin is a fixed point under rotation (R @ [0,0,0] = [0,0,0]),
       sternum superior cannot move
    4. Apply point-to-plane ICP with rotation-only constraint for refinement

    This eliminates ICP "slide" and honors anatomical stability of upper sternum.

    Args:
        subject: Subject data structure
        prone_ribcage_mesh_path: Path to prone ribcage mesh
        supine_ribcage_seg_path: Path to supine ribcage segmentation
        orientation_flag: Image orientation (default 'RAI')
        plot_for_debug: Whether to plot intermediate results
        w_rib: Weight for ribcage alignment term
        w_sternum: Weight for sternum inferior term (high value locks sternum axis)

    Returns:
        Dictionary with transformation results and statistics
    """

    print("\n" + "="*80)
    print("FIXED STERNUM ALIGNMENT")
    print("="*80)

    # ==========================================================
    # PHASE 1: LOAD DATA
    # ==========================================================

    if "prone" not in subject.scans or "supine" not in subject.scans:
        raise ValueError(f"Subject {subject.subject_id} missing prone or supine data")

    # Get anatomical landmarks
    anat_prone = subject.scans["prone"].anatomical_landmarks
    anat_supine = subject.scans["supine"].anatomical_landmarks

    if anat_prone.sternum_superior is None or anat_prone.sternum_inferior is None:
        raise ValueError(f"Subject {subject.subject_id} (prone) missing sternum landmarks")
    if anat_supine.sternum_superior is None or anat_supine.sternum_inferior is None:
        raise ValueError(f"Subject {subject.subject_id} (supine) missing sternum landmarks")

    sternum_prone = np.vstack([anat_prone.sternum_superior, anat_prone.sternum_inferior])
    sternum_supine = np.vstack([anat_supine.sternum_superior, anat_supine.sternum_inferior])

    nipple_prone = np.vstack([anat_prone.nipple_left, anat_prone.nipple_right])
    nipple_supine = np.vstack([anat_supine.nipple_left, anat_supine.nipple_right])

    # Get registrar landmarks
    prone_scan_data = subject.scans["prone"]
    supine_scan_data = subject.scans["supine"]
    landmark_prone_ave_raw = get_landmarks_as_array(prone_scan_data, "average")
    landmark_supine_ave_raw = get_landmarks_as_array(supine_scan_data, "average")

    # Load ribcage data
    import morphic
    prone_ribcage = morphic.Mesh(str(prone_ribcage_mesh_path))
    from breast_metadata_mdv.examples.images.visualise_image_and_mesh import align_prone_supine as aps
    prone_ribcage_mesh_coords = aps.get_surface_mesh_coords(prone_ribcage, res=26)

    supine_ribcage_mask = breast_metadata.readNIFTIImage(
        str(supine_ribcage_seg_path), orientation_flag, swap_axes=True
    )
    supine_ribcage_pc = extract_contour_points(supine_ribcage_mask, 20000)
    supine_ribcage_pc = filter_point_cloud_asymmetric(
        points=supine_ribcage_pc,
        reference=supine_ribcage_pc,
        tol_min=0,
        tol_max=5,
        axis=2
    )

    print(f"Loaded prone ribcage: {prone_ribcage_mesh_coords.shape[0]} points")
    print(f"Loaded supine ribcage: {supine_ribcage_pc.shape[0]} points")

    # ==========================================================
    # PHASE 2: CENTER ON STERNUM SUPERIOR (ANCHOR POINT)
    # ==========================================================

    anchor_prone = sternum_prone[0]  # Sternum superior (prone)
    anchor_supine = sternum_supine[0]  # Sternum superior (supine)

    print(f"\nAnchoring alignment:")
    print(f"  Prone sternum superior: {anchor_prone}")
    print(f"  Supine sternum superior: {anchor_supine}")

    # Center all prone data on prone sternum superior - direct implementation
    prone_rib_centered = prone_ribcage_mesh_coords - anchor_prone
    prone_sternum_centered = sternum_prone - anchor_prone
    prone_nipple_centered = nipple_prone - anchor_prone
    prone_landmarks_centered = landmark_prone_ave_raw - anchor_prone

    # Center all supine data on supine sternum superior - direct implementation
    supine_rib_centered = supine_ribcage_pc - anchor_supine
    supine_sternum_centered = sternum_supine - anchor_supine
    supine_nipple_centered = nipple_supine - anchor_supine
    supine_landmarks_centered = landmark_supine_ave_raw - anchor_supine

    # Verify sternum superior is at origin
    assert np.allclose(prone_sternum_centered[0], [0, 0, 0], atol=1e-10)
    assert np.allclose(supine_sternum_centered[0], [0, 0, 0], atol=1e-10)
    print("✓ Sternum superior locked at origin (0, 0, 0)")

    # ==========================================================
    # PHASE 3: ROTATION-ONLY OPTIMIZATION
    # ==========================================================

    print("\n" + "-"*80)
    print("PHASE 3: Initial Rotation-Only Alignment")
    print("-"*80)

    initial_angles = np.zeros(3)

    result = minimize(
        rotation_only_objective,
        initial_angles,
        args=(
            prone_rib_centered,
            supine_rib_centered,
            prone_sternum_centered[1:2],  # Sternum inferior only
            supine_sternum_centered[1:2],
            w_rib,
            w_sternum
        ),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-6}
    )

    print(f"Optimization converged: {result.success}")
    print(f"Message: {result.message}")
    print(f"Optimal angles (degrees): {np.degrees(result.x)}")

    R_optimal = rotation_matrix_from_euler(result.x)

    # Apply rotation to all prone data (still centered)
    prone_rib_rotated = apply_rotation_only(prone_rib_centered, R_optimal)
    prone_sternum_rotated = apply_rotation_only(prone_sternum_centered, R_optimal)
    prone_nipple_rotated = apply_rotation_only(prone_nipple_centered, R_optimal)
    prone_landmarks_rotated = apply_rotation_only(prone_landmarks_centered, R_optimal)

    # Evaluate initial fit (still in centered coordinates)
    from scipy.spatial import cKDTree
    tree = cKDTree(supine_rib_centered)
    distances, _ = tree.query(prone_rib_rotated, k=1)

    print(f"\nInitial alignment quality (centered coordinates):")
    print(f"  Mean ribcage error: {np.mean(distances):.2f} mm")
    print(f"  Std ribcage error: {np.std(distances):.2f} mm")

    # Check sternum alignment
    sternum_error_centered = np.linalg.norm(
        prone_sternum_rotated - supine_sternum_centered, axis=1
    )
    print(f"  Sternum superior error: {sternum_error_centered[0]:.4f} mm (should be ~0)")
    print(f"  Sternum inferior error: {sternum_error_centered[1]:.2f} mm")

    # Debug visualization after initial rotation
    if plot_for_debug:
        print("\n  Plotting initial rotation alignment...")
        # Un-center for visualization - direct implementation
        prone_rib_rotated_vis = prone_rib_rotated + anchor_supine
        prone_sternum_rotated_vis = prone_sternum_rotated + anchor_supine
        supine_rib_vis = supine_rib_centered + anchor_supine

        sternum_lists = [prone_sternum_rotated_vis, sternum_supine]
        plot_all(point_cloud=supine_rib_vis, mesh_points=prone_rib_rotated_vis,
                anat_landmarks=sternum_lists)

    # ==========================================================
    # PHASE 4: POINT-TO-PLANE ICP WITH ROTATION-ONLY CONSTRAINT
    # ==========================================================

    print("\n" + "-"*80)
    print("PHASE 4: Point-to-Plane ICP Refinement (Rotation-Only)")
    print("-"*80)

    R_icp, supine_rib_aligned, icp_info = run_fixed_sternum_icp(
        source_pts_centered=prone_rib_rotated,
        target_pts_centered=supine_rib_centered,
        max_correspondence_distance=10.0,
        max_iterations=200,
        verbose=True
    )

    # Combined rotation: R_total = R_icp @ R_optimal
    R_total = R_icp @ R_optimal

    # Apply total rotation to all prone data (still centered)
    prone_rib_final = apply_rotation_only(prone_rib_centered, R_total)
    prone_sternum_final = apply_rotation_only(prone_sternum_centered, R_total)
    prone_nipple_final = apply_rotation_only(prone_nipple_centered, R_total)
    prone_landmarks_final = apply_rotation_only(prone_landmarks_centered, R_total)

    # Final evaluation (centered)
    tree = cKDTree(supine_rib_centered)
    distances_final, _ = tree.query(prone_rib_final, k=1)

    print(f"\nFinal alignment quality (centered coordinates):")
    print(f"  Mean ribcage error: {np.mean(distances_final):.2f} mm")
    print(f"  Std ribcage error: {np.std(distances_final):.2f} mm")
    print(f"  ICP fitness: {icp_info['fitness']:.4f}")
    print(f"  ICP inlier RMSE: {icp_info['inlier_rmse']:.2f} mm")

    sternum_error_final = np.linalg.norm(
        prone_sternum_final - supine_sternum_centered, axis=1
    )
    print(f"  Sternum superior error: {sternum_error_final[0]:.6f} mm (locked)")
    print(f"  Sternum inferior error: {sternum_error_final[1]:.2f} mm")

    # Debug visualization after ICP refinement
    if plot_for_debug:
        print("\n  Plotting ICP refinement result...")
        # Un-center for visualization - direct implementation
        prone_rib_final_vis = prone_rib_final + anchor_supine
        prone_sternum_final_vis = prone_sternum_final + anchor_supine
        supine_rib_vis = supine_rib_centered + anchor_supine

        sternum_lists = [prone_sternum_final_vis, sternum_supine]
        plot_all(point_cloud=supine_rib_vis, mesh_points=prone_rib_final_vis,
                anat_landmarks=sternum_lists)

        # Visualize alignment quality with error colors
        try:
            plot_evaluate_alignment(
                supine_pts=supine_rib_vis,
                transformed_prone_mesh=prone_rib_final_vis,
                distances=distances_final,
                idxs=np.arange(len(distances_final)),
                worst_n=60,
                cmap="viridis",
                point_size=3,
                arrow_scale=20,
                show_scalar_bar=True,
                return_data=False
            )
        except Exception as e:
            print(f"  Could not visualize alignment errors: {e}")

    # ==========================================================
    # PHASE 5: CREATE TRANSFORMATION MATRIX (for compatibility only)
    # ==========================================================

    print("\n" + "-"*80)
    print("PHASE 5: Create Transformation Matrix")
    print("-"*80)

    # Create 4x4 transformation matrix for external tools that need it
    # T = [R, t; 0, 1] where t = anchor_supine - R @ anchor_prone
    # This matrix transforms from original prone coordinates to original supine coordinates
    T_total = np.eye(4)
    T_total[:3, :3] = R_total
    T_total[:3, 3] = anchor_supine - R_total @ anchor_prone

    print(f"Transformation matrix created (for external use)")
    print(f"  Translation component: {T_total[:3, 3]}")
    print(f"  Rotation angles (degrees): {np.degrees(result.x)}")

    # Verify sternum superior is at origin in both coordinate systems
    sternum_check_prone = np.linalg.norm(prone_sternum_final[0])
    sternum_check_supine = np.linalg.norm(supine_sternum_centered[0])
    print(f"  Prone sternum superior at origin: {sternum_check_prone:.10f} mm")
    print(f"  Supine sternum superior at origin: {sternum_check_supine:.10f} mm")

    # ==========================================================
    # PHASE 6: CALCULATE DISPLACEMENTS (in centered coordinates)
    # ==========================================================

    print("\n" + "-"*80)
    print("PHASE 6: Calculate Displacements (Sternum Superior = Origin)")
    print("-"*80)

    # Landmark displacements (relative to sternum at origin)
    lm_disp_rel_sternum = supine_landmarks_centered - prone_landmarks_final
    lm_disp_mag_rel_sternum = np.linalg.norm(lm_disp_rel_sternum, axis=1)

    # Nipple displacements (relative to sternum at origin)
    nipple_disp_rel_sternum = supine_nipple_centered - prone_nipple_final
    nipple_disp_mag_rel_sternum = np.linalg.norm(nipple_disp_rel_sternum, axis=1)

    # Assign landmarks to left/right breast based on supine nipple positions
    dist_to_left = np.linalg.norm(supine_landmarks_centered - supine_nipple_centered[0], axis=1)
    dist_to_right = np.linalg.norm(supine_landmarks_centered - supine_nipple_centered[1], axis=1)
    is_left_breast = dist_to_left < dist_to_right

    # Calculate displacements relative to nipple
    # Landmark displacement relative to nipple = landmark displacement - nipple displacement
    nipple_disp_left_vec = nipple_disp_rel_sternum[0]
    nipple_disp_right_vec = nipple_disp_rel_sternum[1]

    closest_nipple_disp_vec = np.where(
        is_left_breast[:, np.newaxis],
        nipple_disp_left_vec,
        nipple_disp_right_vec
    )

    lm_disp_rel_nipple = lm_disp_rel_sternum - closest_nipple_disp_vec
    lm_disp_mag_rel_nipple = np.linalg.norm(lm_disp_rel_nipple, axis=1)

    # Separate by side for plotting (positions relative to nipple)
    left_nipple_prone_pos = prone_nipple_final[0]
    right_nipple_prone_pos = prone_nipple_final[1]

    lm_prone_left = prone_landmarks_final[is_left_breast]
    lm_disp_left = lm_disp_rel_sternum[is_left_breast]
    X_left = lm_prone_left - left_nipple_prone_pos
    V_left = lm_disp_left - nipple_disp_left_vec

    lm_prone_right = prone_landmarks_final[~is_left_breast]
    lm_disp_right = lm_disp_rel_sternum[~is_left_breast]
    X_right = lm_prone_right - right_nipple_prone_pos
    V_right = lm_disp_right - nipple_disp_right_vec

    print(f"Calculated displacements for {len(landmark_prone_ave_raw)} landmarks")
    print(f"  Coordinate system: Centered (sternum superior = origin)")
    print(f"  Mean landmark displacement: {np.mean(lm_disp_mag_rel_sternum):.2f} mm")
    print(f"  Left breast nipple displacement: {nipple_disp_mag_rel_sternum[0]:.2f} mm")
    print(f"  Right breast nipple displacement: {nipple_disp_mag_rel_sternum[1]:.2f} mm")

    # ==========================================================
    # PHASE 7: RETURN RESULTS
    # ==========================================================

    print("\n" + "="*80)
    print("ALIGNMENT COMPLETE")
    print("="*80)
    print("Note: All returned coordinates are CENTERED (sternum superior = origin)")
    print("      Displacements are truly relative to sternum at (0,0,0)")

    results = {
        "vl_id": subject.subject_id,
        "T_total": T_total,  # For external tools only
        "R_total": R_total,

        # Quality metrics
        "sternum_error": sternum_error_final,
        "ribcage_error_mean": float(np.mean(distances_final)),
        "ribcage_error_std": float(np.std(distances_final)),
        "ribcage_inlier_RMSE": icp_info['inlier_rmse'],

        # CENTERED coordinates (sternum superior = origin)
        "sternum_prone_transformed": prone_sternum_final,  # Centered
        "sternum_supine": supine_sternum_centered,  # Centered
        "nipple_prone_transformed": prone_nipple_final,  # Centered
        "nipple_supine": supine_nipple_centered,  # Centered
        "landmark_prone_ave_transformed": prone_landmarks_final,  # Centered
        "landmark_supine_ave": supine_landmarks_centered,  # Centered

        # Displacement vectors and magnitudes (relative to sternum at origin)
        "nipple_displacement_vectors": nipple_disp_rel_sternum,
        "nipple_displacement_magnitudes": nipple_disp_mag_rel_sternum,
        "ld_ave_displacement_vectors": lm_disp_rel_sternum,
        "ld_ave_displacement_magnitudes": lm_disp_mag_rel_sternum,

        # Displacement relative to nipple
        "ld_ave_rel_nipple_vectors": lm_disp_rel_nipple,
        "ld_ave_rel_nipple_magnitudes": lm_disp_mag_rel_nipple,
        "ld_ave_rel_nipple_vectors_base_left": X_left,
        "ld_ave_rel_nipple_vectors_left": V_left,
        "ld_ave_rel_nipple_vectors_base_right": X_right,
        "ld_ave_rel_nipple_vectors_right": V_right,

        # Original anchor positions (for reference/conversion if needed)
        "anchor_prone": anchor_prone,  # Original prone sternum superior position
        "anchor_supine": anchor_supine,  # Original supine sternum superior position

        "method": "fixed_sternum_rotation_only_centered",
        "coordinate_system": "centered_on_sternum_superior"
    }

    return results


if __name__ == "__main__":
    print("Fixed Sternum Alignment Module")
    print("This module should be imported and called from main.py")
