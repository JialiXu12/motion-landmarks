"""
Sternum-Fixed Alignment — Orchestrator

This module is the main entry point for prone-to-supine ribcage alignment.
It orchestrates the full pipeline:
    1. Load data (mesh, point cloud, landmarks)
    2. Preprocess (center, initial rotation, region filtering)
    3. Align (plane-to-point ICP)
    4. Transform landmarks to supine frame
    5. Calculate errors and metrics
    6. Return results

All algorithmic functions live in dedicated modules:
    - alignment_preprocessing.py: centering, initial rotation, filtering
    - surface_to_point_alignment.py: plane-to-point ICP solver
    - alignment_utils.py: shared math/mesh utilities
    - alignment_viz.py: visualization
    - alignment_reporting.py: publication-ready reporting
"""

import numpy as np
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

from alignment_utils import (  # noqa: F401 — re-exported for callers
    filter_point_cloud_asymmetric,
    filter_point_cloud_to_match_mesh_region,
    filter_mutual_region,
    get_surface_mesh_coords,
    get_mesh_elements,
    get_mesh_elements_2,
    get_mesh_with_selected_elements,
    plot_mesh_elements,
    plot_filter_debug,
    apply_transform_to_coords,
    inverse_transform_to_source_frame,
    svd_rotation_point_to_point,
)
from alignment_viz import (  # noqa: F401 — re-exported for callers
    visualize_alignment_errors,
    visualize_alignment_during_iteration,
)
from alignment_reporting import (  # noqa: F401 — re-exported for callers
    print_alignment_accuracy_report,
    aggregate_alignment_statistics,
    print_cohort_alignment_report,
)
from alignment_preprocessing import (
    preprocess_for_alignment,
    compute_initial_alignment,
    select_mesh_elements,
)
from alignment_deprecated import (  # noqa: F401 — re-exported for callers
    cleanup_spine_region,
    optimal_sternum_fixed_alignment,
)
from readers import load_subject
from surface_to_point_alignment import surface_to_point_align


def compute_landmark_displacements(
        landmark_prone_raw: np.ndarray,
        landmark_supine_raw: np.ndarray,
        R: np.ndarray,
        source_anchor: np.ndarray,
        target_anchor: np.ndarray,
        sternum_prone_transformed: np.ndarray,
        sternum_supine: np.ndarray,
        nipple_prone_transformed: np.ndarray,
        nipple_supine: np.ndarray,
        nipple_disp_rel_sternum: np.ndarray,
) -> dict:
    """
    Compute landmark displacements relative to sternum and nipple.

    This is a pure function: given raw landmark positions and the alignment
    transform, it returns all displacement metrics without side effects.

    Args:
        landmark_prone_raw: (N, 3) raw prone landmarks for one registrar
        landmark_supine_raw: (N, 3) raw supine landmarks for one registrar
        R: (3, 3) rotation matrix from ICP alignment
        source_anchor: (3,) prone sternum superior (rotation center)
        target_anchor: (3,) supine sternum superior (translation target)
        sternum_prone_transformed: (2, 3) transformed sternum [superior, inferior]
        sternum_supine: (2, 3) supine sternum [superior, inferior]
        nipple_prone_transformed: (2, 3) transformed nipples [left, right]
        nipple_supine: (2, 3) supine nipples [left, right]
        nipple_disp_rel_sternum: (2, 3) nipple displacement vectors rel sternum

    Returns:
        Dictionary with displacement keys (no prefix — caller adds prefix).
    """
    # Transform prone landmarks to supine frame
    landmark_prone_transformed = apply_transform_to_coords(
        landmark_prone_raw, R, source_anchor, target_anchor
    )

    # Reference points
    ref_sternum_prone = sternum_prone_transformed[0]
    ref_sternum_supine = sternum_supine[0]

    # Positions relative to sternum
    lm_pos_prone_rel_sternum = landmark_prone_transformed - ref_sternum_prone
    lm_pos_supine_rel_sternum = landmark_supine_raw - ref_sternum_supine

    # Displacement relative to sternum
    lm_disp_rel_sternum = lm_pos_supine_rel_sternum - lm_pos_prone_rel_sternum
    lm_disp_mag_rel_sternum = np.linalg.norm(lm_disp_rel_sternum, axis=1)

    # Determine breast side for each landmark
    dist_to_left = np.linalg.norm(landmark_supine_raw - nipple_supine[0], axis=1)
    dist_to_right = np.linalg.norm(landmark_supine_raw - nipple_supine[1], axis=1)
    is_left_breast = dist_to_left < dist_to_right

    nipple_disp_left_vec = nipple_disp_rel_sternum[0]
    nipple_disp_right_vec = nipple_disp_rel_sternum[1]

    closest_nipple_disp_vec = np.where(
        is_left_breast[:, np.newaxis],
        nipple_disp_left_vec,
        nipple_disp_right_vec
    )

    # Displacement relative to nipple
    lm_disp_rel_nipple = lm_disp_rel_sternum - closest_nipple_disp_vec
    lm_disp_mag_rel_nipple = np.linalg.norm(lm_disp_rel_nipple, axis=1)

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

    return {
        'prone_transformed': landmark_prone_transformed,
        'supine': landmark_supine_raw,
        'displacement_magnitudes': lm_disp_mag_rel_sternum,
        'displacement_vectors': lm_disp_rel_sternum,
        'prone_rel_sternum': lm_pos_prone_rel_sternum,
        'supine_rel_sternum': lm_pos_supine_rel_sternum,
        'rel_nipple_vectors': lm_disp_rel_nipple,
        'rel_nipple_magnitudes': lm_disp_mag_rel_nipple,
        'rel_nipple_vectors_base_left': X_left,
        'rel_nipple_vectors_left': V_left,
        'rel_nipple_vectors_base_right': X_right,
        'rel_nipple_vectors_right': V_right,
    }


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
        pc_inferior_trim: float = 0.0,
        mutual_region_padding_reciprocal: float = 0.0,
        use_initial_rotation: bool = False,
        verbose: bool = True
) -> dict:
    """
    Main alignment function with sternum superior fixed at origin.

    This function orchestrates the full prone-to-supine alignment pipeline:
    preprocessing -> ICP alignment -> transform -> errors -> results.

    Args:
        subject: Subject object containing scan data and landmarks
        prone_ribcage_mesh_path: Path to prone ribcage .mesh file
        supine_ribcage_seg_path: Path to supine ribcage segmentation .nii.gz
        orientation_flag: Image orientation (default: 'RAI')
        plot_for_debug: Whether to show debug plots
        max_correspondence_distance: Max distance for ICP correspondences (default: 15mm)
        max_iterations: Max ICP iterations (default: 500)
        trim_percentage: Fraction of worst correspondences to trim (default: 0.1)
        selected_elements: List of element indices for alignment (e.g. [0,1,2,5]).
                          If None, all elements are used.
        visualize_iterations: If True, show visualization during ICP iterations
        visualize_every_n: Show visualization every N iterations (default: 10)
        pc_inferior_trim: Trim inferior points from supine point cloud (mm).
                         For per-subject cleanup (e.g. VL54 = 15.0). Default 0.
        mutual_region_padding_reciprocal: Anterior padding (mm) for reciprocal
                         source filtering in Step 2. Default 0 (tight cut).
        use_initial_rotation: If True, compute Rodrigues rotation from sternum
                         SI axes as ICP starting point. Default False (identity).
        verbose: Print progress information

    Returns:
        Dictionary containing transformation, error metrics, landmarks, and
        displacement vectors (see end of function for full key listing).
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

    # Get scan data (registrar landmarks loaded per-registrar in section 6)
    prone_scan_data = subject.scans["prone"]
    supine_scan_data = subject.scans["supine"]

    # Load prone ribcage mesh
    prone_ribcage = morphic.Mesh(str(prone_ribcage_mesh_path))

    # Load supine ribcage segmentation
    supine_ribcage_mask = breast_metadata.readNIFTIImage(
        str(supine_ribcage_seg_path), orientation_flag, swap_axes=True
    )
    supine_ribcage_pc = extract_contour_points(supine_ribcage_mask, 20000)
    plot_all(point_cloud=supine_ribcage_pc)

    # Get element centers and select subset (from alignment_preprocessing)
    elem_info = select_mesh_elements(
        mesh=prone_ribcage,
        selected_elements=selected_elements,
        vis_res=26,
        verbose=verbose,
        plot_for_debug=plot_for_debug,
    )
    prone_ribcage_mesh_coords = elem_info['mesh_coords']
    prone_ribcage_alignment_coords = elem_info['alignment_coords']
    centers_array = elem_info['centers_array']
    num_elements = elem_info['num_elements']

    # ==========================================================
    # 2. PREPROCESS + 3. ALIGN (multi-start ICP)
    # ==========================================================
    # Common preprocessing args
    _prep_kwargs = dict(
        mesh=prone_ribcage,
        target_pts=supine_ribcage_pc,
        source_sternum_sup=sternum_prone[0],
        target_sternum_sup=sternum_supine[0],
        source_sternum_inf=sternum_prone[1],
        target_sternum_inf=sternum_supine[1],
        selected_elements=selected_elements,
        res=10,
        mutual_region_padding=15.0,
        mutual_region_padding_inferior=0.0,
        mutual_region_padding_reciprocal=mutual_region_padding_reciprocal,
        pc_inferior_trim=pc_inferior_trim,
    )

    # Common ICP args
    _icp_kwargs = dict(
        vl_id=subject.subject_id,
        mesh=prone_ribcage,
        source_sternum_sup=sternum_prone[0],
        target_sternum_sup=sternum_supine[0],
        max_distance=max_correspondence_distance,
        max_iterations=max_iterations,
        convergence_threshold=1e-6,
        trim_percentage=trim_percentage,
        res=10,
        elems=selected_elements,
        point_to_point_weight=0.0,
        target_region_filter=False,
    )

    # Build list of starts: always include identity, optionally add Rodrigues
    starts = [False]
    if use_initial_rotation:
        starts.append(True)

    best_R, best_T, best_info = None, None, None
    best_rmse = np.inf
    best_label = ""

    for use_rot in starts:
        label = "Rodrigues" if use_rot else "identity"
        if verbose:
            print(f"\n  --- ICP start: {label} ---")

        prep = preprocess_for_alignment(
            **_prep_kwargs,
            use_initial_rotation=use_rot,
            verbose=verbose,
            debug_filter_plot=plot_for_debug,
        )

        R, T_total, info = surface_to_point_align(
            **_icp_kwargs,
            target_pts=prep['target_pts_filtered'],
            src_mask=prep['src_mask'],
            R_init=prep['R_init'],
            verbose=verbose,
        )

        rmse = info['euclidean_rmse']
        if verbose:
            print(f"  {label} result: RMSE = {rmse:.4f} mm, "
                  f"iterations = {info['iterations']}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_R, best_T, best_info = R, T_total, info
            best_label = label

        # Short-circuit: identity converged well, skip Rodrigues run
        if not use_rot and rmse < 4.0 and len(starts) > 1:
            if verbose:
                print(f"  Identity RMSE ({rmse:.4f} mm) < 4.0 mm — skipping Rodrigues start")
            break

    if verbose and len(starts) > 1:
        print(f"\n  Multi-start winner: {best_label} (RMSE = {best_rmse:.4f} mm)")

    R, T_total, info = best_R, best_T, best_info

    # Filtered supine PC used for alignment
    supine_ribcage_pc_alignment = info['target_pts_filtered']

    # ==========================================================
    # 4. TRANSFORM ALL PRONE DATA TO SUPINE FRAME
    # ==========================================================
    source_anchor = sternum_prone[0]
    target_anchor = sternum_supine[0]

    sternum_prone_transformed = apply_transform_to_coords(
        sternum_prone, R, source_anchor, target_anchor
    )
    nipple_prone_transformed = apply_transform_to_coords(
        nipple_prone, R, source_anchor, target_anchor
    )
    prone_ribcage_transformed = apply_transform_to_coords(
        prone_ribcage_mesh_coords, R, source_anchor, target_anchor
    )
    centers_array_transformed = apply_transform_to_coords(
        centers_array, R, source_anchor, target_anchor
    )

    if selected_elements is not None:
        prone_selected_transformed = apply_transform_to_coords(
            prone_ribcage_alignment_coords, R, source_anchor, target_anchor
        )
    else:
        prone_selected_transformed = prone_ribcage_transformed

    # Visualize alignment results
    if selected_elements is not None:
        plot_mesh_elements(prone_selected_transformed, centers_array_transformed[selected_elements],
                          selected_elements, supine_ribcage_pc_alignment)

    plot_mesh_elements(prone_ribcage_transformed, centers_array_transformed,
                      range(num_elements), supine_ribcage_pc)

    # ==========================================================
    # 5. CALCULATE ERRORS AND METRICS
    # ==========================================================
    sternum_error = np.linalg.norm(sternum_prone_transformed[0] - sternum_supine[0])

    error_full, mapped_idx_full = breast_metadata.closest_distances(
        prone_ribcage_transformed, supine_ribcage_pc
    )
    rib_error_mag_full = np.linalg.norm(error_full, axis=1)

    ribcage_error_mean = float(np.mean(rib_error_mag_full))
    ribcage_error_std = float(np.std(rib_error_mag_full))
    ribcage_error_rmse = float(np.sqrt(np.mean(rib_error_mag_full ** 2)))

    error_selected, mapped_idx_selected = breast_metadata.closest_distances(
        prone_selected_transformed, supine_ribcage_pc_alignment
    )
    rib_error_mag_selected = np.linalg.norm(error_selected, axis=1)

    selected_error_mean = float(np.mean(rib_error_mag_selected))
    selected_error_std = float(np.std(rib_error_mag_selected))
    selected_error_rmse = float(np.sqrt(np.mean(rib_error_mag_selected ** 2)))

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
        print(f"  Note: Trim percentage (fixed at {trim_percentage*100:.0f}%) was applied during optimization")
        print(f"  Iterations: {info['iterations']}")

        print(f"\n{'='*60}")
        print(f"PUBLICATION SUMMARY:")
        print(f"{'='*60}")
        print_alignment_accuracy_report(rib_error_mag_full, sternum_error, info)

    # ==========================================================
    # 6. CALCULATE NIPPLE DISPLACEMENTS (shared across registrars)
    # ==========================================================
    ref_sternum_prone = sternum_prone_transformed[0]
    ref_sternum_supine = sternum_supine[0]

    nipple_pos_prone_rel_sternum = nipple_prone_transformed - ref_sternum_prone
    nipple_pos_supine_rel_sternum = nipple_supine - ref_sternum_supine
    nipple_disp_rel_sternum = nipple_pos_supine_rel_sternum - nipple_pos_prone_rel_sternum
    nipple_disp_mag_rel_sternum = np.linalg.norm(nipple_disp_rel_sternum, axis=1)

    nipple_disp_left_vec = nipple_disp_rel_sternum[0]
    nipple_disp_right_vec = nipple_disp_rel_sternum[1]

    # ==========================================================
    # 7. COMPUTE DISPLACEMENTS FOR EACH REGISTRAR
    # ==========================================================
    # Registrar name -> results dict prefix mapping
    registrar_prefixes = {
        'anthony': 'ld_anthony',
        'holly': 'ld_holly',
        'average': 'ld_ave',
    }

    registrar_displacement_results = {}
    for reg_name, prefix in registrar_prefixes.items():
        lm_prone_raw = get_landmarks_as_array(prone_scan_data, reg_name)
        lm_supine_raw = get_landmarks_as_array(supine_scan_data, reg_name)

        if lm_prone_raw.shape[0] == 0 or lm_supine_raw.shape[0] == 0:
            if verbose:
                print(f"  Skipping registrar '{reg_name}': no landmarks found")
            continue

        disp = compute_landmark_displacements(
            landmark_prone_raw=lm_prone_raw,
            landmark_supine_raw=lm_supine_raw,
            R=R,
            source_anchor=source_anchor,
            target_anchor=target_anchor,
            sternum_prone_transformed=sternum_prone_transformed,
            sternum_supine=sternum_supine,
            nipple_prone_transformed=nipple_prone_transformed,
            nipple_supine=nipple_supine,
            nipple_disp_rel_sternum=nipple_disp_rel_sternum,
        )

        # Store with prefix
        for key, value in disp.items():
            registrar_displacement_results[f"{prefix}_{key}"] = value

    # ==========================================================
    # 8. CREATE TRANSFORMATION MATRIX
    # ==========================================================
    T_total = np.eye(4)
    T_total[:3, :3] = R
    T_total[:3, 3] = target_anchor - R @ source_anchor

    # ==========================================================
    # 9. VISUALIZATION (if requested)
    # ==========================================================
    if plot_for_debug:
        try:
            sternum_lists = [sternum_prone_transformed, sternum_supine]
            plot_all(
                point_cloud=supine_ribcage_pc,
                mesh_points=prone_ribcage_transformed,
                anat_landmarks=sternum_lists,
            )
        except Exception as e:
            print(f"Could not generate debug plots: {e}")

    # ==========================================================
    # 10. VISUALIZATION (with MRI — disabled by default)
    # ==========================================================
    visualise_mri = False
    if visualise_mri:
        prone_scan = subject.scans["prone"].scan_object
        supine_scan = subject.scans["supine"].scan_object

        orientation_flag = 'RAI'

        prone_image_grid = breast_metadata.SCANToPyvistaImageGrid(prone_scan, orientation_flag)
        supine_image_grid = breast_metadata.SCANToPyvistaImageGrid(supine_scan, orientation_flag)

        prone_image_sitk = breast_metadata.PyvistaImageGridToSITKImage(prone_image_grid)
        prone_image_sitk = sitk.Cast(prone_image_sitk, sitk.sitkUInt8)
        supine_image_sitk = breast_metadata.PyvistaImageGridToSITKImage(supine_image_grid)
        supine_image_sitk = sitk.Cast(supine_image_sitk, sitk.sitkUInt8)

        T_prone_to_supine = np.linalg.inv(T_total)
        affine = sitk.AffineTransform(3)
        affine.SetTranslation(T_prone_to_supine[:3, 3])
        affine.SetMatrix(T_prone_to_supine[:3, :3].ravel())

        prone_image_transformed = sitk.Resample(prone_image_sitk, supine_image_sitk, affine, sitk.sitkLinear, 1.0)
        prone_image_transformed = sitk.Cast(prone_image_transformed, sitk.sitkUInt8)

        prone_scan_transformed = breast_metadata.SITKToScan(prone_image_transformed, orientation_flag, load_dicom=False,
                                                            swap_axes=True)
        prone_image_transformed_grid = breast_metadata.SCANToPyvistaImageGrid(prone_scan_transformed, orientation_flag)

        def get_px_coords(scan_obj, points):
            return np.array([scan_obj.getPixelCoordinates(p) for p in points])

        sternum_prone_px = get_px_coords(prone_scan_transformed, sternum_prone_transformed)
        sternum_supine_px = get_px_coords(supine_scan, sternum_supine)

        lm_ave_prone_transformed = registrar_displacement_results.get('ld_ave_prone_transformed', np.empty((0, 3)))
        lm_ave_supine = registrar_displacement_results.get('ld_ave_supine', np.empty((0, 3)))
        lm_prone_trans_px = get_px_coords(prone_scan_transformed, lm_ave_prone_transformed)
        lm_supine_px = get_px_coords(supine_scan, lm_ave_supine)

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

        plotter = pv.Plotter()
        plotter.add_text("Landmark Displacement After Alignment", font_size=24)

        plotter.add_points(lm_ave_supine, render_points_as_spheres=True, color='green', point_size=10,
                           label='Supine Landmarks')
        plotter.add_points(lm_ave_prone_transformed, render_points_as_spheres=True, color='red', point_size=10,
                           label='Aligned Prone Landmarks')

        global_disp_vectors = lm_ave_supine - lm_ave_prone_transformed
        for start_point, vector in zip(lm_ave_prone_transformed, global_disp_vectors):
            plotter.add_arrows(start_point, vector, mag=1.0, color='yellow')

        opacity = np.linspace(0, 0.1, 100)
        plotter.add_volume(prone_image_transformed_grid, opacity=opacity, cmap='grey', show_scalar_bar=False)
        plotter.add_volume(supine_image_grid, opacity=opacity, cmap='coolwarm', show_scalar_bar=False)
        plotter.add_points(supine_ribcage_pc, color="tan", label='Point cloud', point_size=2,
                           render_points_as_spheres=True)
        plotter.add_points(sternum_prone_transformed, render_points_as_spheres=True, color='black', point_size=10,
                           label='Aligned Prone Sternum')
        plotter.add_points(sternum_supine, render_points_as_spheres=True, color='blue', point_size=10,
                           label='Supine Sternum')
        plotter.add_points(nipple_prone_transformed, render_points_as_spheres=True, color='pink', point_size=8,
                           label='Aligned Prone Nipples')
        plotter.add_points(nipple_supine, render_points_as_spheres=True, color='pink', point_size=8,
                           label='Supine Nipples')

        plotter.add_points(np.array([[0, 0, 0]]),
                           render_points_as_spheres=True,
                           color='orange',
                           point_size=15,
                           label='Origin (0,0,0)')

        plotter.add_legend()
        plotter.show()

    # ==========================================================
    # 11. RETURN RESULTS DICTIONARY
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

        # Transformed anatomical landmarks
        'sternum_prone_transformed': sternum_prone_transformed,
        'sternum_supine': sternum_supine,
        'nipple_prone_transformed': nipple_prone_transformed,
        'nipple_supine': nipple_supine,

        # Nipple positions relative to sternum
        'nipple_prone_rel_sternum': nipple_pos_prone_rel_sternum,
        'nipple_supine_rel_sternum': nipple_pos_supine_rel_sternum,

        # Nipple displacements
        'nipple_displacement_magnitudes': nipple_disp_mag_rel_sternum,
        'nipple_displacement_vectors': nipple_disp_rel_sternum,
        'nipple_disp_left_vec': nipple_disp_left_vec,
        'nipple_disp_right_vec': nipple_disp_right_vec,

        # Algorithm info
        'alignment_info': info,
        'method': 'optimal_sternum_fixed_svd'
    }

    # Merge per-registrar displacement results (ld_anthony_*, ld_holly_*, ld_ave_*)
    results.update(registrar_displacement_results)

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
            dicom_root=None,
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
            selected_elements=[0, 1, 6, 7, 8, 9, 14, 15, 16, 17, 22, 23],
            orientation_flag='RAI',
            plot_for_debug=False,
            use_initial_rotation=True,
            mutual_region_padding_reciprocal=15,
            max_correspondence_distance=1e6,
            trim_percentage=0,
            visualize_iterations=True,
            visualize_every_n=50)
