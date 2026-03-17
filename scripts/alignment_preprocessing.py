"""
Alignment Preprocessing

Prepares prone ribcage mesh and supine ribcage point cloud for ICP alignment:
    - Centering: translate both datasets so sternum superior is at origin
    - Initial alignment: rotate prone SI axis onto supine SI axis (Rodrigues)
    - Mesh element selection: choose anterior elements for alignment
    - Mutual region filtering: clip mesh and point cloud to overlapping region
    - Per-subject point cloud cleanup (e.g. inferior trim)
"""

import numpy as np
from alignment_utils import (
    filter_point_cloud_asymmetric,
    filter_point_cloud_to_match_mesh_region,
    get_surface_mesh_coords,
    get_mesh_elements,
    get_mesh_elements_2,
    get_mesh_with_selected_elements,
    plot_mesh_elements,
    plot_filter_debug,
    filter_mutual_region,
)

# Re-exports for backward compatibility (moved to alignment_deprecated.py)
from alignment_deprecated import (
    filter_anterior_by_widest_point,
    plot_anterior_filter,
)


# ---------------------------------------------------------------------------
# 1. Initial landmark-based alignment (Rodrigues' rotation)
# ---------------------------------------------------------------------------
def compute_initial_alignment(
        prone_si_vector: np.ndarray,
        supine_si_vector: np.ndarray,
        verbose: bool = True,
) -> np.ndarray:
    """
    Compute an initial rotation that aligns the prone sternum axis to supine.

    Uses only the sternum superior->inferior vector (the one reliable rigid
    landmark pair). Computes the minimal rotation that maps the prone SI
    direction onto the supine SI direction. The remaining rotation around
    the SI axis is left for ICP to resolve.

    The rotation axis is the cross product of the two unit vectors, and
    the angle is derived from their dot product.

    Args:
        prone_si_vector:  (3,) sternum_inf - sternum_sup in prone (raw coords)
        supine_si_vector: (3,) sternum_inf - sternum_sup in supine (raw coords)
        verbose: print alignment details

    Returns:
        R_init: (3, 3) rotation matrix (prone -> supine)
    """
    a = np.asarray(prone_si_vector, dtype=np.float64)
    b = np.asarray(supine_si_vector, dtype=np.float64)

    # Normalise
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    # Rotation axis and angle
    v = np.cross(a, b)
    s = np.linalg.norm(v)       # sin(angle)
    c = np.dot(a, b)            # cos(angle)

    if s < 1e-10:
        # Vectors are (anti-)parallel
        if c > 0:
            R_init = np.eye(3)
        else:
            # 180 degree rotation -- pick any perpendicular axis
            perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(a, perp)
            axis = axis / np.linalg.norm(axis)
            R_init = 2.0 * np.outer(axis, axis) - np.eye(3)
    else:
        # Rodrigues' rotation formula via skew-symmetric matrix
        vx = np.array([
            [0,    -v[2],  v[1]],
            [v[2],  0,    -v[0]],
            [-v[1], v[0],  0   ],
        ])
        R_init = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))

    if verbose:
        angle_deg = np.degrees(np.arccos(np.clip(c, -1, 1)))
        # Verify: rotated prone SI should match supine SI
        a_rotated = R_init @ a
        residual = np.degrees(np.arccos(np.clip(np.dot(a_rotated, b), -1, 1)))
        print(f"  Initial alignment (sternum axis only):")
        print(f"    Prone  SI direction: [{a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f}]")
        print(f"    Supine SI direction: [{b[0]:+.3f}, {b[1]:+.3f}, {b[2]:+.3f}]")
        print(f"    Rotation angle: {angle_deg:.1f} deg")
        print(f"    Residual angle after rotation: {residual:.4f} deg")

    return R_init


# ---------------------------------------------------------------------------
# 2. Mesh element selection and visualization
# ---------------------------------------------------------------------------
def select_mesh_elements(
        mesh,
        selected_elements: list = None,
        vis_res: int = 26,
        verbose: bool = True,
        plot_for_debug: bool = False,
) -> dict:
    """
    Get element centers and optionally select a subset of mesh elements.

    Args:
        mesh: morphic.Mesh object (prone ribcage)
        selected_elements: list of element indices, None = all elements
        vis_res: mesh sampling resolution for visualization (default 26)
        verbose: print element selection info
        plot_for_debug: show debug plots of element selection

    Returns:
        dict with:
            'mesh_coords': (N,3) full mesh surface points at vis_res
            'alignment_coords': (M,3) selected element points at vis_res
            'centers_array': (E,3) element center positions
            'num_elements': total number of elements in mesh
    """
    mesh_coords = get_surface_mesh_coords(mesh, res=vis_res)
    centers_array, num_elements = get_mesh_elements_2(mesh)

    if plot_for_debug:
        plot_mesh_elements(mesh_coords, centers_array, range(num_elements))

    if selected_elements is not None:
        alignment_coords = get_mesh_with_selected_elements(
            mesh, selected_elements, res=vis_res
        )
        if verbose:
            print(f"  Using {len(selected_elements)}/{num_elements} elements for alignment")
            print(f"  Selected elements: {selected_elements}")
            print(f"  Alignment points: {alignment_coords.shape[0]} "
                  f"(full mesh: {mesh_coords.shape[0]})")
        if plot_for_debug:
            selected_centers = centers_array[selected_elements]
            plot_mesh_elements(alignment_coords, selected_centers,
                               selected_elements)
    else:
        alignment_coords = mesh_coords
        if verbose:
            print(f"  Using all {num_elements} elements for alignment")

    return {
        'mesh_coords': mesh_coords,
        'alignment_coords': alignment_coords,
        'centers_array': centers_array,
        'num_elements': num_elements,
    }


# ---------------------------------------------------------------------------
# 3. Main preprocessing pipeline
# ---------------------------------------------------------------------------
def preprocess_for_alignment(
        mesh,
        target_pts: np.ndarray,
        source_sternum_sup: np.ndarray,
        target_sternum_sup: np.ndarray,
        source_sternum_inf: np.ndarray,
        target_sternum_inf: np.ndarray,
        selected_elements: list = None,
        res: int = 10,
        mutual_region_padding: float = 15.0,
        mutual_region_padding_inferior: float = 0.0,
        mutual_region_padding_reciprocal: float = 0.0,
        skip_reciprocal: bool = False,
        pc_inferior_trim: float = 0.0,
        pc_superior_trim: float = 0.0,
        use_initial_rotation: bool = False,
        verbose: bool = True,
        debug_filter_plot: bool = False,
        debug_filter_save_path: str = None,
) -> dict:
    """
    Full preprocessing pipeline for prone-to-supine alignment.

    Steps:
        1. Per-subject point cloud cleanup (inferior trim)
        2. Center both datasets on sternum superior
        3. Optionally compute initial rotation (Rodrigues, sternum SI axis)
        4. Sample source mesh points (with element selection)
        5. Mutual region filtering (clip both to overlapping bbox)

    Args:
        mesh: morphic.Mesh object (prone ribcage) or (N,3) point cloud
        target_pts: (M, 3) supine ribcage point cloud (raw coordinates)
        source_sternum_sup: (3,) prone sternum superior position
        target_sternum_sup: (3,) supine sternum superior position
        source_sternum_inf: (3,) prone sternum inferior position
        target_sternum_inf: (3,) supine sternum inferior position
        selected_elements: list of element indices, None = all elements
        res: mesh sampling resolution per element axis (for ICP, default 10)
        mutual_region_padding: padding (mm) around source bbox for target filter
        mutual_region_padding_inferior: padding (mm) for inferior Z boundary
        mutual_region_padding_reciprocal: padding (mm) for reciprocal filter
        skip_reciprocal: skip reciprocal source filtering step
        pc_inferior_trim: trim inferior points from target PC by this many mm
                          (for per-subject cleanup, e.g. VL54 needs 15.0)
        pc_superior_trim: trim superior points from target PC by this many mm
                          (for per-subject cleanup when superior segmentation
                          extends beyond the prone mesh coverage)
        use_initial_rotation: if True, compute Rodrigues rotation from sternum
                          SI axes as ICP starting point. Default False (identity).
        verbose: print progress
        debug_filter_plot: show interactive filter debug plot
        debug_filter_save_path: path to save filter debug screenshot

    Returns:
        dict with:
            'R_init': (3,3) initial rotation matrix (identity if use_initial_rotation=False)
            'src_mask': (N,) bool mask for source points (None if no filtering)
            'target_pts_filtered': (K,3) filtered target points (raw coords)
            'filter_info': dict with bbox details (None if no filtering)
    """
    src_ss = np.asarray(source_sternum_sup, dtype=np.float64).flatten()
    tgt_ss = np.asarray(target_sternum_sup, dtype=np.float64).flatten()
    src_si = np.asarray(source_sternum_inf, dtype=np.float64).flatten()
    tgt_si = np.asarray(target_sternum_inf, dtype=np.float64).flatten()
    target_pts = np.asarray(target_pts, dtype=np.float64)

    # ── 1. Per-subject point cloud cleanup ──
    if pc_inferior_trim > 0:
        z_min = target_pts[:, 2].min()
        n_before = target_pts.shape[0]
        target_pts = target_pts[target_pts[:, 2] > z_min + pc_inferior_trim]
        if verbose:
            print(f"  PC inferior trim: {pc_inferior_trim:.1f}mm -> "
                  f"{n_before} -> {target_pts.shape[0]} points")

    if pc_superior_trim > 0:
        z_max = target_pts[:, 2].max()
        n_before = target_pts.shape[0]
        target_pts = target_pts[target_pts[:, 2] < z_max - pc_superior_trim]
        if verbose:
            print(f"  PC superior trim: {pc_superior_trim:.1f}mm -> "
                  f"{n_before} -> {target_pts.shape[0]} points")

    # ── 2. Compute initial rotation (optional Rodrigues SI axis alignment) ──
    if use_initial_rotation:
        prone_si = src_si - src_ss
        supine_si = tgt_si - tgt_ss
        R_init = compute_initial_alignment(prone_si, supine_si, verbose=verbose)
    else:
        R_init = np.eye(3)
        if verbose:
            print(f"  Initial rotation: identity (Rodrigues disabled)")

    # ── 3. Sample source mesh points for region filtering ──
    use_point_cloud = isinstance(mesh, np.ndarray)
    if use_point_cloud:
        src_pts_initial = np.asarray(mesh, dtype=np.float64)
    else:
        # Import here to avoid circular dependency at module level
        from surface_to_point_alignment import compute_mesh_points_and_normals
        Xi = mesh.grid(res, method='center')
        src_pts_initial, _ = compute_mesh_points_and_normals(
            mesh, Xi, elems=selected_elements
        )

    # ── 4. Mutual region filtering ──
    # Apply R_init to source points for correct bounding box computation:
    # Center on sternum, apply initial rotation, then filter
    src_centered_rotated = (R_init @ (src_pts_initial - src_ss).T).T
    tgt_centered = target_pts - tgt_ss

    # filter_mutual_region expects world coords + sternum offsets.
    # We pass already-centered data with zero offsets.
    origin = np.zeros(3)

    src_mask = None
    filter_info = None

    if selected_elements is not None or use_point_cloud:
        # Shared anterior Y boundary: min of mesh and PC Y-min (sternum-centered).
        # Both bboxes use shared_y_min - padding_reciprocal as Y-min
        # so neither anterior extent is clipped.
        shared_y_min = min(src_centered_rotated[:, 1].min(),
                           tgt_centered[:, 1].min())
        if verbose:
            print(f"  Shared anterior Y-min: {shared_y_min:.1f} "
                  f"(mesh={src_centered_rotated[:, 1].min():.1f}, "
                  f"PC={tgt_centered[:, 1].min():.1f}, "
                  f"padding_reciprocal={mutual_region_padding_reciprocal:.1f}, "
                  f"effective={shared_y_min - mutual_region_padding_reciprocal:.1f})")

        src_mask, tgt_filtered_centered, filter_info = filter_mutual_region(
            src_pts=src_centered_rotated,
            tgt_pts=tgt_centered,
            src_ss=origin,
            tgt_ss=origin,
            padding=mutual_region_padding,
            padding_inferior=mutual_region_padding_inferior,
            padding_reciprocal=mutual_region_padding_reciprocal,
            anterior_y_min=shared_y_min,
            verbose=verbose,
            skip_reciprocal=skip_reciprocal,
        )
        # Convert filtered target back to world coords
        target_pts_filtered = tgt_filtered_centered + tgt_ss

        if debug_filter_plot and src_mask is not None:
            plot_filter_debug(
                src_pts=src_centered_rotated,
                tgt_pts=tgt_centered,
                src_mask=src_mask,
                tgt_filtered=tgt_filtered_centered,
                src_ss=origin,
                tgt_ss=origin,
                filter_info=filter_info,
                title="Mutual Region Filter Debug",
                save_path=debug_filter_save_path,
            )
    else:
        target_pts_filtered = target_pts

    return {
        'R_init': R_init,
        'src_mask': src_mask,
        'target_pts_filtered': target_pts_filtered,
        'filter_info': filter_info,
    }


# ---------------------------------------------------------------------------
# __main__: standalone preprocessing demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    import morphic
    import external.breast_metadata_mdv.breast_metadata as breast_metadata
    from utils import extract_contour_points
    from utils_plot import plot_all
    from structures import Subject
    from readers import read_anatomical_landmarks, load_subject

    # ==================================================================
    # Configuration
    # ==================================================================
    vl_ids = [54]
    ROOT_PATH_MRI = Path(r'U:\projects\volunteer_camri\old_data\mri_t2')
    SOFT_TISSUE_ROOT = Path(r'U:\projects\dashboard\picker_points')
    ANATOMICAL_JSON_BASE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results")
    PRONE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh")
    SUPINE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage")

    orientation_flag = 'RAI'

    SELECTED_ELEMENTS = [0, 1, 6, 7, 8, 9, 14, 15, 16, 17, 22, 23]

    for vl_id in vl_ids:
        vl_id_str = f"VL{vl_id:05d}"
        print(f"\n{'='*60}")
        print(f"ALIGNMENT PREPROCESSING DEMO")
        print(f"Subject: {vl_id_str}")
        print(f"{'='*60}")

        subject = load_subject(
            vl_id=vl_id,
            positions=["prone", "supine"],
            dicom_root=None,
            anatomical_json_base_root=ANATOMICAL_JSON_BASE_ROOT,
            soft_tissue_root=SOFT_TISSUE_ROOT,
        )

        anat_prone = subject.scans["prone"].anatomical_landmarks
        anat_supine = subject.scans["supine"].anatomical_landmarks

        # Load mesh and point cloud
        prone_mesh_file = PRONE_RIBCAGE_ROOT / f"{vl_id_str}_ribcage_prone.mesh"
        supine_seg_file = SUPINE_RIBCAGE_ROOT / f"rib_cage_{vl_id_str}.nii.gz"

        prone_ribcage = morphic.Mesh(str(prone_mesh_file))
        supine_ribcage_mask = breast_metadata.readNIFTIImage(
            str(supine_seg_file), orientation_flag, swap_axes=True
        )
        supine_ribcage_pc = extract_contour_points(supine_ribcage_mask, 20000)

        # Run preprocessing
        prep = preprocess_for_alignment(
            mesh=prone_ribcage,
            target_pts=supine_ribcage_pc,
            source_sternum_sup=anat_prone.sternum_superior,
            target_sternum_sup=anat_supine.sternum_superior,
            source_sternum_inf=anat_prone.sternum_inferior,
            target_sternum_inf=anat_supine.sternum_inferior,
            selected_elements=SELECTED_ELEMENTS,
            mutual_region_padding=15.0,
            pc_inferior_trim=15.0 if vl_id == 54 else 0.0,
            verbose=True,
            debug_filter_plot=True,
        )

        print(f"\n  Preprocessing results:")
        print(f"    R_init:\n{prep['R_init']}")
        print(f"    src_mask: {np.sum(prep['src_mask'])} / {len(prep['src_mask'])} kept"
              if prep['src_mask'] is not None else "    src_mask: None (no filtering)")
        print(f"    target_pts_filtered: {prep['target_pts_filtered'].shape[0]} points")
