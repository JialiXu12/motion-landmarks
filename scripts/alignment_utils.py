"""
Alignment Utilities

Shared functions used by alignment.py, surface_to_point_alignment.py,
and alignment_preprocessing.py:
    - Point cloud filtering
    - Mesh element extraction, selection, and visualization
    - Filter debug visualization
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# 1. Point cloud filtering
# ---------------------------------------------------------------------------
def filter_point_cloud_asymmetric(
        points: np.ndarray,
        reference: np.ndarray,
        tol_min: float,
        tol_max: float,
        axis: int,
) -> np.ndarray:
    """
    Filter points along an axis using different tolerances for min and max bounds.

    Args:
        points: (N, 3) point cloud to filter
        reference: (M, 3) reference point cloud defining the axis range
        tol_min: tolerance added to the minimum bound (mm)
        tol_max: tolerance subtracted from the maximum bound (mm)
        axis: axis index (0=X, 1=Y, 2=Z)

    Returns:
        Filtered (K, 3) point cloud (K <= N)
    """
    min_ref = np.min(reference[:, axis])
    max_ref = np.max(reference[:, axis])

    keep_idx = [idx for idx, pt in enumerate(points)
                if min_ref + tol_min < pt[axis] < max_ref - tol_max]

    return points[keep_idx]


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
                           (default 30 mm)
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


# ---------------------------------------------------------------------------
# 2. Mesh element extraction and selection
# ---------------------------------------------------------------------------
def get_surface_mesh_coords(morphic_mesh, res, elems=None):
    """
    Extract 3D coordinates from a surface mesh.

    Args:
        morphic_mesh: morphic.Mesh object
        res: number of material points per element axis
        elems: list of element indices to extract (None or empty for all)

    Returns:
        (N, 3) mesh coordinates
    """
    if elems is None:
        elems = []

    Xi = morphic_mesh.grid(res, method='center')
    NPPE = Xi.shape[0]

    if len(elems) == 0:
        NE = morphic_mesh.elements.size()
        mesh_coords = np.zeros((NE * NPPE, 3))
        for i, element in enumerate(morphic_mesh.elements):
            eid = element.id
            mesh_coords[i * NPPE:(i + 1) * NPPE, :] = morphic_mesh.elements[eid].evaluate(Xi)
    else:
        NE = len(elems)
        mesh_coords = np.zeros((NE * NPPE, 3))
        for i, element in enumerate(elems):
            mesh_coords[i * NPPE:(i + 1) * NPPE, :] = morphic_mesh.elements[element].evaluate(Xi)

    return mesh_coords


def get_mesh_elements(mesh):
    """
    Extract element centers from a morphic mesh.

    Args:
        mesh: morphic.Mesh object

    Returns:
        centers_array: (N, 3) element center coordinates
        num_elements: total number of elements
    """
    first_element = list(mesh.elements)[0]
    mesh_dims = len(first_element.basis)
    print(f"INFO: Mesh has {mesh_dims}D elements with basis {first_element.basis}")

    centers = []
    num_elements = mesh.elements.size()
    print(f"INFO: Mesh has {num_elements} elements")

    for i in range(num_elements):
        elem_coords = get_surface_mesh_coords(mesh, 3, elems=[i])
        center_idx = elem_coords.shape[0] // 2
        center = elem_coords[center_idx, :]
        centers.append(center)

    centers_array = np.array(centers)

    if centers_array.ndim == 1:
        centers_array = centers_array.reshape(1, 3)
    elif centers_array.ndim != 2 or centers_array.shape[1] != 3:
        raise ValueError(f"centers_array has unexpected shape {centers_array.shape}. Expected (N, 3).")

    return centers_array, num_elements


def get_mesh_elements_2(mesh):
    """
    Extract element centers from a morphic mesh (faster, uses grid directly).

    Args:
        mesh: morphic.Mesh object

    Returns:
        centers_array: (N, 3) element center coordinates
        num_elements: total number of elements
    """
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


def get_mesh_with_selected_elements(
        morphic_mesh,
        selected_elements: list,
        res: int = 26,
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


# ---------------------------------------------------------------------------
# 3. Visualization
# ---------------------------------------------------------------------------
def plot_mesh_elements(mesh_points, centers_array, element_indices, ribcage_point_cloud=None):
    """
    Visualize element centers with labels,
    optionally including the mesh surface and ribcage point cloud.

    Args:
        mesh_points: (N, 3) array of mesh point coordinates to display
        centers_array: (M, 3) array of element center coordinates for labeling
        element_indices: iterable of element index labels
        ribcage_point_cloud: (N, 3) array of ribcage point cloud coordinates (optional)

    Returns:
        centers_array: (N, 3) array of element center coordinates
    """
    import pyvista as pv

    labels = [str(i) for i in element_indices]

    plt = pv.Plotter()

    plt.add_points(
        mesh_points,
        color='gray',
        render_points_as_spheres=True,
        point_size=2,
        label='3D coordinates of a surface mesh'
    )

    plt.add_point_labels(
        centers_array,
        labels=labels,
        font_size=14,
        text_color='black',
        point_size=10,
        point_color='red',
        always_visible=True,
        shadow=True,
        name="element_labels"
    )

    if ribcage_point_cloud is not None:
        plt.add_points(ribcage_point_cloud, color='tan', label='Point cloud',
                       point_size=2, render_points_as_spheres=True)

    if ribcage_point_cloud is not None:
        plt.add_legend([
            ['Mesh Surface', '#FFCCCC'],
            ['Element Centers', 'red'],
            ['Ribcage Point Cloud', 'tan']
        ])

    plt.add_axes()
    plt.show()

    return centers_array


def plot_filter_debug(
        src_pts: np.ndarray,
        tgt_pts: np.ndarray,
        src_mask: np.ndarray,
        tgt_filtered: np.ndarray,
        src_ss: np.ndarray,
        tgt_ss: np.ndarray,
        filter_info: dict,
        title: str = "Filter Debug",
        save_path: str = None,
        show_edge_analysis: bool = True,
) -> None:
    """
    Debug visualization showing the filtering results and bounding boxes.

    Plots:
    - Original source points (gray, semi-transparent)
    - Filtered source points (red)
    - Original target points (light blue, semi-transparent)
    - Filtered target points (blue)
    - Bounding boxes used for filtering (wireframe)
    - Edge analysis: points near bbox boundaries (if show_edge_analysis=True)
    """
    try:
        import pyvista as pv
    except ImportError:
        print("  WARNING: pyvista not available, skipping filter debug plot")
        return

    from pathlib import Path

    plotter = pv.Plotter()
    plotter.set_background('white')

    # Center points for visualization (in sternum-centered coords)
    src_centered = src_pts - src_ss
    tgt_centered = tgt_pts - tgt_ss
    src_filtered_centered = src_centered[src_mask]
    tgt_filtered_centered = tgt_filtered - tgt_ss

    # Original source (gray, transparent)
    plotter.add_points(
        pv.PolyData(src_centered),
        color='gray', point_size=3, opacity=0.2,
        label=f'Src Original ({len(src_centered)} pts)',
    )

    # Filtered source (red)
    plotter.add_points(
        pv.PolyData(src_filtered_centered),
        color='red', point_size=5, opacity=0.8,
        label=f'Src Filtered ({np.sum(src_mask)} pts)',
    )

    # Original target (light blue, transparent)
    plotter.add_points(
        pv.PolyData(tgt_centered),
        color='lightblue', point_size=3, opacity=0.2,
        label=f'Tgt Original ({len(tgt_centered)} pts)',
    )

    # Filtered target (blue)
    plotter.add_points(
        pv.PolyData(tgt_filtered_centered),
        color='steelblue', point_size=5, opacity=0.8,
        label=f'Tgt Filtered ({len(tgt_filtered_centered)} pts)',
    )

    # Draw bounding boxes as wireframes
    src_bbox_min = filter_info['src_bbox_min']
    src_bbox_max = filter_info['src_bbox_max']
    tgt_bbox_min = filter_info['tgt_bbox_min']
    tgt_bbox_max = filter_info['tgt_bbox_max']

    # Source bbox (padded) - used to filter target
    src_box = pv.Box(bounds=[
        src_bbox_min[0], src_bbox_max[0],
        src_bbox_min[1], src_bbox_max[1],
        src_bbox_min[2], src_bbox_max[2],
    ])
    plotter.add_mesh(src_box, style='wireframe', color='orange', line_width=2,
                     label='Src BBox (padded)')

    # Target bbox (no padding) - used to filter source
    tgt_box = pv.Box(bounds=[
        tgt_bbox_min[0], tgt_bbox_max[0],
        tgt_bbox_min[1], tgt_bbox_max[1],
        tgt_bbox_min[2], tgt_bbox_max[2],
    ])
    plotter.add_mesh(tgt_box, style='wireframe', color='blue', line_width=2,
                     label='Tgt BBox (reciprocal)')

    # Add stats text
    n_src_removed = len(src_pts) - np.sum(src_mask)
    n_tgt_removed = len(tgt_pts) - len(tgt_filtered)
    stats_text = (
        f"Src: {len(src_pts)} -> {np.sum(src_mask)} (-{n_src_removed})\n"
        f"Tgt: {len(tgt_pts)} -> {len(tgt_filtered)} (-{n_tgt_removed})\n"
        f"Src BBox X: [{src_bbox_min[0]:.0f}, {src_bbox_max[0]:.0f}]\n"
        f"Src BBox Y: [{src_bbox_min[1]:.0f}, {src_bbox_max[1]:.0f}]\n"
        f"Src BBox Z: [{src_bbox_min[2]:.0f}, {src_bbox_max[2]:.0f}]\n"
        f"Tgt BBox X: [{tgt_bbox_min[0]:.0f}, {tgt_bbox_max[0]:.0f}]\n"
        f"Tgt BBox Y: [{tgt_bbox_min[1]:.0f}, {tgt_bbox_max[1]:.0f}]\n"
        f"Tgt BBox Z: [{tgt_bbox_min[2]:.0f}, {tgt_bbox_max[2]:.0f}]"
    )
    plotter.add_text(stats_text, position='upper_right', font_size=9, color='black')

    # Mark origin (sternum)
    plotter.add_points(
        pv.PolyData(np.array([[0, 0, 0]])),
        color='green', point_size=15, render_points_as_spheres=True,
        label='Sternum (origin)',
    )

    # Edge analysis: highlight points near the bbox boundaries
    if show_edge_analysis:
        edge_margin = 5.0  # mm from boundary

        near_xmin = np.abs(tgt_filtered_centered[:, 0] - tgt_bbox_min[0]) < edge_margin
        near_xmax = np.abs(tgt_filtered_centered[:, 0] - tgt_bbox_max[0]) < edge_margin
        near_ymin = np.abs(tgt_filtered_centered[:, 1] - tgt_bbox_min[1]) < edge_margin
        near_ymax = np.abs(tgt_filtered_centered[:, 1] - tgt_bbox_max[1]) < edge_margin
        near_zmin = np.abs(tgt_filtered_centered[:, 2] - tgt_bbox_min[2]) < edge_margin
        near_zmax = np.abs(tgt_filtered_centered[:, 2] - tgt_bbox_max[2]) < edge_margin

        print(f"\n  Edge analysis (points within {edge_margin}mm of boundary):")
        print(f"    X-min ({tgt_bbox_min[0]:.0f}): {np.sum(near_xmin)} pts")
        print(f"    X-max ({tgt_bbox_max[0]:.0f}): {np.sum(near_xmax)} pts")
        print(f"    Y-min ({tgt_bbox_min[1]:.0f}): {np.sum(near_ymin)} pts")
        print(f"    Y-max ({tgt_bbox_max[1]:.0f}): {np.sum(near_ymax)} pts")
        print(f"    Z-min ({tgt_bbox_min[2]:.0f}): {np.sum(near_zmin)} pts")
        print(f"    Z-max ({tgt_bbox_max[2]:.0f}): {np.sum(near_zmax)} pts")

        lateral_edge_pts = tgt_filtered_centered[near_ymin | near_ymax]
        if len(lateral_edge_pts) > 0:
            plotter.add_points(
                pv.PolyData(lateral_edge_pts),
                color='yellow', point_size=8, opacity=1.0,
                render_points_as_spheres=True,
                label=f'Lateral edge pts ({len(lateral_edge_pts)})',
            )

    plotter.add_legend(face=None, bcolor='white')
    plotter.add_title(title)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(save_path))
        print(f"  Filter debug plot saved: {save_path}")

    plotter.show()


# ---------------------------------------------------------------------------
# 4. Mutual region filtering (mesh <-> point cloud)
# ---------------------------------------------------------------------------
def filter_mutual_region(
        src_pts: np.ndarray,
        tgt_pts: np.ndarray,
        src_ss: np.ndarray,
        tgt_ss: np.ndarray,
        padding: float = 15.0,
        padding_inferior: float = 0.0,
        padding_reciprocal: float = 0.0,
        anterior_y_min: float = None,
        verbose: bool = False,
        skip_reciprocal: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Filter target point cloud and source mesh points to their mutual overlap.

    Both datasets are shifted to sternum-centered space before comparison.

    Step 1 — Filter target to source bbox (with asymmetric padding).
    Step 2 — Filter source to the (filtered) target bbox (reciprocal,
             no extra padding so mesh points beyond the PC extent are removed).
             Can be skipped with skip_reciprocal=True.

    Args:
        src_pts: (N, 3) source mesh points (world coords)
        tgt_pts: (M, 3) target point cloud (world coords)
        src_ss:  (3,) source sternum superior position
        tgt_ss:  (3,) target sternum superior position
        padding: extra margin (mm) on each side of the source bounding box
        padding_inferior: margin (mm) for inferior (Z-min) side only;
                          overrides ``padding`` for that boundary
        padding_reciprocal: extra margin (mm) on each side of the target
                            bounding box in Step 2 (default 0 = tight cut)
        anterior_y_min: shared anterior Y boundary (sternum-centered) for both
                        steps. If provided, both bboxes use this as Y-min,
                        ensuring neither mesh nor PC anterior points are clipped.
        verbose: print filtering summary
        skip_reciprocal: if True, skip Step 2 (reciprocal source filtering).
                         Use this for cleaner straight cuts when the mesh
                         and point cloud have different extents.

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

    if anterior_y_min is not None:
        src_bbox_min[1] = anterior_y_min - padding_reciprocal

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
        tgt_extent_min = tgt_centered.min(axis=0)
        tgt_extent_max = tgt_centered.max(axis=0)
        print(f"    Target original extent (sternum-centered): "
              f"X[{tgt_extent_min[0]:.0f},{tgt_extent_max[0]:.0f}] "
              f"Y[{tgt_extent_min[1]:.0f},{tgt_extent_max[1]:.0f}] "
              f"Z[{tgt_extent_min[2]:.0f},{tgt_extent_max[2]:.0f}]")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            if tgt_extent_min[i] > src_bbox_min[i]:
                print(f"    NOTE: {axis}-min limited by target extent, not source bbox")
            if tgt_extent_max[i] < src_bbox_max[i]:
                print(f"    NOTE: {axis}-max limited by target extent, not source bbox")

    # --- Step 2: reciprocal — filter source to (filtered) target bbox ---
    # Padding is applied only on the anterior side (Y-min in RAI coords)
    # so mesh points extending anteriorly beyond the PC are kept, while
    # other boundaries remain tight.
    if skip_reciprocal:
        src_inside = np.ones(len(src_pts), dtype=bool)
        tgt_filt_centered = tgt_filtered - tgt_ss
        tgt_bbox_min = tgt_filt_centered.min(axis=0)
        tgt_bbox_max = tgt_filt_centered.max(axis=0)
        if verbose:
            print(f"  Reciprocal source filtering: SKIPPED (all source points kept)")
    else:
        tgt_filt_centered = tgt_filtered - tgt_ss

        recip_pad_min = np.array([padding_reciprocal, padding_reciprocal, 0.0])
        recip_pad_max = np.array([padding_reciprocal, 0.0, 0.0])
        tgt_bbox_min = tgt_filt_centered.min(axis=0) - recip_pad_min
        tgt_bbox_max = tgt_filt_centered.max(axis=0) + recip_pad_max

        if anterior_y_min is not None:
            tgt_bbox_min[1] = anterior_y_min - padding_reciprocal

        src_inside = np.all(
            (src_centered >= tgt_bbox_min) & (src_centered <= tgt_bbox_max),
            axis=1,
        )

        if verbose:
            n_removed = int(np.sum(~src_inside))
            pad_str = f", {padding_reciprocal:.0f}mm anterior padding" if padding_reciprocal > 0 else ", no padding"
            print(f"  Reciprocal source filtering (to target extent{pad_str}):")
            print(f"    Source: {len(src_pts)} pts  ->  "
                  f"{int(np.sum(src_inside))} pts kept, "
                  f"{n_removed} removed")
            print(f"    Target bbox (sternum-centered{pad_str}): "
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
# 5. Coordinate transforms
# ---------------------------------------------------------------------------
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


def inverse_transform_to_source_frame(coords, R, source_anchor, target_anchor):
    """
    Inverse transform coordinates from target (supine) frame back to source (prone) frame.

    This is the reverse of apply_transform_to_coords:
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
