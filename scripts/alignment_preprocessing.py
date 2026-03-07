"""
Alignment Preprocessing

Prepares prone ribcage mesh and supine ribcage point cloud for ICP alignment:
    - Centering: translate both datasets so sternum superior is at origin
    - Mesh element selection: choose anterior elements for alignment
    - Point cloud filtering: remove spine, axial extremes, lateral outliers
    - Mutual region filtering: clip mesh and point cloud to overlapping region
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Dict


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
# 2. Mutual region filtering (mesh <-> point cloud)
# ---------------------------------------------------------------------------
def filter_mutual_region(
        src_pts: np.ndarray,
        tgt_pts: np.ndarray,
        src_ss: np.ndarray,
        tgt_ss: np.ndarray,
        padding: float = 15.0,
        padding_inferior: float = 0.0,
        padding_reciprocal: float = 0.0,
        y_clip_max: float = None,
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
        y_clip_max: if not None, clip the Y-max boundary of both source
                    and target bounding boxes to this value (in sternum-
                    centered coords). Typically set to the Y midpoint of
                    the original full mesh/point cloud + padding, to keep
                    only the anterior half.
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

    # Clip Y-max boundary to keep only anterior region
    if y_clip_max is not None:
        src_y_max_orig = src_bbox_max[1]
        src_bbox_max[1] = min(src_bbox_max[1], y_clip_max)
        if verbose:
            print(f"  Source Y clip: Y-max {src_y_max_orig:.0f} -> {src_bbox_max[1]:.0f} "
                  f"(y_clip_max={y_clip_max:.0f})")

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
    if skip_reciprocal:
        src_inside = np.ones(len(src_pts), dtype=bool)
        tgt_filt_centered = tgt_filtered - tgt_ss
        tgt_bbox_min = tgt_filt_centered.min(axis=0)
        tgt_bbox_max = tgt_filt_centered.max(axis=0)
        if verbose:
            print(f"  Reciprocal source filtering: SKIPPED (all source points kept)")
    else:
        tgt_filt_centered = tgt_filtered - tgt_ss
        tgt_bbox_min = tgt_filt_centered.min(axis=0) - padding_reciprocal
        tgt_bbox_max = tgt_filt_centered.max(axis=0) + padding_reciprocal

        # Clip Y-max boundary to keep only anterior region
        if y_clip_max is not None:
            tgt_y_max_orig = tgt_bbox_max[1]
            tgt_bbox_max[1] = min(tgt_bbox_max[1], y_clip_max)
            if verbose:
                print(f"  Target Y clip: Y-max {tgt_y_max_orig:.0f} -> {tgt_bbox_max[1]:.0f} "
                      f"(y_clip_max={y_clip_max:.0f})")

        src_inside = np.all(
            (src_centered >= tgt_bbox_min) & (src_centered <= tgt_bbox_max),
            axis=1,
        )

        if verbose:
            n_removed = int(np.sum(~src_inside))
            pad_str = f", {padding_reciprocal:.0f}mm padding" if padding_reciprocal > 0 else ", no padding"
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
# 3. Mesh element extraction and selection
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
# 4. Mesh element visualization
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
# 5. Initial landmark-based alignment
# ---------------------------------------------------------------------------
def compute_initial_alignment(
        prone_si_vector: np.ndarray,
        supine_si_vector: np.ndarray,
        verbose: bool = True,
) -> np.ndarray:
    """
    Compute an initial rotation that aligns the prone sternum axis to supine.

    Uses only the sternum superior→inferior vector (the one reliable rigid
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
        R_init: (3, 3) rotation matrix (prone → supine)
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
            # 180° rotation — pick any perpendicular axis
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
        print(f"    Rotation angle: {angle_deg:.1f}°")
        print(f"    Residual angle after rotation: {residual:.4f}°")

    return R_init


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

    # Element selection: list of element indices for anterior ribcage
    # Set to None to use all elements
    SELECTED_ELEMENTS = [0, 1, 6, 7, 8, 9, 14, 15, 16, 17, 22, 23]

    # Target region padding (mm)
    TARGET_REGION_PADDING = 15.0

    for vl_id in vl_ids:
        vl_id_str = f"VL{vl_id:05d}"
        print(f"\n{'='*60}")
        print(f"ALIGNMENT PREPROCESSING")
        print(f"Subject: {vl_id_str}")
        print(f"{'='*60}")

        # ==============================================================
        # 1. Load subject data (anatomical landmarks via load_subject)
        # ==============================================================
        # load_subject constructs paths as:
        #   <ANATOMICAL_JSON_BASE_ROOT>/<position>/landmarks/VL<id>_skeleton_data_<position>_t2.json
        # with fallback to .../landmarks/combined/...
        subject = load_subject(
            vl_id=vl_id,
            positions=["prone", "supine"],
            dicom_root=None,  # skip DICOM loading (not needed for preprocessing)
            anatomical_json_base_root=ANATOMICAL_JSON_BASE_ROOT,
            soft_tissue_root=SOFT_TISSUE_ROOT,
        )

        anat_prone = subject.scans["prone"].anatomical_landmarks
        anat_supine = subject.scans["supine"].anatomical_landmarks

        if anat_prone.sternum_superior is None or anat_supine.sternum_superior is None:
            raise ValueError(f"Subject {vl_id_str} missing sternum superior landmarks")

        sternum_prone_sup = anat_prone.sternum_superior
        sternum_supine_sup = anat_supine.sternum_superior

        print(f"  Prone sternum superior:  {sternum_prone_sup}")
        print(f"  Supine sternum superior: {sternum_supine_sup}")

        # ==============================================================
        # 2. Load prone ribcage mesh
        # ==============================================================
        prone_mesh_file = PRONE_RIBCAGE_ROOT / f"{vl_id_str}_ribcage_prone.mesh"
        prone_ribcage = morphic.Mesh(str(prone_mesh_file))
        prone_mesh_coords = get_surface_mesh_coords(prone_ribcage, res=26)
        print(f"  Prone mesh: {prone_ribcage.elements.size()} elements, "
              f"{prone_mesh_coords.shape[0]} points")

        # ==============================================================
        # 3. Load supine ribcage point cloud
        # ==============================================================
        supine_seg_file = SUPINE_RIBCAGE_ROOT / f"rib_cage_{vl_id_str}.nii.gz"
        supine_ribcage_mask = breast_metadata.readNIFTIImage(
            str(supine_seg_file), orientation_flag, swap_axes=True
        )
        supine_ribcage_pc = extract_contour_points(supine_ribcage_mask, 20000)
        print(f"  Supine point cloud: {supine_ribcage_pc.shape[0]} points")

        # Per-subject point cloud cleanup
        if vl_id == 54:
            z_min = supine_ribcage_pc[:, 2].min()
            n_before = supine_ribcage_pc.shape[0]
            supine_ribcage_pc = supine_ribcage_pc[
                supine_ribcage_pc[:, 2] > z_min + 15.0
            ]
            print(f"  VL00054: removed inferior 20mm -> "
                  f"{n_before} -> {supine_ribcage_pc.shape[0]} points")

        # ==============================================================
        # 4. Center both on sternum superior (origin = sternum)
        # ==============================================================
        sternum_prone_inf = anat_prone.sternum_inferior
        sternum_supine_inf = anat_supine.sternum_inferior

        prone_mesh_centered = prone_mesh_coords - sternum_prone_sup
        supine_pc_centered = supine_ribcage_pc - sternum_supine_sup

        print(f"\n  After centering on sternum superior:")
        print(f"    Prone mesh range:  X[{prone_mesh_centered[:,0].min():.1f}, {prone_mesh_centered[:,0].max():.1f}] "
              f"Y[{prone_mesh_centered[:,1].min():.1f}, {prone_mesh_centered[:,1].max():.1f}] "
              f"Z[{prone_mesh_centered[:,2].min():.1f}, {prone_mesh_centered[:,2].max():.1f}]")
        print(f"    Supine PC range:   X[{supine_pc_centered[:,0].min():.1f}, {supine_pc_centered[:,0].max():.1f}] "
              f"Y[{supine_pc_centered[:,1].min():.1f}, {supine_pc_centered[:,1].max():.1f}] "
              f"Z[{supine_pc_centered[:,2].min():.1f}, {supine_pc_centered[:,2].max():.1f}]")

        # Plot sternum-centered mesh and point cloud (before initial alignment)
        print("\n  [Plot 1a] Sternum-centered BEFORE initial alignment")
        plot_all(
            point_cloud=supine_pc_centered,
            mesh_points=prone_mesh_centered,
            anat_landmarks=np.array([[0, 0, 0]]),
        )

        # ==============================================================
        # 4b. Initial alignment: align sternum SI axis (prone → supine)
        # ==============================================================
        # Only use sternum superior→inferior direction. This is the one
        # reliable rigid constraint. Rotation around the SI axis is left
        # for ICP to resolve — nipples move too much to be useful here.
        prone_si = sternum_prone_inf - sternum_prone_sup
        supine_si = sternum_supine_inf - sternum_supine_sup

        R_init = compute_initial_alignment(
            prone_si_vector=prone_si,
            supine_si_vector=supine_si,
            verbose=True,
        )

        # Apply initial rotation to centered prone data
        prone_mesh_centered = (R_init @ prone_mesh_centered.T).T

        print(f"\n  After initial alignment:")
        print(f"    Prone mesh range:  X[{prone_mesh_centered[:,0].min():.1f}, {prone_mesh_centered[:,0].max():.1f}] "
              f"Y[{prone_mesh_centered[:,1].min():.1f}, {prone_mesh_centered[:,1].max():.1f}] "
              f"Z[{prone_mesh_centered[:,2].min():.1f}, {prone_mesh_centered[:,2].max():.1f}]")
        print(f"    Supine PC range:   X[{supine_pc_centered[:,0].min():.1f}, {supine_pc_centered[:,0].max():.1f}] "
              f"Y[{supine_pc_centered[:,1].min():.1f}, {supine_pc_centered[:,1].max():.1f}] "
              f"Z[{supine_pc_centered[:,2].min():.1f}, {supine_pc_centered[:,2].max():.1f}]")

        # Plot after initial alignment with sternum inferior landmarks
        print("\n  [Plot 1b] Sternum-centered AFTER initial alignment")

        prone_si_rotated = R_init @ prone_si
        supine_si_centered = supine_si  # already centered

        import pyvista as pv
        plotter = pv.Plotter()
        plotter.set_background('white')

        # Mesh and point cloud
        plotter.add_points(pv.PolyData(prone_mesh_centered),
                           color='red', point_size=2, opacity=0.4,
                           label='Prone mesh')
        plotter.add_points(pv.PolyData(supine_pc_centered),
                           color='blue', point_size=2, opacity=0.4,
                           label='Supine PC')

        # Sternum superior (origin) — green
        plotter.add_points(pv.PolyData(np.array([[0, 0, 0]])),
                           color='green', point_size=15,
                           render_points_as_spheres=True,
                           label='Sternum sup (origin)')

        # Sternum inferior — prone (orange) vs supine (purple)
        # These should overlap closely after alignment
        plotter.add_points(pv.PolyData(prone_si_rotated.reshape(1, 3)),
                           color='orange', point_size=15,
                           render_points_as_spheres=True,
                           label='Prone sternum inf')
        plotter.add_points(pv.PolyData(supine_si_centered.reshape(1, 3)),
                           color='purple', point_size=15,
                           render_points_as_spheres=True,
                           label='Supine sternum inf')

        plotter.add_legend(face=None, bcolor='white')
        plotter.add_axes()
        plotter.add_title(f"Initial Alignment (sternum axis) - {vl_id_str}")
        plotter.show()

        # ==============================================================
        # 5. Visualize mesh elements and select anterior elements
        # ==============================================================
        centers_array, num_elements = get_mesh_elements_2(prone_ribcage)
        # Apply same centering + initial rotation to element centers
        centers_centered = (R_init @ (centers_array - sternum_prone_sup).T).T

        print(f"\n  Mesh has {num_elements} elements")
        print("  [Plot 2] All mesh elements with labels (choose anterior elements)")
        plot_mesh_elements(prone_mesh_centered, centers_centered, range(num_elements),
                           ribcage_point_cloud=supine_pc_centered)

        # ==============================================================
        # 6. Select mesh elements and extract coordinates
        # ==============================================================
        if SELECTED_ELEMENTS is not None:
            selected_mesh_coords = get_mesh_with_selected_elements(
                prone_ribcage, SELECTED_ELEMENTS, res=26
            )
            # Apply same centering + initial rotation
            selected_mesh_centered = (R_init @ (selected_mesh_coords - sternum_prone_sup).T).T
            selected_centers = centers_centered[SELECTED_ELEMENTS]

            print(f"\n  Selected {len(SELECTED_ELEMENTS)}/{num_elements} elements")
            print(f"  Selected mesh points: {selected_mesh_centered.shape[0]}")

            # Plot selected elements
            print("  [Plot 3] Selected anterior mesh elements")
            plot_mesh_elements(selected_mesh_centered, selected_centers,
                               SELECTED_ELEMENTS, ribcage_point_cloud=supine_pc_centered)
        else:
            selected_mesh_coords = prone_mesh_coords
            selected_mesh_centered = prone_mesh_centered
            print(f"\n  Using all {num_elements} elements (no selection)")

        # ==============================================================
        # 7. Filter point cloud to match selected mesh region
        # ==============================================================
        # Both datasets are already in sternum-centered space (and the
        # mesh has the initial rotation applied), so pass zeros for the
        # sternum offsets — filter_mutual_region subtracts them internally.
        origin = np.zeros(3)

        # Compute Y clip from original full mesh and point cloud (already
        # in sternum-centered + initial-alignment space)
        Y_CLIP_PADDING = 15.0
        mesh_y_mid = (prone_mesh_centered[:, 1].min() + prone_mesh_centered[:, 1].max()) / 2.0
        pc_y_mid = (supine_pc_centered[:, 1].min() + supine_pc_centered[:, 1].max()) / 2.0
        y_clip_max = max(mesh_y_mid, pc_y_mid) + Y_CLIP_PADDING
        print(f"  Y clip: mesh midpoint={mesh_y_mid:.1f}, PC midpoint={pc_y_mid:.1f}, "
              f"y_clip_max={y_clip_max:.1f}")

        src_mask, supine_pc_filtered_centered, filter_info = filter_mutual_region(
            src_pts=selected_mesh_centered,
            tgt_pts=supine_pc_centered,
            src_ss=origin,
            tgt_ss=origin,
            padding=TARGET_REGION_PADDING,
            padding_inferior=0.0,
            padding_reciprocal=TARGET_REGION_PADDING,
            y_clip_max=y_clip_max,
            verbose=True,
            skip_reciprocal=False,
        )

        selected_mesh_filtered = selected_mesh_centered[src_mask]

        print(f"\n  After mutual region filtering:")
        print(f"    Mesh points:  {selected_mesh_centered.shape[0]} -> {selected_mesh_filtered.shape[0]}")
        print(f"    Point cloud:  {supine_pc_centered.shape[0]} -> {supine_pc_filtered_centered.shape[0]}")

        # ==============================================================
        # 8. Plot filtered mesh and point cloud (ready for alignment)
        # ==============================================================
        print("\n  [Plot 4] Filtered mesh and point cloud (alignment-ready)")
        plot_all(
            point_cloud=supine_pc_filtered_centered,
            mesh_points=selected_mesh_filtered,
            anat_landmarks=np.array([[0, 0, 0]]),
        )

        # Debug filter visualization with bounding boxes
        print("  [Plot 5] Filter debug with bounding boxes")
        plot_filter_debug(
            src_pts=selected_mesh_centered,
            tgt_pts=supine_pc_centered,
            src_mask=src_mask,
            tgt_filtered=supine_pc_filtered_centered,
            src_ss=origin,
            tgt_ss=origin,
            filter_info=filter_info,
            title=f"Mutual Region Filter - {vl_id_str}",
        )

        print(f"\n{'='*60}")
        print(f"Preprocessing complete for {vl_id_str}")
        print(f"{'='*60}")
