import pyvista as pv
import breast_metadata
import mesh_tools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.cm as cm

def visualise_landmarks_distance(vl_id, position, skin_mask, rib_mask,
                                 soft_tissue_landmarks, all_results):

    vl_id_str_formatted = f"VL{vl_id:05d}"

    # Load mask grids
    skin_mask_image_grid = breast_metadata.SCANToPyvistaImageGrid(skin_mask, 'RAI')
    rib_mask_image_grid = breast_metadata.SCANToPyvistaImageGrid(rib_mask, 'RAI')

    # Get closest points from results
    closest_points_skin = all_results[vl_id][position]["anthony"]["skin_points"]
    closest_points_rib = all_results[vl_id][position]["anthony"]["rib_points"]

    # Initialize plotter
    plotter = pv.Plotter()

    # Add masks
    skin_mask_threshold = skin_mask_image_grid.threshold(value=0.5)
    rib_mask_threshold = rib_mask_image_grid.threshold(value=0.5)
    plotter.add_mesh(skin_mask_threshold, color='lightskyblue', opacity=0.2, show_scalar_bar=False)
    plotter.add_mesh(rib_mask_threshold, color='lavender', opacity=0.2, show_scalar_bar=False)

    # Add landmarks
    landmarks_coords = np.array(list(soft_tissue_landmarks.values()))
    plotter.add_points(landmarks_coords, color='red', render_points_as_spheres=True, point_size=12)

    # Add closest points
    plotter.add_points(np.array(list(closest_points_skin.values())), color='green', render_points_as_spheres=True,
                       point_size=10)
    plotter.add_points(np.array(list(closest_points_rib.values())), color='blue', render_points_as_spheres=True,
                       point_size=10)

    # Draw lines: landmarks to skin
    for lm_name, lm_coord in soft_tissue_landmarks.items():
        line = pv.Line(lm_coord, closest_points_skin[lm_name])
        plotter.add_mesh(line, color="yellow", line_width=3)

    # Draw lines: landmarks to rib
    for lm_name, lm_coord in soft_tissue_landmarks.items():
        line = pv.Line(lm_coord, closest_points_rib[lm_name])
        plotter.add_mesh(line, color="magenta", line_width=3)

    # Add legend and text
    legend_entries = [
        ['Skin mask', 'lightskyblue'], ['Rib mask', 'lavender'], ['Landmarks', 'red'],
        ['Closest skin points', 'green'], ['Closest rib points', 'blue'],
        ['Distance to skin', 'yellow'], ['Distance to rib', 'magenta']
    ]
    plotter.add_legend(legend_entries, bcolor='w')
    plotter.add_text(f"{vl_id_str_formatted} {position} ", position='upper_left', font_size=10,
                     color='black')
    labels = dict(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    plotter.add_axes(**labels)
    plotter.view_xy()
    plotter.show()



def plot_all(
    scan=None,
    mask=None,
    point_cloud=None,
    mesh=None,
    mesh_points=None,
    anat_landmarks=None,
    soft_tissue_landmarks=None,
):
    """
        Plots various breast-related data (scan, mask, point cloud, mesh, and landmarks)
        using PyVista, skipping any input that is not provided or is None.

        Inputs are made optional by setting their default value to None.

        Args:
            scan (object, optional): Scan data, convertible to PyVista ImageGrid.
            mask (object, optional): Mask data, convertible to PyVista ImageGrid.
            point_cloud (pv.DataSet or array-like, optional): Point cloud data.
            mesh (object, optional): Mesh data, convertible to meshio then PyVista mesh.
            anat_landmarks (pv.DataSet or array-like or list of such, optional): Anatomical landmarks.
            soft_tissue_landmarks (pv.DataSet or array-like or list of such, optional): Soft tissue landmarks.
        """

    orientation_flag = "RAI"
    plotter = pv.Plotter()
    opacity = np.linspace(0, 0.4, 100)

    # 1. Plot Scan Volume
    if scan is not None:
        try:
            image_grid = breast_metadata.SCANToPyvistaImageGrid(scan, orientation_flag)
            plotter.add_volume(
                image_grid,
                scalars='values',
                cmap='gray',
                opacity=opacity
            )
            # plotter.add_volume(image_grid, opacity='sigmoid_6', cmap='coolwarm', show_scalar_bar=False)
            print("INFO: Plotted Scan Volume.")
        except Exception as e:
            print(f"WARNING: Could not plot scan volume. Error: {e}")

    # 2. Plot Segmentation Mask
    if mask is not None:
        try:
            mask_image_grid = breast_metadata.SCANToPyvistaImageGrid(mask, orientation_flag)
            # Threshold the mask to get the surface/shape
            thresholded_mask = mask_image_grid.threshold(value=0.5)
            plotter.add_mesh(
                thresholded_mask,
                color='lightskyblue',
                opacity=0.2,
                show_scalar_bar=False
            )
            print("INFO: Plotted Segmentation Mask.")
        except Exception as e:
            print(f"WARNING: Could not plot segmentation mask. Error: {e}")

    # 3. Plot Point Cloud
    if point_cloud is not None:
        try:
            plotter.add_points(
                point_cloud,
                color="tan",
                label='Point cloud',
                point_size=2,
                render_points_as_spheres=True
            )
            print("INFO: Plotted Point Cloud.")
        except Exception as e:
            print(f"WARNING: Could not plot point cloud. Error: {e}")

    # 4. Plot Mesh
    if mesh is not None:
        try:
            mesh_meshio = mesh_tools.morphic_to_meshio(mesh, triangulate=True, res=4, exterior_only=True)
            plotter.add_mesh(
                mesh_meshio,
                show_edges=False,
                color='#FFCCCC',
                style="surface",
                opacity=0.5,
                label='Surface_mesh'
            )
            print("INFO: Plotted Surface Mesh.")
        except Exception as e:
            print(f"WARNING: Could not plot mesh. Error: {e}")

    if mesh_points is not None:
        try:
            plotter.add_points(
                mesh_points,
                color='gray',
                render_points_as_spheres=True,
                point_size=3,
                label='3D coordinates of a surface mesh'
            )
            print("INFO: Plotted 3D coordinates of a Surface Mesh.")
        except Exception as e:
            print(f"WARNING: Could not plot 3D coordinates of a Surface Mesh. Error: {e}")

    # 5. Plot Anatomical Landmarks (Handling multiple landmarks)
    if anat_landmarks is not None:
        # Check if the input is a list/tuple of landmarks or a single landmark set
        if isinstance(anat_landmarks, (list, tuple)):
            for i, landmark_set in enumerate(anat_landmarks):
                try:
                    plotter.add_points(
                        landmark_set,
                        color='red',
                        render_points_as_spheres=True,
                        point_size=10,
                        label=f'Anatomical landmark Set {i + 1}'
                    )
                except Exception as e:
                    print(f"WARNING: Could not plot anatomical landmark set {i + 1}. Error: {e}")
            print("INFO: Plotted multiple Anatomical Landmark sets.")
        else:
            try:
                plotter.add_points(
                    anat_landmarks,
                    color='red',
                    render_points_as_spheres=True,
                    point_size=10,
                    label='Anatomical landmarks'
                )
                print("INFO: Plotted single Anatomical Landmark set.")
            except Exception as e:
                print(f"WARNING: Could not plot anatomical landmarks. Error: {e}")

    # 6. Plot Soft Tissue Landmarks (Handling multiple landmarks)
    if soft_tissue_landmarks is not None:
        # Check if the input is a list/tuple of landmarks or a single landmark set
        if isinstance(soft_tissue_landmarks, (list, tuple)):
            for i, landmark_set in enumerate(soft_tissue_landmarks):
                try:
                    plotter.add_points(
                        landmark_set,
                        color='green',  # Changed color for visual distinction
                        render_points_as_spheres=True,
                        point_size=12,
                        label=f'Soft Tissue landmark Set {i + 1}'
                    )
                except Exception as e:
                    print(f"WARNING: Could not plot soft tissue landmark set {i + 1}. Error: {e}")
            print("INFO: Plotted multiple Soft Tissue Landmark sets.")
        else:
            try:
                plotter.add_points(
                    soft_tissue_landmarks,
                    color='green',  # Changed color for visual distinction
                    render_points_as_spheres=True,
                    point_size=12,
                    label='Soft Tissue landmarks'
                )
                print("INFO: Plotted single Soft Tissue Landmark set.")
            except Exception as e:
                print(f"WARNING: Could not plot soft tissue landmarks. Error: {e}")

    cam_pos = [
        [378.6543811782509, 309.86957859927173, 490.0925163131904],
        [-35.140107028421866, 15.381214203758844, -17.515439847958362],
        [-0.5214522629375, -0.48151960895256674, 0.7044333919339199]
    ]
    plotter.camera_position = cam_pos

    # --- Final Step ---
    plotter.show()

    cam_pos = plotter.camera_position
    # print("Camera position:", cam_pos[0])
    # print("Camera focal point:", cam_pos[1])
    # print("Camera view up:", cam_pos[2])
    return plotter




# --- I. Configuration for Planes ---
# Define the 3D coordinate system (0: X, 1: Y, 2: Z)
# X (0) = Right/Left, Y (1) = Ant/Post, Z (2) = Inf/Sup
PLANE_CONFIG = {
    'Coronal': {
        'axes': (0, 2), 'shape': 'Circle',
        'xlabel': "right-left (mm)", 'ylabel': "inf-sup (mm)",
        'quadrants_left': ('UO', 'UI', 'LO', 'LI'),
        'quadrants_right': ('UI', 'UO', 'LI', 'LO')
    },
    'Sagittal': {
        'axes': (1, 2), 'shape': 'SemiCircle',
        'xlabel': "ant-post (mm)", 'ylabel': "inf-sup (mm)",
        'quadrants_left': ('', 'upper', '', 'lower'),
        'quadrants_right': ('', 'upper', '', 'lower')
    },
    'Axial': {
        'axes': (0, 1), 'shape': 'SemiCircle',
        'xlabel': "right-left (mm)", 'ylabel': "ant-post (mm)",
        'quadrants_left': ('outer', 'inner', '', ''),
        'quadrants_right': ('inner','outer',  '', '')
    }
}
lims = (-400, 400)
radius = lims[1]


def _plot_plane(
        plane_name: str,
        config: dict,
        base_point_left: np.ndarray,
        vector_left: np.ndarray,
        base_point_right: np.ndarray,
        vector_right: np.ndarray,
        title: str
):
    """Generates and displays a single figure with two subplots (Left and Right) for one plane."""

    AXIS_X, AXIS_Y = config['axes']
    shape = config['shape']

    fig, (ax_right, ax_left) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    fig.suptitle(f"{plane_name} Plane: Direction of landmark motion from prone to supine\n({title})",
                 fontsize=14)

    arc_center_x, arc_center_y = 0, 0

    # Determine arc angles for semicircles
    start_angle_right, end_angle_right = 0, 360  # Default to full circle
    start_angle_left, end_angle_left = 0, 360

    if plane_name == 'Sagittal':
        # Y (Ant/Post) vs Z (Inf/Sup). Breast is anterior (positive Y in this projection)
        arc_center_x, arc_center_y = radius, 0
        start_angle_right, end_angle_right = 90, 270
        start_angle_left, end_angle_left = 90, 270
    elif plane_name == 'Axial':
        # X (R/L) vs Y (Ant/Post). Breast is anterior (positive Y in this projection)
        arc_center_x, arc_center_y = 0, radius
        start_angle_right, end_angle_right = 180, 360
        start_angle_left, end_angle_left = 180, 360


        # --- Plotting and Formatting ---

    # Process Right Breast (ax_right)
    ax_right.set_title(f"Right Breast", loc='left', fontsize=12)
    ax_right.quiver(
        base_point_right[:, AXIS_X], base_point_right[:, AXIS_Y],
        vector_right[:, AXIS_X], vector_right[:, AXIS_Y],
        angles='xy', scale_units='xy', scale=1, color='black'
    )

    # Process Left Breast (ax_left)
    ax_left.set_title(f"Left Breast", loc='left', fontsize=12)
    ax_left.quiver(
        base_point_left[:, AXIS_X], base_point_left[:, AXIS_Y],
        vector_left[:, AXIS_X], vector_left[:, AXIS_Y],
        angles='xy', scale_units='xy', scale=1, color='black'
    )

    # Apply common formatting to both subplots
    for col, ax in enumerate([ax_right, ax_left]):
        ax.set_xlabel(config['xlabel'])
        ax.set_ylabel(config['ylabel'])
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')

        # Center dot and quadrant lines
        ax.plot(0, 0, 'ro', markersize=8, zorder=5)  # Nipple
        ####use prone


        # --- Conditional Quadrant/Center Lines ---
        if plane_name == 'Coronal':
            # Coronal needs both horizontal (y=0) and vertical (x=0) lines
            ax.axhline(0, color='red', lw=1, zorder=0)  # Horizontal line
            ax.axvline(0, color='red', lw=1, zorder=0)  # Vertical line
        elif plane_name == 'Sagittal':
            # Sagittal only needs the horizontal line (y=0)
            ax.axhline(0, color='red', lw=1, zorder=0)  # Horizontal line
        elif plane_name == 'Axial':
            # Axial only needs the vertical line (x=0)
            ax.axvline(0, color='red', lw=1, zorder=0)  # Vertical line

        # --- Add Bounding Shape (Circle or Semicircle) ---
        current_start_angle, current_end_angle = (start_angle_right, end_angle_right) if col == 0 else (
            start_angle_left, end_angle_left)

        if shape == 'Circle':
            bounding_shape = Circle((0, 0), radius, fill=False, color='black', lw=1)
            ax.add_artist(bounding_shape)
        else:  # SemiCircle/Arc
            # Draw the curved part (Arc)
            arc_shape = Arc((arc_center_x, arc_center_y), width=radius * 2, height=radius * 2,
                            angle=0,
                            theta1=current_start_angle, theta2=current_end_angle,
                            color='black', lw=1)
            ax.add_artist(arc_shape)

            # Draw the straight line across the diameter (chest wall)
            # The diameter line is along the X-axis of the current plot (Z-axis or X-axis anatomically)
            ax.plot([-radius, radius], [0, 0], color='black', lw=1)

        # --- Add Quadrant Labels ---
        text_offset = radius * 0.85
        quadrants = config['quadrants_right'] if col == 0 else config['quadrants_left']

        # Quadrant positions (Top-Right, Top-Left, Bottom-Right, Bottom-Left)
        ax.text(text_offset, text_offset, quadrants[0], ha='center', va='center', fontsize=14)
        ax.text(-text_offset, text_offset, quadrants[1], ha='center', va='center', fontsize=14)
        ax.text(text_offset, -text_offset, quadrants[2], ha='center', va='center', fontsize=14)
        ax.text(-text_offset, -text_offset, quadrants[3], ha='center', va='center', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_vector_three_views(
        base_point_left: np.ndarray,
        vector_left: np.ndarray,
        base_point_right: np.ndarray,
        vector_right: np.ndarray,
        title: str
):
    """Generates three separate figures, one for each plane (Coronal, Sagittal, Axial)."""

    # Coronal Figure
    _plot_plane('Coronal', PLANE_CONFIG['Coronal'], base_point_left, vector_left, base_point_right, vector_right, title)

    # Sagittal Figure
    _plot_plane('Sagittal', PLANE_CONFIG['Sagittal'], base_point_left, vector_left, base_point_right, vector_right, title)

    # Axial Figure
    _plot_plane('Axial', PLANE_CONFIG['Axial'], base_point_left, vector_left, base_point_right, vector_right, title)

    # Note: plt.show() is called inside _plot_plane for immediate display of each figure.


def _extract_subject_data(
        alignment_results_all: Dict[int, Dict],
        vl_id: int,
        registrar_key: str = 'r1'  # Use 'r1' for Registrar 1, 'r2' for Registrar 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the four necessary NumPy arrays for plotting a single subject.
    Assumes all arrays are already in the alignment_results dictionary.
    """
    alignment_results = alignment_results_all.get(vl_id, {})

    # Construct the dictionary keys based on the registrar
    base_left_key = f'{registrar_key}_rel_nipple_vectors_base_left'
    vector_left_key = f'{registrar_key}_rel_nipple_vectors_left'
    base_right_key = f'{registrar_key}_rel_nipple_vectors_base_right'
    vector_right_key = f'{registrar_key}_rel_nipple_vectors_right'

    # Retrieve and ensure they are numpy arrays (assuming they are stored as such)
    base_point_left = np.array(alignment_results.get(base_left_key, []))
    vector_left = np.array(alignment_results.get(vector_left_key, []))
    base_point_right = np.array(alignment_results.get(base_right_key, []))
    vector_right = np.array(alignment_results.get(vector_right_key, []))

    # Basic check for empty or improperly shaped data (should be (N, 3))
    if base_point_left.ndim != 2 or base_point_left.shape[1] != 3:
        raise ValueError(f"Data for VL_ID {vl_id} is missing or malformed for key: {base_left_key}")

    return base_point_left, vector_left, base_point_right, vector_right


# --- ADAPTED PLOTTING CORE FUNCTION ---

def _plot_plane_multi_subject(
        plane_name: str,
        config: dict,
        subject_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],  # {VL_ID: (bL, vL, bR, vR)}
        title: str
):
    """
    Generates and displays a single figure with two subplots (Left and Right) for one plane,
    plotting data for multiple subjects.
    """

    AXIS_X, AXIS_Y = config['axes']
    shape = config['shape']

    fig, (ax_right, ax_left) = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    fig.suptitle(f"{plane_name} Plane: Direction of landmark motion from prone to supine)",
                 fontsize=14)

    arc_center_x, arc_center_y = 0, 0

    # Determine arc angles for semicircles
    start_angle_right, end_angle_right = 0, 360
    start_angle_left, end_angle_left = 0, 360

    if plane_name == 'Sagittal':
        arc_center_x, arc_center_y = radius, 0
        start_angle_right, end_angle_right = 90, 270
        start_angle_left, end_angle_left = 90, 270
    elif plane_name == 'Axial':
        arc_center_x, arc_center_y = 0, radius
        start_angle_right, end_angle_right = 180, 360
        start_angle_left, end_angle_left = 180, 360

    # Get a colormap and cycle through colors for subjects
    # Using 'viridis' colormap, but any qualitative map (e.g., 'Dark2', 'Set1') works well.
    colors = cm.get_cmap('viridis', len(subject_data))

    # --- Plotting Quivers for Each Subject ---

    for i, (vl_id, data) in enumerate(subject_data.items()):
        # Unpack the tuple: base_left, vector_left, base_right, vector_right
        base_point_left, vector_left, base_point_right, vector_right = data
        color = colors(i)
        label = f"VL ID: {vl_id}"

        # Process Right Breast (ax_right)
        ax_right.quiver(
            base_point_right[:, AXIS_X], base_point_right[:, AXIS_Y],
            vector_right[:, AXIS_X], vector_right[:, AXIS_Y],
            angles='xy', scale_units='xy', scale=1, color=color, label=label,
            width=0.003  # Adjust width for visibility if many vectors
        )

        # Process Left Breast (ax_left)
        ax_left.quiver(
            base_point_left[:, AXIS_X], base_point_left[:, AXIS_Y],
            vector_left[:, AXIS_X], vector_left[:, AXIS_Y],
            angles='xy', scale_units='xy', scale=1, color=color, label=label,
            width=0.003
        )

    # --- Apply common formatting to both subplots ---
    for col, ax in enumerate([ax_right, ax_left]):
        # Set titles for the specific breast side
        ax.set_title(f"Right Breast" if col == 0 else f"Left Breast", loc='left', fontsize=12)

        ax.set_xlabel(config['xlabel'])
        ax.set_ylabel(config['ylabel'])

        current_xlim = lims
        current_ylim = lims

        # if plane_name == 'Sagittal':
        #     # Constraint: X-axis (Z, Inf/Sup) starts from 0. Y-axis (Y, Ant/Post) maintains original limits.
        #     current_xlim = (0, lims[1])  # Starts X from 0
        #     # The axes now only show Superior and Anterior/Posterior
        # elif plane_name == 'Axial':
        #     # Constraint: Y-axis (Y, Ant/Post) starts from 0. X-axis (X, R/L) maintains original limits.
        #     current_ylim = (0, lims[1])  # Starts Y from 0
        #     # The axes now only show Anterior and Left/Right

        ax.set_xlim(current_xlim)
        ax.set_ylim(current_ylim)


        ax.set_aspect('equal', adjustable='box')

        # Center dot (Nipple)
        ax.plot(0, 0, marker='o', color='red', markersize=6, zorder=5, label='Nipple')

        # --- Conditional Quadrant/Center Lines ---
        if plane_name == 'Coronal':
            ax.axhline(0, color='red', lw=1, zorder=0)
            ax.axvline(0, color='red', lw=1, zorder=0)
        elif plane_name == 'Sagittal':
            ax.axhline(0, color='red', lw=1, zorder=0)
        elif plane_name == 'Axial':
            ax.axvline(0, color='red', lw=1, zorder=0)

        # --- Add Bounding Shape (Circle or Semicircle) ---
        current_start_angle, current_end_angle = (start_angle_right, end_angle_right) if col == 0 else (
            start_angle_left, end_angle_left)

        if shape == 'Circle':
            bounding_shape = Circle((0, 0), radius, fill=False, color='black', lw=1)
            ax.add_artist(bounding_shape)
        else:  # SemiCircle/Arc
            arc_shape = Arc((arc_center_x, arc_center_y), width=radius * 2, height=radius * 2,
                            angle=0, theta1=current_start_angle, theta2=current_end_angle,
                            color='black', lw=1)
            ax.add_artist(arc_shape)
            # Diameter line (chest wall)
            ax.plot([-radius, radius], [0, 0], color='black', lw=1)

        # --- Add Quadrant Labels ---
        text_offset = radius * 0.85
        quadrants = config['quadrants_right'] if col == 0 else config['quadrants_left']

        ax.text(text_offset, text_offset, quadrants[0], ha='center', va='center', fontsize=12)
        ax.text(-text_offset, text_offset, quadrants[1], ha='center', va='center', fontsize=12)
        ax.text(text_offset, -text_offset, quadrants[2], ha='center', va='center', fontsize=12)
        ax.text(-text_offset, -text_offset, quadrants[3], ha='center', va='center', fontsize=12)

        # Add a legend to identify subjects
        ax.legend(title="Subject", loc='best', fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- MAIN PLOTTING FUNCTION ---

def plot_vector_three_views_multi_subject(
        alignment_results_all: Dict[int, Dict],
        title: str,
        registrar_key: str = 'r1'
):
    """
    Generates three separate figures (Coronal, Sagittal, Axial) for multiple subjects.

    1. Processes the alignment_results_all dictionary to extract the necessary
       base points and vectors for all subjects.
    2. Calls the plotting helper function for each plane.
    """

    subject_data = {}

    # 1. Process all subjects
    for vl_id in alignment_results_all.keys():
        try:
            # Extract the 4 arrays for this subject/registrar
            data = _extract_subject_data(alignment_results_all, vl_id, registrar_key)
            subject_data[vl_id] = data
        except ValueError as e:
            print(f"Skipping VL_ID {vl_id} due to data extraction error: {e}")
            continue

    if not subject_data:
        print("No valid subject data found for plotting.")
        return

    # 2. Plotting for each plane
    _plot_plane_multi_subject('Coronal', PLANE_CONFIG['Coronal'], subject_data, title)
    _plot_plane_multi_subject('Sagittal', PLANE_CONFIG['Sagittal'], subject_data, title)
    _plot_plane_multi_subject('Axial', PLANE_CONFIG['Axial'], subject_data, title)


#
#
# def plot_vector_three_views(
#         base_point_left: np.ndarray,
#         vector_left: np.ndarray,
#         base_point_right: np.ndarray,
#         vector_right: np.ndarray
# ):
#     # --- I. Configuration for Planes ---
#     # Define the 3D coordinate system (0: X, 1: Y, 2: Z)
#     # X (0) = Right/Left, Y (1) = Ant/Post, Z (2) = Inf/Sup
#
#     # Define the bounding shape required for each plane
#     # Shape: 'Circle' (360°) or 'SemiCircle' (180°)
#
#     PLANE_CONFIG = {
#         'Coronal': {
#             'axes': (0, 2), 'shape': 'Circle',
#             'xlabel': "right-left (mm)", 'ylabel': "inf-sup (mm)",
#             'quadrants_left': ('UO', 'UI', 'LO', 'LI'),
#             'quadrants_right': ('UI', 'UO', 'LI', 'LO')
#         },
#         'Sagittal': {
#             'axes': (1, 2), 'shape': 'SemiCircle',
#             'xlabel': "ant-post (mm)", 'ylabel': "inf-sup (mm)",
#             # Sagittal view needs to start from the center line (posterior/anterior)
#             'quadrants_left': ('U', 'A', 'D', 'P'),
#             'quadrants_right': ('U', 'A', 'D', 'P')
#         },
#         'Axial': {
#             'axes': (0, 1), 'shape': 'SemiCircle',
#             'xlabel': "right-left (mm)", 'ylabel': "ant-post (mm)",
#             # Axial view needs to start from the center line (right/left)
#             'quadrants_left': ('A', 'O', 'P', 'I'),
#             'quadrants_right': ('A', 'I', 'P', 'O')
#         }
#     }
#
#     # Define plot limits
#     lims = (-100, 100)
#     radius = lims[1]
#
#     # Create the figure with 3 rows and 2 columns
#     fig, axes = plt.subplots(3, 2, figsize=(12, 18), sharex=True, sharey=True)
#     fig.suptitle(
#         "Direction of landmark motion from prone to supine (R1 only)\n(with respect to the nipple) - All Views",
#         fontsize=16)
#
#     # --- II. Plotting Loop ---
#     for row, (plane_name, config) in enumerate(PLANE_CONFIG.items()):
#         AXIS_X, AXIS_Y = config['axes']
#         shape = config['shape']
#
#         # Determine the start/end angles for the semicircle based on the plane
#         # These angles define the arc that represents the "rough breast shape"
#         if plane_name == 'Sagittal':
#             # Looking at Y (Ant/Post) vs Z (Inf/Sup).
#             # The breast is anterior to the center (Y > 0 or 90 to 270 degrees)
#             start_angle_right, end_angle_right = 90, 270  # Right plot (positive X axis is usually to the left of the viewer)
#             start_angle_left, end_angle_left = 90, 270  # Left plot (negative X axis is usually to the right of the viewer)
#         elif plane_name == 'Axial':
#             # Looking at X (Right/Left) vs Y (Ant/Post). Breast is anterior (Y > 0)
#             start_angle_right, end_angle_right = 0, 180  # Right plot (Right side is positive X)
#             start_angle_left, end_angle_left = 0, 180  # Left plot (Left side is negative X)
#         else:
#             # Coronal: full circle
#             start_angle_right, end_angle_right = 0, 360
#             start_angle_left, end_angle_left = 0, 360
#
#             # --- Plot Right Breast (Column 0) ---
#         ax_right = axes[row, 0]
#         ax_right.set_title(f"{plane_name} plane - Right breast", loc='left', fontsize=12)
#         ax_right.quiver(
#             base_point_right[:, AXIS_X], base_point_right[:, AXIS_Y],
#             vector_right[:, AXIS_X], vector_right[:, AXIS_Y],
#             angles='xy', scale_units='xy', scale=1, color='black'
#         )
#
#         # --- Plot Left Breast (Column 1) ---
#         ax_left = axes[row, 1]
#         ax_left.set_title(f"{plane_name} plane - Left breast", loc='left', fontsize=12)
#         ax_left.quiver(
#             base_point_left[:, AXIS_X], base_point_left[:, AXIS_Y],
#             vector_left[:, AXIS_X], vector_left[:, AXIS_Y],
#             angles='xy', scale_units='xy', scale=1, color='black'
#         )
#
#         # --- III. Formatting for Both Axes in the Row ---
#         for col, ax in enumerate([ax_right, ax_left]):
#             ax.set_xlabel(config['xlabel'])
#             ax.set_ylabel(config['ylabel'])
#
#             # Set limits and aspect ratio (applied to all)
#             ax.set_xlim(lims)
#             ax.set_ylim(lims)
#             ax.set_aspect('equal', adjustable='box')
#
#             # Add center dot and quadrant lines
#             ax.plot(0, 0, 'ro', markersize=8, zorder=5)  # Nipple
#             ax.axhline(0, color='red', lw=1, zorder=0)
#             ax.axvline(0, color='red', lw=1, zorder=0)
#
#             # --- Add Bounding Shape (Circle or Semicircle) ---
#             current_start_angle, current_end_angle = (start_angle_right, end_angle_right) if col == 0 else (
#                 start_angle_left, end_angle_left)
#
#             if shape == 'Circle':
#                 bounding_shape = Circle((0, 0), radius, fill=False, color='black', lw=1)
#             else:  # SemiCircle/Arc
#                 # Draw the curved part (Arc)
#                 arc_shape = Arc((0, 0), width=radius * 2, height=radius * 2,
#                                 angle=0,
#                                 theta1=current_start_angle, theta2=current_end_angle,
#                                 color='black', lw=1)
#                 ax.add_artist(arc_shape)
#
#                 # Draw the straight line across the diameter (center line/chest wall)
#                 # For Sagittal and Axial, the breast is generally located where Y > 0 (anterior)
#                 if plane_name == 'Sagittal':  # X-axis is Y (Ant/Post), Y-axis is Z (Inf/Sup)
#                     # Diameter line is along the Y-axis (Z-axis in the plot)
#                     ax.plot([-radius, radius], [0, 0], color='black', lw=1)  # The diameter line for Y-Z view
#                 elif plane_name == 'Axial':  # X-axis is X (R/L), Y-axis is Y (Ant/Post)
#                     # Diameter line is along the X-axis (X-axis in the plot)
#                     ax.plot([-radius, radius], [0, 0], color='black', lw=1)  # The diameter line for X-Y view
#
#                 # The full shape is the Arc plus the diameter line, so we skip adding the arc as the single bounding_shape
#                 continue  # Skip adding bounding_shape at the end of the loop
#
#             ax.add_artist(bounding_shape)
#
#             # Add Quadrant Labels
#             text_offset = radius * 0.85
#             quadrants = config['quadrants_right'] if col == 0 else config['quadrants_left']
#
#             # Top-Right (+X, +Y): quadrants[0]
#             ax.text(text_offset, text_offset, quadrants[0], ha='center', va='center', fontsize=14)
#             # Top-Left (-X, +Y): quadrants[1]
#             ax.text(-text_offset, text_offset, quadrants[1], ha='center', va='center', fontsize=14)
#             # Bottom-Right (+X, -Y): quadrants[2]
#             # Note: For Saggital/Axial, the "bottom" quadrants might be fully or partially excluded by the semicircle.
#             ax.text(text_offset, -text_offset, quadrants[2], ha='center', va='center', fontsize=14)
#             # Bottom-Left (-X, -Y): quadrants[3]
#             ax.text(-text_offset, -text_offset, quadrants[3], ha='center', va='center', fontsize=14)
#
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()


    #
    # # X=right-left and Y=inf-sup
    # AXIS_X = 0
    # AXIS_Y = 2
    #
    # # Define plot limits
    # lims = (-100, 100)
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8.5))
    # fig.suptitle("Direction of landmark motion from prone to supine (R1 only)\n(with respect to the nipple)",
    #              fontsize=16)
    #
    # # --- Plot 1: Right Breast ---
    # ax1.set_title("Coronal plane\nRight breast", loc='left', fontsize=12)
    # ax1.quiver(
    #     base_point_right[:, AXIS_X], base_point_right[:, AXIS_Y],  # Arrow base (relative prone pos)
    #     vector_right[:, AXIS_X], vector_right[:, AXIS_Y],  # Arrow vector (relative displacement)
    #     angles='xy', scale_units='xy', scale=1, color='black'
    # )
    #
    # # --- Plot 2: Left Breast ---
    # ax2.set_title("Coronal plane\nLeft breast", loc='left', fontsize=12)
    # ax2.quiver(
    #     base_point_left[:, AXIS_X], base_point_left[:, AXIS_Y],  # Arrow base (relative prone pos)
    #     vector_left[:, AXIS_X], vector_left[:, AXIS_Y],  # Arrow vector (relative displacement)
    #     angles='xy', scale_units='xy', scale=1, color='black'
    # )
    #
    # # --- Format both plots ---
    # for ax in [ax1, ax2]:
    #     ax.set_xlabel("right-left (mm)")
    #     ax.set_ylabel("inf-sup (mm)")
    #
    #     # Set limits and aspect ratio
    #     ax.set_xlim(lims)
    #     ax.set_ylim(lims)
    #     ax.set_aspect('equal', adjustable='box')
    #
    #     # Add red nipple dot and quadrant lines
    #     ax.plot(0, 0, 'ro', markersize=8, zorder=5)  # Nipple
    #     ax.axhline(0, color='red', lw=1, zorder=0)
    #     ax.axvline(0, color='red', lw=1, zorder=0)
    #
    #     # Add outer circle
    #     circle = Circle((0, 0), lims[1], fill=False, color='black', lw=1)
    #     ax.add_artist(circle)
    #
    # # --- Add Quadrant Labels ---
    # # Note: These are mirrored for left vs. right
    # text_offset = lims[1] * 0.85
    # # Right Breast Quadrants
    # ax1.text(text_offset, text_offset, 'UI', ha='center', va='center', fontsize=14)
    # ax1.text(-text_offset, text_offset, 'UO', ha='center', va='center', fontsize=14)
    # ax1.text(text_offset, -text_offset, 'LI', ha='center', va='center', fontsize=14)
    # ax1.text(-text_offset, -text_offset, 'LO', ha='center', va='center', fontsize=14)
    #
    # # Left Breast Quadrants
    # ax2.text(text_offset, text_offset, 'UO', ha='center', va='center', fontsize=14)
    # ax2.text(-text_offset, text_offset, 'UI', ha='center', va='center', fontsize=14)
    # ax2.text(text_offset, -text_offset, 'LO', ha='center', va='center', fontsize=14)
    # ax2.text(-text_offset, -text_offset, 'LI', ha='center', va='center', fontsize=14)
    #
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()
    #



# def plot_all_subjects():