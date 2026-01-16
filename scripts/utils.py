from pathlib import Path
import pyvista as pv
import numpy as np
import math
import copy
from typing import Dict, List, Tuple, Optional, Callable, Dict, Any
from skimage.segmentation import find_boundaries
from scipy.spatial import cKDTree
import external.breast_metadata_mdv.breast_metadata as breast_metadata
from breast_metadata_mdv.examples.images.visualise_image_and_mesh import align_prone_supine as aps
from collections import defaultdict
import json
import morphic
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import time
from utils_plot import plot_all, plot_vector_three_views, visualise_landmarks_distance
from scipy.spatial import KDTree
import os
import open3d as o3d


try:
    from .structures import Subject, RegistrarData, ScanData
except ImportError:
    from structures import Subject, RegistrarData, ScanData



def calculate_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Calculates the Euclidean distance between two 3D points."""
    # This check prevents errors if a landmark was None
    if point_a is None or point_b is None:
        return float('inf')
    return float(np.linalg.norm(point_a - point_b))

def copy_subject(subject: Subject) -> Subject:
    # 1. Copy the simple, top-level attributes
    new_subject = Subject(
        subject_id=subject.subject_id,
        age=subject.age,
        weight=subject.weight,
        height=subject.height
    )

    # 2. Manually copy the 'scans' dictionary
    new_scans_dict = {}
    for position, original_scan_data in subject.scans.items():
        # Create a new ScanData object
        new_scan_data = ScanData(
            position=original_scan_data.position,
            scan_object=original_scan_data.scan_object,
            anatomical_landmarks=copy.deepcopy(original_scan_data.anatomical_landmarks),
            registrar_data=copy.deepcopy(original_scan_data.registrar_data)
        )
        new_scans_dict[position] = new_scan_data

    new_subject.scans = new_scans_dict
    return new_subject

def find_corresponding_landmarks(
        all_subjects: Dict[int, Subject]
) -> Tuple[Dict[int, List[List[str]]], Dict[int, Subject]]:
    """
    Checks for corresponding landmarks between registrars "a" and "b"
    across "prone" and "supine" positions for each volunteer.

    A correspondence is valid if:
    1. A landmark from 'a' has exactly one match in 'b' in the prone position (<= 3mm).
    2. The corresponding supine landmarks (matched by name) are also within 3mm.
    3. The landmark types match (e.g., "cyst_1" from 'a' and "cyst_1" from 'b').
    4. The landmark type is not 'fibroadenoma'.

    Args:
        all_subjects: The main data dictionary from your main.py:
                      {vl_id: {"prone": SubjectData, "supine": SubjectData}}

    Returns:
        A dictionary where each volunteer ID maps to a list of corresponding
        landmark name pairs. e.g., {9: [["cyst_1", "cyst_1"], ["lymph_1", "lymph_2"]]}
    """
    print("\n--- Find corresponding landmarks between registrars ---")

    corre = {}

    # Loop over each volunteer in the main data object
    for vl_id, subject in all_subjects.items():

        # --- 1. Check if we have all 4 data quadrants ---
        if "prone" not in subject.scans or "supine" not in subject.scans:
            continue

        prone_data = subject.scans["prone"]
        supine_data = subject.scans["supine"]

        if ("anthony" not in prone_data.registrar_data or
                "holly" not in prone_data.registrar_data or
                "anthony" not in supine_data.registrar_data or
                "holly" not in supine_data.registrar_data):
            # Skip subject if missing any registrar data
            continue

        # --- 2. Get the landmark dictionaries ---
        prone_a_lms = prone_data.registrar_data["anthony"].soft_tissue_landmarks
        prone_b_lms = prone_data.registrar_data["holly"].soft_tissue_landmarks
        supine_a_lms = supine_data.registrar_data["anthony"].soft_tissue_landmarks
        supine_b_lms = supine_data.registrar_data["holly"].soft_tissue_landmarks

        corre[vl_id] = []

        # --- 3. Iterate over landmarks from registrar 'a' (prone) ---
        for lm_a_name, lm_a_prone_coord in prone_a_lms.items():

            # --- 4. Find matches in registrar 'b' (prone) ---
            prone_matches_names = []
            for lm_b_name, lm_b_prone_coord in prone_b_lms.items():
                if calculate_distance(lm_a_prone_coord, lm_b_prone_coord) <= 3:
                    prone_matches_names.append(lm_b_name)

            # Only proceed if there is exactly one match
            if len(prone_matches_names) != 1:
                continue

            lm_b_name = prone_matches_names[0]

            # --- 5. Check corresponding supine landmarks ---
            # We rely on the landmark names (e.g., "cyst_1") being stable
            # and present in both prone and supine data.
            if (lm_a_name not in supine_a_lms or
                    lm_b_name not in supine_b_lms):
                # This landmark (e.g. "cyst_1") doesn't exist in the supine
                # data for one of the registrars, so we skip it.
                continue

            lm_a_supine_coord = supine_a_lms[lm_a_name]
            lm_b_supine_coord = supine_b_lms[lm_b_name]

            # --- 6. Evaluate all conditions ---
            distance_valid = calculate_distance(lm_a_supine_coord, lm_b_supine_coord) <= 3
            print(lm_a_name,": ", calculate_distance(lm_a_supine_coord, lm_b_supine_coord))
            # Check type match. The type is the part *before* the underscore.
            # e.g., "cyst_1" -> "cyst"
            type_a = lm_a_name.split('_')[0]
            type_b = lm_b_name.split('_')[0]

            types_match = (type_a == type_b)
            types_excluded = (type_a == 'fibroadenoma')

            if distance_valid and types_match and not types_excluded:
                # This is a valid correspondence.
                # We store the landmark *names*, not the old indices.
                corre[vl_id].append([lm_a_name, lm_b_name])

        # --- 7. Remove duplicates (if any) and finalize ---
        if vl_id in corre:
            unique_correspondences = set(tuple(pair) for pair in corre[vl_id])
            corre[vl_id] = sorted([list(pair) for pair in unique_correspondences])

    # --- 8. Filter out volunteers with no correspondences ---
    # This replaces the old `remove_volunteer` logic in a non-destructive way.
    corre = {k: v for k, v in corre.items() if v}

    all_subjects_filtered = {}

    for vl_id, corr_pairs in corre.items():
        original_subject = all_subjects[vl_id]
        filtered_subject = copy_subject(original_subject)

        matching_lm_names_r1 = {pair[0] for pair in corr_pairs}
        matching_lm_names_r2 = {pair[1] for pair in corr_pairs}

        for position, scan in filtered_subject.scans.items():
            original_reg_data = original_subject.scans[position].registrar_data
            scan.registrar_data = {}  # Clear all registrar data

            # Filter Registrar 1
            if "anthony" in original_reg_data:
                original_landmarks_r1 = original_reg_data["anthony"].soft_tissue_landmarks
                filtered_landmarks_r1 = {
                    lm_name: lm_coord for lm_name, lm_coord in original_landmarks_r1.items()
                    if lm_name in matching_lm_names_r1
                }
                if filtered_landmarks_r1:
                    scan.registrar_data["anthony"] = RegistrarData(
                        soft_tissue_landmarks=filtered_landmarks_r1
                    )

            # Filter Registrar 2
            if "holly" in original_reg_data:
                original_landmarks_r2 = original_reg_data["holly"].soft_tissue_landmarks
                filtered_landmarks_r2 = {
                    lm_name: lm_coord for lm_name, lm_coord in original_landmarks_r2.items()
                    if lm_name in matching_lm_names_r2
                }
                if filtered_landmarks_r2:
                    scan.registrar_data["holly"] = RegistrarData(
                        soft_tissue_landmarks=filtered_landmarks_r2
                    )

        all_subjects_filtered[vl_id] = filtered_subject

    return corre, all_subjects_filtered


def add_averaged_landmarks(all_subjects_filtered, correspondences):
    averaged_subjects = {}
    for vl_id, subj in all_subjects_filtered.items():
        pairs = correspondences.get(vl_id, [])
        subj_copy = copy_subject(subj)
        for pos, scan in subj_copy.scans.items():
            averaged_landmarks = {}
            orig_reg = all_subjects_filtered[vl_id].scans[pos].registrar_data
            r1 = orig_reg.get('anthony')
            r2 = orig_reg.get('holly')
            if not r1 or not r2:
                continue
            for name_a, name_b in pairs:
                a = r1.soft_tissue_landmarks.get(name_a)
                b = r2.soft_tissue_landmarks.get(name_b)
                if a is None or b is None:
                    continue
                a_arr = np.asarray(a, dtype=float)
                b_arr = np.asarray(b, dtype=float)
                averaged_landmarks[name_a] = (a_arr + b_arr) / 2.0
            if averaged_landmarks:
                # Preserve original registrars (if present) and add 'average'
                new_reg = {}
                if 'anthony' in orig_reg:
                    new_reg['anthony'] = orig_reg['anthony']
                if 'holly' in orig_reg:
                    new_reg['holly'] = orig_reg['holly']
                # Add averaged landmarks under the 'average' key
                new_reg['average'] = RegistrarData(soft_tissue_landmarks=averaged_landmarks)
                scan.registrar_data = new_reg
            else:
                scan.registrar_data = {}
        averaged_subjects[vl_id] = subj_copy
    return averaged_subjects


def generate_image_coordinates(image_shape, spacing):
    x, y, z = np.mgrid[0:image_shape[0],0:image_shape[1],0:image_shape[2]]
    x = x*spacing[0]
    y = y*spacing[1]
    z = z*spacing[2]
    image_coor = np.vstack((x.ravel(),y.ravel(),z.ravel())).transpose()
    return image_coor, x, y, z


def extract_contour_points(mask, nb_points):
    labels = mask.values.copy()
    boundaries = find_boundaries(labels,mode = 'inner').astype(np.uint8)
    image_coordinates,x,y,z = generate_image_coordinates(labels.shape,mask.spacing)
    points = np.array(image_coordinates+mask.origin)
    points =points[np.array( boundaries.ravel()).astype(bool),:]
    if (nb_points < len(points)):
        step = math.trunc(len(points)/nb_points)
        indx = range(0,len(points), step)
        return points[indx,:]
    else:
        return points


def calculate_landmark_distances(
        all_subjects: Dict[int, Subject],
        masks_path: Path
) -> Dict:
    """
    Calculates the shortest distance from all soft-tissue landmarks to
    the skin and rib masks.

    This function uses a k-d tree for efficient nearest-neighbor lookup.


    Args:
        all_subjects: The main dictionary of loaded Subject objects.
        masks_path: The primary path to the NIFTI masks folder.

    Returns:
        A nested dictionary with the results:
        {
            vl_id: {
                "prone": {
                    "anthony": {
                        "skin_distances": {"cyst_1": 10.2, ...},
                        "skin_points": {"cyst_1": [x,y,z], ...},
                        "rib_distances": {...},
                        "rib_points": {...}
                    },
                    "holly": {...}
                },
                "supine": {...}
            }
        }
    """
    print("\n--- Calculating nearest distances between landmarks and nipples, rib, and skin ---")

    all_results = {}

    for vl_id, subject in all_subjects.items():
        all_results[vl_id] = {}
        vl_id_str_formatted = f"VL{vl_id:05d}"  # e.g., "VL00009"

        for position, scan in subject.scans.items():
            all_results[vl_id][position] = {}

            # --- 1. Find and Load Skin Mask ---
            skin_mask_path = masks_path / position / "body" / f"body_{vl_id_str_formatted}.nii.gz"

            skin_kd_tree = None
            if skin_mask_path.exists():
                skin_mask = breast_metadata.readNIFTIImage(str(skin_mask_path), 'RAI', True)
                skin_points = extract_contour_points(skin_mask, 100000)
                skin_kd_tree = cKDTree(skin_points)
            else:
                print(f"Skin mask not found for {vl_id_str_formatted} ({position}). Skipping.")
                continue

            # # for debug
            # plotter = pv.Plotter()
            # skin_mask_grid = breast_metadata.SCANToPyvistaImageGrid(skin_mask, 'RAI')
            # skin_mask_threshold = skin_mask_grid.threshold(value=0.5)
            # plotter.add_mesh(skin_mask_threshold, color='lightskyblue', opacity=0.2, show_scalar_bar=False)
            # plotter.add_points(skin_points,color='blue', point_size=1, render_points_as_spheres=True)
            # plotter.show()

            # --- 2. Find and Load Rib Mask ---
            rib_mask_path = masks_path / position / "rib_cage" / f"rib_cage_{vl_id_str_formatted}.nii.gz"

            rib_kd_tree = None
            if rib_mask_path.exists():
                rib_mask = breast_metadata.readNIFTIImage(str(rib_mask_path), 'RAI', True)
                rib_points = extract_contour_points(rib_mask, 100000)
                rib_kd_tree = cKDTree(rib_points)
            else:
                print(f"Rib mask not found for {vl_id_str_formatted} ({position}). Skipping.")
                continue  # Skip this whole position if no masks

            # --- 3. Iterate Registrars and Landmarks ---
            for registrar_name, reg_data in scan.registrar_data.items():

                # Prepare dictionaries to store results for this registrar
                distances_skin = {}
                closest_points_skin = {}
                distances_rib = {}
                closest_points_rib = {}
                skin_neighborhood_avg = {}
                rib_neighborhood_avg = {}

                soft_tissue_landmarks_dict = reg_data.soft_tissue_landmarks
                for lm_name, lm_coord in soft_tissue_landmarks_dict.items():
                    # Query Skin KDTree
                    skin_distance, skin_index = skin_kd_tree.query(lm_coord)
                    distances_skin[lm_name] = skin_distance
                    closest_point_on_skin = skin_points[skin_index]
                    closest_points_skin[lm_name] = closest_point_on_skin

                    skin_neighborhood_distances, _ = skin_kd_tree.query(closest_point_on_skin, k=11)
                    skin_avg_distance = np.mean(skin_neighborhood_distances[1:])
                    skin_neighborhood_avg[lm_name] = skin_avg_distance

                    # Query Rib KDTree
                    rib_distance, rib_index = rib_kd_tree.query(lm_coord)
                    distances_rib[lm_name] = rib_distance
                    closest_point_on_rib = rib_points[rib_index]
                    closest_points_rib[lm_name] = closest_point_on_rib

                    rib_neighborhood_distances, _ = rib_kd_tree.query(closest_point_on_rib, k=11)
                    rib_avg_distance = np.mean(rib_neighborhood_distances[1:])
                    rib_neighborhood_avg[lm_name] = rib_avg_distance

                # Store all results for this registrar
                all_results[vl_id][position][registrar_name] = {
                    "skin_distances": distances_skin,
                    "skin_points": closest_points_skin,
                    "rib_distances": distances_rib,
                    "rib_points": closest_points_rib,
                    "skin_neighborhood_avg": skin_neighborhood_avg,
                    "rib_neighborhood_avg": rib_neighborhood_avg
                }

            # visualise_landmarks_distance(vl_id,position,skin_mask, rib_mask, scan.registrar_data["anthony"].soft_tissue_landmarks, all_results)

    return all_results



def analyse_landmark_distances(
        distance_results: Dict
) -> Dict:
    """
    Analyses landmark distances AND local mask densities.
    """

    all_skin_distances = []
    all_rib_distances = []

    all_skin_neighborhood_avgs = []
    all_rib_neighborhood_avgs = []

    try:
        for vl_id, positions in distance_results.items():
            for position, data_for_pos in positions.items():

                # Loop through registrars for landmark-specific data
                for registrar_name, data in data_for_pos.items():
                    if registrar_name in ['anthony', 'holly']:
                        if isinstance(data, dict):
                            all_skin_distances.extend(data.get('skin_distances', {}).values())
                            all_rib_distances.extend(data.get('rib_distances', {}).values())
                            all_skin_neighborhood_avgs.extend(data.get('skin_neighborhood_avg', {}).values())
                            all_rib_neighborhood_avgs.extend(data.get('rib_neighborhood_avg', {}).values())

    except Exception as e:
        print(f"Error analyzing results: {e}. Dictionary may be in wrong format.")
        return {}

    # Calculate stats
    avg_skin = np.mean(all_skin_distances) if all_skin_distances else 0.0
    std_skin = np.std(all_skin_distances) if all_skin_distances else 0.0
    avg_rib = np.mean(all_rib_distances) if all_rib_distances else 0.0
    std_rib = np.std(all_rib_distances) if all_rib_distances else 0.0
    avg_skin_neighborhood = np.mean(all_skin_neighborhood_avgs) if all_skin_neighborhood_avgs else 0.0
    avg_rib_neighborhood = np.mean(all_rib_neighborhood_avgs) if all_rib_neighborhood_avgs else 0.0

    return {
        "landmark_to_skin_avg": avg_skin,
        "landmark_to_skin_std": std_skin,
        "landmark_to_rib_avg": avg_rib,
        "landmark_to_rib_std": std_rib,
        "landmark_count": len(all_skin_distances),
        "mask_skin_neighborhood_avg": avg_skin_neighborhood,
        "mask_rib_neighborhood_avg": avg_rib_neighborhood
    }


def calculate_clockface_coordinates(
        all_subjects: Dict[int, Subject]
) -> Dict:
    """
    Calculates the clock-face position, quadrant, and distance to the nipple
    for all soft-tissue landmarks.

    The "clock" is viewed from the front (anterior), where:
    - 12 o'clock is Superior (+Z)
    - 3 o'clock is Right (-X)
    - 6 o'clock is Inferior (-Z)
    - 9 o'clock is Left (+X)

    Args:
        all_subjects: The main dictionary of loaded Subject objects.

    Returns:
        A nested dictionary with the results:
        {
            vl_id: {
                "prone": {
                    "anthony": {
                        "cyst_1": {"side": "LB", "time": "1:00",
                                   "quadrant": "UO", "dist_to_nipple": 15.2},
                        ...
                    }, ...
                }, ...
            }
        }
    """

    all_results = defaultdict(dict)

    for vl_id, subject in all_subjects.items():
        for position, scan in subject.scans.items():

            all_results[vl_id][position] = defaultdict(dict)
            anat_landmarks = scan.anatomical_landmarks

            # --- 1. Validation ---
            # We need nipples and a sternum point for the midline.
            if not (anat_landmarks.nipple_left is not None and
                    anat_landmarks.nipple_right is not None and
                    anat_landmarks.sternum_superior is not None):
                print(f"Skipping clockface for VL{vl_id} ({position}): Missing core anatomical landmarks.")
                continue

            # Use superior sternum as the midline
            midline_x = anat_landmarks.sternum_superior[0]

            for registrar_name, reg_data in scan.registrar_data.items():
                for lm_name, landmark in reg_data.soft_tissue_landmarks.items():

                    # --- 2. Determine Breast Side (RAI) ---
                    # Right is -X, Left is +X
                    if landmark[0] > midline_x:
                        side = 'LB'
                        nipple = anat_landmarks.nipple_left
                    else:
                        side = 'RB'
                        nipple = anat_landmarks.nipple_right

                    # Skip if the determined nipple is missing
                    if nipple is None:
                        continue

                    # --- 3. Calculate Distance to Nipple ---
                    dist_to_nipple = float(np.linalg.norm(landmark - nipple))

                    # --- 4. Calculate Clock Time (RAI) ---
                    # Relative coordinates in the Coronal (XZ) plane
                    x_rel = landmark[0] - nipple[0]  # Right/Left axis
                    z_rel = landmark[2] - nipple[2]  # Inferior/Superior axis

                    dist_xz = np.sqrt(x_rel ** 2 + z_rel ** 2)

                    if dist_xz <= 10:
                        clock = 'central'
                    else:
                        # "Up" (12 o'clock) is Z direction, so y_arg = x_rel
                        # "Right" (3 o'clock) is -X direction, so x_arg = z_rel
                        angle_rad = np.arctan2(x_rel, z_rel)

                        # Convert radians to hours
                        hour_decimal = 6 * angle_rad / math.pi
                        if hour_decimal < 0:
                            hour_decimal = 12 + hour_decimal

                        # This rounding logic is copied from your old function
                        whole_hour = math.floor(hour_decimal)
                        min_frac = hour_decimal - whole_hour
                        min_val = min_frac * 60.

                        if whole_hour == 0: whole_hour = 12

                        if min_val < 15:
                            clock = f"{int(whole_hour)}:00"
                        elif min_val < 45:
                            clock = f"{int(whole_hour)}:30"
                        else:
                            whole_hour += 1
                            if whole_hour == 13: whole_hour = 1
                            clock = f"{int(whole_hour)}:00"


                    # --- 5. Determine Quadrant ---
                    if clock == 'central':
                        quadrant = 'central'
                    else:
                        # Vertical: Superior is +Z
                        q_v = 'U' if landmark[2] > nipple[2] else 'L'

                        # Horizontal: Inner (Medial) vs Outer (Lateral)
                        if side == 'LB':  # Left Breast (+X)
                            q_h = 'O' if landmark[0] > nipple[0] else 'I'
                        else:  # Right Breast (-X)
                            q_h = 'O' if landmark[0] < nipple[0] else 'I'

                        quadrant = q_v + q_h

                    # --- 6. Store Results ---
                    all_results[vl_id][position][registrar_name][lm_name] = {
                        "side": side,
                        "time": clock,
                        "quadrant": quadrant,
                        "dist_to_nipple": dist_to_nipple
                    }

    # Convert defaultdicts to regular dicts for cleaner output
    return all_results


def get_landmarks_as_array(scan: ScanData, registrar_name: str) -> np.ndarray:
    """
    Safely extracts soft-tissue landmarks from a scan object for a
    given registrar and returns them as a clean (N, 3) NumPy array.
    """
    # 1. Check if registrar data exists
    if registrar_name not in scan.registrar_data:
        print(f"Warning: Registrar '{registrar_name}' not found in {scan.position} scan.")
        return np.empty((0, 3))  # Return an empty, correctly-shaped array

    # 2. Get the dictionary of landmarks
    landmarks_dict = scan.registrar_data[registrar_name].soft_tissue_landmarks
    if not landmarks_dict:
        return np.empty((0, 3))  # No landmarks

    # 3. Filter for valid points and convert to array in one step
    filtered_list = [p for p in landmarks_dict.values() if p is not None and p.shape == (3,)]

    if not filtered_list:
        return np.empty((0, 3))  # No *valid* landmarks

    return np.array(filtered_list)



def apply_transform(points, T):
    if points.shape[0] == 0:
        return np.empty((0, 3))  # Return empty if no points
    ones = np.ones((len(points), 1))
    return (T @ np.hstack((points, ones)).T)[:-1, :].T


def plot_evaluate_alignment(
        supine_pts: np.ndarray,
        transformed_prone_mesh,
        distances: np.ndarray = None,
        idxs: np.ndarray = None,
        worst_n: int = 50,
        cmap: str = "viridis",
        point_size: int = 5,
        arrow_scale: float = 1.0,
        show_scalar_bar: bool = True,
        return_data: bool = False
):
    """
    Visualise the supine point cloud colored by nearest-neighbour distance to the
    transformed prone mesh/point-cloud and draw arrows for the worst N points.

    Parameters:
    - supine_pts: (M,3) numpy array of supine point coordinates.
    - transformed_prone_pts_or_mesh: either a (K,3) numpy array of transformed
      prone points, or a PyVista mesh/PolyData object already transformed into
      the supine frame.
    - distances: .
    - worst_n: number of worst points to draw arrows for.
    - cmap, point_size, arrow_scale: visual params.
    - show_scalar_bar: whether to show the scalar bar for distances.
    - return_data: if True, returns a dict with computed distances and nearest indices.

    Returns:
    - If return_data is True: {"distances": distances, "nearest_indices": idxs}
      otherwise returns None.

    Notes:
    - This function does not modify any global state. It uses PyVista for
      interactive 3D rendering.
    """
    # Validate inputs
    supine_pts = np.asarray(supine_pts)
    if supine_pts.ndim != 2 or supine_pts.shape[1] != 3:
        raise ValueError("supine_pts must be shape (N,3)")

    # Resolve transformed prone coordinates array for KD-tree and arrows
    prone_pts_for_tree = None
    prone_mesh_for_plot = None
    try:
        # If the caller passed a PyVista object, extract points for KDTree
        if isinstance(transformed_prone_mesh, pv.PolyData) or isinstance(transformed_prone_mesh, pv.UnstructuredGrid):
            prone_mesh_for_plot = transformed_prone_mesh
            prone_pts_for_tree = np.asarray(prone_mesh_for_plot.points)
        else:
            prone_pts_for_tree = np.asarray(transformed_prone_mesh)
    except Exception:
        prone_pts_for_tree = np.asarray(transformed_prone_mesh)

    if prone_pts_for_tree is None or len(prone_pts_for_tree) == 0:
        raise ValueError("transformed_prone_pts_or_mesh contains no points")

    # Create PyVista objects
    sup_pd = pv.PolyData(supine_pts)
    sup_pd["dist"] = distances

    plotter = pv.Plotter()

    # Add supine points coloured by distance
    clim = (0.0, float(np.percentile(distances, 99))) if len(distances) > 0 else None
    plotter.add_points(
        sup_pd,
        scalars="dist",
        cmap=cmap,
        point_size=point_size,
        render_points_as_spheres=True,
        lighting=False,
        scalar_bar_args={"title": "Distance (mm)"} if show_scalar_bar else None,
        clim=clim
    )

    # Overlay prone mesh or points
    if prone_mesh_for_plot is not None:
        # semi-transparent mesh if available
        try:
            plotter.add_mesh(prone_mesh_for_plot, color="tan", opacity=0.5, label="Transformed prone mesh")
        except Exception:
            # fallback to points
            plotter.add_points(prone_pts_for_tree, color="tan", point_size=3, render_points_as_spheres=True)
    else:
        plotter.add_points(prone_pts_for_tree, color="tan", point_size=3, render_points_as_spheres=True)

    # Draw arrows for the worst N supine points (largest distances)
    worst_n_use = min(worst_n, supine_pts.shape[0])
    if worst_n_use > 0:
        worst_idx = np.argsort(distances)[-worst_n_use:]
        worst_targets = supine_pts[worst_idx]
        nearest_prone_pts = prone_pts_for_tree[idxs[worst_idx]]
        vectors = nearest_prone_pts - worst_targets

        # Build a PolyData of the worst target locations and attach vectors
        worst_pd = pv.PolyData(worst_targets)
        # Ensure vectors shape is (N,3)
        worst_pd["vecs"] = vectors
        # Create glyph arrows from the vectors
        try:
            arrows = worst_pd.glyph(orient="vecs", scale=False, factor=arrow_scale)
            plotter.add_mesh(arrows, color="red", label=f"Top {worst_n_use} errors")
        except Exception:
            # If glyph fails (e.g. very small vectors), add simple line segments instead
            for p, v in zip(worst_targets, vectors):
                line = pv.Line(p, p + v * arrow_scale)
                plotter.add_mesh(line, color="red")

    plotter.add_axes()
    plotter.add_legend()

    cam_pos = [
        [378.6543811782509, 309.86957859927173, 490.0925163131904],
        [-35.140107028421866, 15.381214203758844, -17.515439847958362],
        [-0.5214522629375, -0.48151960895256674, 0.7044333919339199]
    ]
    plotter.camera_position = cam_pos

    plotter.show()

    if return_data:
        return {"distances": distances, "nearest_indices": idxs}


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
#

# def clean_ribcage_point_cloud(
#         pc_data: np.ndarray,
#         image_grid: pv.ImageData,
#         config_filepath: Optional[str] = None,  # File path for JSON configuration
#         asym_filter_func: Optional[Callable] = None,
#         # --- Geometric Cropping Parameters (Default values) ---
#         z_voxels_bottom: float = 20,
#         z_voxels_top: float = 5,
#         x_spine_offset: float = 25.0,
#         y_spine_offset: float = 60.0,
#         x_lateral_margin: float = 90.0,
#         y_posterior_margin: float = 60.0,
#         z_inferior_margin: float = 25.0,
#         # --- Statistical Outlier Removal (SOR) Parameters (Default values) ---
#         sor_k_neighbors: int = 50,
#         sor_std_multiplier: float = 1.0,
#         run_plot_all: bool = True
# ) -> np.ndarray:
#     # ----------------------------------------------------
#     # I. Configuration Loading and Setup
#     # ----------------------------------------------------
#     params = {
#         'z_voxels_bottom': z_voxels_bottom,
#         'z_voxels_top': z_voxels_top,
#         'x_spine_offset': x_spine_offset,
#         'y_spine_offset': y_spine_offset,
#         'x_lateral_margin': x_lateral_margin,
#         'y_posterior_margin': y_posterior_margin,
#         'z_inferior_margin': z_inferior_margin,
#         'sor_k_neighbors': sor_k_neighbors,
#         'sor_std_multiplier': sor_std_multiplier
#     }
#
#     if config_filepath and os.path.exists(config_filepath):
#         # Load parameters from file if it exists
#         try:
#             with open(config_filepath, 'r') as f:
#                 loaded_params = json.load(f)
#             # Update function's local variables with loaded values
#             params.update(loaded_params)
#             print(f"INFO: Loaded parameters from {config_filepath}")
#         except Exception as e:
#             print(f"WARNING: Could not load parameters from file: {e}. Using defaults/provided args.")
#
#     # Update local variables for use in the rest of the function
#     z_voxels_bottom = params['z_voxels_bottom']
#     z_voxels_top = params['z_voxels_top']
#     x_spine_offset = params['x_spine_offset']
#     y_spine_offset = params['y_spine_offset']
#     x_lateral_margin = params['x_lateral_margin']
#     y_posterior_margin = params['y_posterior_margin']
#     z_inferior_margin = params['z_inferior_margin']
#     sor_k_neighbors = params['sor_k_neighbors']
#     sor_std_multiplier = params['sor_std_multiplier']
#
#     # ----------------------------------------------------
#     # II. Filtering Logic
#     # ----------------------------------------------------
#     print("original point cloud size:", pc_data.shape)
#     if run_plot_all:
#         plot_all(point_cloud=pc_data)
#
#     # supine_ribcage_pc = supine_ribcage_pc[supine_ribcage_pc[:, 0] > -120.]
#     # print("remove outlier near arm, supine ribcage point cloud size:", supine_ribcage_pc.shape)
#     # plot_all(point_cloud=supine_ribcage_pc)
#
#     # remove points on the top and bottom axial slices
#     pc_data = filter_point_cloud_asymmetric(
#         points=pc_data,
#         reference=pc_data,
#         tol_min=image_grid.spacing[2] * z_voxels_bottom,  # Remove 20 voxels from the bottom/MIN side
#         tol_max=image_grid.spacing[2] * z_voxels_top,  # Remove 5 voxels from the top/MAX side
#         axis=2
#     )
#
#     print("remove top and bottom axial slices, point cloud size:", pc_data.shape)
#     if run_plot_all:
#         plot_all(point_cloud=pc_data)
#
#     # remove points on the back side of the ribcage around the spine
#     x_median = np.median(pc_data[:, 0])
#     pc_data = pc_data[
#         (pc_data[:, 0] <= (x_median - x_spine_offset)) |
#         (pc_data[:, 0] >= (x_median + x_spine_offset)) |
#         (pc_data[:, 1] < np.max(pc_data[:, 1]) - y_spine_offset)
#         ]
#
#     print("remove spine region, point cloud size:", pc_data.shape)
#     if run_plot_all:
#         plot_all(point_cloud=pc_data)
#
#     # x_offset = 100.
#     # y_median = np.median(supine_ribcage_pc[:, 1])
#     # y_offset = 50.
#     # supine_ribcage_pc = supine_ribcage_pc[
#     # (supine_ribcage_pc[:, 0] <= (x_median - x_offset)) |
#     # (supine_ribcage_pc[:, 0] >= (x_median + x_offset)) |
#     # (supine_ribcage_pc[:, 1] <= (y_median - y_offset)) |
#     # (supine_ribcage_pc[:, 1] >= (y_median + y_offset))
#     # ]
#     #
#     # plot_all(point_cloud=supine_ribcage_pc)
#
#
#     pc_data = pc_data[
#         (pc_data[:, 0] <= (x_median - x_lateral_margin)) |
#         (pc_data[:, 0] >= (x_median + x_lateral_margin)) |
#         (pc_data[:, 1] < np.max(pc_data[:, 1]) - y_posterior_margin) |
#         (pc_data[:, 2] >= (np.min(pc_data[:, 2] + z_inferior_margin)))
#         ]
#
#     print("remove posterior inferior points, point cloud size:", pc_data.shape)
#     if run_plot_all:
#         plot_all(point_cloud=pc_data)
#
#
#     point_cloud_data = pc_data.copy()
#     # 1. Build a KD-Tree for fast nearest neighbor lookups
#     try:
#         tree = KDTree(point_cloud_data)
#     except Exception as e:
#         print(f"ERROR: Could not build KDTree. Ensure point_cloud_data is a valid (N, 3) NumPy array. Error: {e}")
#         return point_cloud_data
#
#     # 2. Find the distance to the K-th nearest neighbor for every point
#     # Query K+1 because the first neighbor is the point itself (distance 0).
#     distances, _ = tree.query(point_cloud_data, k=sor_k_neighbors + 1)
#     kth_dist = distances[:, sor_k_neighbors]
#
#     # 3. Calculate the statistical threshold (Mean + alpha * StdDev)
#     mean_dist = np.mean(kth_dist)
#     std_dev = np.std(kth_dist)
#     threshold = mean_dist + sor_std_multiplier * std_dev
#
#     # 4. Filter the points
#     # Keep points whose k-th neighbor distance is SMALLER than the calculated threshold
#     inliers_mask = kth_dist < threshold
#     point_cloud_data_filtered = point_cloud_data[inliers_mask]
#
#     # Update the original variable
#     pc_data = point_cloud_data_filtered
#     print("remove outliers, point cloud size:", pc_data.shape)
#     if run_plot_all:
#         plot_all(point_cloud=pc_data)
#
#     # ----------------------------------------------------
#     # III. Configuration Saving
#     # ----------------------------------------------------
#     if config_filepath and not os.path.exists(config_filepath):
#         # Save the parameters used (including loaded ones) only if the file didn't exist initially
#         try:
#             config_dir = os.path.dirname(config_filepath)
#             if config_dir and not os.path.exists(config_dir):
#                 # Create the directory and any necessary parent directories
#                 os.makedirs(config_dir, exist_ok=True)
#             with open(config_filepath, 'w') as f:
#                 json.dump(params, f, indent=4)
#             print(f"INFO: Saved current parameters to {config_filepath}")
#
#         except Exception as e:
#             print(f"WARNING: Could not save parameters to file: {e}")
#
#     return pc_data


def align_prone_to_supine(
        subject: Subject,
        prone_ribcage_mesh_path: Path,
        supine_ribcage_seg_path: Path,
        orientation_flag: str = 'RAI',
        plot_for_debug: bool = False
) -> dict:
    """
    Adapts the alignment function to work with the new Subject data structure.
    Aligns a prone ribcage mesh to a supine ribcage segmentation.

    Returns a dictionary containing all transformation results and statistics.
    """

    # ==========================================================
    # %% LOAD DATA
    # ==========================================================

    # --- 1. Get Prone and Supine Scans from Subject ---
    if "prone" not in subject.scans or "supine" not in subject.scans:
        raise ValueError(f"Subject {subject.subject_id} is missing prone or supine scan data.")

    # prone_scan = subject.scans["prone"].scan_object
    # supine_scan = subject.scans["supine"].scan_object

    # --- 2. Convert Scans to Pyvista Image Grids ---
    # prone_image_grid = breast_metadata.SCANToPyvistaImageGrid(prone_scan, orientation_flag)
    # supine_image_grid = breast_metadata.SCANToPyvistaImageGrid(supine_scan, orientation_flag)

    # --- 3. Load Anatomical Landmarks from Subject ---
    anat_prone = subject.scans["prone"].anatomical_landmarks
    anat_supine = subject.scans["supine"].anatomical_landmarks

    if anat_prone.sternum_superior is None or anat_prone.sternum_inferior is None:
        raise ValueError(f"Subject {subject.subject_id} (prone) is missing sternum landmarks for alignment.")
    sternum_prone = np.vstack([anat_prone.sternum_superior, anat_prone.sternum_inferior])

    if anat_supine.sternum_superior is None or anat_supine.sternum_inferior is None:
        raise ValueError(f"Subject {subject.subject_id} (supine) is missing sternum landmarks for alignment.")
    sternum_supine = np.vstack([anat_supine.sternum_superior, anat_supine.sternum_inferior])

    nipple_prone = np.vstack([anat_prone.nipple_left, anat_prone.nipple_right])
    nipple_supine = np.vstack([anat_supine.nipple_left, anat_supine.nipple_right])

    # --- 4. Load Registrar Landmarks from Subject ---
    # --- Get the scan objects once ---
    prone_scan_data = subject.scans["prone"]
    supine_scan_data = subject.scans["supine"]

    # --- Use the helper function to get clean arrays ---
    # landmark_prone_r1_raw = get_landmarks_as_array(prone_scan_data, "anthony")
    # landmark_supine_r1_raw = get_landmarks_as_array(supine_scan_data, "anthony")
    # landmark_prone_r2_raw = get_landmarks_as_array(prone_scan_data, "holly")
    # landmark_supine_r2_raw = get_landmarks_as_array(supine_scan_data, "holly")
    landmark_prone_ave_raw = get_landmarks_as_array(prone_scan_data, "average")
    landmark_supine_ave_raw = get_landmarks_as_array(supine_scan_data, "average")

    # --- 5. Load Ribcage Mesh and Mask ---
    prone_ribcage = morphic.Mesh(str(prone_ribcage_mesh_path))
    prone_ribcage_mesh_coords = aps.get_surface_mesh_coords(prone_ribcage, res=26)

    supine_ribcage_mask = breast_metadata.readNIFTIImage(str(supine_ribcage_seg_path), orientation_flag, swap_axes=True)

    supine_ribcage_pc = extract_contour_points(supine_ribcage_mask, 20000)

    supine_ribcage_pc = filter_point_cloud_asymmetric(
        points=supine_ribcage_pc,
        reference=supine_ribcage_pc,
        tol_min=0,  # Remove 20 voxels from the bottom/MIN side
        tol_max=5,  # Remove 5 voxels from the top/MAX side
        axis=2
    )

    # # --- 6. Clean up Supine Point Cloud ---
    # vl_id_str = subject.subject_id
    # supine_ribcage_pc = clean_ribcage_point_cloud(
    #     pc_data=supine_ribcage_pc,
    #     image_grid=supine_image_grid,
    #     config_filepath=f'../output/config/{vl_id_str}_config.json',
    #     asym_filter_func=filter_point_cloud_asymmetric,
    #     # --- Geometric Cropping Parameters (Default values) ---
    #     z_voxels_bottom=20,
    #     z_voxels_top=5,
    #     x_spine_offset=25.0,
    #     y_spine_offset=60.0,
    #     x_lateral_margin=90.0,
    #     y_posterior_margin=60.0,
    #     z_inferior_margin=25.0,
    #     # --- Statistical Outlier Removal (SOR) Parameters (Default values) ---
    #     sor_k_neighbors=50,
    #     sor_std_multiplier=1.0,
    #     run_plot_all=True
    # )


    # ==========================================================
    # %% INITIAL POINT TO POINT ALIGNMENT
    # ==========================================================
    rot_angle_init = [0., 0., 0.]
    translation_init = list(
        breast_metadata.find_centroid(sternum_supine.T) - breast_metadata.find_centroid(sternum_prone.T))
    T_init = rot_angle_init + translation_init

    #   First iteration: optimise transformation matrix by performing kd-tree of point clouds between
    #   the landmarks in prone and supine
    print("\nInitial point to point alignment\n============")
    prone_points = [prone_ribcage_mesh_coords, sternum_prone]
    supine_points = [supine_ribcage_pc, sternum_supine]
    # T_optimal is transformation matrix:  the upper-left 3×3 submatrix represents rotation and scaling,
    # and the last column represents translation
    T_optimal, res_optimal = breast_metadata.run_optimisation(breast_metadata.combined_objective_function, T_init,
                                                              prone_points, supine_points)
    print("success:", res_optimal.success)
    print("message:", res_optimal.message)

    # %% APPLY TRANSFORMATION AND EVALUATE FIT
    print("\nEvaluate initial alignment\n============")
    prone_ribcage_mesh_transformed = apply_transform(prone_ribcage_mesh_coords, T_optimal)
    prone_sternum_transformed = apply_transform(sternum_prone, T_optimal)

    # evaluate sternum fit
    error, mapped_idx = breast_metadata.closest_distances(sternum_supine, prone_sternum_transformed)
    sternum_error = np.linalg.norm(error, axis=1)
    print(f"Sternum initial alignment error: {sternum_error} mm")

    # evaluate ribcage fit
    error, mapped_idx = breast_metadata.closest_distances(supine_ribcage_pc, prone_ribcage_mesh_transformed)
    rib_error_mag = np.linalg.norm(error, axis=1)
    print(f"Mean ribcage initial alignment error: {np.mean(rib_error_mag)} mm")
    # # show statistics and distribution of projection errors
    # aps.summary_stats(rib_error_mag)
    # aps.plot_histogram(rib_error_mag, 5)

    if plot_for_debug:
        plot_all(point_cloud=supine_ribcage_pc, mesh_points=prone_ribcage_mesh_transformed)

    # Visualise supine point-cloud coloured by nearest distance to the transformed prone mesh
    # and draw arrows at the worst residual points.

    # try:
    #     plot_evaluate_alignment(
    #         supine_pts=supine_ribcage_pc,
    #         transformed_prone_mesh=prone_ribcage_mesh_transformed,
    #         distances=rib_error_mag,
    #         idxs=mapped_idx,
    #         worst_n=60,
    #         cmap="viridis",
    #         point_size=3,
    #         arrow_scale=20,
    #         show_scalar_bar=True,
    #         return_data=False
    #     )
    # except Exception as e:
    #     print(f"Could not visualise ribcage errors: {e}")


    # ==========================================================
    # %% REFINE WITH POINT-TO-PLANE ICP
    # ==========================================================
    print("\nRefine with point to plane alignment\n============")

    target_pts = np.asarray(prone_ribcage_mesh_transformed, dtype=np.float64)
    source_pts = np.asarray(supine_ribcage_pc, dtype=np.float64)
    # run Open3D point-to-plane ICP to refine prone -> supine alignment
    T_icp, supine_ribcage_refined, icp_result = run_point_to_plane_icp(
        source_pts=source_pts,
        target_pts=target_pts,
        max_correspondence_distance=10.0,
        max_iterations=200,
        delta=1.0
    )

    if plot_for_debug:
        plot_all(point_cloud=supine_ribcage_refined, mesh_points=prone_ribcage_mesh_transformed)

    # update the prone point cloud for subsequent plotting and evaluation
    supine_ribcage_refined_inlier = icp_result['inlier_source_pts']

    if plot_for_debug:
        plot_all(point_cloud=supine_ribcage_refined_inlier, mesh_points=prone_ribcage_mesh_transformed)

    print("\nEvaluate point to plane alignment\n============")
    if icp_result is not None:
        try:
            print(f"ICP refinement: fitness={icp_result['fitness']:.4f}, inlier_rmse={icp_result['inlier_rmse']:.4f}")
        except Exception:
            print("ICP refinement completed.")

    # evaluate ribcage fit
    error, mapped_idx = breast_metadata.closest_distances(supine_ribcage_refined,prone_ribcage_mesh_transformed)
    rib_error_mag = np.linalg.norm(error, axis=1)
    print(f"Mean ribcage alignment error (absolute): {np.mean(rib_error_mag)} mm")

    # Visualise supine point-cloud coloured by nearest distance to the transformed prone mesh
    # and draw arrows at the worst residual points.
    # try:
    #     plot_evaluate_alignment(
    #         supine_pts=supine_ribcage_refined,
    #         transformed_prone_mesh=prone_ribcage_mesh_transformed,
    #         distances=rib_error_mag,
    #         idxs=mapped_idx,
    #         worst_n=60,
    #         cmap="viridis",
    #         point_size=3,
    #         arrow_scale=20,
    #         show_scalar_bar=True,
    #         return_data=False
    #     )
    # except Exception as e:
    #     print(f"Could not visualise ribcage errors: {e}")

    # show statistics and distribution of projection errors
    print(f"\nSummary of ribcage alignment error (absolute) after ICP")
    aps.summary_stats(rib_error_mag)
    # aps.plot_histogram(rib_error_mag, 5)


    T_icp_inv = np.linalg.inv(T_icp)
    T_total = T_icp_inv @ T_optimal
    prone_ribcage_aligned_final = apply_transform(prone_ribcage_mesh_coords, T_total)
    prone_sternum_aligned_final = apply_transform(sternum_prone, T_total)

    error, mapped_idx = breast_metadata.closest_distances(sternum_supine, prone_sternum_aligned_final)
    sternum_error = np.linalg.norm(error, axis=1)
    print(f"Sternum alignment error: {sternum_error} mm")

    if plot_for_debug:
        error_check, mapped_idx = breast_metadata.closest_distances(supine_ribcage_pc, prone_ribcage_aligned_final)
        rib_error_mag_check = np.linalg.norm(error_check, axis=1)
        print(f"Mean ribcage alignment error for checking: {np.mean(rib_error_mag_check)} mm")

    # ==========================================================
    # %% Landmark displacement after alignment
    # ==========================================================

    # 1. Apply Transform to Prone Data
    # ----------------------------------------------------------
    landmark_prone_transformed = apply_transform(landmark_prone_ave_raw, T_total)
    nipple_prone_transformed = apply_transform(nipple_prone, T_total)

    # Define Reference Points (Sternum Superior)
    ref_sternum_prone = prone_sternum_aligned_final[0]
    ref_sternum_supine = sternum_supine[0]

    # 2. Calculate Displacements Relative to Sternum
    # ----------------------------------------------------------
    # This removes global chest translation, isolating soft tissue deformation.
    # Landmark positions relative to sternum
    lm_pos_prone_rel_sternum = landmark_prone_transformed - ref_sternum_prone
    lm_pos_supine_rel_sternum = landmark_supine_ave_raw - ref_sternum_supine

    # Vector: Change in position relative to sternum (Deformation Vector)
    lm_disp_rel_sternum = lm_pos_supine_rel_sternum - lm_pos_prone_rel_sternum
    lm_disp_mag_rel_sternum = np.linalg.norm(lm_disp_rel_sternum, axis=1)

    # Nipple positions relative to sternum
    nipple_pos_prone_rel_sternum = nipple_prone_transformed - ref_sternum_prone
    nipple_pos_supine_rel_sternum = nipple_supine - ref_sternum_supine

    # Vector: Nipple displacement relative to sternum
    nipple_disp_rel_sternum = nipple_pos_supine_rel_sternum - nipple_pos_prone_rel_sternum
    nipple_disp_mag_rel_sternum = np.linalg.norm(nipple_disp_rel_sternum, axis=1)

    # Extract individual nipple vectors for later masking
    nipple_disp_left_vec = nipple_disp_rel_sternum[0]
    nipple_disp_right_vec = nipple_disp_rel_sternum[1]

    # 3. Associate Landmarks with Closest Nipple (Left vs Right)
    # ----------------------------------------------------------
    # Calculate Euclidean distance to each supine nipple to determine side
    dist_to_left = np.linalg.norm(landmark_supine_ave_raw - nipple_supine[0], axis=1)
    dist_to_right = np.linalg.norm(landmark_supine_ave_raw - nipple_supine[1], axis=1)

    # Create a boolean mask where True means the landmark is on the Left breast
    is_left_breast = dist_to_left < dist_to_right

    # Assign the relevant nipple displacement vector to each landmark
    closest_nipple_disp_vec = np.where(
        is_left_breast[:, np.newaxis],
        nipple_disp_left_vec,
        nipple_disp_right_vec
    )

    # 4. Calculate Displacements Relative to Nipple
    # ----------------------------------------------------------
    # How much did the landmark move *compared* to how much the nipple moved?

    lm_disp_rel_nipple = lm_disp_rel_sternum - closest_nipple_disp_vec
    lm_disp_mag_rel_nipple = np.linalg.norm(lm_disp_rel_nipple, axis=1)

    # 5. Separate Data by Side for Plotting/Analysis
    # ----------------------------------------------------------
    left_nipple_prone_pos = nipple_prone_transformed[0]
    right_nipple_prone_pos = nipple_prone_transformed[1]

    # --- Left Breast ---
    lm_prone_left = landmark_prone_transformed[is_left_breast]
    lm_disp_left = lm_disp_rel_sternum[is_left_breast]

    # X: Initial position relative to the prone nipple (for quiver plot origin)
    X_left = lm_prone_left - left_nipple_prone_pos
    # V: Differential movement vector (Nipple vector - Landmark vector)
    V_left = lm_disp_left - nipple_disp_left_vec

    # --- Right Breast ---
    lm_prone_right = landmark_prone_transformed[~is_left_breast]
    lm_disp_right = lm_disp_rel_sternum[~is_left_breast]

    # X: Initial position relative to the prone nipple
    X_right = lm_prone_right - right_nipple_prone_pos
    # V: Differential movement vector
    V_right = lm_disp_right - nipple_disp_right_vec


    # ==========================================================
    # %% PLOT
    # ==========================================================
    title_sternum = "Landmark Displacement Relative to Sternal Superior (Jugular Notch)"
    lm_pos_left_rel_sternum = lm_pos_prone_rel_sternum[is_left_breast]
    lm_pos_right_rel_sternum = lm_pos_prone_rel_sternum[~is_left_breast]
    plot_vector_three_views(lm_pos_left_rel_sternum, lm_disp_left,
                            lm_pos_right_rel_sternum, lm_disp_right, title_sternum)

    title_nipple = "Landmark Displacement Relative to Nipples"
    plot_vector_three_views(X_left, V_left, X_right, V_right, title_nipple)

    ''''
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
    sternum_prone_px = get_px_coords(prone_scan_transformed, prone_sternum_aligned_final)
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
    plotter.add_points(prone_sternum_aligned_final, render_points_as_spheres=True, color='black', point_size=10,
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

    # %%   scalar colour map (red-blue) to show alignment of prone transformed and supine MRIs
    breast_metadata.visualise_alignment_with_landmarks(
        supine_image_sitk, prone_image_transformed, sternum_supine_px[0], sternum_prone_px[0],
        orientation='axial')
    breast_metadata.visualise_alignment_with_landmarks(
        supine_image_sitk, prone_image_transformed, sternum_supine_px[1], sternum_prone_px[1],
        orientation='axial')

    # Loop through each landmark and create a visualization
    for i in range(len(lm_supine_px)):
        print(f"Visualizing alignment for landmark #{i + 1}")
        breast_metadata.visualise_alignment_with_landmarks(
            supine_image_sitk,
            prone_image_transformed,
            lm_supine_px[i],
            lm_prone_trans_px[i],
            orientation='axial'
        )

    # Loop through each landmark and create a visualization
    for i in range(len(lm_supine_px)):
        print(f"Visualizing alignment for landmark #{i + 1}")
        breast_metadata.visualise_alignment_with_landmarks(
            supine_image_sitk,
            prone_image_transformed,
            lm_supine_px[i],
            lm_prone_trans_px[i],
            orientation='sagittal'
        )

    # Loop through each landmark and create a visualization
    for i in range(len(lm_supine_px)):
        print(f"Visualizing alignment for landmark #{i + 1}")
        breast_metadata.visualise_alignment_with_landmarks(
            supine_image_sitk,
            prone_image_transformed,
            lm_supine_px[i],
            lm_prone_trans_px[i],
            orientation='coronal'
        )

    '''


    # ==========================================================
    # %% RETURN RESULTS
    # ==========================================================
    results = {
        "vl_id": subject.subject_id,
        "T_total": T_total,
        "sternum_error": sternum_error,
        "ribcage_error_mean": np.mean(rib_error_mag),
        "ribcage_error_std": np.std(rib_error_mag),
        "ribcage_inlier_RMSE": icp_result['inlier_rmse'],
        "sternum_prone_transformed": prone_sternum_aligned_final,
        "sternum_supine": sternum_supine,
        "nipple_prone_transformed": nipple_prone_transformed,
        "nipple_supine": nipple_supine,
        "nipple_displacement_vectors": nipple_disp_rel_sternum,
        "nipple_displacement_magnitudes": nipple_disp_mag_rel_sternum,
        "landmark_prone_ave_transformed": lm_pos_prone_rel_sternum,
        "landmark_supine_ave": lm_pos_supine_rel_sternum,
        "ld_ave_displacement_vectors": lm_disp_rel_sternum,
        "ld_ave_displacement_magnitudes": lm_disp_mag_rel_sternum,
        "ld_ave_rel_nipple_vectors": lm_disp_rel_nipple,
        "ld_ave_rel_nipple_magnitudes": lm_disp_mag_rel_nipple,
        "ld_ave_rel_nipple_vectors_base_left":X_left,
        "ld_ave_rel_nipple_vectors_left":V_left,
        "ld_ave_rel_nipple_vectors_base_right":X_right,
        "ld_ave_rel_nipple_vectors_right":V_right

        # "r1_displacement_vectors": landmark_r1_displacement_vectors,
        # "r1_displacement_magnitudes": landmark_r1_displacement_magnitudes,
        # "r2_displacement_vectors": landmark_r2_displacement_vectors,
        # "r2_displacement_magnitudes": landmark_r2_displacement_magnitudes,
        # "r1_rel_nipple_vectors": landmark_r1_rel_nipple_vectors,
        # "r1_rel_nipple_magnitudes": landmark_r1_rel_nipple_mag,
        # "r2_rel_nipple_vectors": landmark_r2_rel_nipple_vectors,
        # "r2_rel_nipple_magnitudes": landmark_r2_rel_nipple_mag,
        # "r1_rel_nipple_vectors_base_left":X_left_r1,
        # "r1_rel_nipple_vectors_left":V_left_r1,
        # "r1_rel_nipple_vectors_base_right":X_right_r1,
        # "r1_rel_nipple_vectors_right":V_right_r1,
        # "r2_rel_nipple_vectors_base_left":X_left_r2,
        # "r2_rel_nipple_vectors_left":V_left_r2,
        # "r2_rel_nipple_vectors_base_right":X_right_r2,
        # "r2_rel_nipple_vectors_right":V_right_r2,
        # "prone_image_transformed": prone_image_transformed
    }

    return results


def compute_icp_metrics(src_pts: np.ndarray, tgt_pts: np.ndarray, tgt_normals: np.ndarray, max_correspondence_distance: float = np.inf):
    """
    Compute both Euclidean RMSE and point-to-plane RMSE between src_pts and tgt_pts using nearest-neighbours.
    Returns a dict with keys: euclidean_rmse, point_to_plane_rmse, inlier_fraction, n_inliers
    """
    src = np.asarray(src_pts, dtype=float)
    tgt = np.asarray(tgt_pts, dtype=float)
    if src.size == 0 or tgt.size == 0:
        return {"euclidean_rmse": None, "point_to_plane_rmse": None, "inlier_fraction": 0.0, "n_inliers": 0}

    tree = cKDTree(tgt)
    dists, idxs = tree.query(src)
    valid = dists <= max_correspondence_distance
    n_inliers = int(np.count_nonzero(valid))
    inlier_fraction = float(n_inliers) / float(len(src)) if len(src) > 0 else 0.0

    euclidean_rmse = float(np.sqrt(np.mean(dists[valid] ** 2))) if n_inliers > 0 else None

    # point-to-plane residuals
    if tgt_normals is None or n_inliers == 0:
        ptp_rmse = None
    else:
        N = np.asarray(tgt_normals)[idxs[valid]]
        P = src[valid]
        Q = tgt[idxs[valid]]
        residuals = np.einsum('ij,ij->i', N, (P - Q))
        ptp_rmse = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size > 0 else None

    return {
        "euclidean_rmse": euclidean_rmse,
        "point_to_plane_rmse": ptp_rmse,
        "inlier_fraction": inlier_fraction,
        "n_inliers": n_inliers
    }


def run_point_to_plane_icp(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    target_normals: Optional[np.ndarray] = None,
    max_correspondence_distance: float = 10.0,
    max_iterations: int = 50,
    method: str = 'open3d',
    delta: float = 1.0,
    reweight_strategy: str = 'huber',
    huber_delta: float = 2.0,
    verbose: bool = False
) -> tuple:
    """
    Flexible point-to-plane ICP. Two modes:
      - method='open3d' : use Open3D registration_icp (point-to-plane) and return its transform + metrics
      - method='reweighted' : run an iterative reweighted least-squares point-to-plane ICP implemented in numpy

    Returns (T_total, source_transformed, info) where info includes diagnostics and both RMSEs.
    """
    source_pts = np.asarray(source_pts, dtype=np.float64)
    target_pts = np.asarray(target_pts, dtype=np.float64)
    if source_pts.size == 0 or target_pts.size == 0:
        return np.eye(4), source_pts.copy(), {"it": 0, "euclidean_rmse": None, "ptp_rmse": None}

    # Helper: estimate normals on target if not provided using Open3D
    if target_normals is None:
        try:
            tgt_pcd = o3d.geometry.PointCloud()
            tgt_pcd.points = o3d.utility.Vector3dVector(target_pts)
            # heuristic radius
            radius = max(5.0, max_correspondence_distance * 1.5)
            tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))
            target_normals = np.asarray(tgt_pcd.normals)
        except Exception:
            target_normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (target_pts.shape[0], 1))

    if method == 'open3d':
        # Build Open3D point clouds
        src_pcd = o3d.geometry.PointCloud()
        tgt_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(source_pts)
        tgt_pcd.points = o3d.utility.Vector3dVector(target_pts)
        # ensure target normals exist in Open3D pcd
        try:
            tgt_pcd.normals = o3d.utility.Vector3dVector(target_normals)
        except Exception:
            pass

        init_T = np.eye(4)
        icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)

        #The Huber Loss is a robust loss function that behaves like L2 (Least Squares) for small errors (inliers)
        # and like L1 (Least Absolute Error) for large errors (outliers).
        loss = o3d.pipelines.registration.HuberLoss(k=delta) # k is the delta parameter
        # The Tukey Biweight Loss is the closest practical equivalent to an aggressive inverse weighting scheme for outlier rejection.
        # loss = o3d.pipelines.registration.TukeyLoss(k=delta)
        try:
            result = o3d.pipelines.registration.registration_icp(
                src_pcd,
                tgt_pcd,
                max_correspondence_distance,
                init_T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
                icp_criteria
            )
        except Exception as e:
            if verbose:
                print(f"Open3D ICP failed: {e}")
            return np.eye(4), source_pts.copy(), {"it": 0}

        T_icp = np.asarray(result.transformation)
        src_h = np.hstack((source_pts, np.ones((source_pts.shape[0], 1))))
        source_aligned = (T_icp @ src_h.T)[:3, :].T

        # Re-use the target point cloud created earlier (or recreate it for simplicity)
        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(target_pts)

        source_pcd_aligned = o3d.geometry.PointCloud()
        source_pcd_aligned.points = o3d.utility.Vector3dVector(source_aligned)

        # Find correspondences after final alignment
        # The result includes the indices of correspondences and their distances
        corr_result = o3d.pipelines.registration.evaluate_registration(
            source_pcd_aligned,
            tgt_pcd,
            max_correspondence_distance,
            T_icp  # The transformation is T_icp @ init_T, which is T_icp here
        )

        # The 'correspondence_set' is a NumPy array where each row is [source_index, target_index]
        correspondence_indices = np.asarray(corr_result.correspondence_set)

        # The indices of the *aligned source points* that are considered inliers
        inlier_source_indices = correspondence_indices[:, 0]

        # Filter the source_aligned array to get only the inlier points
        inlier_source_pts = source_aligned[inlier_source_indices]

        metrics = compute_icp_metrics(source_aligned, target_pts, target_normals, max_correspondence_distance)

        info = {
            "it": result.convergence_status if hasattr(result, 'convergence_status') else max_iterations,
            "fitness": float(getattr(result, 'fitness', np.nan)),
            "inlier_rmse": float(getattr(result, 'inlier_rmse', np.nan)),
            "euclidean_rmse": metrics["euclidean_rmse"],
            "point_to_plane_rmse": metrics["point_to_plane_rmse"],
            "n_inliers": metrics["n_inliers"],
            "inlier_fraction": metrics["inlier_fraction"],
            "inlier_source_pts": inlier_source_pts
        }
        return T_icp, source_aligned, info

    # else method == 'reweighted'
    # iterative reweighted point-to-plane ICP implemented in numpy
    tree = cKDTree(target_pts)
    src = source_pts.copy()
    T_total = np.eye(4)
    prev_ptp_rmse = np.inf

    for it in range(max_iterations):
        dists, idxs = tree.query(src)
        valid_mask = dists <= max_correspondence_distance
        if not np.any(valid_mask):
            if verbose:
                print("No valid correspondences in reweighted ICP iteration")
            break

        P = src[valid_mask]
        Q = target_pts[idxs[valid_mask]]
        N = target_normals[idxs[valid_mask]]

        # point-to-plane residuals r = n^T (p - q)
        r = np.einsum('ij,ij->i', N, (P - Q))

        # compute weights
        if reweight_strategy == 'huber':
            abs_r = np.abs(r)
            w = np.where(abs_r <= huber_delta, 1.0, huber_delta / (abs_r + 1e-12))
        elif reweight_strategy == 'inv':
            w = 1.0 / (np.abs(r) + 1e-6)
            # cap weights to avoid extreme values
            w = np.minimum(w, 100.0)
        else:
            # uniform weights
            w = np.ones_like(r)

        # Build A and b for linearised point-to-plane: A_i = [ (p x n)^T , n^T ], b_i = n^T (q - p)
        pxn = np.cross(P, N)
        A = np.hstack((pxn, N))
        b = np.einsum('ij,ij->i', N, (Q - P))

        # Apply weights by scaling rows
        W_sqrt = np.sqrt(w)[:, None]
        Aw = W_sqrt * A
        bw = W_sqrt[:, 0] * b

        try:
            x, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        except Exception as e:
            if verbose:
                print(f"Lstsq failed in reweighted ICP: {e}")
            break

        w_rot = x[:3]
        t_trans = x[3:]

        # incremental transform
        # Rodrigues
        theta = np.linalg.norm(w_rot)
        if theta < 1e-12:
            R_delta = np.eye(3)
        else:
            k = w_rot / theta
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R_delta = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        T_delta = np.eye(4)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = t_trans

        # apply to src
        src_h = np.hstack((src, np.ones((src.shape[0], 1))))
        src = (T_delta @ src_h.T)[:3, :].T

        # accumulate transform
        T_total = T_delta @ T_total

        # compute metrics
        metrics = compute_icp_metrics(src, target_pts, target_normals, max_correspondence_distance)
        ptp_rmse = metrics["point_to_plane_rmse"] if metrics["point_to_plane_rmse"] is not None else np.inf

        if verbose:
            print(f"Reweighted ICP iter {it+1}: ptp_rmse={ptp_rmse:.6f}, eucl_rmse={metrics['euclidean_rmse']}, n_inliers={metrics['n_inliers']}")

        # convergence check (small change in ptp RMSE)
        if np.abs(prev_ptp_rmse - ptp_rmse) < 1e-6:
            break
        prev_ptp_rmse = ptp_rmse

    info = {
        "it": it + 1,
        "euclidean_rmse": metrics.get("euclidean_rmse"),
        "point_to_plane_rmse": metrics.get("point_to_plane_rmse"),
        "n_inliers": metrics.get("n_inliers"),
        "inlier_fraction": metrics.get("inlier_fraction")
    }

    return T_total, src, info


def save_results_to_excel(
        excel_path: Path,
        correspondences: Dict[int, List[List[str]]],
        all_subjects: Dict[int, Subject],
        distance_results: Dict,
        clockface_results: Dict,
        alignment_results_all: Dict[int, Dict],
):
    """
    Gathers all analysis results from the various dictionaries
    and saves them to a single, comprehensive Excel file.

    This function now works with averaged landmarks only (no separate registrar sheets).
    The alignment_results use 'ld_ave_' prefix for landmark displacement data.
    """
    print("\nSaving all results to Excel...\n============")

    # --- Internal helper to build rows for averaged landmarks ---
    def _build_averaged_data() -> List[Dict[str, any]]:

        all_rows = []

        # Master loop: Iterate through the corresponding landmarks
        for vl_id, pairs in correspondences.items():

            # --- Get Subject-Level Data ---
            if vl_id not in all_subjects:
                continue
            subject = all_subjects[vl_id]
            align_res = alignment_results_all.get(vl_id)  # Safe get

            # Get subject-level alignment data (nipples, sternum, and ribcage)
            if align_res:
                nipple_disp_mag = align_res.get("nipple_displacement_magnitudes", [None, None])
                nipple_disp_vec = align_res.get("nipple_displacement_vectors", [[None] * 3, [None] * 3])
                nipple_prone_transformed = align_res.get("nipple_prone_transformed", [[None] * 3, [None] * 3])
                nipple_supine = align_res.get("nipple_supine", [[None] * 3, [None] * 3])
                sternum_prone_transformed = align_res.get("sternum_prone_transformed", [[None] * 3, [None] * 3])
                sternum_supine = align_res.get("sternum_supine", [[None] * 3, [None] * 3])
                ribcage_error_mean = align_res.get("ribcage_error_mean", None)
                ribcage_error_std = align_res.get("ribcage_error_std", None)
                ribcage_inlier_RMSE = align_res.get("ribcage_inlier_RMSE", None)
                T_total = align_res.get("T_total", None)
            else:
                nipple_disp_mag = [None, None]
                nipple_disp_vec = [[None] * 3, [None] * 3]
                nipple_prone_transformed = [[None] * 3, [None] * 3]
                nipple_supine = [[None] * 3, [None] * 3]
                sternum_prone_transformed = [[None] * 3, [None] * 3]
                sternum_supine = [[None] * 3, [None] * 3]
                ribcage_error_mean = None
                ribcage_error_std = None
                ribcage_inlier_RMSE = None
                T_total = None

            # --- Loop through each landmark pair for this subject ---
            for i, pair in enumerate(pairs):

                # For averaged landmarks, use the first name in the pair (they should be the same)
                lm_name = pair[0]

                # Get landmark-specific alignment data (from alignment_results with 'ld_ave_' prefix)
                if align_res:
                    lm_prone_ave_transformed = align_res.get("landmark_prone_ave_transformed", [[None] * 3] * len(pairs))[i]
                    lm_supine_ave = align_res.get("landmark_supine_ave", [[None] * 3] * len(pairs))[i]
                    lm_disp_mag = align_res.get("ld_ave_displacement_magnitudes", [None] * len(pairs))[i]
                    lm_disp_vec = align_res.get("ld_ave_displacement_vectors", [[None] * 3] * len(pairs))[i]
                    lm_rel_mag = align_res.get("ld_ave_rel_nipple_magnitudes", [None] * len(pairs))[i]
                    lm_rel_vec = align_res.get("ld_ave_rel_nipple_vectors", [[None] * 3] * len(pairs))[i]
                else:
                    lm_prone_ave_transformed = [None] * 3
                    lm_supine_ave = [None] * 3
                    lm_disp_mag = None
                    lm_disp_vec = [None] * 3
                    lm_rel_mag = None
                    lm_rel_vec = [None] * 3

                # --- Helper to safely get nested dictionary data ---
                # Note: distance_results and clockface_results use "average" as registrar name
                def get_data(results_dict: Dict, position: str, data_key: str, default: any = None) -> any:
                    try:
                        if data_key in ["skin_distances", "rib_distances", "skin_neighborhood_avg", "rib_neighborhood_avg"]:
                            # This data is in distance_results
                            return results_dict[vl_id][position]["average"][data_key][lm_name]
                        else:
                            # This data is in clockface_results
                            return results_dict[vl_id][position]["average"][lm_name][data_key]
                    except (KeyError, TypeError):
                        return default

                # --- Build the row for this single landmark ---
                row_data = {
                    'VL_ID': vl_id,
                    'Age': subject.age,
                    'Height [m]': subject.height,
                    'Weight [kg]': subject.weight,
                    'Landmark name': lm_name,
                    'Landmark type': lm_name.split('_')[0] if '_' in lm_name else lm_name,

                    'landmark side (prone)': get_data(clockface_results, "prone", "side"),
                    'Distance to nipple (prone) [mm]': get_data(clockface_results, "prone", "dist_to_nipple"),
                    'Distance to rib cage (prone) [mm]': get_data(distance_results, "prone", "rib_distances"),
                    'Distance to skin (prone) [mm]': get_data(distance_results, "prone", "skin_distances"),
                    'Time (prone)': get_data(clockface_results, "prone", "time"),
                    'Quadrant (prone)': get_data(clockface_results, "prone", "quadrant"),

                    'landmark side (supine)': get_data(clockface_results, "supine", "side"),
                    'Distance to nipple (supine) [mm]': get_data(clockface_results, "supine", "dist_to_nipple"),
                    'Distance to rib cage (supine) [mm]': get_data(distance_results, "supine", "rib_distances"),
                    'Distance to skin (supine) [mm]': get_data(distance_results, "supine", "skin_distances"),
                    'Time (supine)': get_data(clockface_results, "supine", "time"),
                    'Quadrant (supine)': get_data(clockface_results, "supine", "quadrant"),

                    "ribcage error mean": ribcage_error_mean,
                    "ribcage error std": ribcage_error_std,
                    "ribcage inlier RMSE": ribcage_inlier_RMSE,

                    "sternum superior prone transformed x": sternum_prone_transformed[0][0],
                    "sternum superior prone transformed y": sternum_prone_transformed[0][1],
                    "sternum superior prone transformed z": sternum_prone_transformed[0][2],
                    "sternum superior supine x": sternum_supine[0][0],
                    "sternum superior supine y": sternum_supine[0][1],
                    "sternum superior supine z": sternum_supine[0][2],

                    "left nipple prone transformed x": nipple_prone_transformed[0][0],
                    "left nipple prone transformed y": nipple_prone_transformed[0][1],
                    "left nipple prone transformed z": nipple_prone_transformed[0][2],
                    "right nipple prone transformed x": nipple_prone_transformed[1][0],
                    "right nipple prone transformed y": nipple_prone_transformed[1][1],
                    "right nipple prone transformed z": nipple_prone_transformed[1][2],

                    "left nipple supine x": nipple_supine[0][0],
                    "left nipple supine y": nipple_supine[0][1],
                    "left nipple supine z": nipple_supine[0][2],
                    "right nipple supine x": nipple_supine[1][0],
                    "right nipple supine y": nipple_supine[1][1],
                    "right nipple supine z": nipple_supine[1][2],

                    "landmark ave prone transformed x": lm_prone_ave_transformed[0],
                    "landmark ave prone transformed y": lm_prone_ave_transformed[1],
                    "landmark ave prone transformed z": lm_prone_ave_transformed[2],
                    "landmark ave supine x": lm_supine_ave[0],
                    "landmark ave supine y": lm_supine_ave[1],
                    "landmark ave supine z": lm_supine_ave[2],

                    'Left nipple displacement [mm]': nipple_disp_mag[0],
                    'Right nipple displacement [mm]': nipple_disp_mag[1],

                    'Left nipple displacement vector vx': nipple_disp_vec[0][0],
                    'Left nipple displacement vector vy': nipple_disp_vec[0][1],
                    'Left nipple displacement vector vz': nipple_disp_vec[0][2],

                    'Right nipple displacement vector vx': nipple_disp_vec[1][0],
                    'Right nipple displacement vector vy': nipple_disp_vec[1][1],
                    'Right nipple displacement vector vz': nipple_disp_vec[1][2],

                    'Landmark displacement [mm]': lm_disp_mag,
                    'Landmark displacement relative to nipple [mm]': lm_rel_mag,

                    'Landmark displacement vector vx': lm_disp_vec[0],
                    'Landmark displacement vector vy': lm_disp_vec[1],
                    'Landmark displacement vector vz': lm_disp_vec[2],

                    'Landmark relative to nipple vector vx': lm_rel_vec[0],
                    'Landmark relative to nipple vector vy': lm_rel_vec[1],
                    'Landmark relative to nipple vector vz': lm_rel_vec[2],

                    'Mask skin neighborhood avg (prone)': get_data(distance_results, "prone", "skin_neighborhood_avg"),
                    'Mask rib neighborhood avg (prone)': get_data(distance_results, "prone", "rib_neighborhood_avg"),
                    'Mask skin neighborhood avg (supine)': get_data(distance_results, "supine", "skin_neighborhood_avg"),
                    'Mask rib neighborhood avg (supine)': get_data(distance_results, "supine", "rib_neighborhood_avg"),
                }
                all_rows.append(row_data)

        print(json.dumps(all_rows, indent=2))

        return all_rows


    # --- Main function logic ---

    # Create the output directory if it doesn't exist
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    # Build data for averaged landmarks only
    print(f"Building data for averaged landmark positions...")
    ave_data = _build_averaged_data()
    df_ave = pd.DataFrame(ave_data)

    # Get VL_IDs from new data
    new_vl_ids = df_ave['VL_ID'].unique() if not df_ave.empty else []

    # Load existing data if file exists
    df_existing = pd.DataFrame()
    if excel_path.exists():
        try:
            df_existing = pd.read_excel(excel_path, sheet_name='processed_ave_data', engine='openpyxl')
            if not df_existing.empty:
                # Filter out rows for VL_IDs that we're updating
                df_existing = df_existing[~df_existing['VL_ID'].isin(new_vl_ids)].copy()
        except ValueError:
            print(f"Warning: Sheet 'processed_ave_data' not found. It will be created.")
        except Exception as e:
            print(f"Error reading 'processed_ave_data' sheet: {e}. Starting new sheet.")

    # Combine existing and new data
    print("Combining dataframes...")
    df_combined = pd.concat([df_existing, df_ave], ignore_index=True)
    df_combined = df_combined.sort_values(by=['VL_ID'], kind='stable').reset_index(drop=True)

    # Determine write mode
    write_mode = 'a' if excel_path.exists() else 'w'
    writer_kwargs = {'engine': 'openpyxl', 'mode': write_mode}
    if write_mode == 'a':
        writer_kwargs['if_sheet_exists'] = 'replace'

    try:
        # Use ExcelWriter as a context manager for safe file handling
        with pd.ExcelWriter(excel_path, **writer_kwargs) as writer:
            df_combined.to_excel(writer, sheet_name='processed_ave_data', index=False)
        print(f"Data successfully saved to {excel_path}")
    except Exception as e:
        print(f"An error occurred while saving: {e}")


def save_raw_data_to_excel(
        excel_path: Path,
        all_subjects: Dict[int, Subject],
):
    # --- Build raw_data sheet (one row per subject) ---
    # Build flattened raw_rows: one row per subject, per position, per registrar, per landmark
    raw_rows = []

    for vl_id, subject in all_subjects.items():
        age = getattr(subject, 'age', None)
        height = getattr(subject, 'height', None)
        weight = getattr(subject, 'weight', None)

        # iterate positions and registrars
        for position in ('prone', 'supine'):
            scan = subject.scans.get(position) if hasattr(subject, 'scans') else None
            if scan is None or not hasattr(scan, 'registrar_data'):
                continue

            for registrar_name, reg_data in scan.registrar_data.items():
                # map registrar name to id (keep same convention as processed_data)
                registrar_id = 1 if registrar_name.lower() == 'anthony' else 2 if registrar_name.lower() == 'holly' else None

                landmarks_dict = {}
                try:
                    landmarks_dict = reg_data.soft_tissue_landmarks or {}
                except Exception:
                    landmarks_dict = {}

                # if no landmarks, still emit a row to capture subject/registrar metadata? skip to minimize rows
                if not landmarks_dict:
                    continue

                for lm_name, lm_coord in landmarks_dict.items():
                    # coerce coordinate to list of floats
                    try:
                        coord = np.asarray(lm_coord, dtype=float).reshape(3).tolist()
                        x, y, z = coord
                    except Exception:
                        x = y = z = None

                    raw_rows.append({
                        'Registrar': registrar_id,
                        'Registrar_name': registrar_name,
                        'VL_ID': vl_id,
                        'Age': age,
                        'Height [m]': height,
                        'Weight [kg]': weight,
                        'position': position,
                        'Landmark name': lm_name,
                        'Landmark type': lm_name.split('_')[0] if isinstance(lm_name,
                                                                             str) and '_' in lm_name else lm_name,
                        'x': x,
                        'y': y,
                        'z': z
                    })

    df_raw = pd.DataFrame(raw_rows)
    new_vl_ids = df_raw['VL_ID'].unique()
    df_existing = pd.DataFrame()

    if excel_path.exists():
        try:
            df_existing = pd.read_excel(excel_path, sheet_name='raw_data', engine='openpyxl')
            if not df_existing.empty:
                df_existing_filtered = df_existing[~df_existing['VL_ID'].isin(new_vl_ids)].copy()
                df_existing = df_existing_filtered
        except ValueError as e:
            print(f"Warning: Sheet '{'raw_data'}' not found. Creating new sheet.")

    df_combined = pd.concat([df_existing, df_raw], ignore_index=True)
    df_combined = df_combined.sort_values(by=['Registrar', 'VL_ID'], kind='stable').reset_index(drop=True)

    # Determine the mode: 'a' if file exists, 'w' if it's brand new
    write_mode = 'a' if excel_path.exists() else 'w'
    writer_kwargs = {'engine': 'openpyxl', 'mode': write_mode}
    if write_mode == 'a':
        writer_kwargs['if_sheet_exists'] = 'replace'

    try:
        # Use ExcelWriter as a context manager for safe file handling
        with pd.ExcelWriter(excel_path, **writer_kwargs) as writer:
            # Save raw data sheet
            df_combined.to_excel(writer, sheet_name='raw_data', index=False)
        print(f"Raw data successfully saved to {excel_path}")
    except Exception as e:
        print(f"An error occurred while saving raw data: {e}")
