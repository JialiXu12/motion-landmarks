from pathlib import Path
import pyvista as pv
import numpy as np
import math
import copy
from typing import Dict, List, Tuple
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

                for lm_name, lm_coord in reg_data.soft_tissue_landmarks.items():
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
                    # This check is now simpler
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


def align_prone_to_supine(
        subject: Subject,
        prone_ribcage_mesh_path: Path,
        supine_ribcage_seg_path: Path,
        orientation_flag: str = 'RAI'
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

    prone_scan = subject.scans["prone"].scan_object
    supine_scan = subject.scans["supine"].scan_object

    # --- 2. Convert Scans to Pyvista Image Grids ---
    prone_image_grid = breast_metadata.SCANToPyvistaImageGrid(prone_scan, orientation_flag)
    supine_image_grid = breast_metadata.SCANToPyvistaImageGrid(supine_scan, orientation_flag)

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
    landmark_prone_r1 = get_landmarks_as_array(prone_scan_data, "anthony")
    landmark_supine_r1 = get_landmarks_as_array(supine_scan_data, "anthony")
    landmark_prone_r2 = get_landmarks_as_array(prone_scan_data, "holly")
    landmark_supine_r2 = get_landmarks_as_array(supine_scan_data, "holly")

    # --- 5. Load Ribcage Mesh and Mask ---
    prone_ribcage = morphic.Mesh(str(prone_ribcage_mesh_path))
    prone_ribcage_mesh_coords = aps.get_surface_mesh_coords(prone_ribcage, res=26)

    supine_ribcage_mask = breast_metadata.readNIFTIImage(str(supine_ribcage_seg_path), orientation_flag, swap_axes=True)

    supine_ribcage_pc = extract_contour_points(supine_ribcage_mask, 20000)

    # --- 6. Clean up Supine Point Cloud ---
    supine_ribcage_pc = supine_ribcage_pc[supine_ribcage_pc[:, 0] > -120.]
    #   remove points on the top and bottom axial slices
    supine_ribcage_pc = aps.filter_point_cloud(
        supine_ribcage_pc, supine_ribcage_pc, supine_image_grid.spacing[2] * 10, axis=2)
    #   remove points on the back side of the ribcage around the spine
    supine_ribcage_pc = supine_ribcage_pc[
        (supine_ribcage_pc[:, 0] <= -10.) | (supine_ribcage_pc[:, 0] >= 40.) |
        (supine_ribcage_pc[:, 1] < np.max(supine_ribcage_pc[:, 1]) - 60)]

    # ==========================================================
    # %% INITIAL ALIGNMENT
    # ==========================================================
    rot_angle_init = [0., 0., 0.]
    translation_init = list(
        breast_metadata.find_centroid(sternum_supine.T) - breast_metadata.find_centroid(sternum_prone.T))
    T_init = rot_angle_init + translation_init

    #   First iteration: optimise transformation matrix by performing kd-tree of point clouds between
    #   the landmarks in prone and supine
    print("\nPERFORMING OPTIMISATION\n============")
    prone_points = [prone_ribcage_mesh_coords, sternum_prone]
    supine_points = [supine_ribcage_pc, sternum_supine]
    T_optimal, res_optimal = breast_metadata.run_optimisation(breast_metadata.combined_objective_function, T_init,
                                                              prone_points, supine_points)
    print(f"\nProne-to-supine ribcage transformation:\n {T_optimal}")
    print("6 DoFs:", res_optimal.x)

    # ==========================================================
    # %% APPLY TRANSFORMATION
    # ==========================================================
    T_inv = np.linalg.inv(T_optimal)

    ribcage_prone_mesh_transformed = apply_transform(prone_ribcage_mesh_coords, T_optimal)
    sternum_prone_transformed = apply_transform(sternum_prone, T_optimal)
    sternum_supine_transformed = apply_transform(sternum_supine, T_inv)
    nipple_prone_transformed = apply_transform(nipple_prone, T_optimal)
    nipple_supine_transformed = apply_transform(nipple_supine, T_inv)

    landmark_prone_r1_transformed = apply_transform(landmark_prone_r1, T_optimal)
    landmark_prone_r2_transformed = apply_transform(landmark_prone_r2, T_optimal)
    landmark_supine_r1_transformed = apply_transform(landmark_supine_r1, T_inv)
    landmark_supine_r2_transformed = apply_transform(landmark_supine_r2, T_inv)

    # ==========================================================
    # %% EVALUATE DISPLACEMENT
    # ==========================================================

    # Evaluate absolute displacement of landmarks and nipples
    landmark_r1_displacement_vectors = landmark_supine_r1 - landmark_prone_r1_transformed
    landmark_r1_displacement_magnitudes = np.linalg.norm(landmark_r1_displacement_vectors, axis=1)
    landmark_r2_displacement_vectors = landmark_supine_r2 - landmark_prone_r2_transformed
    landmark_r2_displacement_magnitudes = np.linalg.norm(landmark_r2_displacement_vectors, axis=1)

    nipple_displacement_vectors = nipple_supine - nipple_prone_transformed
    nipple_displacement_magnitudes = np.linalg.norm(nipple_displacement_vectors, axis=1)
    left_nipple_displacement_vectors = nipple_displacement_vectors[0]
    right_nipple_displacement_vectors = nipple_displacement_vectors[1]

    # --- Relative Displacements ---
    # Create a boolean mask where True means the landmark is closer to the left nipple
    dist_to_left = np.linalg.norm(landmark_supine_r1 - nipple_supine[0], axis=1)
    dist_to_right = np.linalg.norm(landmark_supine_r1 - nipple_supine[1], axis=1)
    is_closer_to_left = dist_to_left < dist_to_right

    relevant_nipple_vectors = np.where(
        is_closer_to_left[:, np.newaxis],  # Condition
        left_nipple_displacement_vectors,  # Value if True
        right_nipple_displacement_vectors  # Value if False
    )

    # Registrar 1
    landmark_r1_rel_nipple_vectors = relevant_nipple_vectors - landmark_r1_displacement_vectors
    landmark_r1_rel_nipple_mag = np.linalg.norm(landmark_r1_rel_nipple_vectors, axis=1)

    # Registrar 2
    landmark_r2_rel_nipple_vectors = relevant_nipple_vectors - landmark_r2_displacement_vectors
    landmark_r2_rel_nipple_mag = np.linalg.norm(landmark_r2_rel_nipple_vectors, axis=1)

    # %% Landmark displacement relative to nipple displacement
    left_nipple_prone = nipple_prone_transformed[0]
    right_nipple_prone = nipple_prone_transformed[1]

    prone_pos_left = landmark_prone_r1_transformed[is_closer_to_left]
    land_disp_left = landmark_r1_displacement_vectors[is_closer_to_left]
    X_left = left_nipple_prone - prone_pos_left
    V_left = left_nipple_displacement_vectors - land_disp_left

    prone_pos_right = landmark_prone_r1_transformed[~is_closer_to_left]
    land_disp_right = landmark_r1_displacement_vectors[~is_closer_to_left]
    X_right = right_nipple_prone - prone_pos_right
    V_right = right_nipple_displacement_vectors - land_disp_right

    # ==========================================================
    # %% PLOT
    # ==========================================================
    # X=right-left and Y=inf-sup
    AXIS_X = 0
    AXIS_Y = 2

    # Define plot limits
    lims = (-60, 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8.5))
    fig.suptitle("Direction of landmark motion from prone to supine (R1 only)\n(with respect to the nipple)",
                 fontsize=16)

    # --- Plot 1: Right Breast ---
    ax1.set_title("Coronal plane\nRight breast", loc='left', fontsize=12)
    ax1.quiver(
        X_right[:, AXIS_X], X_right[:, AXIS_Y],  # Arrow base (relative prone pos)
        V_right[:, AXIS_X], V_right[:, AXIS_Y],  # Arrow vector (relative displacement)
        angles='xy', scale_units='xy', scale=1, color='black'
    )

    # --- Plot 2: Left Breast ---
    ax2.set_title("Coronal plane\nLeft breast", loc='left', fontsize=12)
    ax2.quiver(
        X_left[:, AXIS_X], X_left[:, AXIS_Y],  # Arrow base (relative prone pos)
        V_left[:, AXIS_X], V_left[:, AXIS_Y],  # Arrow vector (relative displacement)
        angles='xy', scale_units='xy', scale=1, color='black'
    )

    # --- Format both plots ---
    for ax in [ax1, ax2]:
        ax.set_xlabel("right-left (mm)")
        ax.set_ylabel("inf-sup (mm)")

        # Set limits and aspect ratio
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')

        # Add red nipple dot and quadrant lines
        ax.plot(0, 0, 'ro', markersize=8, zorder=5)  # Nipple
        ax.axhline(0, color='red', lw=1, zorder=0)
        ax.axvline(0, color='red', lw=1, zorder=0)

        # Add outer circle
        circle = Circle((0, 0), lims[1], fill=False, color='black', lw=1)
        ax.add_artist(circle)

    # --- Add Quadrant Labels ---
    # Note: These are mirrored for left vs. right
    text_offset = lims[1] * 0.85
    # Right Breast Quadrants
    ax1.text(text_offset, text_offset, 'UI', ha='center', va='center', fontsize=14)
    ax1.text(-text_offset, text_offset, 'UO', ha='center', va='center', fontsize=14)
    ax1.text(text_offset, -text_offset, 'LI', ha='center', va='center', fontsize=14)
    ax1.text(-text_offset, -text_offset, 'LO', ha='center', va='center', fontsize=14)

    # Left Breast Quadrants
    ax2.text(text_offset, text_offset, 'UO', ha='center', va='center', fontsize=14)
    ax2.text(-text_offset, text_offset, 'UI', ha='center', va='center', fontsize=14)
    ax2.text(text_offset, -text_offset, 'LO', ha='center', va='center', fontsize=14)
    ax2.text(-text_offset, -text_offset, 'LI', ha='center', va='center', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # ==========================================================
    # %% PRINT RESULTS & EVALUATE FIT
    # ==========================================================
    print("--- Landmark Displacement ---")
    print("--- Registrar 1 ---")
    for i in range(len(landmark_supine_r1)):
        print(f"Landmark {i + 1}:")
        print(f"  Displacement Vector: {landmark_r1_displacement_vectors[i]}")
        print(f"  Displacement Magnitude: {landmark_r1_displacement_magnitudes[i]:.2f} mm")

    print("--- Registrar 2 ---")
    for i in range(len(landmark_supine_r2)):
        print(f"Landmark {i + 1}:")
        print(f"  Displacement Vector: {landmark_r2_displacement_vectors[i]}")
        print(f"  Displacement Magnitude: {landmark_r2_displacement_magnitudes[i]:.2f} mm")

    # evaluate sternum fit
    error, mapped_idx = breast_metadata.closest_distances(sternum_supine, sternum_prone_transformed)
    sternum_error = np.linalg.norm(error, axis=1)
    print(f"Sternum alignment error: {sternum_error} mm")

    # evaluate ribcage fit
    error, mapped_idx = breast_metadata.closest_distances(supine_ribcage_pc, ribcage_prone_mesh_transformed)
    rib_error_mag = np.linalg.norm(error, axis=1)
    print(f"Ribcage alignment error: {rib_error_mag} mm")

    # show statistics and distribution of projection errors
    aps.summary_stats(rib_error_mag)
    aps.plot_histogram(rib_error_mag, 5)

    # ==========================================================
    # %% RESAMPLE & PLOT
    # ==========================================================
    #   convert Pyvista image grid to SITK image
    prone_image_sitk = breast_metadata.PyvistaImageGridToSITKImage(prone_image_grid)
    prone_image_sitk = sitk.Cast(prone_image_sitk, sitk.sitkUInt8)
    supine_image_sitk = breast_metadata.PyvistaImageGridToSITKImage(supine_image_grid)
    supine_image_sitk = sitk.Cast(supine_image_sitk, sitk.sitkUInt8)

    #   initialise affine transformation matrix
    dimensions = 3
    affine = sitk.AffineTransform(dimensions)
    #   set transformation matrix from prone to supine
    T_prone_to_supine = np.linalg.inv(T_optimal)
    affine.SetTranslation(T_prone_to_supine[:3, 3])
    affine.SetMatrix(T_prone_to_supine[:3, :3].ravel())

    #   transform prone image to supine coordinate system
    #   sitk.Resample(input_image, reference_image, transform) takes a transformation matrix that maps points
    #   from the reference_image (output space) to it's corresponding location on the input_image (input space)
    prone_image_transformed = sitk.Resample(prone_image_sitk, supine_image_sitk, affine, sitk.sitkLinear, 1.0)
    prone_image_transformed = sitk.Cast(prone_image_transformed, sitk.sitkUInt8)

    #   get pixel coordinates of landmarks
    prone_scan_transformed = breast_metadata.SITKToScan(prone_image_transformed, orientation_flag, load_dicom=False,
                                                        swap_axes=True)
    prone_image_transformed_grid = breast_metadata.SCANToPyvistaImageGrid(prone_scan_transformed, orientation_flag)

    sternum_prone_transformed_px = prone_scan_transformed.getPixelCoordinates(sternum_prone_transformed)
    sternum_supine_px = supine_scan.getPixelCoordinates(sternum_supine)
    nipple_prone_transformed_px = prone_scan_transformed.getPixelCoordinates(nipple_prone_transformed)
    nipple_supine_px = supine_scan.getPixelCoordinates(nipple_supine_transformed)
    landmark_prone_r1_transformed_px = prone_scan_transformed.getPixelCoordinates(landmark_prone_r1_transformed)
    landmark_prone_r2_transformed_px = prone_scan_transformed.getPixelCoordinates(landmark_prone_r2_transformed)
    landmark_supine_r1_px = supine_scan.getPixelCoordinates(landmark_supine_r1)
    landmark_supine_r2_px = supine_scan.getPixelCoordinates(landmark_supine_r2)

    # %%   plot
    # #the prone and supine ribcage point clouds before and after alignment
    plotter = pv.Plotter()
    plotter.add_text("Landmark Displacement After Alignment", font_size=24)

    # Plot the target supine landmarks (e.g., in green)
    plotter.add_points(landmark_supine_r1, render_points_as_spheres=True, color='green', point_size=10,
                       label='Supine Landmarks'
                       )
    # Plot the aligned prone landmarks (e.g., in red)
    plotter.add_points(landmark_prone_r1_transformed, render_points_as_spheres=True, color='red', point_size=10,
                       label='Aligned Prone Landmarks'
                       )
    # Add arrows to show the displacement vectors
    for start_point, vector in zip(landmark_prone_r1_transformed, landmark_r1_displacement_vectors):
        plotter.add_arrows(start_point, vector, mag=1.0, color='yellow')

    opacity = np.linspace(0, 0.1, 100)

    plotter.add_volume(prone_image_transformed_grid, opacity=opacity, cmap='grey', show_scalar_bar=False)
    plotter.add_volume(supine_image_grid, opacity=opacity, cmap='coolwarm', show_scalar_bar=False)

    plotter.add_points(sternum_prone_transformed, render_points_as_spheres=True, color='black', point_size=10,
                       label='Aligned Prone Sternum'
                       )
    plotter.add_points(sternum_supine, render_points_as_spheres=True, color='white', point_size=10,
                       label='Supine Sternum'
                       )
    plotter.add_points(nipple_prone_transformed, render_points_as_spheres=True, color='pink', point_size=8,
                       label='Aligned Prone Nipples'
                       )
    plotter.add_points(nipple_supine, render_points_as_spheres=True, color='pink', point_size=8,
                       label='Supine Nipples'
                       )
    plotter.add_legend()
    plotter.show(auto_close=False, interactive_update=True)
    time.sleep(5)
    plotter.close()

    # %%   scalar colour map (red-blue) to show alignment of prone transformed and supine MRIs
    breast_metadata.visualise_alignment_with_landmarks(
        supine_image_sitk, prone_image_transformed, sternum_supine_px[0], sternum_prone_transformed_px[0],
        orientation='axial')
    breast_metadata.visualise_alignment_with_landmarks(
        supine_image_sitk, prone_image_transformed, sternum_supine_px[1], sternum_prone_transformed_px[1],
        orientation='axial')

    # Loop through each landmark and create a visualization
    for i in range(len(landmark_supine_r1_px)):
        print(f"Visualizing alignment for landmark #{i + 1}")
        breast_metadata.visualise_alignment_with_landmarks(
            supine_image_sitk,
            prone_image_transformed,
            landmark_supine_r1_px[i],
            landmark_prone_r1_transformed_px[i],
            orientation='axial'
        )

    # Loop through each landmark and create a visualization
    for i in range(len(landmark_supine_r1_px)):
        print(f"Visualizing alignment for landmark #{i + 1}")
        breast_metadata.visualise_alignment_with_landmarks(
            supine_image_sitk,
            prone_image_transformed,
            landmark_supine_r1_px[i],
            landmark_prone_r1_transformed_px[i],
            orientation='sagittal'
        )

    # Loop through each landmark and create a visualization
    for i in range(len(landmark_supine_r1_px)):
        print(f"Visualizing alignment for landmark #{i + 1}")
        breast_metadata.visualise_alignment_with_landmarks(
            supine_image_sitk,
            prone_image_transformed,
            landmark_supine_r1_px[i],
            landmark_prone_r1_transformed_px[i],
            orientation='coronal'
        )

        # %%   visualise relative displacement vectors using glyphs
        points = pv.PolyData(landmark_prone_r1_transformed)

        # Attach the relative vectors as data to these points
        points['relative_motion'] = landmark_r1_rel_nipple_vectors

        # Generate arrows (glyphs) based on the vector data
        # The arrows are oriented and scaled by  'relative_motion' vectors
        arrows = points.glyph(
            orient='relative_motion',
            scale='relative_motion',
            factor=1.0  # Adjust this factor to make arrows larger or smaller
        )

        plotter = pv.Plotter()
        plotter.add_mesh(arrows, color='yellow', label='Relative Motion')
        plotter.add_points(landmark_supine_r1, color='red', render_points_as_spheres=True, point_size=10,
                           label='Supine Landmarks')
        plotter.add_legend()
        plotter.add_axes()
        plotter.show(auto_close=False, interactive_update=True)
        time.sleep(5)
        plotter.close()


    # ==========================================================
    # %% RETURN RESULTS (Refactored to a dictionary)
    # ==========================================================
    results = {
        "vl_id": subject.subject_id,
        "T_optimal": T_optimal,
        "T_degrees_translation": res_optimal.x,
        "sternum_error": sternum_error,
        "ribcage_error_mean": np.mean(rib_error_mag),
        "ribcage_error_std": np.std(rib_error_mag),
        "nipple_displacement_vectors": nipple_displacement_vectors,
        "nipple_displacement_magnitudes": nipple_displacement_magnitudes,
        "r1_displacement_vectors": landmark_r1_displacement_vectors,
        "r1_displacement_magnitudes": landmark_r1_displacement_magnitudes,
        "r2_displacement_vectors": landmark_r2_displacement_vectors,
        "r2_displacement_magnitudes": landmark_r2_displacement_magnitudes,
        "r1_rel_nipple_vectors": landmark_r1_rel_nipple_vectors,
        "r1_rel_nipple_magnitudes": landmark_r1_rel_nipple_mag,
        "r2_rel_nipple_vectors": landmark_r2_rel_nipple_vectors,
        "r2_rel_nipple_magnitudes": landmark_r2_rel_nipple_mag,
        "prone_image_transformed": prone_image_transformed
    }

    return results


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

    This function assumes the i-th landmark in 'correspondences[vl_id]'
    matches the i-th entry in the alignment_results arrays (e.g.,
    'r1_displacement_magnitudes'[i]).
    """

    # --- Internal helper to build rows for one registrar ---
    def _build_registrar_data(
            registrar_name: str,
            registrar_id: int
    ) -> List[Dict[str, any]]:

        all_rows = []

        # Master loop: Iterate through the corresponding landmarks
        for vl_id, pairs in correspondences.items():

            # --- Get Subject-Level Data ---
            if vl_id not in all_subjects:
                continue
            subject = all_subjects[vl_id]
            align_res = alignment_results_all.get(vl_id)  # Safe get

            # Get subject-level alignment data (nipples)
            if align_res:
                nipple_disp_mag = align_res.get("nipple_displacement_magnitudes", [None, None])
                nipple_disp_vec = align_res.get("nipple_displacement_vectors", [[None] * 3, [None] * 3])
            else:
                nipple_disp_mag = [None, None]
                nipple_disp_vec = [[None] * 3, [None] * 3]

            # --- Loop through each landmark pair for this subject ---
            for i, pair in enumerate(pairs):

                # Get the correct landmark name and alignment data for this registrar
                if registrar_id == 1:
                    lm_name = pair[0]
                    reg_key = "r1"
                else:  # registrar_id == 2
                    lm_name = pair[1]
                    reg_key = "r2"

                # Get landmark-specific alignment data
                if align_res:
                    lm_disp_mag = align_res.get(f"{reg_key}_displacement_magnitudes", [None] * len(pairs))[i]
                    lm_disp_vec = align_res.get(f"{reg_key}_displacement_vectors", [[None] * 3] * len(pairs))[i]
                    lm_rel_mag = align_res.get(f"{reg_key}_rel_nipple_magnitudes", [None] * len(pairs))[i]
                    lm_rel_vec = align_res.get(f"{reg_key}_rel_nipple_vectors", [[None] * 3] * len(pairs))[i]
                else:
                    lm_disp_mag, lm_disp_vec, lm_rel_mag, lm_rel_vec = None, [None] * 3, None, [None] * 3

                # --- Helper to safely get nested dictionary data ---
                def get_data(results_dict: Dict, position: str, data_key: str, default: any = None) -> any:
                    try:
                        if data_key in ["skin_distances", "rib_distances", "skin_neighborhood_avg"]:
                            # This data is in distance_results
                            return results_dict[vl_id][position][registrar_name][data_key][lm_name]
                        else:
                            # This data is in clockface_results
                            return results_dict[vl_id][position][registrar_name][lm_name][data_key]
                    except (KeyError, TypeError):
                        return default

                print(clockface_results[vl_id]["prone"])
                print(clockface_results[vl_id]["prone"][registrar_name])
                print(clockface_results[vl_id]["prone"][registrar_name][lm_name])
                print(clockface_results[vl_id]["prone"][registrar_name][lm_name]["side"])


                # --- Build the row for this single landmark ---
                row_data = {
                    'Registrar': registrar_id,
                    'VL_ID': vl_id,
                    'Age': subject.age,
                    'Height [m]': subject.height,
                    'Weight [kg]': subject.weight,
                    'Landmark Name': lm_name,
                    'Landmark type': lm_name.split('_')[0],

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

                    'Landmark displacement [mm]': lm_disp_mag,
                    'Landmark displacement relative to nipple [mm]': lm_rel_mag,
                    'Left nipple displacement [mm]': nipple_disp_mag[0],
                    'Right nipple displacement [mm]': nipple_disp_mag[1],

                    'Landmark displacement vector vx': lm_disp_vec[0],
                    'Landmark displacement vector vy': lm_disp_vec[1],
                    'Landmark displacement vector vz': lm_disp_vec[2],

                    'Landmark relative to nipple vector vx': lm_rel_vec[0],
                    'Landmark relative to nipple vector vy': lm_rel_vec[1],
                    'Landmark relative to nipple vector vz': lm_rel_vec[2],

                    'Left nipple displacement vector vx': nipple_disp_vec[0][0],
                    'Left nipple displacement vector vy': nipple_disp_vec[0][1],
                    'Left nipple displacement vector vz': nipple_disp_vec[0][2],

                    'Right nipple displacement vector vx': nipple_disp_vec[1][0],
                    'Right nipple displacement vector vy': nipple_disp_vec[1][1],
                    'Right nipple displacement vector vz': nipple_disp_vec[1][2]
                }
                all_rows.append(row_data)
        print(json.dumps(all_rows, indent=2))

        return all_rows

    # --- Main function logic ---

    # Create the output directory if it doesn't exist
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    # Build data for Registrar 1 ("anthony")
    print(f"Building data for Registrar 1 ...")
    r1_data = _build_registrar_data(
        registrar_name="anthony",
        registrar_id=1
    )
    df_r1 = pd.DataFrame(r1_data)

    # Build data for Registrar 2 ("holly")
    print(f"Building data for Registrar 2...")
    r2_data = _build_registrar_data(
        registrar_name="holly",
        registrar_id=2
    )
    df_r2 = pd.DataFrame(r2_data)

    # Combine and save
    print("Combining dataframes...")
    df_combined = pd.concat([df_r1, df_r2], ignore_index=True)

    df_combined.to_excel(excel_path, index=False)
    print(f"Data successfully saved to {excel_path}")

