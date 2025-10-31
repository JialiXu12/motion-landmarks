import pyvista as pv
import numpy as np
from typing import Dict, List

try:
    from .structures import Subject, MRImage
except ImportError:
    from structures import Subject, MRImage



def mri_to_pyvista(mri_image: MRImage) -> pv.ImageData:
    """
    Converts MRImage dataclass into a PyVista UniformGrid.
    """
    # 1. Get the data from the MRImage object
    image_array = mri_image.image_array  # This has shape (Z, Y, X)
    spacing = mri_image.spacing  # This is (X_sp, Y_sp, Z_sp)
    origin = mri_image.origin  # This is (X_o, Y_o, Z_o)

    # 2. Get dimensions in (X, Y, Z) order
    # The image_array shape is (Z, Y, X), so we reverse it.
    dimensions = (
        image_array.shape[2],  # X dim
        image_array.shape[1],  # Y dim
        image_array.shape[0]  # Z dim
    )

    # 3. Create the PyVista ImageData
    grid = pv.ImageData(
        dimensions=dimensions,
        spacing=spacing,
        origin=origin
    )

    # 4. Add the pixel data
    # Use .ravel(order="F") to efficiently re-order the
    # (Z, Y, X) numpy array into the (X, Y, Z) VTK/PyVista expects.
    grid.point_data["values"] = image_array.ravel(order="F")

    return grid




def calculate_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Calculates the Euclidean distance between two 3D points."""
    # This check prevents errors if a landmark was None
    if point_a is None or point_b is None:
        return float('inf')
    return np.linalg.norm(point_a - point_b)


def find_corresponding_landmarks(
        all_subjects: Dict[int, Subject]
) -> Dict[int, List[List[str]]]:
    """
    Checks for corresponding landmarks between registrars "a" and "b"
    across "prone" and "supine" positions for each volunteer.

    A correspondence is valid if:
    1. A landmark from 'a' has exactly one match in 'b' in the prone position (<= 3mm).
    2. The corresponding supine landmarks (matched by name) are also within 3mm.
    3. The landmark types match (e.g., "cyst_1" from 'a' and "cyst_1" from 'b').
    4. The landmark type is not 'fibroadenoma'.

    Args:
        all_subject_data: The main data dictionary from your main.py:
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

        if ("ben_reviewed" not in prone_data.registrar_data or
                "holly" not in prone_data.registrar_data or
                "ben_reviewed" not in supine_data.registrar_data or
                "holly" not in supine_data.registrar_data):
            # Skip subject if missing any registrar data
            continue

        # --- 2. Get the landmark dictionaries ---
        prone_a_lms = prone_data.registrar_data["ben_reviewed"].soft_tissue_landmarks
        prone_b_lms = prone_data.registrar_data["holly"].soft_tissue_landmarks
        supine_a_lms = supine_data.registrar_data["ben_reviewed"].soft_tissue_landmarks
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

    return corre