from pathlib import Path
from readers import load_subject
from typing import Dict
from utils import mri_to_pyvista, find_corresponding_landmarks
import pyvista as pv
import json
from structures import Subject


# --- Define all your root paths ---
ROOT_PATH_MRI = Path(r'U:\projects\volunteer_camri\old_data\mri_t2')
SOFT_TISSUE_ROOT = Path(r'U:\projects\dashboard\picker_points')
ANATOMICAL_JSON_BASE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results")

# --- Define subjects to load ---
VL_IDS = [9]
POSITIONS = ["prone", "supine"]

# This dictionary will store all loaded subjects
all_subjects: Dict[int, Subject] = {}

# --- Loop through subjects ONLY ---
for vl_id in VL_IDS:
    vl_id_str = f"VL{vl_id:05d}"
    print(f"--- Loading Subject: {vl_id_str} ---")

    # Call the new "load one subject" function
    subject = load_subject(
        vl_id=vl_id,
        positions=POSITIONS,  # Pass both positions
        dicom_root=ROOT_PATH_MRI,
        anatomical_json_base_root=ANATOMICAL_JSON_BASE_ROOT,
        soft_tissue_root=SOFT_TISSUE_ROOT
    )

    # Add to dict if any scans were successfully loaded
    if subject.scans:
        all_subjects[vl_id] = subject

# --- Example: Accessing the new data ---
if 9 in all_subjects:
    print("\n--- Example: Accessing data for VL00009 ---")

    # Access shared data
    subject_9 = all_subjects[9]
    print(f"  ID: {subject_9.subject_id}")
    print(f"  Age: {subject_9.age}")

    # Access position-specific data
    if "prone" in subject_9.scans:
        prone_scan = subject_9.scans["prone"]
        print(f"  Prone Image Shape: {prone_scan.mri_image.image_array.shape}")

    if "supine" in subject_9.scans:
        supine_scan = subject_9.scans["supine"]
        print(f"  Supine Left Nipple: {supine_scan.anatomical_landmarks.nipple_left}")

        # Access registrar data for a specific scan
        print("\nRegistrar Data:")
        for registrar_name in supine_scan.registrar_data:
            print(f"  Registrar: {registrar_name}")


# --- 2. Run the analysis function ---
print("\n--- Running landmark correspondence analysis ---")
correspondences = find_corresponding_landmarks(all_subjects)

print("\n--- Analysis Results ---")
print(json.dumps(correspondences, indent=2))


#
# # --- How to access the new data ---
# print(f"Data for Subject: {subject.subject_id}")
# print(f"Anatomical Left Nipple: {subject.anatomical_landmarks.nipple_left}")
#
# print("\nRegistrar Data:")
# for registrar_name, data in subject.registrar_data.items():
#     print(f"  Registrar: {registrar_name}")
#
#     # You can access their specific landmarks
#     if "cyst" in data.soft_tissue_landmarks:
#         print(f"    Cyst 1 position: {data.soft_tissue_landmarks['cyst']}")
#
#     print(f"    Total landmarks found: {len(data.soft_tissue_landmarks)}")
#

# --- CONVERT AND PLOT ---
subject = all_subjects[9]
# 1. Get the MRImage object
mri_data = subject.scans['supine'].mri_image

# 2. Convert to PyVista grid
pv_grid = mri_to_pyvista(mri_data)

# 3. Plot the grid!
# (e.g., as a volume)
print(f"Plotting volume for {subject.subject_id}...")
pv_grid.plot(volume=True, cmap="bone")

# (e.g., as orthogonal slices)
# pv_grid.plot_orthogonal(opacity=0.5, use_panel=False)