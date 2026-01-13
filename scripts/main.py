from pathlib import Path
from readers import load_subject
from typing import Dict
from utils import (
    find_corresponding_landmarks,
    add_averaged_landmarks,
    calculate_landmark_distances,
    analyse_landmark_distances,
    calculate_clockface_coordinates,
    align_prone_to_supine,
    save_results_to_excel,
    save_raw_data_to_excel
)
import pyvista as pv
import json
from scripts.utils_plot import plot_vector_three_views_multi_subject
from structures import Subject
import numpy as np
import external.breast_metadata_mdv.breast_metadata as breast_metadata


#%% --- Define all your root paths ---
ROOT_PATH_MRI = Path(r'U:\projects\volunteer_camri\old_data\mri_t2')
SOFT_TISSUE_ROOT = Path(r'U:\projects\dashboard\picker_points')
ANATOMICAL_JSON_BASE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results")
SEGMENTATION_ROOT = Path(r'U:\sandbox\jxu759\volunteer_seg\results')

PRONE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh")
SUPINE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage")

OUTPUT_DIR = Path("../output")
# EXCEL_FILE_PATH = OUTPUT_DIR / "landmark_results_v3_2025_12.xlsx"
EXCEL_FILE_PATH = OUTPUT_DIR / "landmark_results_v5_2026_01_13.xlsx"

# OUTPUT_DIR_T_Matrix = Path(r"../output/transformation_matrix")
OUTPUT_DIR_T_Matrix = Path(r"../output/transformation_matrix_v2")
OUTPUT_DIR_T_Matrix.mkdir(parents=True, exist_ok=True)

#%% --- Define subjects to load ---
# VL_IDS = [54,56,57,58,59,60,61,63,64,65,66,67,68,69,70,71,72,74,75,76,77,78,79,81,82,84,85,86,87,88,89]
VL_IDS = [81]
# VL_IDS = [9,10,11,12,14,15,17,18,19,20,22,25,27,28,29,30,31]
# VL_IDS = [32,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
# VL_IDS = [51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69]
# VL_IDS = [70,71,72,73,74,75,76,77,78,79,81,82,83,84,85,
#           86,87,88,89]

print("Number of participants in total: ", len(VL_IDS))
# VL_IDS = [9,11,12,14,15,17,18,19,20,22,25,29,30,31,32,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,52,
#            54,56,57,58,59,60,61,63,64,65,66,67,68,69,70,71,72,74,75,76,77,78,79,81,82,84,85,86,87,88,89]
POSITIONS = ["prone", "supine"]

# This dictionary will store all loaded subjects
all_subjects: Dict[int, Subject] = {}

#%% --- Loop through subjects  ---
for vl_id in VL_IDS:
    vl_id_str = f"VL{vl_id:05d}"
    # EXCEL_FILE_PATH = OUTPUT_DIR / f"{vl_id_str}.xlsx"
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

#%% --- save raw data ---
save_raw_data_to_excel(EXCEL_FILE_PATH, all_subjects)

#%% --- Process data ---
# --- 1. calculate distances ---
correspondences, all_subjects_filtered = find_corresponding_landmarks(all_subjects)
all_subjects_filtered = add_averaged_landmarks(all_subjects_filtered, correspondences)

distance_results = calculate_landmark_distances(all_subjects_filtered, SEGMENTATION_ROOT)
distance_stats = analyse_landmark_distances(distance_results)

# --- 2. Print the results ---
print("\n--- Landmark distance Analysis (All Subjects) ---")
print(f"    Avg. 10-Neighbor Dist Skin: {distance_stats.get('mask_skin_neighborhood_avg', 0):.2f} mm")
print(f"    Avg. 10-Neighbor Dist Rib: {distance_stats.get('mask_rib_neighborhood_avg', 0):.2f} mm")

# --- 3. Calculate Clockface Coordinates ---
print("\n---Calculating clockface coordinates for filtered landmarks ---")
clockface_results = calculate_clockface_coordinates(all_subjects_filtered)

# print(clockface_results[9]["prone"])
# print(clockface_results[9]["prone"]["anthony"])
# print(clockface_results[9]["prone"]["anthony"]["cyst_1"])
# print(clockface_results[9]["prone"]["anthony"]["cyst_1"]["side"])


# --- 3. Alignment ---
print("\n--- Starting Prone-to-Supine Alignment ---")
alignment_results_all = {}  # Dictionary to store results for all subjects

for vl_id, filtered_subject in all_subjects_filtered.items():
    vl_id_str = filtered_subject.subject_id  # e.g., "VL00009"
    print(f"\n--- Running Alignment for {vl_id_str} ---")

    try:
        # --- 1. Build the paths for this subject ---
        prone_mesh_file = PRONE_RIBCAGE_ROOT / f"{vl_id_str}_ribcage_prone.mesh"
        supine_seg_file = SUPINE_RIBCAGE_ROOT / f"rib_cage_{vl_id_str}.nii.gz"

        # --- 2. Check that files exist ---
        if not prone_mesh_file.exists():
            print(f"Skipping: Prone mesh not found at {prone_mesh_file}")
            continue
        if not supine_seg_file.exists():
            print(f"Skipping: Supine segmentation not found at {supine_seg_file}")
            continue

        # --- 3. Call the alignment function ---
        alignment_results = align_prone_to_supine(
            subject=filtered_subject,
            prone_ribcage_mesh_path=prone_mesh_file,
            supine_ribcage_seg_path=supine_seg_file,
            orientation_flag='RAI',
            plot_for_debug=False
        )

        # --- 4. Store and print summary ---
        alignment_results_all[vl_id] = alignment_results

        print(f"  Alignment for {vl_id_str} COMPLETE")
        print(f"  Ribcage Error (Mean): {alignment_results['ribcage_error_mean']:.2f} mm")
        # if alignment_results['r1_displacement_magnitudes'].shape[0] > 0:
        #     avg_disp = np.mean(alignment_results['r1_displacement_magnitudes'])
        #     print(f"  Avg. Landmark Disp. (Anthony): {avg_disp:.2f} mm")

    except Exception as e:
        print(f"!!! Alignment for {vl_id_str} FAILED: {e}")

#%% --- 4. Save results to Excel ---
save_results_to_excel(
    excel_path=EXCEL_FILE_PATH,
    correspondences=correspondences,
    all_subjects=all_subjects_filtered,
    distance_results=distance_results,
    clockface_results=clockface_results,
    alignment_results_all=alignment_results_all,
)


for vl_id, filtered_subject in all_subjects_filtered.items():
    vl_id_str = f"VL{vl_id:05d}"
    matrix_path = OUTPUT_DIR_T_Matrix / f"{vl_id_str}_transform_matrix.npy"
    np.save(matrix_path, alignment_results_all[vl_id]["T_total"])

    # loaded_matrix = np.load('transform_matrix.npy')
#%% --
plot_vector_three_views_multi_subject(
    alignment_results_all=alignment_results_all,
    registrar_key="ld_ave",
    title="Displacement of landmarks from prone to supine",
)




#
# # --- Example: Accessing the new data ---
# if 9 in all_subjects:
#     print("\n--- Example: Accessing data for VL00009 ---")
#
#     # Access shared data
#     subject_9 = all_subjects[9]
#     print(f"  ID: {subject_9.subject_id}")
#     print(f"  Age: {subject_9.age}")
#
#     # Access position-specific data
#     if "prone" in subject_9.scans:
#         prone_scan = subject_9.scans["prone"]
#         print(f"  Prone Image Shape: {prone_scan.scan_object.values.shape}")
#
#     if "supine" in subject_9.scans:
#         supine_scan = subject_9.scans["supine"]
#         print(f"  Supine Left Nipple: {supine_scan.anatomical_landmarks.nipple_left}")
#
#         # Access registrar data for a specific scan
#         print("\nRegistrar Data:")
#         for registrar_name in supine_scan.registrar_data:
#             print(f"  Registrar: {registrar_name}")
#
#
#
#
# #
# # for registrar_name, data in subject.registrar_data.items():
# #     print(f"  Registrar: {registrar_name}")
# #
# #     # You can access their specific landmarks
# #     if "cyst" in data.soft_tissue_landmarks:
# #         print(f"    Cyst 1 position: {data.soft_tissue_landmarks['cyst']}")
# #
# #     print(f"    Total landmarks found: {len(data.soft_tissue_landmarks)}")
# #
#
#
# anat_landmarks = all_subjects[9].scans["supine"].anatomical_landmarks
# anat_points = [
#         anat_landmarks.nipple_left,
#         anat_landmarks.nipple_right,
#         anat_landmarks.sternum_superior,
#         anat_landmarks.sternum_inferior
#     ]
# # Convert the list of arrays into a single (N, 3) array
# anat_points_np = np.array([p for p in anat_points if p is not None and p.size > 0])
#
# soft_tissue_landmarks =  all_subjects[9].scans["supine"].registrar_data["anthony"].soft_tissue_landmarks.values()
# soft_tissue_list = list(soft_tissue_landmarks)
# anthony_points_np = np.array([p for p in soft_tissue_list if p is not None and p.size > 0])
#
# closest_skin_point = list(distance_results[9]['supine']['anthony']['skin_points'].values())
# closest_skin_point_supine_anthony = np.array([p for p in closest_skin_point if p is not None and p.size > 0])
#
#
#
#
# # --- CONVERT AND PLOT ---
# subject = all_subjects[9]
# # 1. Get the MRImage object
# scan_obj = subject.scans['supine'].scan_object
#
# # 2. Convert to PyVista grid
# # pv_grid = mri_to_pyvista(scan_obj)
# pv_grid = breast_metadata.SCANToPyvistaImageGrid(scan_obj, 'RAI')
# # anat_mesh = pv.PolyData(anat_points_np)
# # soft_tissue_ld_mesh = pv.PolyData(list(soft_tissue_landmarks))
#
#
# # 3. Plot the grid!
# # # (e.g., as a volume)
# # print(f"Plotting volume for {subject.subject_id}...")
# # pv_grid.plot(volume=True, cmap="bone")
#
# plotter = pv.Plotter()
# opacity = np.linspace(0, 0.2, 100)
# plotter.add_volume(pv_grid, scalars='values', cmap='gray', opacity=opacity)
#
# # (e.g., as orthogonal slices)
# # pv_grid.plot_orthogonal(opacity=0.5, use_panel=False)
#
# plotter.add_points(anat_points_np, color='red', render_points_as_spheres=True, point_size=10, label='Anatomical landmarks')
# plotter.add_points(anthony_points_np[0], color='blue', point_size=10, render_points_as_spheres=True, label='Soft Tissue Landmarks')
# plotter.add_points(closest_skin_point_supine_anthony[0], color='green', point_size=10, render_points_as_spheres=True, label='Closest Skin Points')
# plotter.add_legend()
# print(f"Plotting {subject.subject_id} (supine)...")
# plotter.show()