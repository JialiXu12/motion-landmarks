"""
Stage 1: Process Data — Load, filter, distances, clockface, save.

No alignment dependency. Can run independently of main_alignment.py.

Steps:
    1. load_subject() for all VL_IDs
    2. save_raw_data_to_excel() — raw per-registrar positions
    3. find_corresponding_landmarks() → correspondences
    4. add_averaged_landmarks() → adds "average" key
    5. calculate_landmark_distances() — per-registrar
    6. calculate_clockface_coordinates() — per-registrar
    7. save_processed_data_to_excel() — 3 sheets (r1, r2, ave)
"""

from pathlib import Path
from typing import Dict

from readers import load_subject
from structures import Subject
from utils import (
    find_corresponding_landmarks,
    add_averaged_landmarks,
    calculate_landmark_distances,
    analyse_landmark_distances,
    calculate_clockface_coordinates,
    save_raw_data_to_excel,
    save_processed_data_to_excel,
)


# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_PATH_MRI = Path(r'U:\projects\volunteer_camri\old_data\mri_t2')
SOFT_TISSUE_ROOT = Path(r'U:\projects\dashboard\picker_points')
ANATOMICAL_JSON_BASE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results")
SEGMENTATION_ROOT = Path(r'U:\sandbox\jxu759\volunteer_seg\results')

OUTPUT_DIR = Path("../output")
EXCEL_FILE_PATH = OUTPUT_DIR / "landmark_results_v7_2026_03_10.xlsx"

# ── Subjects ───────────────────────────────────────────────────────────────
# VL_IDS = [9,11,12,14,15,17,18,19,20,22,25,29,30,31]
# VL_IDS = [32,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50]
# VL_IDS = [51,52,54,56,57,58,59,60,61,63,64,65,66,67,68,69]
# VL_IDS = [70,71,72,74,75,76,77,78,79,81,82,84,85,86,87,88,89]
VL_IDS = [12]
POSITIONS = ["prone", "supine"]

print(f"Number of participants: {len(VL_IDS)}")

# ── 1. Load subjects ──────────────────────────────────────────────────────
all_subjects: Dict[int, Subject] = {}

for vl_id in VL_IDS:
    vl_id_str = f"VL{vl_id:05d}"
    print(f"--- Loading Subject: {vl_id_str} ---")

    subject = load_subject(
        vl_id=vl_id,
        positions=POSITIONS,
        dicom_root=ROOT_PATH_MRI,
        anatomical_json_base_root=ANATOMICAL_JSON_BASE_ROOT,
        soft_tissue_root=SOFT_TISSUE_ROOT,
    )

    if subject.scans:
        all_subjects[vl_id] = subject

# ── 2. Save raw data ──────────────────────────────────────────────────────
save_raw_data_to_excel(EXCEL_FILE_PATH, all_subjects)

# ── 3-4. Correspondences + averaged landmarks ─────────────────────────────
correspondences, all_subjects_filtered = find_corresponding_landmarks(all_subjects)
all_subjects_filtered = add_averaged_landmarks(all_subjects_filtered, correspondences)

# ── 5. Distances ──────────────────────────────────────────────────────────
distance_results = calculate_landmark_distances(all_subjects_filtered, SEGMENTATION_ROOT)
distance_stats = analyse_landmark_distances(distance_results)

print("\n--- Landmark Distance Analysis (All Subjects) ---")
print(f"    Avg. 10-Neighbor Dist Skin: {distance_stats.get('mask_skin_neighborhood_avg', 0):.2f} mm")
print(f"    Avg. 10-Neighbor Dist Rib:  {distance_stats.get('mask_rib_neighborhood_avg', 0):.2f} mm")

# ── 6. Clockface ─────────────────────────────────────────────────────────
print("\n--- Calculating clockface coordinates ---")
clockface_results = calculate_clockface_coordinates(all_subjects_filtered)

# ── 7. Save processed data (distances + clockface, 3 registrar sheets) ───
save_processed_data_to_excel(
    excel_path=EXCEL_FILE_PATH,
    correspondences=correspondences,
    all_subjects=all_subjects_filtered,
    distance_results=distance_results,
    clockface_results=clockface_results,
)

print("\n=== main_process_data.py complete ===")
