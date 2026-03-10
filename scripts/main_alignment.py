"""
Stage 2: Alignment — Load, align, compute per-registrar displacements, save.

Can run independently of main_process_data.py.
Requires prone ribcage mesh and supine ribcage segmentation files.

Steps:
    1. load_subject() for all VL_IDs
    2. find_corresponding_landmarks() → correspondences
    3. add_averaged_landmarks() → adds "average" registrar
    4. For each subject: align_prone_to_supine_optimal()
       (internally computes ld_anthony_*, ld_holly_*, ld_ave_* displacements)
    5. save_alignment_results_to_excel() — 3 sheets (r1, r2, ave)
    6. Save transformation matrices as .npy
"""

from pathlib import Path
from typing import Dict

import numpy as np

from readers import load_subject
from structures import Subject
from utils import (
    find_corresponding_landmarks,
    add_averaged_landmarks,
    save_alignment_results_to_excel,
    save_alignment_metrics,
)
from alignment import align_prone_to_supine_optimal


# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_PATH_MRI = Path(r'U:\projects\volunteer_camri\old_data\mri_t2')
SOFT_TISSUE_ROOT = Path(r'U:\projects\dashboard\picker_points')
ANATOMICAL_JSON_BASE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results")

PRONE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh")
SUPINE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage")

OUTPUT_DIR = Path("../output")
EXCEL_FILE_PATH = OUTPUT_DIR / "landmark_results_v7_2026_03_10.xlsx"

OUTPUT_DIR_T_MATRIX = Path(r"../output/alignment/transformation_matrix_v7")
OUTPUT_DIR_T_MATRIX.mkdir(parents=True, exist_ok=True)

# ── Subjects ───────────────────────────────────────────────────────────────
# VL_IDS = [9,11,12,14,15,17,18,19,20,22,25,29,30,31]
# VL_IDS = [32,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50]
# VL_IDS = [51,52,54,56,57,58,59,60,61,63,64,65,66,67,68,69]
# VL_IDS = [70,71,72,74,75,76,77,78,79,81,82,84,85,86,87,88,89]
VL_IDS = [22]
POSITIONS = ["prone", "supine"]

# Per-subject point cloud inferior trim (mm).
# Subjects with segmentation artifacts extending below the ribcage.
PC_INFERIOR_TRIM = {
    22: 55.0,
    54: 15.0,
}

print(f"Number of participants: {len(VL_IDS)}")

# ── 1. Load subjects ──────────────────────────────────────────────────────
all_subjects: Dict[int, Subject] = {}

for vl_id in VL_IDS:
    vl_id_str = f"VL{vl_id:05d}"
    print(f"--- Loading Subject: {vl_id_str} ---")

    subject = load_subject(
        vl_id=vl_id,
        positions=POSITIONS,
        # dicom_root=ROOT_PATH_MRI,
        dicom_root=None,
        anatomical_json_base_root=ANATOMICAL_JSON_BASE_ROOT,
        soft_tissue_root=SOFT_TISSUE_ROOT,
    )

    if subject.scans:
        all_subjects[vl_id] = subject

# ── 2-3. Correspondences + averaged landmarks ─────────────────────────────
correspondences, all_subjects_filtered = find_corresponding_landmarks(all_subjects)
all_subjects_filtered = add_averaged_landmarks(all_subjects_filtered, correspondences)

# ── 4. Alignment ─────────────────────────────────────────────────────────
print("\n--- Starting Prone-to-Supine Alignment ---")
print(f"Subjects: {list(all_subjects_filtered.keys())}")
alignment_results_all = {}

for vl_id, filtered_subject in all_subjects_filtered.items():
    vl_id_str = filtered_subject.subject_id
    print(f"\n--- Running Alignment for {vl_id_str} ---")

    try:
        prone_mesh_file = PRONE_RIBCAGE_ROOT / f"{vl_id_str}_ribcage_prone.mesh"
        supine_seg_file = SUPINE_RIBCAGE_ROOT / f"rib_cage_{vl_id_str}.nii.gz"

        if not prone_mesh_file.exists():
            print(f"Skipping: Prone mesh not found at {prone_mesh_file}")
            continue
        if not supine_seg_file.exists():
            print(f"Skipping: Supine segmentation not found at {supine_seg_file}")
            continue

        alignment_results = align_prone_to_supine_optimal(
            subject=filtered_subject,
            prone_ribcage_mesh_path=prone_mesh_file,
            supine_ribcage_seg_path=supine_seg_file,
            orientation_flag='RAI',
            plot_for_debug=True,
            selected_elements=[0, 1, 6, 7, 8, 9, 14, 15, 16, 17, 22, 23],
            use_initial_rotation=True,
            mutual_region_padding_reciprocal=15,
            pc_inferior_trim=PC_INFERIOR_TRIM.get(vl_id, 0.0),
            max_correspondence_distance=1e6,
            trim_percentage=0,
            visualize_iterations=False,
            visualize_every_n=50
        )


        alignment_results_all[vl_id] = alignment_results

        print(f"  Alignment for {vl_id_str} COMPLETE")
        print(f"  Ribcage Error (Mean): {alignment_results['ribcage_error_mean']:.2f} mm")

    except Exception as e:
        print(f"!!! Alignment for {vl_id_str} FAILED: {e}")

print(f"\nSubjects with alignment results: {list(alignment_results_all.keys())}")

# ── 5. Save alignment results to Excel (3 registrar sheets) ──────────────
save_alignment_results_to_excel(
    excel_path=EXCEL_FILE_PATH,
    correspondences=correspondences,
    all_subjects=all_subjects_filtered,
    alignment_results_all=alignment_results_all,
)

# ── 6. Save transformation matrices + alignment metrics ──────────────────
for vl_id, results in alignment_results_all.items():
    vl_id_str = f"VL{vl_id:05d}"
    matrix_path = OUTPUT_DIR_T_MATRIX / f"{vl_id_str}_transform_matrix.npy"
    np.save(matrix_path, results["T_total"])
    save_alignment_metrics(OUTPUT_DIR_T_MATRIX, vl_id, results)
    print(f"Saved transformation matrix + metrics for {vl_id_str}")

print("\n=== main_alignment.py complete ===")
