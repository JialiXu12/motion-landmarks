# Motion Landmarks — User Guide

## Overview

Motion Landmarks is a pipeline for quantifying breast tissue landmark displacement between prone (face-down) and supine (face-up) MRI positions. It loads MRI data and manually annotated landmarks, aligns ribcage structures between positions using rigid-body registration, and computes displacement metrics. Statistical analysis is performed in a separate project.

---

## Quick Start

### Prerequisites

- Python 3.10+ with conda environment `camri` (or equivalent with all dependencies)
- Access to network drives:
  - `U:\projects\volunteer_camri\old_data\mri_t2` (DICOM images)
  - `U:\projects\dashboard\picker_points` (soft tissue annotations)
  - `U:\sandbox\jxu759\volunteer_seg\results` (anatomical landmarks + segmentations)
  - `U:\sandbox\jxu759\volunteer_prone_mesh` (prone ribcage mesh files)
- Required Python packages: numpy, scipy, pandas, openpyxl, morphic, pyvista, SimpleITK, matplotlib, scikit-image, scikit-learn, open3d

### Running the Full Pipeline

```bash
cd scripts/
python main.py
```

This runs all stages: load data, compute distances, compute clock positions, align prone-to-supine, save results, and plot displacement vectors.

### Running Stages Independently

**Stage 1+2 — Process data (no alignment):**
```bash
python main_process_data.py
```
Loads subjects, computes landmark distances to skin/rib cage, computes clock-face coordinates, and saves to Excel (columns A-V in `processed_*` sheets).

**Stage 3 — Alignment only:**
```bash
python main_alignment.py
```
Loads subjects, runs prone-to-supine alignment, computes landmark displacements, and merges results into existing Excel sheets (columns W onwards). Also saves transformation matrices as `.npy` files and alignment error metrics as `.json` sidecar files.

**Note:** Statistical analysis is performed in a separate project that reads the Excel output from the stages above.

---

## Configuring Which Subjects to Process

Each entry point script has a `VL_IDS` list near the top. Edit this list to control which subjects are processed:

```python
# Process a single subject
VL_IDS = [9]

# Process a batch
VL_IDS = [9, 11, 12, 14, 15, 17, 18, 19, 20, 22, 25, 29, 30, 31]

# Full cohort (uncomment all batches)
VL_IDS = [9,11,12,14,15,17,18,19,20,22,25,29,30,31,
          32,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,
          51,52,54,56,57,58,59,60,61,63,64,65,66,67,68,69,
          70,71,72,74,75,76,77,78,79,81,82,84,85,86,87,88,89]
```

---

## Configuring Alignment Parameters

In `main_alignment.py` (or `main.py`), the alignment call accepts several parameters:

```python
alignment_results = align_prone_to_supine_optimal(
    subject=filtered_subject,
    prone_ribcage_mesh_path=prone_mesh_file,
    supine_ribcage_seg_path=supine_seg_file,
    orientation_flag='RAI',               # Coordinate system convention
    plot_for_debug=True,                  # Show intermediate 3D plots
    selected_elements=[0,1,6,7,8,9,14,15,16,17,22,23],  # Anterior rib elements
    use_initial_rotation=True,            # SVD pre-rotation
    mutual_region_padding_reciprocal=15,  # Region overlap filtering
    max_correspondence_distance=1e6,      # KD-tree search radius (mm)
    trim_percentage=0,                    # Outlier rejection fraction
    visualize_iterations=True,            # Show ICP convergence
    visualize_every_n=50                  # Plot every N iterations
)
```

| Parameter | Default | When to Change |
|-----------|---------|---------------|
| `plot_for_debug` | `True` | Set `False` for batch processing (no GUI) |
| `selected_elements` | `[0,1,6,7,...]` | Change if ribcage mesh topology differs |
| `trim_percentage` | `0` | Increase (e.g., 0.1) if alignment is noisy |
| `visualize_iterations` | `True` | Set `False` for batch processing |

---

## Applying Saved Alignment Results

After alignment has been run and transformation matrices saved, you can reload, compute displacements, save to Excel, and visualize — all without re-running alignment:

```bash
# Default: apply alignment for VL00009
python apply_saved_alignment.py

# Specify one or more subjects
python apply_saved_alignment.py --vl_id 9 22 31

# Specify T_matrix directory and Excel output path
python apply_saved_alignment.py --vl_id 9 22 --t_matrix_dir ../output/alignment/transformation_matrix_v7 --excel_path ../output/landmark_results_v7.xlsx

# Skip the 3D plot (useful for batch processing)
python apply_saved_alignment.py --vl_id 9 22 31 --no_plot
```

This script:
1. Loads the saved 4x4 transformation matrix (`.npy`) and alignment error metrics (`.json` sidecar)
2. Transforms all prone data (mesh, sternum, nipples, landmarks)
3. Computes per-registrar displacement vectors (anthony, holly, average)
4. Saves results to Excel (same format as `main_alignment.py`)
5. Displays an interactive 3D plot (unless `--no_plot`):
   - Supine ribcage point cloud (tan)
   - Aligned prone ribcage mesh (blue)
   - Sternum superior & inferior — prone-transformed (black) vs supine (blue)
   - Nipples — prone-transformed (red) vs supine (green)
   - Soft-tissue landmarks with displacement arrows (yellow)

### Using Programmatically

```python
from apply_saved_alignment import run_apply_saved_alignment, build_alignment_results

# Full pipeline: load, transform, compute displacements, save Excel, plot
run_apply_saved_alignment(vl_ids=[9, 22], no_plot=True)

# Or build alignment results dict for a single subject (same format as align_prone_to_supine_optimal output)
results = build_alignment_results(vl_id=9, t_matrix_dir=Path("../output/alignment/transformation_matrix_v7"))
```

---

## Output Files

### Excel Workbook

**Location:** `output/landmark_results_v7_2026_03_10.xlsx`

**Sheets:**

| Sheet | Contents | Produced By |
|-------|----------|-------------|
| `raw_data` | Raw landmark coordinates per registrar per position | `main.py` or `main_process_data.py` |
| `processed_r1_data` | Anthony's landmarks — distances, clock, alignment | Stage 2 + Stage 3 |
| `processed_r2_data` | Holly's landmarks — same structure | Stage 2 + Stage 3 |
| `processed_ave_data` | Averaged landmarks — same structure | Stage 2 + Stage 3 |
| `demographic` | Subject demographics (age, height, weight) | `main.py` |

**Key columns in processed sheets:**

| Column | Description | Unit |
|--------|-------------|------|
| VL_ID | Subject identifier | — |
| Landmark name | e.g., "cyst_1" | — |
| Distance to nipple (prone) | 3D distance from landmark to nipple | mm |
| Distance to skin (prone) | Closest distance to skin surface | mm |
| Distance to rib cage (prone) | Closest distance to rib cage | mm |
| Time (prone) | Clock position relative to nipple | hours (1-12) |
| Quadrant (prone) | UO, UI, LO, or LI | — |
| ribcage error rmse | Full-mesh alignment error | mm |
| ribcage anterior rmse | Anterior-only alignment error | mm |
| Landmark displacement [mm] | How far the landmark moved | mm |
| Landmark displacement vx/vy/vz | Displacement vector components | mm |

### Transformation Matrices & Metrics

**Location:** `output/alignment/transformation_matrix_v7/`

Each subject produces two files:
- `VL{id}_transform_matrix.npy` — 4x4 NumPy array (homogeneous transformation)
- `VL{id}_alignment_metrics.json` — JSON sidecar with ribcage error metrics (RMSE, mean, std for full mesh and anterior region, plus sternum error)

**Usage:**
```python
import numpy as np

T = np.load("output/alignment/transformation_matrix_v7/VL00009_transform_matrix.npy")
# Transform prone point to supine frame:
point_prone = np.array([x, y, z, 1])
point_supine = T @ point_prone  # first 3 elements are x, y, z
```

The JSON sidecar is loaded automatically by `apply_saved_alignment.py` and `load_alignment_metrics()`. These metrics are produced during alignment and cannot be recomputed from the T_matrix alone.

### Figures

**Location:** `output/figs/`

| Subdirectory | Contents |
|-------------|----------|
| `clock_analysis/` | Clock position rotation polar plots |
| `landmark vectors/` | 3-panel displacement vector plots (axial/coronal/sagittal) |
| Root | Alignment comparison, convergence, displacement mechanism |

---

## Understanding the Pipeline

### What is "Correspondence"?

Two human annotators (Anthony and Holly) independently identify soft-tissue landmarks in each MRI. The pipeline finds **corresponding** landmarks — those where both annotators agree on the position (within 3mm) in both prone and supine scans. Only corresponding landmarks are used for displacement analysis. An **averaged** registrar is also created by averaging the two annotators' coordinates.

### What is "Alignment"?

The prone and supine MRI scans are in different coordinate frames. The alignment step finds a rigid-body transformation (rotation + translation) that maps the prone ribcage onto the supine ribcage. This is done by:
1. Fixing the sternum superior at the origin
2. Rotating the prone ribcage mesh to match the supine ribcage point cloud
3. Applying the same transform to all prone landmarks

After alignment, displacement vectors represent true tissue deformation, not just coordinate frame differences.

### What is "Clock Position"?

Each landmark's position is expressed as a clock time (1-12 o'clock) and quadrant relative to the nipple:
- 12 o'clock = superior (towards head)
- 3 o'clock = lateral (away from midline, but depends on breast side)
- 6 o'clock = inferior (towards feet)
- 9 o'clock = medial (towards midline)

Quadrants: UO = Upper Outer, UI = Upper Inner, LO = Lower Outer, LI = Lower Inner

---

## Troubleshooting

### "Registrar 'average' not found in prone scan"

This warning appears when running `alignment.py` directly (its `__main__` block), because it does not call `add_averaged_landmarks()` first. It is harmless — anthony and holly results are still computed. When using `main_alignment.py` or `main.py`, the averaged landmarks are created before alignment.

### Alignment fails for a subject

Common causes:
- **Missing files:** prone mesh (`*_ribcage_prone.mesh`) or supine segmentation (`rib_cage_*.nii.gz`) not found
- **Poor mesh quality:** some subjects have low-quality ribcage meshes
- **Convergence failure:** ICP did not converge — try increasing `max_iterations` or adjusting `trim_percentage`

### Excel has stale columns from old code

Delete the output Excel file and re-run from scratch. Old column names from previous code versions persist in existing files.

### ModuleNotFoundError

Ensure you are using the correct Python environment with all dependencies installed. The project requires specialized packages (`morphic`, `breast_metadata`) that are available as git submodules in `external/`.

---

## Project Structure

```
motion-landmarks/
├── scripts/                    # All Python source code
│   ├── main.py                 # Full combined pipeline (backward compatible)
│   ├── main_process_data.py    # Stage 1+2: load, distances, clockface
│   ├── main_alignment.py       # Stage 3: alignment + displacement
│   ├── apply_saved_alignment.py # Load saved T_matrix, compute displacements, save Excel, visualize
│   ├── structures.py           # Dataclasses (Subject, ScanData, etc.)
│   ├── readers.py              # Data loading from disk
│   ├── utils.py                # Core utilities (distances, Excel I/O)
│   ├── alignment.py            # Alignment orchestrator
│   ├── alignment_preprocessing.py # Initial rotation, mesh selection
│   ├── alignment_utils.py      # Transform math, point cloud filtering
│   ├── surface_to_point_alignment.py # ICP solver
│   └── utils_plot.py           # 3D visualization (PyVista)
├── external/                   # Git submodules
│   ├── breast_metadata_mdv/    # DICOM/NIfTI I/O
│   ├── bmw/                    # Breast biomechanics
│   └── automesh/               # Mesh generation
├── output/                     # Results (Excel, .npy, figures)
└── docs/                       # Documentation
```

---

## Running Tests

```bash
cd scripts/

# Test the script separation (main.py → main_process_data.py + main_alignment.py)
python test_script_separation.py

# Test per-registrar displacement computation
python test_per_registrar_displacements.py

# Test apply_saved_alignment (transform + plot)
python test_apply_saved_alignment.py

# Test alignment functions (requires full environment)
python test_surface_to_point_alignment.py
```

Note: Some tests require the full conda environment with all dependencies (`morphic`, `breast_metadata`, etc.). The `test_apply_saved_alignment.py` tests are self-contained and run with standard Python 3.10+ (only needs `numpy` and `pyvista`).
