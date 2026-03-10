# Motion Landmarks — Technical Specification

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Document the full technical specification of the motion-landmarks project — a medical imaging pipeline that quantifies breast tissue landmark displacement between prone and supine MRI positions.

**Architecture:** A modular Python pipeline with 3 stages: data loading, distance/clockface computation, and rigid-body alignment. Data flows through `Subject` dataclasses, with Excel as the intermediate persistence layer between stages. Statistical analysis is performed in a separate project.

**Tech Stack:** Python 3.10+, NumPy, SciPy, pandas, morphic (mesh), Open3D, PyVista, SimpleITK, matplotlib

---

## 1. Project Purpose

This pipeline tracks soft-tissue breast landmarks (cysts, lymph nodes) across two MRI body positions — **prone** (face-down) and **supine** (face-up) — to characterize tissue deformation patterns. The clinical goal is to understand how lesion positions shift between surgical planning (supine) and imaging (prone) positions.

### Key Outputs

| Output | Description |
|--------|-------------|
| Landmark displacement vectors | 3D vectors showing how each landmark moves from prone to supine |
| Displacement magnitudes | Euclidean distance of each displacement (mm) |
| Clock-face positions | Landmark positions expressed as clock time relative to nipple |
| Ribcage alignment error | Quality metric for the prone-to-supine registration |

---

## 2. Coordinate System

All coordinates use the **RAI** (Right-Anterior-Inferior) convention:

| Axis | Direction | Clinical Meaning |
|------|-----------|-----------------|
| X | Right → Left | Medial-Lateral |
| Y | Anterior → Posterior | Front-Back |
| Z | Superior → Inferior | Head-Foot |

Origin is determined by the DICOM image origin. After alignment, the **sternum superior** is fixed at origin.

---

## 3. Data Structures

### 3.1 Core Dataclasses (`structures.py`)

```
Subject
├── subject_id: str                    # "VL00009"
├── age: Optional[str]
├── weight: Optional[float]           # kg
├── height: Optional[float]           # m
└── scans: Dict[str, ScanData]        # {"prone": ..., "supine": ...}
    └── ScanData
        ├── position: str              # "prone" or "supine"
        ├── scan_object: Scan          # 3D DICOM volume (nullable)
        ├── anatomical_landmarks: AnatomicalLandmarks
        │   ├── nipple_left: ndarray(3,)
        │   ├── nipple_right: ndarray(3,)
        │   ├── sternum_superior: ndarray(3,)
        │   └── sternum_inferior: ndarray(3,)
        └── registrar_data: Dict[str, RegistrarData]
            # {"anthony": ..., "holly": ..., "average": ...}
            └── RegistrarData
                └── soft_tissue_landmarks: Dict[str, ndarray(3,)]
                    # {"cyst_1": [x,y,z], "lymph_2": [x,y,z], ...}
```

### 3.2 Alignment Results Dictionary

Returned by `align_prone_to_supine_optimal()`:

```
{
    # Transformation
    'T_total': ndarray(4,4),           # Homogeneous transform matrix
    'R': ndarray(3,3),                 # Rotation component

    # Error metrics — full mesh
    'ribcage_error_rmse': float,       # RMSE of all mesh-to-cloud distances
    'ribcage_error_mean': float,
    'ribcage_error_std': float,

    # Error metrics — anterior (inlier) region
    'ribcage_inlier_rmse': float,
    'ribcage_inlier_mean': float,
    'ribcage_inlier_std': float,
    'sternum_error': float,            # Distance between aligned sternums

    # Transformed anatomical landmarks
    'sternum_prone_transformed': ndarray(2,3),
    'sternum_supine': ndarray(2,3),
    'nipple_prone_transformed': ndarray(2,3),
    'nipple_supine': ndarray(2,3),

    # Nipple displacements
    'nipple_displacement_magnitudes': ndarray(2,),
    'nipple_displacement_vectors': ndarray(2,3),
    'nipple_disp_left_vec': ndarray(3,),
    'nipple_disp_right_vec': ndarray(3,),

    # Per-registrar landmarks (prefixed: ld_anthony_*, ld_holly_*, ld_ave_*)
    '{prefix}_prone_transformed': ndarray(N,3),
    '{prefix}_supine': ndarray(N,3),
    '{prefix}_displacement_magnitudes': ndarray(N,),
    '{prefix}_displacement_vectors': ndarray(N,3),
    '{prefix}_rel_nipple_magnitudes': ndarray(N,),
    '{prefix}_rel_nipple_vectors': ndarray(N,3),
    ...
}
```

---

## 4. Pipeline Stages

### Stage 1: Data Loading (`readers.py`)

**Entry point:** `load_subject(vl_id, positions, dicom_root, anatomical_json_base_root, soft_tissue_root)`

**Input files per subject:**

| File | Path Pattern | Format |
|------|-------------|--------|
| DICOM series | `{dicom_root}/VL{id:05d}/{position}/` | DICOM |
| Anatomical landmarks | `{anat_root}/{position}/landmarks/VL{id}_skeleton_data_{position}_t2.json` | JSON |
| Soft tissue (Anthony) | `{soft_tissue_root}/ben_reviewed/VL{id}/point.{type}_{n}.json` | JSON |
| Soft tissue (Holly) | `{soft_tissue_root}/holly/VL{id}/point.{type}_{n}.json` | JSON |

**Anatomical JSON structure:**
```json
{
  "bodies": {
    "Jiali-test": {
      "landmarks": {
        "nipple-l": {"3d_position": {"x": ..., "y": ..., "z": ...}},
        "nipple-r": {"3d_position": {...}},
        "sternal-superior": {"3d_position": {...}},
        "sternal-inferior": {"3d_position": {...}}
      }
    }
  }
}
```

**Soft tissue JSON structure:**
```json
{
  "status": "accepted",
  "type": "cyst",
  "prone_point": {"point": {"x": ..., "y": ..., "z": ...}},
  "supine_point": {"point": {"x": ..., "y": ..., "z": ...}}
}
```

### Stage 2: Correspondence & Distances (`utils.py`)

**Landmark correspondence** (`find_corresponding_landmarks`):
1. For each landmark in Anthony's prone: find Holly's prone landmarks within 3mm
2. Accept only unique 1-to-1 matches
3. Verify same pair is also within 3mm in supine
4. Exclude fibroadenoma landmarks
5. Return: `{vl_id: [["cyst_1", "cyst_1"], ...]}`

**Average landmarks** (`add_averaged_landmarks`):
- For each corresponding pair: `average = (anthony_coord + holly_coord) / 2`
- Adds `"average"` key to `registrar_data` in each scan

**Distance calculation** (`calculate_landmark_distances`):
- Load segmentation masks (skin, rib cage) from NIfTI files
- For each landmark: find closest point on skin surface and rib cage surface
- Compute 10-nearest-neighbor average distance as density metric

**Clock-face coordinates** (`calculate_clockface_coordinates`):
- Project landmark position relative to nipple onto axial plane
- Convert to clock time (12 o'clock = superior)
- Assign quadrant: UO (upper outer), UI (upper inner), LO (lower outer), LI (lower inner)

### Stage 3: Alignment (`alignment.py`)

**Entry point:** `align_prone_to_supine_optimal(subject, prone_ribcage_mesh_path, supine_ribcage_seg_path, ...)`

**Algorithm (2-stage):**

#### Stage 3a: Initial Rotation (SVD)
1. Compute sternum SI axis in prone and supine: `v = sternum_inferior - sternum_superior`
2. Use Rodrigues' rotation formula to align prone SI axis onto supine SI axis
3. Center rotation around sternum superior (sternum superior → origin)

#### Stage 3b: ICP Refinement (Plane-to-Point)
1. Sample prone mesh surface at 26x26 resolution using morphic basis functions
2. Extract supine ribcage point cloud from NIfTI segmentation (20,000 boundary points)
3. Filter mutual region (only overlapping Z/Y ranges between mesh and point cloud)
4. Multi-start ICP with identity + Rodrigues initial rotations:
   - Find nearest neighbors via KD-tree (max radius configurable, default 15mm)
   - Trim worst correspondences (configurable, default 10%)
   - Solve rotation via SVD on normal equations
   - Convergence: RMSE change < 1e-6 or 200 iterations
   - Short-circuit: accept first result with RMSE < 4mm
5. Apply final transform to all prone data (mesh, sternum, nipples, landmarks)

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `selected_elements` | `[0,1,6,7,8,9,14,15,16,17,22,23]` | Anterior ribcage elements for alignment |
| `use_initial_rotation` | `True` | Use SVD pre-rotation |
| `mutual_region_padding_reciprocal` | `15` | Padding for region filtering |
| `max_correspondence_distance` | `1e6` | KD-tree search radius (mm) |
| `trim_percentage` | `0` | Fraction of worst correspondences to discard |
| `max_iterations` | `200` | ICP iteration limit |
| `convergence_threshold` | `1e-6` | RMSE convergence criterion |

#### Landmark Displacement Computation

After alignment, for each registrar (anthony, holly, average):

```
displacement_vector = supine_landmark - prone_landmark_transformed
displacement_magnitude = ||displacement_vector||

# Relative to sternum
prone_rel_sternum = prone_transformed - sternum_superior_prone_transformed
supine_rel_sternum = supine_landmark - sternum_superior_supine

# Relative to nipple
prone_rel_nipple = prone_transformed - nipple_prone_transformed
supine_rel_nipple = supine_landmark - nipple_supine
rel_nipple_vector = supine_rel_nipple - prone_rel_nipple
```

**Note:** Statistical analysis (ANOVA, correlations, BMI effects) is performed in a separate project that reads the Excel output from this pipeline.

---

## 5. File System Layout

### Input Data (Network Drives)

```
U:\projects\volunteer_camri\old_data\mri_t2\     # DICOM images
U:\projects\dashboard\picker_points\               # Soft tissue annotations
U:\sandbox\jxu759\volunteer_seg\results\           # Anatomical JSONs + segmentations
U:\sandbox\jxu759\volunteer_prone_mesh\            # Morphic mesh files
```

### Output Data (Local)

```
output/
├── landmark_results_v7_2026_03_10.xlsx            # Multi-sheet results workbook
├── alignment/
│   └── transformation_matrix_v7/
│       ├── VL00009_transform_matrix.npy           # 4x4 homogeneous matrix
│       ├── VL00009_alignment_metrics.json         # JSON sidecar with error metrics
│       ├── VL00011_transform_matrix.npy
│       ├── VL00011_alignment_metrics.json
│       └── ...
└── figs/
    ├── clock_analysis/                            # Polar/clock plots
    └── landmark vectors/                          # 3-panel displacement plots
```

### Excel Sheet Structure

**raw_data:**
| Column | Type | Description |
|--------|------|-------------|
| Registrar | str | "anthony", "holly" |
| VL_ID | str | "VL00009" |
| Age | str | Subject age |
| Height [m] | float | |
| Weight [kg] | float | |
| position | str | "prone" or "supine" |
| Landmark name | str | "cyst_1", "lymph_2", etc. |
| x, y, z | float | World coordinates (mm) |

**processed_r1_data / processed_r2_data / processed_ave_data:**
| Column Group | Columns | Source Stage |
|-------------|---------|-------------|
| Demographics | VL_ID, Age, Height, Weight, Landmark name, Landmark type | Stage 1 |
| Prone position | landmark side, Distance to nipple/skin/rib [mm], Time, Quadrant | Stage 2 |
| Supine position | (same as prone) | Stage 2 |
| Mask metrics | skin/rib neighborhood avg (prone & supine) | Stage 2 |
| Ribcage error | RMSE, mean, std (full mesh + anterior) | Stage 3 |
| Sternum positions | prone transformed x/y/z, supine x/y/z | Stage 3 |
| Nipple positions | left/right, prone transformed, supine (12 cols) | Stage 3 |
| Displacements | magnitude, vx/vy/vz (absolute + nipple-relative) | Stage 3 |

---

## 6. Module Dependency Graph

```
main.py / main_process_data.py / main_alignment.py
    ├── readers.py
    │   └── external/breast_metadata_mdv    (DICOM/NIfTI I/O)
    ├── structures.py                       (dataclasses)
    ├── utils.py
    │   ├── external/breast_metadata_mdv
    │   ├── scikit-image                    (contour extraction)
    │   ├── pandas / openpyxl               (Excel I/O)
    │   ├── save_alignment_metrics()        (JSON sidecar write)
    │   └── load_alignment_metrics()        (JSON sidecar read)
    ├── apply_saved_alignment.py
    │   ├── utils.py                        (metrics I/O, Excel save, correspondences)
    │   ├── alignment.py                    (compute_landmark_displacements)
    │   └── readers.py                      (load_subject)
    ├── alignment.py
    │   ├── alignment_preprocessing.py
    │   │   └── alignment_utils.py          (mesh element selection)
    │   ├── surface_to_point_alignment.py
    │   │   ├── morphic                     (mesh basis functions)
    │   │   └── scipy.spatial.KDTree        (nearest neighbor)
    │   ├── alignment_utils.py              (transforms, filtering)
    │   └── alignment_viz.py                (debug plots)
    └── utils_plot.py                       (PyVista 3D plots)
```

---

## 7. Transformation Matrix Format

The 4x4 homogeneous transformation matrix `T_total` transforms prone coordinates into the supine frame:

```
T_total = | R11  R12  R13  tx |
          | R21  R22  R23  ty |
          | R31  R32  R33  tz |
          |  0    0    0   1  |

Usage:
  point_supine = T_total @ [point_prone_x, point_prone_y, point_prone_z, 1]^T

Construction:
  T_total[:3, :3] = R                              # 3x3 rotation
  T_total[:3, 3]  = target_anchor - R @ source_anchor  # translation
  where source_anchor = sternum_superior_prone
        target_anchor = sternum_superior_supine
```

The inverse `T_prone_to_supine = inv(T_total)` maps supine back to prone.

### JSON Sidecar — Alignment Error Metrics

Saved alongside each `.npy` file as `VL{id}_alignment_metrics.json`:

```json
{
  "ribcage_error_rmse": 5.23,
  "ribcage_error_mean": 4.01,
  "ribcage_error_std": 3.35,
  "ribcage_inlier_rmse": 3.12,
  "ribcage_inlier_mean": 2.45,
  "ribcage_inlier_std": 1.98,
  "sternum_error": 0.87
}
```

Written by `save_alignment_metrics()` in `utils.py`, read by `load_alignment_metrics()`. These metrics are produced during alignment (mesh-to-cloud comparison) and cannot be recomputed from the saved T_matrix alone — hence the sidecar pattern.

---

## 8. Subject Cohort

- **Total subjects:** ~56 (VL00009 through VL00089, with gaps)
- **Batches:** processed in groups of ~15-17 subjects at a time
- **Exclusions:** subjects missing prone mesh or supine segmentation files
- **Registrars:** 2 human annotators (Anthony, Holly) + computed average
- **Landmarks per subject:** 5-15 soft tissue landmarks (varies)
- **Anatomical landmarks per subject:** 4 fixed (nipple L/R, sternum sup/inf)

---

## 9. Error Metrics

| Metric | Description | Typical Range |
|--------|-------------|--------------|
| Ribcage error RMSE | Full mesh-to-cloud distance after alignment | 3-8 mm |
| Ribcage anterior RMSE | Anterior elements only (inlier region) | 2-5 mm |
| Sternum error | Distance between aligned sternum points | 0-2 mm |
| Landmark displacement | Distance each landmark moves prone→supine | 5-50 mm |
| Correspondence threshold | Max distance for landmark matching between registrars | 3 mm |

---

## 10. Entry Points Summary

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `main.py` | Full combined pipeline | Network drives | Excel + .npy + plots |
| `main_process_data.py` | Stage 1+2 only (no alignment) | Network drives | Excel (columns A-V) |
| `main_alignment.py` | Stage 3 only (alignment) | Network drives | Excel (columns W+) + .npy |
| `apply_saved_alignment.py` | Load saved T_matrix, compute displacements, save to Excel, and visualize | .npy + .json + network drives | Excel + 3D plot |

**Note:** Statistical analysis (`analysis.py` and related plotting scripts) is maintained in a separate project that consumes the Excel output from this pipeline.
